"""
rankn.py — NumPy reference for the Q-WiSE rank-N MWF speech enhancer.

This mirrors the MATLAB implementation in
    acoustic_simulator/mwf/{mwf.m, mwf_compute_rankn_weights.m, mwf_omlsa_postfilter.m}

Pipeline (STFT domain), per utterance:
    1. multichannel STFT
    2. speech / noise frame masks (energy VAD)
    3. masked covariances  Rss (speech frames), Rnn (noise frames)   [PSD]
    4. rank-N MWF weights, eig/inverse-free:
           (phi_s, d) : dominant eigenpair of Rss via POWER ITERATION
                        (reference-column init)               -> no eig
           x = Rnn^{-1} d : complex CONJUGATE GRADIENT        -> no inverse
           w = phi_s * conj(d_ref) * x / (mu + phi_s * d^H x)
       (mu = 1, power_iters = 0 reproduces mwf_2mic / mwf_3mic for N = 2,3)
    5. beamform   Y = w^H X
    6. OMLSA single-channel post-suppressor (near-zero residual)
    7. iSTFT

The covariance + weight + beamform stage (steps 3-5) is what is exported to
ONNX (see build_onnx.py). STFT, VAD and OMLSA stay on the host here; they are
standard DSP and trivially portable.

All array conventions match the MATLAB code. Weights are applied as Y = w^H x.
"""

from dataclasses import dataclass
import numpy as np

try:
    from scipy.special import exp1 as _exp1
except Exception:  # pragma: no cover - fallback if scipy missing
    def _exp1(x):
        # series/continued-fraction-free crude fallback (not used if scipy present)
        x = np.asarray(x, dtype=np.float64)
        g = 0.5772156649015329
        out = -g - np.log(np.maximum(x, 1e-12))
        term = x.copy()
        s = x.copy()
        for k in range(2, 40):
            term = -term * x * (k - 1) / (k * k)
            s = s + term
        return np.maximum(out + s, 0.0)


# ---------------------------------------------------------------------------
#  Configuration (per sample rate)
# ---------------------------------------------------------------------------
@dataclass
class RankNConfig:
    fs: int = 16000
    n_fft: int = 512          # STFT window length
    hop: int = 128            # STFT hop
    ref_mic: int = 0          # 0-based reference microphone
    mu: float = 1.0           # SDW trade-off (>=1 = more suppression)
    power_iters: int = 3      # rank-1 power-iteration refinement steps
    cg_iters: int = 8         # conjugate-gradient iterations (>= n_mics is exact)
    eps_reg: float = 1e-5     # numerical floor / diagonal loading on Rnn
                              # (>= ~1e-6 keeps the fp32 ONNX path well-conditioned)
    # OMLSA
    post_omlsa: bool = True
    omlsa_floor_db: float = -30.0
    omlsa_alpha_dd: float = 0.92
    omlsa_alpha_s: float = 0.90
    omlsa_alpha_d: float = 0.85
    omlsa_win_min: int = 60
    # energy VAD
    vad_speech_pct: float = 60.0   # frames above this energy pct -> speech
    vad_noise_pct: float = 35.0    # frames below this energy pct -> noise


def config_for(fs: int) -> RankNConfig:
    """Return the standard config for a supported sample rate (8 k / 16 k)."""
    if fs == 16000:
        return RankNConfig(fs=16000, n_fft=512, hop=128)
    if fs == 8000:
        # half the FFT/hop -> same ~32 ms / 8 ms time-frequency resolution
        return RankNConfig(fs=8000, n_fft=256, hop=64)
    raise ValueError(f"Unsupported sample rate {fs}; use 8000 or 16000.")


# ---------------------------------------------------------------------------
#  STFT / iSTFT  (sqrt-Hann, COLA)
# ---------------------------------------------------------------------------
def _window(n_fft):
    # periodic Hann, square-rooted (analysis == synthesis), COLA at 75 % overlap
    n = np.arange(n_fft)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * n / n_fft)
    return np.sqrt(hann)


def stft_multi(x, n_fft, hop):
    """x: [L, M] -> X: [M, F, T] complex."""
    if x.ndim == 1:
        x = x[:, None]
    L, M = x.shape
    win = _window(n_fft)
    n_frames = 1 + max(0, (L - n_fft) // hop)
    F = n_fft // 2 + 1
    X = np.zeros((M, F, n_frames), dtype=np.complex128)
    for m in range(M):
        for t in range(n_frames):
            seg = x[t * hop: t * hop + n_fft, m] * win
            X[m, :, t] = np.fft.rfft(seg, n_fft)
    return X


def istft(Y, n_fft, hop, length=None):
    """Y: [F, T] complex -> y: [L] real (overlap-add, COLA normalized)."""
    win = _window(n_fft)
    F, T = Y.shape
    L = (T - 1) * hop + n_fft
    out = np.zeros(L)
    den = np.zeros(L)
    for t in range(T):
        seg = np.fft.irfft(Y[:, t], n_fft) * win
        out[t * hop: t * hop + n_fft] += seg
        den[t * hop: t * hop + n_fft] += win ** 2
    out = out / np.maximum(den, 1e-8)
    if length is not None:
        out = out[:length]
    return out


# ---------------------------------------------------------------------------
#  Energy VAD -> speech / noise frame masks
# ---------------------------------------------------------------------------
def energy_masks(X, ref_mic, speech_pct=60.0, noise_pct=35.0):
    """Return (speech_mask, noise_mask) over frames from the reference mic."""
    P = np.sum(np.abs(X[ref_mic]) ** 2, axis=0)        # [T] frame power
    Pdb = 10 * np.log10(P + 1e-12)
    hi = np.percentile(Pdb, speech_pct)
    lo = np.percentile(Pdb, noise_pct)
    speech = Pdb >= hi
    noise = Pdb <= lo
    if speech.sum() < 3:
        speech = Pdb >= np.percentile(Pdb, 50)
    if noise.sum() < 3:
        noise = Pdb <= np.percentile(Pdb, 50)
    return speech, noise


# ---------------------------------------------------------------------------
#  Covariances (masked average -> PSD, streaming-consistent)
# ---------------------------------------------------------------------------
def covariances(X, speech_mask, noise_mask, eps_reg):
    """X: [M,F,T] -> Rss, Rnn: [F,M,M] complex (Hermitian PSD)."""
    M, F, T = X.shape
    Xf = np.transpose(X, (1, 0, 2))        # [F, M, T]
    ms = speech_mask.astype(np.float64)
    mn = noise_mask.astype(np.float64)
    sw = max(ms.sum(), 1.0)
    nw = max(mn.sum(), 1.0)
    Xs = Xf * ms[None, None, :]
    Xn = Xf * mn[None, None, :]
    Rss = np.matmul(Xs, np.conj(np.transpose(Xf, (0, 2, 1)))) / sw       # [F,M,M]
    Rnn = np.matmul(Xn, np.conj(np.transpose(Xf, (0, 2, 1)))) / nw
    I = np.eye(M)[None]
    Rss = 0.5 * (Rss + np.conj(np.transpose(Rss, (0, 2, 1))))
    Rnn = 0.5 * (Rnn + np.conj(np.transpose(Rnn, (0, 2, 1)))) + eps_reg * I
    return Rss, Rnn


# ---------------------------------------------------------------------------
#  rank-N MWF weights (eig/inverse-free, batched over frequency)
# ---------------------------------------------------------------------------
def _power_iteration(Rs, ref, n_iter, eps):
    """Dominant eigenpair of Hermitian Rs [F,M,M]; returns phi_s [F], d [F,M]
    with d[:, ref] = 1. ref-column init; n_iter refinement steps."""
    F, M, _ = Rs.shape
    u = Rs[:, :, ref].copy()                          # [F, M]
    nu = np.linalg.norm(u, axis=1, keepdims=True)
    safe = nu[:, 0] >= eps
    u = np.where(nu >= eps, u / np.maximum(nu, eps), u)
    e_ref = np.zeros((F, M), dtype=Rs.dtype)
    e_ref[:, ref] = 1.0
    u = np.where(safe[:, None], u, e_ref)
    for _ in range(n_iter):
        y = np.einsum('fij,fj->fi', Rs, u)
        ny = np.linalg.norm(y, axis=1, keepdims=True)
        u = np.where(ny >= eps, y / np.maximum(ny, eps), u)
    # Rayleigh quotient lam = u^H Rs u  (||u|| = 1)
    Rsu = np.einsum('fij,fj->fi', Rs, u)
    lam = np.real(np.sum(np.conj(u) * Rsu, axis=1))   # [F]
    uref = u[:, ref]                                  # [F]
    good = np.abs(uref) > eps
    d = np.where(good[:, None], u / uref[:, None], u)
    phi = np.where(good, np.maximum(lam, 0.0) * np.abs(uref) ** 2,
                   np.maximum(lam, 0.0))
    return phi, d


def _cg_solve(A, b, K, eps):
    """Solve A x = b for Hermitian PD A [F,M,M], b [F,M] by complex CG.
    Exact in <= M iterations. Batched over F."""
    F, M, _ = A.shape
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = np.real(np.sum(np.conj(r) * r, axis=1))      # [F]
    rs0 = np.maximum(rs, eps ** 2)
    for _ in range(K):
        Ap = np.einsum('fij,fj->fi', A, p)
        pAp = np.real(np.sum(np.conj(p) * Ap, axis=1))     # [F]
        a = rs / np.maximum(pAp, eps)
        x = x + a[:, None] * p
        r = r - a[:, None] * Ap
        rs_new = np.real(np.sum(np.conj(r) * r, axis=1))
        beta = rs_new / np.maximum(rs, eps)
        p = r + beta[:, None] * p
        rs = rs_new
    return x


def rankn_weights(Rss, Rnn, ref, mu, power_iters, cg_iters, eps):
    """Rss, Rnn: [F,M,M] -> W: [F,M] complex.  Y = w^H x convention."""
    F, M, _ = Rss.shape
    phi, d = _power_iteration(Rss, ref, power_iters, eps)
    x = _cg_solve(Rnn, d, cg_iters, eps)
    eta = np.sum(np.conj(d) * x, axis=1)              # d^H x  [F]
    dr = d[:, ref]                                    # [F]
    den = mu + phi * eta
    scale = np.where(np.abs(den) > eps, phi * np.conj(dr) / den, 0.0)
    W = scale[:, None] * x                            # [F, M]
    # fallback to e_ref where degenerate
    bad = ~np.isfinite(W).all(axis=1) | (np.abs(den) <= eps)
    if np.any(bad):
        e = np.zeros((F, M), dtype=W.dtype)
        e[:, ref] = 1.0
        W = np.where(bad[:, None], e, W)
    return W


def beamform(W, X):
    """W: [F,M], X: [M,F,T] -> Y: [F,T] = w^H x per bin."""
    return np.einsum('fm,mft->ft', np.conj(W), X)


# ---------------------------------------------------------------------------
#  OMLSA post-suppressor (exact, recursive) — host-side reference
# ---------------------------------------------------------------------------
def omlsa(Y, floor_db=-30.0, a_dd=0.92, a_s=0.90, a_d=0.85, win_min=60):
    F, T = Y.shape
    P = np.abs(Y) ** 2
    Gmin = 10 ** (floor_db / 20)
    n0 = min(8, T)
    lam = np.maximum(P[:, :n0].mean(axis=1), 1e-10)
    S = lam.copy()
    Gp = np.ones(F)
    gam_p = np.ones(F)
    minbuf = []
    out = np.zeros_like(Y)
    for t in range(T):
        p = P[:, t]
        S = a_s * S + (1 - a_s) * p
        minbuf.append(S.copy())
        if len(minbuf) > win_min:
            minbuf.pop(0)
        Smin = np.min(np.stack(minbuf, 0), axis=0)
        Sr = S / np.maximum(Smin, 1e-12)
        ppres = np.clip((Sr - 1.0) / 8.0, 0, 1)
        lam = np.maximum(lam + (1 - ppres) * (a_d * lam + (1 - a_d) * p - lam), 1e-12)
        gamma = p / lam
        xi = np.maximum(a_dd * Gp ** 2 * gam_p + (1 - a_dd) * np.maximum(gamma - 1, 0), 1e-3)
        nu = np.clip(xi / (1 + xi) * gamma, 1e-6, 500)
        G_lsa = xi / (1 + xi) * np.exp(0.5 * _exp1(nu))
        G = np.clip(G_lsa ** ppres * Gmin ** (1 - ppres), Gmin, 1.0)
        out[:, t] = G * Y[:, t]
        Gp = G
        gam_p = gamma
    return out


# ---------------------------------------------------------------------------
#  Top-level enhancement
# ---------------------------------------------------------------------------
def enhance(x, cfg: RankNConfig):
    """x: [L] or [L, M] time-domain -> y: [L] enhanced mono."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    L = x.shape[0]
    X = stft_multi(x, cfg.n_fft, cfg.hop)                       # [M,F,T]
    speech, noise = energy_masks(X, cfg.ref_mic, cfg.vad_speech_pct, cfg.vad_noise_pct)
    Rss, Rnn = covariances(X, speech, noise, cfg.eps_reg)
    W = rankn_weights(Rss, Rnn, cfg.ref_mic, cfg.mu,
                      cfg.power_iters, cfg.cg_iters, cfg.eps_reg)
    Y = beamform(W, X)                                          # [F,T]
    if cfg.post_omlsa:
        Y = omlsa(Y, cfg.omlsa_floor_db, cfg.omlsa_alpha_dd,
                  cfg.omlsa_alpha_s, cfg.omlsa_alpha_d, cfg.omlsa_win_min)
    y = istft(Y, cfg.n_fft, cfg.hop, length=L)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.9
    return y


def enhance_with_onnx(x, cfg: RankNConfig, onnx_session):
    """Same as enhance() but the covariance + weights + beamform stage runs in
    ONNX. STFT, VAD and OMLSA stay on the host."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    L = x.shape[0]
    X = stft_multi(x, cfg.n_fft, cfg.hop)
    speech, noise = energy_masks(X, cfg.ref_mic, cfg.vad_speech_pct, cfg.vad_noise_pct)
    feeds = {
        "Xr": np.real(X).astype(np.float32),
        "Xi": np.imag(X).astype(np.float32),
        "ms": speech.astype(np.float32),
        "mn": noise.astype(np.float32),
    }
    Yr, Yi = onnx_session.run(["Yr", "Yi"], feeds)
    Y = Yr.astype(np.float64) + 1j * Yi.astype(np.float64)
    if cfg.post_omlsa:
        Y = omlsa(Y, cfg.omlsa_floor_db, cfg.omlsa_alpha_dd,
                  cfg.omlsa_alpha_s, cfg.omlsa_alpha_d, cfg.omlsa_win_min)
    y = istft(Y, cfg.n_fft, cfg.hop, length=L)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.9
    return y
