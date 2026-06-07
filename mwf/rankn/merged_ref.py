"""
merged_ref.py — reference for the merged Silero-VAD + rank-N MWF enhancer.

This is the exact pipeline the single merged ONNX implements, written in
NumPy + onnxruntime so it runs and can be validated today, and so the PyTorch
export (build_merged.py) has a bit-for-bit spec to match:

    multichannel wav
        -> reference-mic chunks (64-sample context + 512 / 256)
        -> Silero VAD, SEQUENTIAL with LSTM state carry   (onnxruntime)
        -> speech / noise frame masks  (mapped chunk -> STFT frames)
        -> rank-N MWF (power iteration + conjugate gradient)   [rankn.py]
        -> beamform
        -> OMLSA post-suppressor                              [rankn.py]
        -> clean complex STFT  ->  iSTFT  ->  clean wav

The VAD state carry matters (running Silero chunk-independent disagrees with
the sequential result ~40% of the time), so the merged graph runs it as an
ONNX Scan; here we just loop.
"""

import os
import numpy as np
import onnxruntime as ort

import rankn as R


SILERO_DEFAULT = os.path.expanduser(
    "~/Projects/qwise/silero-vad/src/silero_vad/data/silero_vad.onnx")


def _vad_chunks(ref, fs):
    """Frame the reference mic into Silero inputs [n_chunks, ctx+win] with the
    rolling 64/32-sample context Silero expects."""
    win = 512 if fs == 16000 else 256
    ctx = 64 if fs == 16000 else 32
    n = len(ref) // win
    X = ref[:n * win].reshape(n, win).astype(np.float32)
    ctxs = np.concatenate([np.zeros((1, ctx), np.float32), X[:-1, -ctx:]], 0)
    return np.concatenate([ctxs, X], 1).astype(np.float32), win


def silero_probs(ref, fs, silero_path=SILERO_DEFAULT):
    """Sequential Silero VAD with state carry -> per-chunk speech probability."""
    sess = ort.InferenceSession(silero_path, providers=["CPUExecutionProvider"])
    vad_in, win = _vad_chunks(ref, fs)
    state = np.zeros((2, 1, 128), np.float32)
    sr = np.array(fs, np.int64)
    probs = np.empty(vad_in.shape[0], np.float32)
    for i in range(vad_in.shape[0]):
        out, state = sess.run(["output", "stateN"],
                              {"input": vad_in[i:i+1], "state": state, "sr": sr})
        probs[i] = out[0, 0]
    return probs, win


def masks_from_probs(probs, win, hop, n_frames, thr_hi=0.5, thr_lo=0.35):
    """Map per-chunk VAD probability to per-STFT-frame speech/noise masks."""
    speech = np.zeros(n_frames, bool)
    noise = np.zeros(n_frames, bool)
    for t in range(n_frames):
        c = min((t * hop) // win, len(probs) - 1)
        speech[t] = probs[c] >= thr_hi
        noise[t] = probs[c] <= thr_lo
    if speech.sum() < 3:                      # fallbacks keep covariances stable
        speech = probs_to_frames_(probs, win, hop, n_frames) >= np.percentile(
            probs_to_frames_(probs, win, hop, n_frames), 60)
    if noise.sum() < 3:
        noise = ~speech
    return speech, noise


def probs_to_frames_(probs, win, hop, n_frames):
    pf = np.empty(n_frames, np.float32)
    for t in range(n_frames):
        pf[t] = probs[min((t * hop) // win, len(probs) - 1)]
    return pf


def enhance_onnx(x, fs, vadmask_onnx, silero_path=SILERO_DEFAULT, opts=None):
    """Same pipeline as enhance(), but the spatial stage (masks + rank-N MWF +
    beamform) runs in the merged ONNX `rankn_vadmask_{rate}.onnx`. Silero,
    OMLSA and the STFT/iSTFT stay on the host."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    L = x.shape[0]
    cfg = R.config_for(fs)
    if opts:
        for k, v in opts.items():
            setattr(cfg, k, v)
    X = R.stft_multi(x, cfg.n_fft, cfg.hop)
    probs, _ = silero_probs(x[:, cfg.ref_mic], fs, silero_path)
    sess = ort.InferenceSession(vadmask_onnx, providers=["CPUExecutionProvider"])
    Yr, Yi, vprob = sess.run(["Yr", "Yi", "vprob"], {
        "probs": probs.astype(np.float32),
        "Xr": np.real(X).astype(np.float32),
        "Xi": np.imag(X).astype(np.float32)})
    Y = Yr.astype(np.float64) + 1j * Yi.astype(np.float64)
    if cfg.post_omlsa:
        Y = R.omlsa(Y, cfg.omlsa_floor_db, cfg.omlsa_alpha_dd,
                    cfg.omlsa_alpha_s, cfg.omlsa_alpha_d, cfg.omlsa_win_min)
    if getattr(cfg, "vad_gate", True):
        g = vad_gate(vprob, cfg.hop, fs, thr=getattr(cfg, "gate_thr", 0.5),
                     hangover_s=getattr(cfg, "gate_hangover_s", 0.2),
                     floor=getattr(cfg, "gate_floor", 0.0))
        Y = Y * g[None, :len(Y[0])]
    y = R.istft(Y, cfg.n_fft, cfg.hop, length=L)
    pk = np.max(np.abs(y))
    if pk > 0:
        y = y / pk * 0.9
    return y


def vad_gate(vprob, hop, fs, thr=0.5, hangover_s=0.2, floor=0.0, smooth_s=0.03):
    """Per-frame output gate from VAD probability: 1 during speech (+hangover),
    `floor` elsewhere, with a short raised-cosine smoothing to avoid clicks.
    vprob/gate are per STFT frame; hop/fs set the frame rate."""
    g = (np.asarray(vprob) >= thr).astype(np.float64)
    hg = max(1, int(round(hangover_s * fs / hop)))
    if hg > 1:                                   # dilate speech (max filter)
        gd = g.copy()
        for s in range(1, hg + 1):
            gd[:-s] = np.maximum(gd[:-s], g[s:])
            gd[s:] = np.maximum(gd[s:], g[:-s])
        g = gd
    k = max(1, int(round(smooth_s * fs / hop)))
    if k > 1:                                     # smooth edges
        g = np.convolve(g, np.ones(k) / k, mode="same")
    return floor + (1.0 - floor) * np.clip(g, 0, 1)


def enhance_single(x, fs, single_onnx, opts=None):
    """Run the FULLY merged ONNX (Silero VAD + rank-N MWF embedded in one graph).
    No host-side Silero call — only STFT in, OMLSA + iSTFT out."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    L = x.shape[0]
    cfg = R.config_for(fs)
    if opts:
        for k, v in opts.items():
            setattr(cfg, k, v)
    X = R.stft_multi(x, cfg.n_fft, cfg.hop)
    vad_in, _ = _vad_chunks(x[:, cfg.ref_mic], fs)
    sess = ort.InferenceSession(single_onnx, providers=["CPUExecutionProvider"])
    Yr, Yi, vprob = sess.run(["k_Yr", "k_Yi", "k_vprob"], {
        "vad_in": vad_in.astype(np.float32),
        "state0": np.zeros((2, 1, 128), np.float32),
        "k_Xr": np.real(X).astype(np.float32),
        "k_Xi": np.imag(X).astype(np.float32)})
    Y = Yr.astype(np.float64) + 1j * Yi.astype(np.float64)
    if cfg.post_omlsa:
        Y = R.omlsa(Y, cfg.omlsa_floor_db, cfg.omlsa_alpha_dd,
                    cfg.omlsa_alpha_s, cfg.omlsa_alpha_d, cfg.omlsa_win_min)
    if getattr(cfg, "vad_gate", True):           # silence non-speech regions
        g = vad_gate(vprob, cfg.hop, fs,
                     thr=getattr(cfg, "gate_thr", 0.5),
                     hangover_s=getattr(cfg, "gate_hangover_s", 0.2),
                     floor=getattr(cfg, "gate_floor", 0.0))
        Y = Y * g[None, :len(Y[0])]
    y = R.istft(Y, cfg.n_fft, cfg.hop, length=L)
    pk = np.max(np.abs(y))
    if pk > 0:
        y = y / pk * 0.9
    return y


def enhance(x, fs, silero_path=SILERO_DEFAULT, opts=None):
    """multichannel x [L,M] -> clean mono y [L], using Silero VAD for masks."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    L = x.shape[0]
    cfg = R.config_for(fs)
    if opts:
        for k, v in opts.items():
            setattr(cfg, k, v)

    X = R.stft_multi(x, cfg.n_fft, cfg.hop)                    # [M,F,T]
    n_frames = X.shape[2]

    probs, win = silero_probs(x[:, cfg.ref_mic], fs, silero_path)
    speech, noise = masks_from_probs(probs, win, cfg.hop, n_frames)

    Rss, Rnn = R.covariances(X, speech, noise, cfg.eps_reg)
    W = R.rankn_weights(Rss, Rnn, cfg.ref_mic, cfg.mu,
                        cfg.power_iters, cfg.cg_iters, cfg.eps_reg)
    Y = R.beamform(W, X)
    if cfg.post_omlsa:
        Y = R.omlsa(Y, cfg.omlsa_floor_db, cfg.omlsa_alpha_dd,
                    cfg.omlsa_alpha_s, cfg.omlsa_alpha_d, cfg.omlsa_win_min)
    y = R.istft(Y, cfg.n_fft, cfg.hop, length=L)
    pk = np.max(np.abs(y))
    if pk > 0:
        y = y / pk * 0.9
    return y, speech, noise


# --------------------------------------------------------------------------
#  self-test on a real-speech multichannel mixture
# --------------------------------------------------------------------------
def _si_sdr(est, ref):
    m = min(len(est), len(ref))
    est = est[:m] - est[:m].mean(); ref = ref[:m] - ref[:m].mean()
    a = (est @ ref) / (ref @ ref + 1e-12); t = a * ref
    return 10 * np.log10((t @ t) / (((est - t) @ (est - t)) + 1e-12) + 1e-12)


if __name__ == "__main__":
    import soundfile as sf, scipy.signal as ss
    base = os.path.dirname(os.path.abspath(__file__))
    speech_wav = os.path.expanduser("~/Projects/qwise/silero-vad/tests/data/test.wav")
    for fs in (16000, 8000):
        sp, sr0 = sf.read(speech_wav)
        if sp.ndim > 1:
            sp = sp.mean(1)
        if sr0 != fs:
            sp = ss.resample(sp, int(len(sp) * fs / sr0))
        sp = sp[:fs * 8] / (np.std(sp[:fs * 8]) + 1e-9)
        t = np.arange(len(sp)) / fs
        rotor = sum(np.sin(2*np.pi*133*h*t + h) for h in range(1, 8))
        rotor = rotor / (np.std(rotor) + 1e-9)
        M = 3
        ds = np.linspace(0, 2.0, M); dn = np.linspace(0, 1.0, M) + 0.6
        mix = np.zeros((len(sp), M))
        for m in range(M):
            mix[:, m] = (np.interp(t - ds[m]/fs, t, sp)
                         + 0.9 * np.interp(t - dn[m]/fs, t, rotor))
        ref = np.interp(t - ds[0]/fs, t, sp)
        mix /= np.max(np.abs(mix)) + 1e-9
        y, sm, nm = enhance(mix, fs)
        print(f"fs={fs}: speech frames={int(sm.sum())} noise frames={int(nm.sum())}  "
              f"SI-SDR in={_si_sdr(mix[:, 0], ref):.1f} -> out={_si_sdr(y, ref):.1f} dB "
              f"({_si_sdr(y, ref) - _si_sdr(mix[:, 0], ref):+.1f})")
