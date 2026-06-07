"""
build_merged.py — export the merged VAD-driven rank-N MWF stage to ONNX.

Why this shape (read me)
------------------------
A *single* ONNX that also contains Silero is not practical: Silero's ONNX is a
wrapper of nested `If` subgraphs holding the real network in conditional
branches, and wrapping that in a `Scan` makes onnxruntime crash; its TorchScript
model hides the LSTM state, so torch.onnx cannot trace it either. That is why
every Silero example loops the model on the host. We follow the same pattern.

So the merge happens at the level that is robust: Silero runs as its own model
(host loop, carrying state) and feeds its per-chunk speech probabilities into
ONE ONNX that does everything spatial:

    inputs : probs [N]          Silero speech probability per VAD chunk
             Xr, Xi [M, F, T]   multichannel STFT (real / imag)
    graph  : probs -> per-frame speech/noise masks
             masked covariances -> rank-N MWF (power-iter + CG) -> beamform
    outputs: Yr, Yi [F, T]      clean beamformed STFT ("clean signal matrix")

OMLSA and the iSTFT stay on the host (ONNX has no inverse-STFT op); see
run_merged.py / merged_ref.py. Everything in the graph is plain ONNX ops
(matmul / einsum / gather / scalar) — no eig, no inverse, no control flow.

    pip install numpy onnx onnxruntime spox
    python build_merged.py        # writes rankn_vadmask_16k.onnx and _8k.onnx
"""

import os
import numpy as np
import onnx
from spox import argument, build, Tensor
import spox.opset.ai.onnx.v18 as op

import build_onnx as B          # shared helpers: c, ci, rsum, add, sub, mul, div, relu, cmatvec
from rankn import config_for

THR_HI, THR_LO = 0.5, 0.35      # speech / noise probability thresholds


def build_graph_vadmask(fs, ref_mic=0, mu=1.0, power_iters=3, cg_iters=8, eps=1e-5):
    cfg = config_for(fs)
    hop = cfg.hop
    vad_win = 512 if fs == 16000 else 256
    EPS = B.c(eps)

    Xr = argument(Tensor(np.float32, ("M", "F", "T")))
    Xi = argument(Tensor(np.float32, ("M", "F", "T")))
    probs = argument(Tensor(np.float32, ("N",)))

    # ---- per-frame VAD probability -> speech / noise masks ----------------
    T = op.gather(op.shape(Xr), B.ci([2]))
    N = op.gather(op.shape(probs), B.ci([0]))
    tidx = op.range(B.ci(0), op.squeeze(T, B.ci([0])), B.ci(1))      # [T]
    cidx = op.div(op.mul(tidx, B.ci(hop)), B.ci(vad_win))            # (t*hop)//win
    cidx = op.min([cidx, op.reshape(op.sub(N, B.ci([1])), B.ci([]))])  # clamp <= N-1
    pf = op.gather(probs, cidx, axis=0)                              # [T]
    ms = op.cast(op.greater_or_equal(pf, B.c(THR_HI)), to=np.float32)
    mn = op.cast(op.less_or_equal(pf, B.c(THR_LO)), to=np.float32)

    # ---- masked covariances (Hermitian PSD) ------------------------------
    Xf_r = op.transpose(Xr, perm=[1, 0, 2]); Xf_i = op.transpose(Xi, perm=[1, 0, 2])
    XhT_r = op.transpose(Xf_r, perm=[0, 2, 1])
    XhT_i = op.transpose(op.mul(Xf_i, B.c(-1.0)), perm=[0, 2, 1])

    def cov(mask):
        w3 = op.reshape(mask, B.ci([1, 1, -1]))
        Xw_r = B.mul(Xf_r, w3); Xw_i = B.mul(Xf_i, w3)
        R_r = B.sub(op.matmul(Xw_r, XhT_r), op.matmul(Xw_i, XhT_i))
        R_i = B.add(op.matmul(Xw_r, XhT_i), op.matmul(Xw_i, XhT_r))
        wsum = op.max([B.rsum(mask, 0), B.c(1.0)])
        R_r = B.div(R_r, wsum); R_i = B.div(R_i, wsum)
        R_r = B.mul(B.add(R_r, op.transpose(R_r, perm=[0, 2, 1])), B.c(0.5))
        R_i = B.mul(B.sub(R_i, op.transpose(R_i, perm=[0, 2, 1])), B.c(0.5))
        return R_r, R_i

    Rss_r, Rss_i = cov(ms)
    Rnn_r, Rnn_i = cov(mn)
    R0 = op.gather(Rnn_r, B.ci(0), axis=0)
    I3 = op.unsqueeze(op.eye_like(R0), axes=B.ci([0]))
    Rnn_r = B.add(Rnn_r, B.mul(I3, EPS))

    # ---- rank-1 model (power iteration) ----------------------------------
    ur = op.gather(Rss_r, B.ci(ref_mic), axis=2); ui = op.gather(Rss_i, B.ci(ref_mic), axis=2)
    nrm = op.sqrt(B.rsum(B.add(B.mul(ur, ur), B.mul(ui, ui)), 1))
    ur = B.div(ur, B.add(nrm, EPS)); ui = B.div(ui, B.add(nrm, EPS))
    for _ in range(power_iters):
        yr, yi = B.cmatvec(Rss_r, Rss_i, ur, ui)
        nrm = op.sqrt(B.rsum(B.add(B.mul(yr, yr), B.mul(yi, yi)), 1))
        ur = B.div(yr, B.add(nrm, EPS)); ui = B.div(yi, B.add(nrm, EPS))
    Ru_r, Ru_i = B.cmatvec(Rss_r, Rss_i, ur, ui)
    lam = B.rsum(B.add(B.mul(ur, Ru_r), B.mul(ui, Ru_i)), 1)
    uref_r = op.unsqueeze(op.gather(ur, B.ci(ref_mic), axis=1), axes=B.ci([1]))
    uref_i = op.unsqueeze(op.gather(ui, B.ci(ref_mic), axis=1), axes=B.ci([1]))
    a2 = B.add(B.add(B.mul(uref_r, uref_r), B.mul(uref_i, uref_i)), EPS)
    dr = B.div(B.add(B.mul(ur, uref_r), B.mul(ui, uref_i)), a2)
    di = B.div(B.sub(B.mul(ui, uref_r), B.mul(ur, uref_i)), a2)
    phi = B.mul(B.relu(lam), B.sub(a2, EPS))

    # ---- x = Rnn^-1 d via conjugate gradient -----------------------------
    xr = B.mul(dr, B.c(0.0)); xi = B.mul(di, B.c(0.0))
    rr, ri = dr, di; pr, pi = dr, di
    rs = B.rsum(B.add(B.mul(rr, rr), B.mul(ri, ri)), 1)
    for _ in range(cg_iters):
        Apr, Api = B.cmatvec(Rnn_r, Rnn_i, pr, pi)
        pAp = B.rsum(B.add(B.mul(pr, Apr), B.mul(pi, Api)), 1)
        a = B.div(rs, B.add(pAp, EPS))
        xr = B.add(xr, B.mul(a, pr)); xi = B.add(xi, B.mul(a, pi))
        rr = B.sub(rr, B.mul(a, Apr)); ri = B.sub(ri, B.mul(a, Api))
        rsn = B.rsum(B.add(B.mul(rr, rr), B.mul(ri, ri)), 1)
        beta = B.div(rsn, B.add(rs, EPS))
        pr = B.add(rr, B.mul(beta, pr)); pi = B.add(ri, B.mul(beta, pi)); rs = rsn

    # ---- SDW rank-1 weight + beamform ------------------------------------
    eta_r = B.rsum(B.add(B.mul(dr, xr), B.mul(di, xi)), 1)
    eta_i = B.rsum(B.sub(B.mul(dr, xi), B.mul(di, xr)), 1)
    dref_r = op.unsqueeze(op.gather(dr, B.ci(ref_mic), axis=1), axes=B.ci([1]))
    dref_i = op.unsqueeze(op.gather(di, B.ci(ref_mic), axis=1), axes=B.ci([1]))
    den_r = B.add(B.c(mu), B.mul(phi, eta_r)); den_i = B.mul(phi, eta_i)
    num_r = B.mul(phi, dref_r); num_i = B.mul(phi, B.mul(dref_i, B.c(-1.0)))
    den2 = B.add(B.add(B.mul(den_r, den_r), B.mul(den_i, den_i)), EPS)
    sc_r = B.div(B.add(B.mul(num_r, den_r), B.mul(num_i, den_i)), den2)
    sc_i = B.div(B.sub(B.mul(num_i, den_r), B.mul(num_r, den_i)), den2)
    Wr = B.sub(B.mul(sc_r, xr), B.mul(sc_i, xi))
    Wi = B.add(B.mul(sc_r, xi), B.mul(sc_i, xr))

    Yr = B.add(op.einsum([Wr, Xr], equation="fm,mft->ft"),
               op.einsum([Wi, Xi], equation="fm,mft->ft"))
    Yi = B.sub(op.einsum([Wr, Xi], equation="fm,mft->ft"),
               op.einsum([Wi, Xr], equation="fm,mft->ft"))

    # Also expose the per-frame VAD probability so the host can build the
    # output gate (applied AFTER OMLSA) — needed in the single-file path where
    # the host never sees the raw Silero output.
    vprob = pf                                                  # [T]
    return build({"probs": probs, "Xr": Xr, "Xi": Xi},
                 {"Yr": Yr, "Yi": Yi, "vprob": vprob})


def export(path, fs):
    m = build_graph_vadmask(fs)
    cfg = config_for(fs)
    for k, v in {"fs": str(fs), "n_fft": str(cfg.n_fft), "hop": str(cfg.hop),
                 "vad_win": str(512 if fs == 16000 else 256),
                 "stage": "vad-prob -> masks -> rankN MWF -> beamformed STFT; "
                          "host runs silero loop + omlsa + istft"}.items():
        e = m.metadata_props.add(); e.key, e.value = k, v
    onnx.checker.check_model(m)
    onnx.save(m, path)
    print(f"wrote {path} (fs={fs}, n_fft={cfg.n_fft}, hop={cfg.hop})")


def _set_opset(m, v=18):
    from onnx import helper
    del m.opset_import[:]
    m.opset_import.append(helper.make_opsetid("", v))
    return m


SAFETENSORS_16K = os.path.expanduser(
    "~/Projects/qwise/silero-vad/src/silero_vad/data/silero_vad_16k.safetensors")


def export_single_file(path, fs, safetensors_path=SAFETENSORS_16K):
    """Fuse Silero VAD + rank-N MWF into ONE ONNX:

        inputs : vad_in [N, ctx+win]   reference-mic chunks (64+512 @16k)
                 state0 [2,1,128]       (zeros)
                 k_Xr, k_Xi [M,F,T]     multichannel STFT
        outputs: k_Yr, k_Yi [F,T]       clean beamformed STFT

    The whole 'signal -> silero-vad -> rank-N MWF -> clean matrix' flow runs
    inside the graph. STFT (host) feeds vad_in + Xr/Xi; OMLSA + iSTFT on host.
    Needs the matching Silero weights (silero_vad_16k.safetensors).
    """
    import onnx
    from onnx import compose
    import silero_onnx as SO
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(
            f"Silero weights not found: {safetensors_path}\n"
            "Single-file embedding needs the safetensors weights for this rate.")
    vad = _set_opset(SO.build_silero_scan_model(safetensors_path, fs))
    mwf = _set_opset(compose.add_prefix(build_graph_vadmask(fs), prefix="k_"))
    merged = compose.merge_models(vad, mwf, io_map=[("probs", "k_probs")])
    for k, v in {"fs": str(fs), "stage": "signal -> silero-vad(scan) -> "
                 "rankN MWF -> clean STFT (single file)"}.items():
        e = merged.metadata_props.add(); e.key, e.value = k, v
    onnx.checker.check_model(merged)
    onnx.save(merged, path)
    print(f"wrote {path} (fs={fs}, single-file VAD+MWF, "
          f"{os.path.getsize(path)//1024} KB)")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    # masks-input models (silero looped on host) — both rates
    export(os.path.join(here, "rankn_vadmask_16k.onnx"), 16000)
    export(os.path.join(here, "rankn_vadmask_8k.onnx"), 8000)
    # single fused file with Silero embedded (16 kHz; needs 16k safetensors)
    try:
        export_single_file(os.path.join(here, "rankn_vad_16k.onnx"), 16000)
    except FileNotFoundError as e:
        print("skipped single-file export:", e)
