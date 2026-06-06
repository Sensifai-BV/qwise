"""
build_onnx.py — export the rank-N MWF spatial stage to ONNX.

The exported graph implements steps 3-5 of rankn.py entirely in real
arithmetic (complex values carried as (real, imag) pairs), so it maps onto
plain ONNX ops with no eig and no matrix inverse:

    inputs  : Xr, Xi  [M, F, T]   multichannel STFT (real / imag)
              ms, mn  [T]         speech / noise frame masks (0/1 or soft)
    compute : Rss, Rnn   masked covariances (Hermitian PSD)
              (phi_s, d) dominant eigenpair of Rss via POWER ITERATION
              x = Rnn^-1 d        via complex CONJUGATE GRADIENT
              w = phi_s conj(d_ref) x / (mu + phi_s d^H x)
    outputs : Yr, Yi  [F, T]      beamformed STFT (real / imag),  Y = w^H X

M, F and T are dynamic axes. cg_iters >= M makes the solve exact; the default
8 is exact for up to 8 microphones (raise it for larger arrays).

The STFT-domain math is sample-rate independent, but we export one file per
rate (rankn_8k.onnx, rankn_16k.onnx) with the matching STFT parameters stored
in the model metadata so the host (demo.py) picks n_fft / hop automatically.

Usage:
    python build_onnx.py            # writes rankn_8k.onnx and rankn_16k.onnx
"""

import numpy as np
import onnx
from spox import argument, build, Tensor
import spox.opset.ai.onnx.v18 as op

from rankn import config_for


# --- small helpers on spox Vars -------------------------------------------
F32 = np.float32


def c(v):
    return op.const(np.asarray(v, dtype=F32))


def ci(v):
    return op.const(np.asarray(v, dtype=np.int64))


def rsum(x, axis):
    return op.reduce_sum(x, axes=ci([axis]), keepdims=1)


def add(*xs):
    y = xs[0]
    for x in xs[1:]:
        y = op.add(y, x)
    return y


def sub(a, b):
    return op.sub(a, b)


def mul(*xs):
    y = xs[0]
    for x in xs[1:]:
        y = op.mul(y, x)
    return y


def div(a, b):
    return op.div(a, b)


def relu(x):
    return op.max([x, c(0.0)])


def cmatvec(Ar, Ai, vr, vi):
    """(Ar+iAi) @ (vr+ivi), A:[F,M,M], v:[F,M] -> [F,M] (real, imag)."""
    yr = sub(op.einsum([Ar, vr], equation="fij,fj->fi"),
             op.einsum([Ai, vi], equation="fij,fj->fi"))
    yi = add(op.einsum([Ar, vi], equation="fij,fj->fi"),
             op.einsum([Ai, vr], equation="fij,fj->fi"))
    return yr, yi


def build_graph(ref_mic=0, mu=1.0, power_iters=3, cg_iters=8, eps=1e-5):
    Xr = argument(Tensor(F32, ("M", "F", "T")))
    Xi = argument(Tensor(F32, ("M", "F", "T")))
    ms = argument(Tensor(F32, ("T",)))
    mn = argument(Tensor(F32, ("T",)))

    EPS = c(eps)

    # ---- to [F, M, T] -----------------------------------------------------
    Xf_r = op.transpose(Xr, perm=[1, 0, 2])
    Xf_i = op.transpose(Xi, perm=[1, 0, 2])
    w_s = op.reshape(ms, ci([1, 1, -1]))          # [1,1,T]
    w_n = op.reshape(mn, ci([1, 1, -1]))

    # conj-transpose of Xf over (M,T):  [F,T,M]
    XhT_r = op.transpose(Xf_r, perm=[0, 2, 1])
    XhT_i = op.transpose(op.mul(Xf_i, c(-1.0)), perm=[0, 2, 1])

    def cov(weight3, wsum_src):
        Xw_r = mul(Xf_r, weight3)
        Xw_i = mul(Xf_i, weight3)
        R_r = sub(op.matmul(Xw_r, XhT_r), op.matmul(Xw_i, XhT_i))
        R_i = add(op.matmul(Xw_r, XhT_i), op.matmul(Xw_i, XhT_r))
        wsum = op.max([rsum(wsum_src, 0), c(1.0)])      # scalar-ish [1]
        R_r = div(R_r, wsum)
        R_i = div(R_i, wsum)
        # Hermitian symmetrize
        R_r = mul(add(R_r, op.transpose(R_r, perm=[0, 2, 1])), c(0.5))
        R_i = mul(sub(R_i, op.transpose(R_i, perm=[0, 2, 1])), c(0.5))
        return R_r, R_i

    Rss_r, Rss_i = cov(w_s, ms)
    Rnn_r, Rnn_i = cov(w_n, mn)

    # diagonal loading on Rnn: + eps * I  (I broadcast from a 2-D slice)
    R0 = op.gather(Rnn_r, ci(0), axis=0)            # [M,M]
    I2 = op.eye_like(R0)                            # [M,M]
    I3 = op.unsqueeze(I2, axes=ci([0]))            # [1,M,M]
    Rnn_r = add(Rnn_r, mul(I3, EPS))

    # ---- rank-1 model (phi_s, d) by power iteration -----------------------
    ur = op.gather(Rss_r, ci(ref_mic), axis=2)      # [F,M] reference column
    ui = op.gather(Rss_i, ci(ref_mic), axis=2)
    nrm = op.sqrt(rsum(add(mul(ur, ur), mul(ui, ui)), 1))   # [F,1]
    ur = div(ur, add(nrm, EPS))
    ui = div(ui, add(nrm, EPS))
    for _ in range(power_iters):
        yr, yi = cmatvec(Rss_r, Rss_i, ur, ui)
        nrm = op.sqrt(rsum(add(mul(yr, yr), mul(yi, yi)), 1))
        ur = div(yr, add(nrm, EPS))
        ui = div(yi, add(nrm, EPS))
    Ru_r, Ru_i = cmatvec(Rss_r, Rss_i, ur, ui)
    lam = rsum(add(mul(ur, Ru_r), mul(ui, Ru_i)), 1)        # [F,1] Rayleigh

    uref_r = op.unsqueeze(op.gather(ur, ci(ref_mic), axis=1), axes=ci([1]))  # [F,1]
    uref_i = op.unsqueeze(op.gather(ui, ci(ref_mic), axis=1), axes=ci([1]))
    absu2 = add(mul(uref_r, uref_r), mul(uref_i, uref_i))   # [F,1]
    absu2s = add(absu2, EPS)
    dr = div(add(mul(ur, uref_r), mul(ui, uref_i)), absu2s)
    di = div(sub(mul(ui, uref_r), mul(ur, uref_i)), absu2s)
    phi = mul(relu(lam), absu2)                              # [F,1]

    # ---- x = Rnn^-1 d  via complex conjugate gradient ---------------------
    xr = mul(dr, c(0.0))
    xi = mul(di, c(0.0))
    rr, ri = dr, di
    pr, pi = dr, di
    rs = rsum(add(mul(rr, rr), mul(ri, ri)), 1)             # [F,1]
    for _ in range(cg_iters):
        Apr, Api = cmatvec(Rnn_r, Rnn_i, pr, pi)
        pAp = rsum(add(mul(pr, Apr), mul(pi, Api)), 1)
        a = div(rs, add(pAp, EPS))
        xr = add(xr, mul(a, pr))
        xi = add(xi, mul(a, pi))
        rr = sub(rr, mul(a, Apr))
        ri = sub(ri, mul(a, Api))
        rs_new = rsum(add(mul(rr, rr), mul(ri, ri)), 1)
        beta = div(rs_new, add(rs, EPS))
        pr = add(rr, mul(beta, pr))
        pi = add(ri, mul(beta, pi))
        rs = rs_new

    # ---- SDW rank-1 weight  w = phi conj(d_ref) x / (mu + phi d^H x) ------
    eta_r = rsum(add(mul(dr, xr), mul(di, xi)), 1)          # [F,1]
    eta_i = rsum(sub(mul(dr, xi), mul(di, xr)), 1)
    dref_r = op.unsqueeze(op.gather(dr, ci(ref_mic), axis=1), axes=ci([1]))
    dref_i = op.unsqueeze(op.gather(di, ci(ref_mic), axis=1), axes=ci([1]))
    den_r = add(c(mu), mul(phi, eta_r))
    den_i = mul(phi, eta_i)
    num_r = mul(phi, dref_r)
    num_i = mul(phi, mul(dref_i, c(-1.0)))                  # phi * conj(d_ref)
    den2 = add(add(mul(den_r, den_r), mul(den_i, den_i)), EPS)
    scale_r = div(add(mul(num_r, den_r), mul(num_i, den_i)), den2)
    scale_i = div(sub(mul(num_i, den_r), mul(num_r, den_i)), den2)
    Wr = sub(mul(scale_r, xr), mul(scale_i, xi))            # [F,M]
    Wi = add(mul(scale_r, xi), mul(scale_i, xr))

    # ---- beamform  Y = w^H X  --------------------------------------------
    Yr = add(op.einsum([Wr, Xr], equation="fm,mft->ft"),
             op.einsum([Wi, Xi], equation="fm,mft->ft"))
    Yi = sub(op.einsum([Wr, Xi], equation="fm,mft->ft"),
             op.einsum([Wi, Xr], equation="fm,mft->ft"))

    model = build({"Xr": Xr, "Xi": Xi, "ms": ms, "mn": mn},
                  {"Yr": Yr, "Yi": Yi})
    return model


def export(path, fs, ref_mic=0, mu=1.0, power_iters=3, cg_iters=8, eps=1e-5):
    cfg = config_for(fs)
    model = build_graph(ref_mic=ref_mic, mu=mu, power_iters=power_iters,
                        cg_iters=cg_iters, eps=eps)
    meta = {
        "fs": str(fs), "n_fft": str(cfg.n_fft), "hop": str(cfg.hop),
        "ref_mic": str(ref_mic), "mu": str(mu),
        "power_iters": str(power_iters), "cg_iters": str(cg_iters),
        "stage": "rankn-spatial-mwf (covariance+powiter+cg+beamform); "
                 "host does stft/vad/omlsa/istft",
    }
    for k, v in meta.items():
        e = model.metadata_props.add()
        e.key, e.value = k, v
    model.doc_string = "Q-WiSE rank-N MWF spatial beamformer (ONNX, fp32)"
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"wrote {path}  (fs={fs}, n_fft={cfg.n_fft}, hop={cfg.hop}, "
          f"power_iters={power_iters}, cg_iters={cg_iters})")


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    export(os.path.join(here, "rankn_16k.onnx"), 16000)
    export(os.path.join(here, "rankn_8k.onnx"), 8000)
