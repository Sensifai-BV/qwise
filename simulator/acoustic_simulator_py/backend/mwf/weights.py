"""Beamformer-weight kernels — ports of the three MATLAB weight files.

Each function takes per-bin covariance cubes ``Phi_ss`` and ``Phi_nn``
of shape ``[n_freq, n_ch, n_ch]`` and returns ``W`` of shape
``[n_freq, n_ch]``.

* :func:`compute_mwf_weights`  — Speech-Distortion-Weighted MWF
                                  (``mwf_compute_mwf_weights.m``).
* :func:`compute_mvdr_weights` — MVDR with eigenvector steering
                                  (``mwf_compute_mvdr_weights.m``).
* :func:`compute_gev_weights`  — Generalized-Eigenvalue / Max-SNR beam
                                  (``mwf_compute_gev_weights.m``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh as gen_eigh


# --------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------- #
def _diag_load(trace_val: complex, n_ch: int, eps_reg: float, ratio: float) -> float:
    """Diagonal-loading scalar matching the MATLAB recipe."""
    return float(max(eps_reg, ratio * float(np.real(trace_val)) / n_ch))


def _solve(M: NDArray[np.complex128], n_ch: int) -> NDArray[np.complex128]:
    """Solve ``M X = I`` (returns ``inv(M)``), falling back to pinv."""
    I = np.eye(n_ch, dtype=np.complex128)
    try:
        return np.linalg.solve(M, I)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)


# --------------------------------------------------------------------- #
# SDW-MWF
# --------------------------------------------------------------------- #
def compute_mwf_weights(
    Phi_ss: ArrayLike,
    Phi_nn: ArrayLike,
    ref_channel: int,
    mu: float,
    eps_reg: float = 1e-10,
    diag_load_ratio: float = 1e-4,
) -> NDArray[np.complex128]:
    """Closed-form SDW-MWF:: ``W(f) = Phi_ss · inv(Phi_ss + mu·Phi_nn) · e_ref``.

    ``ref_channel`` is the 1-based mic index, matching MATLAB / ``cfg.mwf.ref_mic``.
    """
    Pss = np.asarray(Phi_ss, dtype=np.complex128)
    Pnn = np.asarray(Phi_nn, dtype=np.complex128)
    n_freq, n_ch, _ = Pss.shape
    W = np.zeros((n_freq, n_ch), dtype=np.complex128)
    e_ref = np.zeros(n_ch, dtype=np.complex128)
    e_ref[int(ref_channel) - 1] = 1.0
    I = np.eye(n_ch, dtype=np.complex128)
    for f in range(n_freq):
        Rs = Pss[f]
        Rn = Pnn[f]
        M = Rs + float(mu) * Rn
        load = _diag_load(np.trace(M), n_ch, eps_reg, diag_load_ratio)
        M = M + load * I
        Minv = _solve(M, n_ch)
        W[f, :] = Rs @ Minv @ e_ref
    return W


# --------------------------------------------------------------------- #
# MVDR
# --------------------------------------------------------------------- #
def compute_mvdr_weights(
    Phi_ss: ArrayLike,
    Phi_nn: ArrayLike,
    ref_channel: int,
    eps_reg: float = 1e-10,
    diag_load_ratio: float = 1e-4,
) -> NDArray[np.complex128]:
    """Per-bin MVDR with steering taken from the principal eigvec of Phi_ss.

    Falls back to a unit vector on the reference channel if the
    denominator ``a^H Phi_nn^{-1} a`` collapses (zero noise covariance).
    """
    Pss = np.asarray(Phi_ss, dtype=np.complex128)
    Pnn = np.asarray(Phi_nn, dtype=np.complex128)
    n_freq, n_ch, _ = Pss.shape
    W = np.zeros((n_freq, n_ch), dtype=np.complex128)
    I = np.eye(n_ch, dtype=np.complex128)
    ref0 = int(ref_channel) - 1
    for f in range(n_freq):
        Rs = Pss[f]
        Rn = Pnn[f]

        # ``Phi_ss`` is already Hermitian after :func:`enforce_psd`, so
        # we feed it to ``eigh`` as-is to stay byte-compatible with the
        # reference Q-WiSE pipeline. ``eigh`` returns eigenvalues in
        # ascending order → ``[:, -1]`` is the principal eigenvector.
        _, V = np.linalg.eigh(Rs)
        a = V[:, -1]
        if abs(a[ref0]) > eps_reg:
            a = a / a[ref0]

        load = _diag_load(np.trace(Rn), n_ch, eps_reg, diag_load_ratio)
        Rn_r = Rn + load * I
        Rn_inv = _solve(Rn_r, n_ch)

        num = Rn_inv @ a
        denom = a.conj() @ num
        if abs(denom) > eps_reg:
            w = num / denom
        else:
            w = np.zeros(n_ch, dtype=np.complex128)
            w[ref0] = 1.0
        W[f, :] = w
    return W


# --------------------------------------------------------------------- #
# GEV / Max-SNR
# --------------------------------------------------------------------- #
def compute_gev_weights(
    Phi_ss: ArrayLike,
    Phi_nn: ArrayLike,
    ref_channel: int,
    eps_reg: float = 1e-10,
    diag_load_ratio: float = 1e-4,
) -> NDArray[np.complex128]:
    """Generalized-Eigenvalue (Max-SNR) beamformer.

    Solves ``A w = λ B w`` with ``A = Phi_ss + ε I``, ``B = Phi_nn + ε I``
    via :func:`scipy.linalg.eigh` (Hermitian GEP — matches the reference
    Q-WiSE Python pipeline byte-for-byte). Falls through to
    ``inv(B) A`` and finally to a unit reference vector on failure.

    ``scipy.linalg.eigh`` returns eigenvalues in ascending order, so the
    principal direction is at ``V[:, -1]``; phase is then normalised to
    the reference mic.
    """
    Pss = np.asarray(Phi_ss, dtype=np.complex128)
    Pnn = np.asarray(Phi_nn, dtype=np.complex128)
    n_freq, n_ch, _ = Pss.shape
    W = np.zeros((n_freq, n_ch), dtype=np.complex128)
    I = np.eye(n_ch, dtype=np.complex128)
    ref0 = int(ref_channel) - 1
    for f in range(n_freq):
        Rs = Pss[f]
        Rn = Pnn[f]
        load_ss = _diag_load(np.trace(Rs), n_ch, eps_reg, diag_load_ratio)
        load_nn = _diag_load(np.trace(Rn), n_ch, eps_reg, diag_load_ratio)
        A = (Rs + Rs.conj().T) / 2.0 + load_ss * I
        B = (Rn + Rn.conj().T) / 2.0 + load_nn * I

        w: NDArray[np.complex128] | None = None
        try:
            # eigh returns eigenvalues ascending; pick the principal one.
            _, V = gen_eigh(A, B)
            w = V[:, -1]
        except Exception:                       # pragma: no cover
            try:
                d, V = np.linalg.eigh(np.linalg.solve(B, A))
                w = V[:, -1]
            except Exception:                   # pragma: no cover
                w = None
        if w is None or not np.all(np.isfinite(w)):
            w = np.zeros(n_ch, dtype=np.complex128)
            w[ref0] = 1.0

        if abs(w[ref0]) > eps_reg:
            w = w / w[ref0]
        W[f, :] = w
    return W


__all__ = [
    "compute_mwf_weights",
    "compute_mvdr_weights",
    "compute_gev_weights",
]
