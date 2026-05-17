"""Spatial-covariance estimation — port of ``mwf_estimate_covariance.m``.

Per-bin outer-product accumulator over masked STFT frames, with a small
diagonal-load floor (``eps_reg``) for numerical stability.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def estimate_covariance(
    X_multi: ArrayLike, mask: ArrayLike, eps_reg: float = 1e-10
) -> NDArray[np.complex128]:
    """Return ``[n_freq, n_ch, n_ch]`` covariance tensor.

    Parameters
    ----------
    X_multi
        Complex multi-channel STFT cube ``[n_ch, n_freq, n_frames]``.
    mask
        Boolean ``[n_frames]`` selecting frames to accumulate.
    eps_reg
        Diagonal-load floor matching the MATLAB / Python references.
    """
    X = np.asarray(X_multi, dtype=np.complex128)
    if X.ndim != 3:
        raise ValueError("X_multi must be [n_ch, n_freq, n_frames]")
    n_ch, n_freq, _ = X.shape
    mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
    idx = np.flatnonzero(mask_arr)

    eps_I = float(eps_reg) * np.eye(n_ch, dtype=np.complex128)
    if idx.size == 0:
        return np.broadcast_to(eps_I, (n_freq, n_ch, n_ch)).copy()

    n_valid = idx.size
    Phi = np.zeros((n_freq, n_ch, n_ch), dtype=np.complex128)
    for f in range(n_freq):
        obs = X[:, f, idx]  # [n_ch, n_valid]
        if n_valid == 1:
            obs = obs.reshape(n_ch, 1)
        R = (obs @ obs.conj().T) / n_valid
        # Sanitise NaN/Inf the way the Python reference does.
        bad = ~np.isfinite(R)
        if bad.any():
            R = R.copy()
            R[bad] = 0.0
        Phi[f] = R + eps_I
    return Phi


def enforce_psd(
    Phi: ArrayLike, n_ch: int, eps_reg: float
) -> NDArray[np.complex128]:
    """Project each ``[n_ch, n_ch]`` matrix onto the PSD cone and re-load.

    Used after computing ``Phi_ss = Phi_yy - Phi_nn`` so negative
    eigenvalues introduced by subtraction don't poison the beamformer.
    """
    Pa = np.asarray(Phi, dtype=np.complex128).copy()
    n_freq = Pa.shape[0]
    I = np.eye(int(n_ch), dtype=np.complex128)
    for f in range(n_freq):
        M = Pa[f]
        M = (M + M.conj().T) / 2.0
        d, V = np.linalg.eigh(M)
        d = np.maximum(d.real, 0.0)
        M = V @ np.diag(d.astype(np.complex128)) @ V.conj().T
        Pa[f] = M + float(eps_reg) * I
    return Pa


__all__ = ["estimate_covariance", "enforce_psd"]
