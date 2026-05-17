"""Apply per-bin beamformer weights to a multi-channel STFT.

Port of ``mwf/mwf_apply_beamformer.m``::

    Y(f, t) = W(f)^H · X_multi(:, f, t)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def apply_beamformer(W: ArrayLike, X_multi: ArrayLike) -> NDArray[np.complex128]:
    """Return ``[n_freq, n_frames]`` single-channel STFT after beamforming."""
    Wa = np.asarray(W, dtype=np.complex128)
    Xa = np.asarray(X_multi, dtype=np.complex128)
    n_ch, n_freq, n_frames = Xa.shape
    if Wa.shape != (n_freq, n_ch):
        raise ValueError(
            f"W shape {Wa.shape} doesn't match (n_freq={n_freq}, n_ch={n_ch})"
        )

    # Vectorised conjugate-inner-product across the mic axis.
    # Y[f, t] = sum_m conj(W[f, m]) * X[m, f, t]
    return np.einsum("fm,mft->ft", np.conj(Wa), Xa)


__all__ = ["apply_beamformer"]
