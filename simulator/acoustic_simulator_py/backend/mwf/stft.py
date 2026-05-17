"""STFT / iSTFT — port of ``mwf/mwf_stft.m`` and ``mwf/mwf_istft.m``.

Delegates to :func:`scipy.signal.stft` and :func:`scipy.signal.istft` so
the batch path is bit-compatible with the reference Q-WiSE Python
pipeline (``/Users/javad/Projects/qwise/mwf.py``). The MATLAB sources
ship their own manual loops because MATLAB lacks an scipy.signal.stft
equivalent; in Python we just call the canonical implementation.

The streaming path in :mod:`backend.mwf.mwf` does NOT use these
functions — it owns its own per-hop FFT loop with a ``sqrt(periodic
Hann)`` analysis window. :func:`_periodic_hann` is exported so the
streaming code shares the exact same window definition.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal as sig
from scipy.signal import windows


def _periodic_hann(n_fft: int) -> NDArray[np.float64]:
    """MATLAB-style ``hann(n_fft, 'periodic')`` — ``sym=False`` in scipy."""
    return windows.hann(int(n_fft), sym=False)


def stft(
    x: ArrayLike, n_fft: int, hop: int, fs: int = 16_000
) -> NDArray[np.complex128]:
    """Short-time Fourier transform — one-sided complex spectrum.

    Returns ``[n_freq, n_frames]`` with ``n_freq = n_fft // 2 + 1``.
    Matches ``scipy.signal.stft`` with ``boundary='zeros'`` /
    ``padded=True`` — the same call the reference Q-WiSE pipeline uses
    so covariance + weight estimates land on identical numerical grids.

    ``fs`` only sets the time axis returned by scipy; it has no effect
    on the complex STFT values themselves. We keep the parameter for
    parity with the reference signature.
    """
    if not (np.isfinite(n_fft) and n_fft > 0):
        raise ValueError("n_fft must be a positive integer")
    if not (np.isfinite(hop) and hop > 0):
        raise ValueError("hop must be a positive integer")

    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    window = _periodic_hann(int(n_fft))
    _, _, Zxx = sig.stft(
        xa,
        fs=int(fs),
        window=window,
        nperseg=int(n_fft),
        noverlap=int(n_fft) - int(hop),
        boundary="zeros",
        padded=True,
    )
    return np.asarray(Zxx, dtype=np.complex128)


def istft(
    X: ArrayLike, n_fft: int, hop: int, fs: int = 16_000
) -> NDArray[np.float64]:
    """Inverse STFT — pairs with :func:`stft`.

    Returns a 1-D real signal. The caller trims to the original input
    length (matches the reference Python's ``clean[:min_mic_len]`` pattern).
    """
    Xa = np.asarray(X, dtype=np.complex128)
    if Xa.ndim != 2:
        raise ValueError("X must be a 2-D STFT matrix [n_freq, n_frames]")
    n_freq, _ = Xa.shape
    if n_freq != int(n_fft) // 2 + 1:
        raise ValueError(
            f"STFT bin count ({n_freq}) does not match n_fft/2+1 "
            f"({int(n_fft) // 2 + 1})."
        )

    window = _periodic_hann(int(n_fft))
    _, xrec = sig.istft(
        Xa,
        fs=int(fs),
        window=window,
        nperseg=int(n_fft),
        noverlap=int(n_fft) - int(hop),
        boundary=True,
    )
    return np.asarray(np.real(xrec), dtype=np.float64)


__all__ = ["stft", "istft"]
