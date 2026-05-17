"""Locate VAD-extracted speech inside the reference mic capture.

Port of ``mwf/mwf_align_vad.m``. Uses :func:`scipy.signal.correlate`
(equivalent to MATLAB ``xcorr``) and returns the non-negative-lag peak.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import correlate


def align_vad(
    vad_signal: ArrayLike, mic_signal: ArrayLike
) -> tuple[int, NDArray[np.float64]]:
    """Return ``(lag, aligned_vad)`` such that the VAD audio aligns to mic.

    Parameters
    ----------
    vad_signal
        The VAD-output (extracted speech) audio.
    mic_signal
        The reference-mic capture. Must be at least as long as
        ``vad_signal``; if it isn't, ``vad_signal`` is clipped to match.

    Returns
    -------
    lag : int
        0-based sample offset where ``vad_signal`` is placed inside
        ``mic_signal``.
    aligned_vad : ndarray
        ``len(mic_signal)``-length array with ``vad_signal`` placed at
        the lag and zeros elsewhere.
    """
    v = np.asarray(vad_signal, dtype=np.float64).reshape(-1)
    m = np.asarray(mic_signal, dtype=np.float64).reshape(-1)

    if v.size > m.size:
        v = v[: m.size]

    corr = correlate(m, v, mode="full")
    # ``correlate(a, b, 'full')`` has length ``len(a) + len(b) - 1``;
    # the lag-0 sample sits at index ``len(v) - 1``. Positive lags
    # advance the VAD signal forward inside the mic capture.
    mid = v.size - 1
    max_lag = m.size - v.size
    seg = corr[mid : mid + max_lag + 1]
    k = int(np.argmax(np.abs(seg)))
    lag = k  # 0-based, in [0, max_lag]

    aligned = np.zeros(m.size, dtype=np.float64)
    copy_end = min(lag + v.size, m.size)
    copy_len = copy_end - lag
    aligned[lag : lag + copy_len] = v[:copy_len]
    return lag, aligned


__all__ = ["align_vad"]
