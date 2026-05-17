"""Frame-level speech / noise mask — port of ``mwf_build_speech_mask.m``.

For each STFT frame, compute RMS energy of the aligned VAD audio inside
the analysis window. Frames whose normalised energy clears the threshold
are flagged speech, then dilated by ``context_frames`` in both
directions to safely include transitions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def build_speech_mask(
    aligned_vad: ArrayLike,
    n_frames: int,
    n_fft: int,
    hop: int,
    threshold: float = 0.01,
    context_frames: int = 3,
) -> NDArray[np.bool_]:
    """Return ``[n_frames]`` boolean mask, ``True`` where speech is present.

    Reads frames directly from ``aligned_vad`` without the
    ``n_fft//2`` pre-pad the MATLAB sibling uses. The reference Q-WiSE
    Python pipeline (``/Users/javad/Projects/qwise/mwf.py``) also reads
    the un-padded signal; keeping that convention makes the covariance
    estimates byte-compatible with the reference.
    """
    aligned = np.asarray(aligned_vad, dtype=np.float64).reshape(-1)

    energy = np.zeros(int(n_frames), dtype=np.float64)
    for t in range(int(n_frames)):
        start = t * int(hop)
        stop = start + int(n_fft)
        if stop > aligned.size:
            break
        frame = aligned[start:stop]
        energy[t] = float(np.sqrt(np.mean(frame * frame)))

    peak = float(energy.max())
    if peak <= 0:
        return np.zeros(int(n_frames), dtype=bool)

    base_mask = (energy / peak) > threshold
    mask = base_mask.copy()

    # Dilate ±context_frames around every hit.
    hits = np.flatnonzero(base_mask)
    for i in hits:
        a = max(0, int(i) - int(context_frames))
        b = min(int(n_frames) - 1, int(i) + int(context_frames))
        mask[a : b + 1] = True
    return mask


__all__ = ["build_speech_mask"]
