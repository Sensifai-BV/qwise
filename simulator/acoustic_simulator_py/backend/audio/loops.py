"""Looped-source helpers — ports of ``load_wav_loop.m`` and ``loop_chunk.m``.

Loaded once at startup and consumed block-by-block by the SourceMixer.
Used for the drone-fan and env-ambient loops, and for an optionally
user-uploaded clean-speech loop.

If the requested WAV is missing we fall back to a band-limited
pink-noise placeholder so the rest of the pipeline still runs — same
recovery the MATLAB sibling does, so the FastAPI server stays reachable
on a fresh checkout that hasn't bundled the WAVs yet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import ArrayLike, NDArray
from scipy import signal as sig

log = logging.getLogger(__name__)

# Resolve relative WAV paths against the repo root, the same way the
# MATLAB version resolves them against the project root.  backend/audio/loops.py
# → parents[2] is the project root.
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_wav_loop(
    wav_path: str | Path,
    target_fs: int,
    loop_sec: float,
    project_root: Path | None = None,
) -> NDArray[np.float64]:
    """Read a WAV (any sample rate / channel count) and tile it.

    Mirrors ``core/load_wav_loop.m``:

    * If ``wav_path`` is relative, resolve it against ``project_root``
      (default: the repo root).
    * If the file doesn't exist, generate a pink-noise placeholder so
      the simulator keeps running.
    * Down-mix to mono, resample to ``target_fs`` if needed.
    * Tile to ``round(loop_sec * target_fs)`` samples.
    * Normalise the whole loop to unit RMS so downstream gain knobs
      have a predictable starting point.
    """
    p = Path(wav_path)
    if not p.is_absolute():
        root = project_root or REPO_ROOT
        p = root / p

    n_target = int(round(float(loop_sec) * int(target_fs)))

    if not p.is_file():
        log.info(
            "[Q-WiSE] %s not found — generating pink-noise placeholder", p
        )
        y = _pink_noise_placeholder(n_target)
    else:
        log.info("[Q-WiSE] Loading %s", p)
        data, fs_orig = sf.read(str(p), dtype="float64")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if fs_orig != target_fs:
            data = sig.resample(data, int(round(len(data) * target_fs / fs_orig)))
        # Tile to at least n_target then trim.
        n_rep = int(np.ceil(n_target / max(len(data), 1)))
        y = np.tile(data, n_rep)[:n_target]

    rms = float(np.sqrt(np.mean(y * y))) + 1e-12
    return (y / rms).astype(np.float64)


def loop_chunk(wav: NDArray[np.float64], ptr: int, n: int) -> NDArray[np.float64]:
    """Read ``n`` samples from a circular buffer starting at 0-based ``ptr``.

    The MATLAB sibling is 1-based; the Python port uses 0-based pointers
    because that matches NumPy indexing and ``np.arange`` semantics.
    """
    L = wav.shape[0]
    if L == 0:
        return np.zeros(int(n), dtype=np.float64)
    idx = (int(ptr) + np.arange(int(n))) % L
    return wav[idx]


def _pink_noise_placeholder(n: int) -> NDArray[np.float64]:
    """Band-limited noise stand-in for a missing source WAV.

    Same recipe as ``load_wav_loop.m``: white noise filtered with a
    1-st-order Butterworth LP at ``Wn = 0.015`` (relative to Nyquist).
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    rng = np.random.default_rng(0)
    w = rng.standard_normal(int(n))
    b, a = sig.butter(1, 0.015)
    return sig.lfilter(b, a, w).astype(np.float64)


__all__ = ["load_wav_loop", "loop_chunk", "REPO_ROOT"]
