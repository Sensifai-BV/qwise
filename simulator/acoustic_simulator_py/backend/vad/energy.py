"""Statistical fallback VAD — port of ``vad/VADEnergy.m``.

Energy + spectral-flatness gate, EMA-smoothed score, hangover release on
the raw binary decision. Used whenever the Q-WiSE neural backend cannot
be loaded.

The math mirrors the MATLAB reference exactly:

    rms     = sqrt(mean(x.^2))
    dB      = 20*log10(rms + 1e-12)
    X       = |FFT(x)|        (half spectrum)
    gmean   = exp(mean(log(X + 1e-12)))
    sfm     = gmean / (mean(X) + 1e-12)
    raw     = (dB > energy_threshold) AND (sfm < sfm_threshold)
    score   = (1 - smooth) * score_prev + smooth * raw
    state   = hangover-extended ``raw``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from ..config import Config


class EnergyVad:
    """Statistical voice-activity detector (RMS dBFS + SFM + hangover)."""

    #: Always ``True`` — the energy VAD has no external dependency.
    ready: bool = True

    def __init__(self, cfg: "Config") -> None:
        self.cfg = cfg.vad  # live reference so UI edits to thresholds apply
        self._smooth = float(cfg.vad.smoothing)
        self._hang_counter: int = 0
        self._state_speech: bool = False
        self._score_ema: float = 0.0

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    def step(self, x: ArrayLike) -> tuple[bool, float]:
        """One frame in, ``(is_speech, smoothed_score)`` out."""
        xf = np.asarray(x, dtype=np.float64).reshape(-1)
        rms = float(np.sqrt(np.mean(xf * xf))) + 1e-12
        db = 20.0 * np.log10(rms)

        # Spectral flatness measure on the half-spectrum (matches MATLAB
        # ``X = X(1:floor(numel(X)/2))`` — note: the DC bin is included,
        # the Nyquist bin is not).
        spec = np.abs(np.fft.fft(xf))
        spec = spec[: spec.size // 2]
        gmean = float(np.exp(np.mean(np.log(spec + 1e-12))))
        amean = float(np.mean(spec)) + 1e-12
        sfm = gmean / amean

        raw = float(db > self.cfg.energy_threshold and sfm < self.cfg.sfm_threshold)

        # Score EMA for the smoother UI trace.
        self._score_ema = (1.0 - self._smooth) * self._score_ema + self._smooth * raw
        score = self._score_ema

        # Hangover release on the raw decision (NOT the EMA score).
        if raw > 0.5:
            self._state_speech = True
            self._hang_counter = int(self.cfg.hang_frames)
        elif self._hang_counter > 0:
            self._hang_counter -= 1
            self._state_speech = True
        else:
            self._state_speech = False

        return self._state_speech, score

    def reset(self) -> None:
        self._hang_counter = 0
        self._state_speech = False
        self._score_ema = 0.0


__all__ = ["EnergyVad"]
