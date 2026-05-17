"""VAD dispatcher — port of ``vad/vad.m``.

Picks one of the two backends (:class:`QwiseVad` or :class:`EnergyVad`)
per ``cfg.vad.backend`` and keeps a ring buffer of recent scores +
binary flags so the UI can render the VAD trace.

Selection rules (matching MATLAB):

* ``'qwise'``   — force the neural backend; if it fails to load, log a
                  warning and fall back to energy.
* ``'energy'``  — force the statistical fallback.
* ``'auto'``    — try the neural backend, silently fall back if missing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .energy import EnergyVad
from .qwise import QwiseVad

if TYPE_CHECKING:
    from ..config import Config

log = logging.getLogger(__name__)


class Vad:
    """Top-level VAD with backend fallback and a ring-buffer score history."""

    cfg: "Config"
    backend: EnergyVad | QwiseVad
    backend_name: str            # 'qwise-vad' | 'energy'
    history: NDArray[np.float64]
    flags: NDArray[np.bool_]
    hist_len: int

    def __init__(self, cfg: "Config") -> None:
        self.cfg = cfg
        self.hist_len = max(
            8,
            int(round(cfg.ui.vad_hist_sec * cfg.fs / cfg.frame_size)),
        )
        self.history = np.zeros(self.hist_len, dtype=np.float64)
        self.flags = np.zeros(self.hist_len, dtype=bool)
        self._hist_ptr: int = 0

        backend_pref = cfg.vad.backend.lower()
        if backend_pref == "energy":
            self.backend = EnergyVad(cfg)
            self.backend_name = "energy"
        else:
            qwise = QwiseVad(cfg)
            if qwise.ready:
                self.backend = qwise
                self.backend_name = "qwise-vad"
            else:
                if backend_pref == "qwise":
                    log.warning(
                        "[Q-WiSE] Neural VAD unavailable; "
                        "falling back to energy VAD."
                    )
                self.backend = EnergyVad(cfg)
                self.backend_name = "energy"
        log.info("[Q-WiSE] VAD backend: %s", self.backend_name)

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    def step(self, x: ArrayLike) -> tuple[bool, float]:
        """One block in → ``(is_speech, smoothed_score)``.

        Also pushes the score + flag into the ring buffer so the UI can
        render the historical trace via :meth:`trace`.
        """
        is_speech, score = self.backend.step(x)
        self.history[self._hist_ptr] = score
        self.flags[self._hist_ptr] = is_speech
        self._hist_ptr = (self._hist_ptr + 1) % self.hist_len
        return is_speech, score

    def trace(self) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Return the ring contents in chronological order (oldest first)."""
        idx = (self._hist_ptr + np.arange(self.hist_len)) % self.hist_len
        return self.history[idx].copy(), self.flags[idx].copy()

    def reset(self) -> None:
        self.history.fill(0.0)
        self.flags.fill(False)
        self._hist_ptr = 0
        if hasattr(self.backend, "reset"):
            self.backend.reset()


def make_vad(cfg: "Config") -> Vad:
    """Factory matching MATLAB's ``vad(cfg)`` constructor."""
    return Vad(cfg)


__all__ = ["Vad", "make_vad"]
