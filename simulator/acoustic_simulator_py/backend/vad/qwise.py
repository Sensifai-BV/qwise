"""Q-WiSE neural VAD — port of ``vad/VADQwise.m`` + ``qwise_ort_helper.py``.

The MATLAB version had four different ONNX-importer fallbacks because
MATLAB releases vary widely; in Python we go straight to
``onnxruntime``, which is the bridge MATLAB's ``qwise_ort_helper.py``
already targeted.

Model quirks (matching the upstream Silero-style v5 export):

* The audio input tensor is ``(1, 576)`` — 64 samples of *previous*
  audio context prepended to each new 512-sample chunk. Feeding a bare
  512-sample chunk returns near-zero probabilities for every frame.
* The LSTM hidden state has shape ``(2, 1, 128)`` and is threaded
  between calls so multi-block inputs share state.
* The model also takes the integer sample-rate as a 0-D ``int64``
  scalar input.

If ``onnxruntime`` is missing, the model file can't be found, or the
session refuses to load, :attr:`ready` stays ``False`` and the
dispatcher (see :mod:`backend.vad.dispatcher`) silently swaps in the
energy VAD.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from ..config import Config

log = logging.getLogger(__name__)

# Resolve repo paths against this file's location: backend/vad/qwise.py
# → parents[2] is the project root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Constants — must match the upstream ONNX checkpoint.
CONTEXT_SIZE = 64
STATE_SHAPE: tuple[int, int, int] = (2, 1, 128)


class QwiseVad:
    """Neural VAD backend wrapping ``models/qwise_vad.onnx``.

    Stateful — keeps both the LSTM hidden state and the 64-sample audio
    context between :meth:`step` calls.
    """

    #: ``True`` once the ONNX session is loaded and the first ``step``
    #: call may proceed. Falls back to ``False`` whenever construction
    #: hits any failure (missing onnxruntime, missing model, etc.).
    ready: bool = False

    def __init__(self, cfg: "Config") -> None:
        self.cfg = cfg.vad
        self.frame_size = int(cfg.vad.qwise_frame)
        self.sr = int(cfg.fs)
        self._threshold = float(cfg.vad.qwise_threshold)
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._ctx = np.zeros(CONTEXT_SIZE, dtype=np.float32)
        self._session = None
        self._in_names: list[str] = []
        self._out_names: list[str] = []

        # --- onnxruntime is optional at install time; degrade gracefully ---
        try:
            import onnxruntime as ort
        except ImportError:
            log.warning(
                "[Q-WiSE] onnxruntime not installed — falling back to energy VAD."
            )
            return

        onnx_path = self._resolve_onnx_path(self.cfg.onnx_path)
        if onnx_path is None:
            log.warning(
                "[Q-WiSE] VAD ONNX model not found at %s (cwd=%s).",
                self.cfg.onnx_path,
                Path.cwd(),
            )
            return

        try:
            so = ort.SessionOptions()
            so.log_severity_level = 3  # silence ORT's banner on import
            self._session = ort.InferenceSession(
                str(onnx_path),
                sess_options=so,
                providers=["CPUExecutionProvider"],
            )
            self._in_names = [i.name for i in self._session.get_inputs()]
            self._out_names = [o.name for o in self._session.get_outputs()]
        except Exception as ex:  # pragma: no cover — broad catch is the contract
            log.warning("[Q-WiSE] Failed to load ONNX VAD model: %s", ex)
            return

        self.ready = True
        log.info(
            "[Q-WiSE] Neural VAD loaded (%s -> %s).",
            ", ".join(self._in_names),
            ", ".join(self._out_names),
        )

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    def step(self, x: ArrayLike) -> tuple[bool, float]:
        """Run one host-block through the model, return ``(is_speech, score)``.

        The host frame (typically ``cfg.frame_size``) is split into 512-sample
        sub-blocks because that is what the ONNX model expects; the LSTM
        state carries naturally across the sub-blocks. The probability of
        the *last* sub-block is what we report — same convention as MATLAB.

        Errors during inference flip :attr:`ready` to ``False`` and return
        silence, so the dispatcher can switch to the energy VAD on the
        next frame.
        """
        if not self.ready or self._session is None:
            return False, 0.0
        try:
            xf = np.asarray(x, dtype=np.float32).reshape(-1)
            nf = self.frame_size
            if xf.size < nf:
                xf = np.concatenate(
                    [xf, np.zeros(nf - xf.size, dtype=np.float32)]
                )
            last_prob = 0.0
            for off in range(0, xf.size - nf + 1, nf):
                last_prob = self._infer_chunk(xf[off : off + nf])
            score = float(last_prob)
            return score > self._threshold, score
        except Exception as ex:  # pragma: no cover
            log.warning("[Q-WiSE] VAD step failed: %s (disabling).", ex)
            self.ready = False
            return False, 0.0

    def reset(self) -> None:
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._ctx = np.zeros(CONTEXT_SIZE, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _infer_chunk(self, chunk: NDArray[np.float32]) -> float:
        """Run a single 512-sample chunk through the ONNX session."""
        # Prepend the 64-sample audio context from the previous chunk.
        x_in = np.ascontiguousarray(
            np.concatenate([self._ctx, chunk]).reshape(1, -1).astype(np.float32)
        )
        feeds = self._build_feeds(x_in)
        outs = self._session.run(self._out_names, feeds)
        prob = float(np.asarray(outs[0]).reshape(-1)[0])
        # Most exports return ``(prob, new_state)``; some return prob only
        # (state implicit) — keep the previous state in that case.
        if len(outs) > 1:
            new_state = np.asarray(outs[1], dtype=np.float32).reshape(STATE_SHAPE)
            self._state = np.ascontiguousarray(new_state)
        self._ctx = chunk[-CONTEXT_SIZE:].astype(np.float32, copy=True)
        return prob

    def _build_feeds(self, x_in: NDArray[np.float32]) -> dict[str, np.ndarray]:
        """Bind ORT feeds by sniffing the input-name heuristics.

        Mirrors ``qwise_ort_helper._pick_feeds``:
        * names containing ``state`` / ``hidden`` / ``h`` / ``c`` / ``h0`` /
          ``c0`` → the LSTM hidden state
        * names containing ``sr`` / ``rate`` → the 0-D ``int64`` sample rate
        * everything else → the audio input
        """
        sr_arr = np.asarray(int(self.sr), dtype=np.int64)
        feeds: dict[str, np.ndarray] = {}
        for name in self._in_names:
            nl = name.lower()
            if nl in {"h", "c", "h0", "c0"} or "state" in nl or "hidden" in nl:
                feeds[name] = self._state
            elif "sr" in nl or "sample_rate" in nl or "rate" in nl:
                feeds[name] = sr_arr
            else:
                feeds[name] = x_in
        return feeds

    @staticmethod
    def _resolve_onnx_path(p: str) -> Path | None:
        """Find the ONNX file relative to cwd or the project root."""
        candidate = Path(p)
        if candidate.is_absolute():
            return candidate if candidate.is_file() else None
        for tried in (Path.cwd() / candidate, REPO_ROOT / candidate):
            if tried.is_file():
                return tried
        return None


__all__ = ["QwiseVad", "CONTEXT_SIZE", "STATE_SHAPE"]
