"""Voice activity detection — ports of ``vad/*.m``.

Exports:
    * :class:`Vad`        — dispatcher (Q-WiSE neural + energy fallback)
    * :class:`QwiseVad`   — neural backend (onnxruntime + qwise_vad.onnx)
    * :class:`EnergyVad`  — statistical fallback (RMS + spectral flatness)
    * :func:`make_vad`    — convenience factory matching MATLAB's ``vad(cfg)``
"""

from .dispatcher import Vad, make_vad
from .energy import EnergyVad
from .qwise import CONTEXT_SIZE, STATE_SHAPE, QwiseVad

__all__ = [
    "Vad",
    "EnergyVad",
    "QwiseVad",
    "CONTEXT_SIZE",
    "STATE_SHAPE",
    "make_vad",
]
