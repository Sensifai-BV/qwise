"""Acoustic-scene primitives — ports of ``core/*.m``.

Available:
    * :mod:`backend.core.geometry`      — port of ``build_geometry.m``
    * :mod:`backend.core.source_mixer`  — port of ``SourceMixer.m``

Pending:
    * ``distance_to_gain``  — port of ``distance_to_gain.m`` utility
    * ``wav_loop``          — ports of ``load_wav_loop.m`` + ``loop_chunk.m``
"""

from .geometry import (
    Geometry,
    build_geometry,
    build_mic_array,
    spherical_offset,
)
from .source_mixer import NMicChangeError, SourceMixer

__all__ = [
    "Geometry",
    "build_geometry",
    "build_mic_array",
    "spherical_offset",
    "SourceMixer",
    "NMicChangeError",
]
