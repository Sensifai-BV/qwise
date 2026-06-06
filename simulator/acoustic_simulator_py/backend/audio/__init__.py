"""Audio I/O — looped sources + recording sessions.

* :func:`backend.audio.load_wav_loop`  — port of ``load_wav_loop.m``
* :func:`backend.audio.loop_chunk`     — port of ``loop_chunk.m`` (0-based pointer)
* :class:`backend.audio.AudioIO`       — port of ``core/AudioIO.m`` (web flavour)

The hardware-mic/speaker bits of the MATLAB AudioIO are gone — the
browser owns capture & playback and feeds frames over the WebSocket.
"""

from .io import DEFAULT_DATA_DIR, AudioIO
from .loops import REPO_ROOT, load_wav_loop, loop_chunk

__all__ = [
    "AudioIO",
    "DEFAULT_DATA_DIR",
    "load_wav_loop",
    "loop_chunk",
    "REPO_ROOT",
]
