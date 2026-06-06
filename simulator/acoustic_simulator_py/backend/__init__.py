"""Q-WiSE acoustic simulator — backend package.

Mirrors the MATLAB project at acoustic_simulator/ folder-for-folder:

    backend.config    ↔  config/
    backend.core      ↔  core/
    backend.vad       ↔  vad/
    backend.mwf       ↔  mwf/
    backend.audio     ↔  core/AudioIO.m + load_wav_loop / loop_chunk
    backend.api       ↔  no MATLAB counterpart — FastAPI + WebSocket layer

Every numerical routine here is expected to match the MATLAB reference
output to within ~1e-9 in the equivalent unit test.
"""

__version__ = "0.1.0"
