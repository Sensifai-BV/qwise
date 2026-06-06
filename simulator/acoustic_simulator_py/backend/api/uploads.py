"""Speech-WAV upload endpoint.

Stores user-supplied clean-speech files under ``<data_dir>/uploads/``
with a UUID filename so different sessions can't collide on a path,
then returns the resolved path + duration. The frontend follows up with
a WebSocket ``load_speech_wav`` control message so the active
:class:`PipelineSession` plumbs it into its :class:`AudioIO`.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

import soundfile as sf
from fastapi import APIRouter, File, HTTPException, UploadFile

from ..audio.io import DEFAULT_DATA_DIR

log = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR_NAME = "uploads"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024            # 50 MB
ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
)


def uploads_root() -> Path:
    """Resolve + create the uploads directory.

    Path is built off the same ``QWISE_DATA_DIR`` :class:`AudioIO` uses
    so the uploaded file is reachable when the WebSocket loads it.
    """
    root = (Path(os.getenv("QWISE_DATA_DIR", str(DEFAULT_DATA_DIR)))
            / UPLOAD_DIR_NAME)
    root.mkdir(parents=True, exist_ok=True)
    return root


@router.post("/api/speech-wav")
async def upload_speech_wav(file: UploadFile = File(...)) -> dict[str, Any]:
    """Accept a multipart audio file and persist it.

    Returns ``{path, name, fs, duration, channels}`` — ``path`` is what
    the frontend ships in the ``load_speech_wav`` control message; the
    other fields are for the "WAV: …" badge in the UI.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="empty file")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file too large")

    original = file.filename or "speech.wav"
    ext = Path(original).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type {ext!r} "
                   f"(allowed: {sorted(ALLOWED_EXTENSIONS)})",
        )

    out_path = uploads_root() / f"{uuid.uuid4().hex}{ext}"
    out_path.write_bytes(contents)

    try:
        info = sf.info(str(out_path))
    except Exception as ex:
        out_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400, detail=f"unreadable audio: {ex}"
        )

    log.info(
        "[Q-WiSE] speech-WAV upload: %s -> %s (%.2f s @ %d Hz, %d ch)",
        original, out_path, info.duration, info.samplerate, info.channels,
    )

    return {
        "path": str(out_path),
        "name": original,
        "fs": int(info.samplerate),
        "duration": float(info.duration),
        "channels": int(info.channels),
    }


__all__ = ["router", "uploads_root", "UPLOAD_DIR_NAME"]
