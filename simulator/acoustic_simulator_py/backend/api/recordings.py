"""Recording catalogue endpoints — Task 18.

Surfaces the session folders Task 10's :class:`AudioIO` writes to disk
so the playback grid can list them and stream individual WAV files.

* ``GET /api/recordings``                                — newest-first list
* ``GET /api/recordings/{session}/files/{filename}``     — single WAV stream

Path safety is the only non-trivial concern here: session + filename
must match a strict identifier regex, and the resolved path is
re-checked against ``app.state.records_root`` so a crafted name like
``../../etc/passwd.wav`` can't escape.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

router = APIRouter()

# Session folders are ``<prefix>_<sub>_<YYYYMMDD>_<HHMMSS>_<msec>``.
# Track names are valid Python identifiers (``mic01``, ``vad``, …) +
# the ``.wav`` extension. Anything outside that vocabulary is rejected
# before we touch the filesystem.
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _records_root(request: Request) -> Path:
    """Read the recordings root the lifespan resolved at startup."""
    root = getattr(request.app.state, "records_root", None)
    if root is None:
        # Tests skip lifespan; lazy-build a sane default so the endpoint
        # still works without a `with TestClient(app)` wrapper.
        from ..audio.io import DEFAULT_DATA_DIR
        from ..config import default
        cfg = default()
        p = Path(cfg.record.dir)
        root = p if p.is_absolute() else DEFAULT_DATA_DIR / p
        request.app.state.records_root = root
    return Path(root)


# --------------------------------------------------------------------- #
# GET /api/recordings
# --------------------------------------------------------------------- #
@router.get("/api/recordings")
async def list_recordings(
    request: Request,
    limit: int = Query(20, ge=1, le=200),
) -> list[dict[str, Any]]:
    """Return up to ``limit`` recording sessions, newest first.

    Each entry is ``{name, mtime, files}`` where ``files`` is a sorted
    list of WAV filenames inside the session folder. Empty folders /
    folders whose name doesn't match the safe regex are skipped.
    """
    root = _records_root(request)
    if not root.is_dir():
        return []

    entries: list[dict[str, Any]] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        if not _SAFE_NAME_RE.match(entry.name):
            continue
        wavs = sorted(
            p.name for p in entry.glob("*.wav")
            if _SAFE_NAME_RE.match(p.name)
        )
        if not wavs:
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:                              # pragma: no cover
            continue
        entries.append({
            "name": entry.name,
            "mtime": mtime,
            "files": wavs,
        })
    entries.sort(key=lambda d: d["mtime"], reverse=True)
    return entries[:limit]


# --------------------------------------------------------------------- #
# GET /api/recordings/{session}/files/{filename}
# --------------------------------------------------------------------- #
@router.get("/api/recordings/{session}/files/{filename}")
async def get_recording_file(
    request: Request, session: str, filename: str
) -> FileResponse:
    """Stream one WAV from a recording session."""
    if not _SAFE_NAME_RE.match(session):
        raise HTTPException(status_code=400, detail="bad session name")
    if not _SAFE_NAME_RE.match(filename) or not filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="bad filename")

    root = _records_root(request).resolve()
    target = (root / session / filename).resolve()
    try:
        target.relative_to(root)
    except ValueError as ex:
        raise HTTPException(
            status_code=400, detail="path outside recordings root"
        ) from ex
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")

    return FileResponse(target, media_type="audio/wav", filename=filename)


__all__ = ["router"]
