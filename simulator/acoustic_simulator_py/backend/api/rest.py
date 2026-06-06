"""REST endpoints — minimal surface for Task 12 + Task 13.

Currently exposes:

* ``GET /healthz``            — liveness probe (HF Space readiness)
* ``GET /api/version``        — backend version string
* ``GET /api/config/default`` — full default config as JSON
* ``GET /api/config/schema``  — flat UI sidebar schema (Task 15)
* ``GET /api/quota``          — per-IP recording quota status (Task 13)

Recording / upload endpoints land in their own tasks (16 wires speech
uploads; 18 surfaces recordings).
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .. import __version__
from ..config import Config, default, ui_sidebar_schema
from ..core import build_geometry
from .ratelimit import client_ip_from_headers

router = APIRouter()


# --------------------------------------------------------------------- #
# Liveness + metadata
# --------------------------------------------------------------------- #
@router.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": __version__})


@router.get("/api/version")
async def version() -> dict[str, str]:
    return {"version": __version__}


# --------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------- #
@router.get("/api/config/default")
async def config_default() -> dict:
    """Return the default Q-WiSE runtime configuration as JSON."""
    cfg: Config = default()
    return cfg.model_dump()


@router.get("/api/config/schema")
async def config_schema() -> dict:
    """Return the UI sidebar entries (one per editable field)."""
    cfg = default()
    return {"sidebar": ui_sidebar_schema(cfg)}


@router.get("/api/geometry")
async def geometry_default() -> dict:
    """Return the default scene geometry as JSON.

    Lets the frontend render real (empty) Plotly cards before the first
    WebSocket ``hello`` lands — no more "Task 17" placeholder text on
    page load. The payload mirrors ``PipelineSession.serialize_geometry``.
    """
    cfg = default()
    geo = build_geometry(cfg)
    return {
        "pos_human":    geo.pos_human.tolist(),
        "pos_img_src":  geo.pos_img_src.tolist(),
        "pos_drone":    geo.pos_drone.tolist(),
        "pos_env":      geo.pos_env.tolist(),
        "pos_mics":     geo.pos_mics.tolist(),
        "dist_speech":  geo.dist_speech.tolist(),
        "dist_drone":   geo.dist_drone.tolist(),
        "dist_env":     geo.dist_env.tolist(),
        "gains_speech": geo.gains_speech.tolist(),
        "gains_drone":  geo.gains_drone.tolist(),
        "gains_env":    geo.gains_env.tolist(),
        "drone_agl":    float(geo.drone_agl),
        "ref_mic":      int(cfg.mwf.ref_mic),
    }


# --------------------------------------------------------------------- #
# Per-IP quota
# --------------------------------------------------------------------- #
@router.get("/api/quota")
async def quota(request: Request) -> dict:
    """Return the caller's current recording-quota snapshot.

    The frontend reads this on page load to render the
    ``"N / 10 recordings left, resets at HH:MM"`` hint and to skip
    showing the Record button when the limit is exhausted.
    """
    # Local import keeps the test path from circular-import-ing app.py.
    from .app import get_rate_limiter

    rl = get_rate_limiter(request.app)
    ip = client_ip_from_headers(
        request.headers.items(),
        fallback=request.client.host if request.client else None,
    )
    status = rl.status(ip)
    payload = status.to_dict()
    payload["ip"] = ip
    return payload


__all__ = ["router"]
