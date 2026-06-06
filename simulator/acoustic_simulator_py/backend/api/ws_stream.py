"""WebSocket ``/ws/stream`` — one full pipeline per connection.

Wire-protocol (versioned via ``hello`` on connect):

Client → Server
    * **Binary** frame  — ``float32 LE`` raw PCM mic block, ``cfg.frame_size``
      samples. Server pads / truncates silently.
    * **Text** frame   — JSON control message. ``{"type": ..., ...}``.
      Recognised types live in :meth:`PipelineSession.apply_control`.

Server → Client
    * On connect — one ``hello`` JSON message announcing the active
      config, the UI sidebar schema, and the binary payload layout.
    * For every processed audio block — one ``frame`` JSON message
      followed by one **binary** payload. The binary is:

          float32 LE, length = (2 + n_mics) * N,
          layout = [mwf (N) | comp (N) | mic[0] (N) | mic[1] (N) | ...]

      ``mwf`` is zeros when MWF is off; ``comp`` is the VAD-feed mono
      signal (matches the SourceMixer ``composite``); the mic columns
      are the full N-channel block.

    * After every control message — one ``ack`` JSON message echoing
      ``{type, kind, ok, ...}``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from numpy.typing import NDArray

from ..config import default, ui_sidebar_schema
from .pipeline import PipelineSession
from .ratelimit import client_ip_from_headers

log = logging.getLogger(__name__)

router = APIRouter()


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _pack_audio(
    mic: NDArray[np.float64],
    comp: NDArray[np.float64],
    mwf: NDArray[np.float64],
) -> bytes:
    """Encode one frame's worth of output audio as float32 LE."""
    n, n_mics = mic.shape
    out = np.empty((2 + n_mics, n), dtype=np.float32)
    out[0, :] = mwf.astype(np.float32, copy=False)
    out[1, :] = comp.astype(np.float32, copy=False)
    # Mic rows: float32 view, mic is [N, n_mics] → transpose to [n_mics, N]
    out[2:, :] = mic.T.astype(np.float32, copy=False)
    return out.tobytes(order="C")


def _hello_payload(session: PipelineSession) -> dict[str, Any]:
    cfg = session.cfg
    return {
        "type": "hello",
        "config": cfg.model_dump(),
        "ui_schema": ui_sidebar_schema(cfg),
        "geometry": session.serialize_geometry(),
        "frame_size": cfg.frame_size,
        "n_mics": cfg.n_mics,
        "fs": cfg.fs,
        "binary_layout": {
            "dtype": "float32",
            "endian": "little",
            "rows": ["mwf", "comp"] + [f"mic{i + 1}" for i in range(cfg.n_mics)],
            "row_length": cfg.frame_size,
        },
        "vad_backend": session.vad.backend_name,
    }


# --------------------------------------------------------------------- #
# WebSocket route
# --------------------------------------------------------------------- #
@router.websocket("/ws/stream")
async def stream(ws: WebSocket) -> None:
    """Bidirectional audio + control channel for one demo session."""
    # Local import keeps the test path from circular-import-ing app.py.
    from .app import get_rate_limiter

    await ws.accept()
    session = PipelineSession()
    rate_limiter = get_rate_limiter(ws.app)
    client_ip = client_ip_from_headers(
        ws.headers.items(),
        fallback=ws.client.host if ws.client else None,
    )
    log.info(
        "[Q-WiSE] WebSocket session opened (vad=%s, ip=%s)",
        session.vad.backend_name, client_ip,
    )

    # Send the hello payload so the client can size buffers + render the sidebar.
    try:
        await ws.send_json(_hello_payload(session))
    except Exception as ex:                 # pragma: no cover
        log.warning("[Q-WiSE] hello send failed: %s", ex)
        return

    try:
        while True:
            msg = await ws.receive()
            kind = msg.get("type")
            if kind == "websocket.disconnect":
                break

            if "text" in msg and msg["text"] is not None:
                # Control message.
                try:
                    control = json.loads(msg["text"])
                except json.JSONDecodeError as ex:
                    await ws.send_json({
                        "type": "ack", "ok": False, "error": f"bad_json: {ex}"
                    })
                    continue
                # Rate-limit recording starts — every other control type
                # passes through unchecked. We consume the quota *before*
                # the pipeline opens a folder so a denied request never
                # leaves an empty directory on disk.
                if control.get("type") == "recording_start":
                    status = rate_limiter.try_consume(client_ip)
                    if not status.allowed:
                        await ws.send_json({
                            "type": "ack",
                            "kind": "recording_start",
                            "ok": False,
                            "error": "rate_limited",
                            "quota": status.to_dict(),
                        })
                        continue
                    ack = session.apply_control(control)
                    ack["quota"] = status.to_dict()
                    await ws.send_json(ack)
                    continue
                ack = session.apply_control(control)
                # Geometry-affecting controls ship the new positions so
                # the 3-D scene plot can redraw without a round-trip.
                if (
                    ack.get("ok")
                    and ack.get("kind") in ("config_patch", "reset")
                ):
                    ack["geometry"] = session.serialize_geometry()
                await ws.send_json(ack)
            elif "bytes" in msg and msg["bytes"] is not None:
                # Mic frame (float32 LE).
                speech = np.frombuffer(msg["bytes"], dtype=np.float32).astype(
                    np.float64, copy=False
                )
                result = session.process_block(speech)
                await ws.send_json({
                    "type": "frame",
                    "frame_idx": result["frame_idx"],
                    "vad_score": result["vad_score"],
                    "is_speech": result["is_speech"],
                    "speech_source": session.speech_source,
                    "drone_on": session.drone_on,
                    "env_on": session.env_on,
                    "vad_on": session.vad_on,
                    "mwf_on": session.mwf_on,
                    "recording": session.recording,
                })
                await ws.send_bytes(
                    _pack_audio(result["mic"], result["comp"], result["mwf"])
                )
            else:
                # Either a connect / close handshake, or something we don't
                # recognise — ignore quietly.
                continue
    except WebSocketDisconnect:
        log.info("[Q-WiSE] WebSocket session closed")
    except Exception as ex:                 # pragma: no cover
        log.warning("[Q-WiSE] WebSocket error: %s", ex)
        try:
            await ws.close(code=1011)
        except Exception:
            pass
    finally:
        # Stop any active recording so the WAV is on disk before the
        # session is GC'd.
        if session.recording:
            try:
                session.stop_recording()
            except Exception:
                pass


__all__ = ["router"]
