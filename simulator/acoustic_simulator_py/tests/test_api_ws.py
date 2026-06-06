"""End-to-end tests for the ``/ws/stream`` WebSocket pipeline.

Spins up the FastAPI ``TestClient`` and exercises the protocol:

* on connect → ``hello`` message with config + schema + binary layout
* binary mic frame → ``frame`` JSON + binary audio payload of the
  expected shape
* control messages → ``ack`` messages with the right ``ok`` flag
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.config import default


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _f32_bytes(arr: np.ndarray) -> bytes:
    return np.asarray(arr, dtype=np.float32).tobytes(order="C")


# --------------------------------------------------------------------- #
# Handshake
# --------------------------------------------------------------------- #
def test_hello_message_on_connect(client) -> None:
    cfg = default()
    with client.websocket_connect("/ws/stream") as ws:
        hello = ws.receive_json()
    assert hello["type"] == "hello"
    assert hello["frame_size"] == cfg.frame_size
    assert hello["n_mics"] == cfg.n_mics
    assert hello["fs"] == cfg.fs
    # Sidebar schema (Task 15 will consume).
    assert "ui_schema" in hello
    assert len(hello["ui_schema"]) > 5
    # Binary layout description matches what _pack_audio produces.
    layout = hello["binary_layout"]
    assert layout["dtype"] == "float32"
    assert layout["row_length"] == cfg.frame_size
    expected_rows = ["mwf", "comp"] + [f"mic{i + 1}" for i in range(cfg.n_mics)]
    assert layout["rows"] == expected_rows

    # Task 17: hello must include the initial scene geometry so the
    # 3-D scene plot can render before any audio arrives.
    geom = hello["geometry"]
    for key in ("pos_human", "pos_drone", "pos_env", "pos_mics", "pos_img_src",
                "dist_speech", "gains_speech", "drone_agl", "ref_mic"):
        assert key in geom, f"hello.geometry missing {key}"
    assert len(geom["pos_mics"]) == cfg.n_mics
    assert len(geom["pos_human"]) == 3
    assert len(geom["pos_drone"]) == 3
    assert geom["ref_mic"] == cfg.mwf.ref_mic


def test_config_patch_ack_carries_updated_geometry(client) -> None:
    """A successful config_patch must echo the freshly-rebuilt geometry
    so the 3-D scene plot can redraw without a round-trip."""
    with client.websocket_connect("/ws/stream") as ws:
        hello = ws.receive_json()
        drone_xy_before = (
            hello["geometry"]["pos_drone"][0],
            hello["geometry"]["pos_drone"][1],
        )
        ws.send_text(json.dumps({
            "type": "config_patch",
            "patch": {"slant_dist": 5.0},
        }))
        ack = ws.receive_json()

    assert ack["ok"] is True
    assert ack["kind"] == "config_patch"
    assert "geometry" in ack
    # Drone moved (distance was 2.5 m, now 5 m → roughly double the
    # x-offset from the speaker).
    drone_xy_after = (ack["geometry"]["pos_drone"][0],
                      ack["geometry"]["pos_drone"][1])
    assert drone_xy_after != drone_xy_before


# --------------------------------------------------------------------- #
# Audio frames
# --------------------------------------------------------------------- #
def test_audio_frame_round_trip_shape(client) -> None:
    """Raw round-trip shape check.

    The pipeline now boots with VAD/MWF/Drone/Env on by default, so this
    test explicitly disables them to keep asserting the "speech-only
    passthrough" invariants (mwf row zeros, comp == mic1).
    """
    cfg = default()
    n = cfg.frame_size
    rng = np.random.default_rng(0)
    frame = rng.standard_normal(n).astype(np.float32)
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()                       # hello

        # Drop everything to off so the row invariants below hold.
        for ctl in (
            {"type": "drone_on", "on": False},
            {"type": "env_on", "on": False},
            {"type": "enable_vad", "on": False},      # also turns MWF off
        ):
            ws.send_text(json.dumps(ctl))
            ws.receive_json()

        ws.send_bytes(_f32_bytes(frame))
        meta = ws.receive_json()
        audio = ws.receive_bytes()

    assert meta["type"] == "frame"
    assert meta["frame_idx"] == 1
    assert meta["vad_score"] == 0.0
    assert meta["is_speech"] is False
    assert meta["vad_on"] is False
    assert meta["mwf_on"] is False
    # Binary payload = (2 + n_mics) * N float32 samples.
    n_rows = 2 + cfg.n_mics
    expected_bytes = n_rows * n * 4
    assert len(audio) == expected_bytes
    out = np.frombuffer(audio, dtype=np.float32).reshape(n_rows, n)
    # MWF row is zeros when mwf_on is off.
    np.testing.assert_array_equal(out[0], np.zeros(n, dtype=np.float32))
    # comp == mic 1 (default composite='mic1'), and mic-1 carries the
    # full noisy mix; with drone/env off it's just the speech frame.
    np.testing.assert_allclose(out[1], out[2], atol=1e-6)


def test_frame_idx_increments_per_block(client) -> None:
    cfg = default()
    n = cfg.frame_size
    frame = np.zeros(n, dtype=np.float32)
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()                       # hello
        seen = []
        for _ in range(3):
            ws.send_bytes(_f32_bytes(frame))
            meta = ws.receive_json()
            ws.receive_bytes()
            seen.append(meta["frame_idx"])
    assert seen == [1, 2, 3]


# --------------------------------------------------------------------- #
# Control messages
# --------------------------------------------------------------------- #
def test_toggle_drone_then_env(client) -> None:
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "drone_on", "on": True}))
        ack1 = ws.receive_json()
        ws.send_text(json.dumps({"type": "env_on", "on": True}))
        ack2 = ws.receive_json()

        # Send one frame and check the meta echoes the toggles.
        cfg = default()
        ws.send_bytes(_f32_bytes(np.zeros(cfg.frame_size, dtype=np.float32)))
        meta = ws.receive_json()
        ws.receive_bytes()

    assert ack1["ok"] and ack1["kind"] == "drone_on"
    assert ack2["ok"] and ack2["kind"] == "env_on"
    assert meta["drone_on"] is True
    assert meta["env_on"] is True


def test_enable_mwf_auto_enables_vad(client) -> None:
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "enable_mwf", "on": True}))
        ack = ws.receive_json()

        cfg = default()
        ws.send_bytes(_f32_bytes(np.zeros(cfg.frame_size, dtype=np.float32)))
        meta = ws.receive_json()
        ws.receive_bytes()

    assert ack["ok"]
    assert meta["vad_on"] is True
    assert meta["mwf_on"] is True


def test_config_patch_rejects_n_mics_change(client) -> None:
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({
            "type": "config_patch",
            "patch": {"n_mics": 7},
        }))
        ack = ws.receive_json()
    assert ack["ok"] is False
    assert "n_mics" in ack["error"]


def test_config_patch_accepts_human_height(client) -> None:
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({
            "type": "config_patch",
            "patch": {"human_height": 1.85},
        }))
        ack = ws.receive_json()
    assert ack["ok"] is True
    assert ack["kind"] == "config_patch"


def test_bad_json_control_returns_error_ack(client) -> None:
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        ws.send_text("{not json}")
        ack = ws.receive_json()
    assert ack["type"] == "ack"
    assert ack["ok"] is False
    assert "bad_json" in ack["error"]


def test_unknown_control_kind_returns_error_ack(client) -> None:
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        ws.send_text(json.dumps({"type": "no_such_thing"}))
        ack = ws.receive_json()
    assert ack["ok"] is False
    assert ack["error"] == "unknown_type"


# --------------------------------------------------------------------- #
# End-to-end: drone + env + VAD + MWF pipeline through the WS
# --------------------------------------------------------------------- #
def test_end_to_end_voiced_block_produces_nonzero_mwf(client) -> None:
    """Enable VAD + MWF, push a voiced tone, expect MWF row != zeros."""
    cfg = default()
    n = cfg.frame_size
    t = np.arange(n) / cfg.fs
    voiced = (
        0.40 * np.sin(2 * math.pi * 220 * t)
        + 0.20 * np.sin(2 * math.pi * 440 * t)
    ).astype(np.float32)

    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()                       # hello
        ws.send_text(json.dumps({"type": "drone_on", "on": True}))
        ws.receive_json()
        ws.send_text(json.dumps({"type": "enable_mwf", "on": True}))
        ws.receive_json()

        last_mwf = None
        for _ in range(4):
            ws.send_bytes(_f32_bytes(voiced))
            ws.receive_json()                   # meta
            audio = ws.receive_bytes()
            arr = np.frombuffer(audio, dtype=np.float32).reshape(
                2 + cfg.n_mics, n
            )
            last_mwf = arr[0]

    assert last_mwf is not None
    assert np.any(np.abs(last_mwf) > 1e-4), "MWF output stayed silent under voiced input"
