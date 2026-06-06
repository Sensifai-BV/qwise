"""Tests for the speech-WAV upload endpoint + the matching WebSocket
``load_speech_wav`` / ``clear_speech_wav`` control flow.
"""

from __future__ import annotations

import io
import json
import os
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient


@pytest.fixture
def app_and_uploads(monkeypatch, tmp_path):
    """Spin up a fresh app with QWISE_DATA_DIR pointed at ``tmp_path``.

    The upload helper resolves the uploads root at *call time* off
    ``QWISE_DATA_DIR``, so we monkeypatch the env var here and only
    import the app module afterwards.

    NB: ``backend.api`` re-exports ``app`` from its submodule, so
    ``import backend.api.app as x`` resolves to the FastAPI instance
    (the attribute), not the module. We go through
    :func:`importlib.import_module`, which always returns the module
    object from ``sys.modules``.
    """
    monkeypatch.setenv("QWISE_DATA_DIR", str(tmp_path))
    import importlib
    app_module = importlib.import_module("backend.api.app")
    uploads_module = importlib.import_module("backend.api.uploads")
    importlib.reload(uploads_module)
    importlib.reload(app_module)
    app = app_module.create_app()
    return app, tmp_path


@pytest.fixture
def speech_wav_bytes() -> bytes:
    """A 0.5 s mono 16 kHz WAV with a 440 Hz tone, in-memory."""
    fs = 16000
    t = np.arange(int(fs * 0.5)) / fs
    sig = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, sig, fs, subtype="PCM_16", format="WAV")
    return buf.getvalue()


# --------------------------------------------------------------------- #
# POST /api/speech-wav
# --------------------------------------------------------------------- #
def test_upload_round_trip(app_and_uploads, speech_wav_bytes) -> None:
    app, data_dir = app_and_uploads
    with TestClient(app) as client:
        r = client.post(
            "/api/speech-wav",
            files={"file": ("voiced.wav", speech_wav_bytes, "audio/wav")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "voiced.wav"
    assert body["fs"] == 16000
    assert body["channels"] == 1
    assert body["duration"] == pytest.approx(0.5, abs=0.01)
    # File landed under <data_dir>/uploads/<uuid>.wav.
    p = Path(body["path"]).resolve()
    assert p.is_file()
    assert (data_dir / "uploads") in p.parents


def test_upload_rejects_empty_body(app_and_uploads) -> None:
    app, _ = app_and_uploads
    with TestClient(app) as client:
        r = client.post(
            "/api/speech-wav",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
    assert r.status_code == 400


def test_upload_rejects_disallowed_extension(app_and_uploads, speech_wav_bytes) -> None:
    app, _ = app_and_uploads
    with TestClient(app) as client:
        r = client.post(
            "/api/speech-wav",
            files={"file": ("not_audio.exe", speech_wav_bytes, "application/octet-stream")},
        )
    assert r.status_code == 415


def test_upload_rejects_oversize_payload(monkeypatch, tmp_path) -> None:
    """File > MAX_UPLOAD_BYTES is refused with 413."""
    monkeypatch.setenv("QWISE_DATA_DIR", str(tmp_path))
    import importlib
    uploads_module = importlib.import_module("backend.api.uploads")
    importlib.reload(uploads_module)
    monkeypatch.setattr(uploads_module, "MAX_UPLOAD_BYTES", 1024)  # tiny cap

    app_module = importlib.import_module("backend.api.app")
    importlib.reload(app_module)

    too_big = b"\x00" * 4096
    with TestClient(app_module.create_app()) as client:
        r = client.post(
            "/api/speech-wav",
            files={"file": ("big.wav", too_big, "audio/wav")},
        )
    assert r.status_code == 413


def test_upload_rejects_unreadable_audio(app_and_uploads) -> None:
    app, _ = app_and_uploads
    with TestClient(app) as client:
        r = client.post(
            "/api/speech-wav",
            files={"file": ("broken.wav", b"NOT A WAV", "audio/wav")},
        )
    assert r.status_code == 400


# --------------------------------------------------------------------- #
# WebSocket load_speech_wav / clear_speech_wav
# --------------------------------------------------------------------- #
def test_ws_load_speech_wav_round_trip(app_and_uploads, speech_wav_bytes) -> None:
    app, _ = app_and_uploads
    with TestClient(app) as client:
        # 1) upload
        r = client.post(
            "/api/speech-wav",
            files={"file": ("voiced.wav", speech_wav_bytes, "audio/wav")},
        )
        path = r.json()["path"]

        # 2) ask the WS to load it
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()                       # hello
            ws.send_text(json.dumps({"type": "load_speech_wav", "path": path}))
            ack = ws.receive_json()
            assert ack["kind"] == "load_speech_wav"
            assert ack["ok"] is True
            assert ack["has_wav"] is True

            # Now clear it.
            ws.send_text(json.dumps({"type": "clear_speech_wav"}))
            ack2 = ws.receive_json()
            assert ack2["kind"] == "clear_speech_wav"
            assert ack2["ok"] is True
            assert ack2["has_wav"] is False


def test_ws_load_speech_wav_rejects_paths_outside_uploads(app_and_uploads) -> None:
    """Path traversal must be denied so a client can't load arbitrary
    files off the server's disk."""
    app, _ = app_and_uploads
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()                       # hello
            ws.send_text(json.dumps({
                "type": "load_speech_wav", "path": "/etc/passwd",
            }))
            ack = ws.receive_json()
            assert ack["ok"] is False
            assert "uploads" in ack["error"]


def test_ws_load_speech_wav_requires_path(app_and_uploads) -> None:
    app, _ = app_and_uploads
    with TestClient(app) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()                       # hello
            ws.send_text(json.dumps({"type": "load_speech_wav"}))
            ack = ws.receive_json()
            assert ack["ok"] is False
            assert "path is required" in ack["error"]
