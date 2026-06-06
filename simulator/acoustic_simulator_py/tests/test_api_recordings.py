"""Tests for the recording catalogue endpoints (Task 18) and the
end-to-end WS recording → file → playback flow.
"""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient


@pytest.fixture
def app_and_data(monkeypatch, tmp_path):
    """Spin up a fresh app with ``QWISE_DATA_DIR`` pointed at ``tmp_path``.

    Two backend modules cache the data root at import time: ``audio.io``
    (resolves ``DEFAULT_DATA_DIR``) and ``api.app`` (resolves
    ``records_root`` in ``lifespan``). Re-import both so the test
    operates against a clean directory.
    """
    monkeypatch.setenv("QWISE_DATA_DIR", str(tmp_path))
    audio_mod = importlib.import_module("backend.audio.io")
    importlib.reload(audio_mod)
    uploads_mod = importlib.import_module("backend.api.uploads")
    importlib.reload(uploads_mod)
    app_mod = importlib.import_module("backend.api.app")
    importlib.reload(app_mod)
    return app_mod.create_app(), tmp_path


def _seed_session(records_dir: Path, name: str, files: dict[str, np.ndarray],
                  fs: int = 16000) -> Path:
    """Drop a fake recording session on disk so the listing endpoint
    has something to surface."""
    sess = records_dir / name
    sess.mkdir(parents=True, exist_ok=True)
    for fname, samples in files.items():
        sf.write(str(sess / fname), samples, fs, subtype="PCM_16")
    return sess


# --------------------------------------------------------------------- #
# GET /api/recordings
# --------------------------------------------------------------------- #
def test_recordings_empty_when_no_sessions(app_and_data) -> None:
    app, _ = app_and_data
    with TestClient(app) as client:
        r = client.get("/api/recordings")
    assert r.status_code == 200
    assert r.json() == []


def test_recordings_lists_sessions_newest_first(app_and_data) -> None:
    app, data_dir = app_and_data
    records_dir = data_dir / "recordings"
    a = _seed_session(records_dir, "qwise_multi_old", {
        "mic01.wav": np.zeros(1600, dtype=np.float32),
    })
    # Sleep so mtimes are strictly ordered (FS resolution can be 1 ms).
    time.sleep(0.01)
    b = _seed_session(records_dir, "qwise_multi_new", {
        "mic01.wav": np.zeros(1600, dtype=np.float32),
        "mic02.wav": np.zeros(1600, dtype=np.float32),
        "vad.wav":   np.zeros(1600, dtype=np.float32),
    })
    with TestClient(app) as client:
        r = client.get("/api/recordings")
    assert r.status_code == 200
    body = r.json()
    assert [e["name"] for e in body] == ["qwise_multi_new", "qwise_multi_old"]
    assert body[0]["files"] == ["mic01.wav", "mic02.wav", "vad.wav"]
    assert body[1]["files"] == ["mic01.wav"]


def test_recordings_skips_empty_folders_and_files(app_and_data) -> None:
    app, data_dir = app_and_data
    records_dir = data_dir / "recordings"
    records_dir.mkdir()
    (records_dir / "qwise_multi_empty").mkdir()        # folder, no WAVs
    (records_dir / "stray.txt").write_text("ignore me")  # not a directory
    _seed_session(records_dir, "qwise_multi_real", {
        "mic01.wav": np.zeros(1600, dtype=np.float32),
    })
    with TestClient(app) as client:
        names = [e["name"] for e in client.get("/api/recordings").json()]
    assert names == ["qwise_multi_real"]


def test_recordings_respects_limit(app_and_data) -> None:
    app, data_dir = app_and_data
    records_dir = data_dir / "recordings"
    for i in range(5):
        time.sleep(0.005)
        _seed_session(records_dir, f"qwise_multi_{i:02d}", {
            "mic01.wav": np.zeros(800, dtype=np.float32),
        })
    with TestClient(app) as client:
        r = client.get("/api/recordings?limit=2")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    # Newest first.
    assert body[0]["name"] == "qwise_multi_04"
    assert body[1]["name"] == "qwise_multi_03"


# --------------------------------------------------------------------- #
# GET /api/recordings/{session}/files/{filename}
# --------------------------------------------------------------------- #
def test_get_file_streams_wav(app_and_data) -> None:
    app, data_dir = app_and_data
    rng = np.random.default_rng(0)
    # Cap below ±1.0 so the PCM_16 quantiser never saturates on edges
    # (a saturated 1.0 reads back as 0.99997, blowing the tolerance).
    samples = np.clip(
        0.3 * rng.standard_normal(8000), -0.9, 0.9
    ).astype(np.float32)
    _seed_session(data_dir / "recordings", "qwise_multi_one", {
        "mic01.wav": samples,
    })
    with TestClient(app) as client:
        r = client.get("/api/recordings/qwise_multi_one/files/mic01.wav")
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("audio/wav")
    # Round-trip the bytes through soundfile. PCM_16 quantisation gives
    # us ≈ 1/32768 ≈ 3e-5 of round-trip error per sample.
    import io
    back, fs = sf.read(io.BytesIO(r.content), dtype="float32")
    assert fs == 16000
    np.testing.assert_allclose(back, samples, atol=1.0 / 32767)


def test_get_file_rejects_traversal(app_and_data) -> None:
    app, _ = app_and_data
    with TestClient(app) as client:
        # The router validates against a strict regex first.
        r1 = client.get("/api/recordings/..%2F..%2Fetc/files/passwd.wav")
        assert r1.status_code in (400, 404)
        # Also reject a non-WAV extension.
        r2 = client.get("/api/recordings/qwise_multi_one/files/notes.txt")
        assert r2.status_code == 400


def test_get_file_404_when_missing(app_and_data) -> None:
    app, data_dir = app_and_data
    _seed_session(data_dir / "recordings", "qwise_multi_real", {
        "mic01.wav": np.zeros(800, dtype=np.float32),
    })
    with TestClient(app) as client:
        r = client.get("/api/recordings/qwise_multi_real/files/mwf.wav")
    assert r.status_code == 404


# --------------------------------------------------------------------- #
# WebSocket record-start → record-stop → listing → file serving
# --------------------------------------------------------------------- #
def test_ws_recording_round_trip_to_file_endpoint(app_and_data) -> None:
    app, _ = app_and_data
    with TestClient(app) as client:
        # Open the WS, record a short session. VAD/MWF default to ON,
        # so the session writes mic + vad + mwf tracks unless we toggle
        # them off — this test exercises the "mic only" path.
        with client.websocket_connect("/ws/stream") as ws:
            hello = ws.receive_json()
            n_mics = hello["n_mics"]
            n = hello["frame_size"]

            # Disable VAD (also disables MWF) so the recording is mic-only.
            ws.send_text(json.dumps({"type": "enable_vad", "on": False}))
            ws.receive_json()

            ws.send_text(json.dumps({"type": "recording_start"}))
            start_ack = ws.receive_json()
            assert start_ack["ok"] is True

            rng = np.random.default_rng(1)
            for _ in range(3):
                ws.send_bytes(rng.standard_normal(n).astype(np.float32).tobytes())
                ws.receive_json()    # frame meta
                ws.receive_bytes()   # binary audio

            ws.send_text(json.dumps({"type": "recording_stop"}))
            stop_ack = ws.receive_json()
            assert stop_ack["ok"] is True

        # Listing must contain the just-finished session with N mic files.
        listing = client.get("/api/recordings").json()
        assert listing, "expected at least one session after recording_stop"
        latest = listing[0]
        expected = [f"mic{m+1:02d}.wav" for m in range(n_mics)]
        assert latest["files"] == expected

        # And we can stream one of them back.
        r = client.get(
            f"/api/recordings/{latest['name']}/files/{expected[0]}"
        )
        assert r.status_code == 200
        assert r.headers.get("content-type", "").startswith("audio/wav")
