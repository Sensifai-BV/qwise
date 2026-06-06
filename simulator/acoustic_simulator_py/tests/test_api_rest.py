"""Tests for the REST endpoints.

Smoke-level: every endpoint returns 200 + a JSON body of the right
shape. Heavier configuration semantics live in ``test_config.py``.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.api.app import app, create_app


client = TestClient(app)


def test_healthz_returns_ok() -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_version_endpoint() -> None:
    r = client.get("/api/version")
    assert r.status_code == 200
    assert "version" in r.json()


def test_config_default_returns_full_config() -> None:
    r = client.get("/api/config/default")
    assert r.status_code == 200
    cfg = r.json()
    # Spot-check every top-level section the frontend will read.
    for key in (
        "fs", "frame_size", "n_mics", "mic_spacing", "mic_geometry",
        "human_height", "mouth_height", "slant_dist", "elev_deg",
        "drone", "env", "mixer", "vad", "mwf", "record", "ui",
        "drone_rpm", "drone_blades", "distance_ref",
    ):
        assert key in cfg, f"missing {key}"
    # Nested submodels arrive as dicts.
    assert isinstance(cfg["env"], dict)
    assert "distance_from_mouth" in cfg["env"]


def test_config_schema_groups_by_section() -> None:
    r = client.get("/api/config/schema")
    assert r.status_code == 200
    body = r.json()
    assert "sidebar" in body
    entries = body["sidebar"]
    # Every entry must carry the four widget hints + a current value.
    for e in entries:
        for key in ("key", "value", "section", "widget", "label"):
            assert key in e, f"entry missing {key}: {e}"
    # ``mwf.method`` and ``n_mics`` are part of the user-approved set.
    keys = {e["key"] for e in entries}
    assert "n_mics" in keys
    assert "mwf.method" in keys


def test_create_app_factory_is_isolated() -> None:
    """``create_app`` builds a fresh instance — useful for parallel tests."""
    a = create_app()
    b = create_app()
    assert a is not b
