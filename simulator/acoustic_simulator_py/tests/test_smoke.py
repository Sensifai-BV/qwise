"""Smoke tests that verify the scaffold itself is wired up.

Task 11 will mirror the MATLAB tests/ suite; for now these prove that
the package imports cleanly and the FastAPI app spins up.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.api.app import app


def test_package_imports() -> None:
    """Every sub-package must be importable without side effects."""
    import backend  # noqa: F401
    import backend.api  # noqa: F401
    import backend.audio  # noqa: F401
    import backend.config  # noqa: F401
    import backend.core  # noqa: F401
    import backend.mwf  # noqa: F401
    import backend.vad  # noqa: F401


def test_healthz_returns_ok() -> None:
    """The healthz endpoint must return 200 OK with a status field."""
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
