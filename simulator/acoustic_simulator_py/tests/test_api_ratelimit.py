"""Integration tests for the rate-limit + quota wiring."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.api.app import app, create_app, get_rate_limiter
from backend.api.ratelimit import RateLimiter


@pytest.fixture
def small_quota_client() -> TestClient:
    """A fresh app with a tiny rate limit so the tests can exhaust it."""
    test_app = create_app()
    # Install a small limiter on the app state so the WS / REST handlers
    # both see it (they look it up via ``get_rate_limiter(app)``).
    test_app.state.rate_limiter = RateLimiter(max_per_window=3, window_sec=60)
    return TestClient(test_app)


# --------------------------------------------------------------------- #
# /api/quota
# --------------------------------------------------------------------- #
def test_quota_endpoint_starts_full(small_quota_client) -> None:
    r = small_quota_client.get("/api/quota")
    assert r.status_code == 200
    body = r.json()
    assert body["limit"] == 3
    assert body["used"] == 0
    assert body["remaining"] == 3
    assert body["allowed"] is True
    assert "ip" in body


def test_quota_endpoint_reflects_consumed_events(small_quota_client) -> None:
    rl: RateLimiter = small_quota_client.app.state.rate_limiter
    ip = small_quota_client.get("/api/quota").json()["ip"]
    rl.try_consume(ip)
    rl.try_consume(ip)
    body = small_quota_client.get("/api/quota").json()
    assert body["used"] == 2
    assert body["remaining"] == 1
    assert body["allowed"] is True


# --------------------------------------------------------------------- #
# Rate-limited WebSocket recording_start
# --------------------------------------------------------------------- #
def test_recording_start_succeeds_within_quota(small_quota_client) -> None:
    with small_quota_client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()                              # hello
        ws.send_text(json.dumps({"type": "recording_start"}))
        ack = ws.receive_json()
        ws.send_text(json.dumps({"type": "recording_stop"}))
        ws.receive_json()
    assert ack["kind"] == "recording_start"
    assert ack["ok"] is True
    quota = ack["quota"]
    assert quota["used"] == 1
    assert quota["remaining"] == 2          # limit 3 → 2 left after start


def test_recording_start_denied_when_quota_exhausted(small_quota_client) -> None:
    """Consume the entire quota, then attempt one more start over the WS.

    The 4th attempt must come back with ``ok=false``, ``error='rate_limited'``,
    and a ``quota`` payload showing ``used == limit``. No session folder
    should have been opened (the limiter consumes the slot *before* the
    pipeline runs ``rec_start_session``).
    """
    rl: RateLimiter = small_quota_client.app.state.rate_limiter

    # Look up the client IP the WS will see and burn the quota directly.
    quota = small_quota_client.get("/api/quota").json()
    ip = quota["ip"]
    for _ in range(rl.max_per_window):
        rl.try_consume(ip)

    with small_quota_client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()                              # hello
        ws.send_text(json.dumps({"type": "recording_start"}))
        ack = ws.receive_json()

    assert ack["type"] == "ack"
    assert ack["kind"] == "recording_start"
    assert ack["ok"] is False
    assert ack["error"] == "rate_limited"
    assert ack["quota"]["used"] == ack["quota"]["limit"]
    assert ack["quota"]["remaining"] == 0
    assert ack["quota"]["allowed"] is False


def test_non_recording_controls_are_not_rate_limited(small_quota_client) -> None:
    """Other control messages must NOT consume quota, no matter how many
    of them go through."""
    rl: RateLimiter = small_quota_client.app.state.rate_limiter
    ip = small_quota_client.get("/api/quota").json()["ip"]

    with small_quota_client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()                              # hello
        for _ in range(20):
            ws.send_text(json.dumps({"type": "drone_on", "on": True}))
            ws.receive_json()
            ws.send_text(json.dumps({"type": "drone_on", "on": False}))
            ws.receive_json()
    assert rl.status(ip).used == 0


# --------------------------------------------------------------------- #
# get_rate_limiter (lazy build path)
# --------------------------------------------------------------------- #
def test_get_rate_limiter_returns_singleton_per_app() -> None:
    a = create_app()
    rl1 = get_rate_limiter(a)
    rl2 = get_rate_limiter(a)
    assert rl1 is rl2
    # Different apps must get independent limiters.
    b = create_app()
    rl3 = get_rate_limiter(b)
    assert rl3 is not rl1
