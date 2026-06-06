"""Tests for :class:`backend.api.ratelimit.RateLimiter`."""

from __future__ import annotations

import time

import pytest

from backend.api.ratelimit import RateLimiter, client_ip_from_headers


# --------------------------------------------------------------------- #
# Constructor sanity
# --------------------------------------------------------------------- #
def test_constructor_rejects_zero_limit() -> None:
    with pytest.raises(ValueError):
        RateLimiter(max_per_window=0, window_sec=60)


def test_constructor_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError):
        RateLimiter(max_per_window=10, window_sec=0)


# --------------------------------------------------------------------- #
# Allow/deny flow
# --------------------------------------------------------------------- #
def test_status_before_any_consume_is_full_quota() -> None:
    rl = RateLimiter(max_per_window=3, window_sec=60)
    s = rl.status("1.2.3.4")
    assert s.allowed is True
    assert s.used == 0
    assert s.remaining == 3
    assert s.limit == 3


def test_try_consume_records_event_until_limit() -> None:
    rl = RateLimiter(max_per_window=3, window_sec=60)
    ip = "203.0.113.7"
    for expected_used in (1, 2, 3):
        s = rl.try_consume(ip)
        assert s.allowed is True
        assert s.used == expected_used
        assert s.remaining == 3 - expected_used

    # The fourth call is denied AND must not add a new event.
    denied = rl.try_consume(ip)
    assert denied.allowed is False
    assert denied.used == 3
    assert denied.remaining == 0

    # ``status`` confirms the deny didn't bump the counter.
    again = rl.status(ip)
    assert again.used == 3
    assert again.allowed is False


def test_separate_ips_have_independent_quotas() -> None:
    rl = RateLimiter(max_per_window=2, window_sec=60)
    rl.try_consume("a")
    rl.try_consume("a")
    assert rl.try_consume("a").allowed is False
    # Different IP — still has full quota.
    assert rl.try_consume("b").allowed is True
    assert rl.status("b").used == 1


# --------------------------------------------------------------------- #
# Window expiry
# --------------------------------------------------------------------- #
def test_events_expire_after_window(monkeypatch: pytest.MonkeyPatch) -> None:
    rl = RateLimiter(max_per_window=2, window_sec=10)
    fake_now = {"t": 1_000_000.0}
    monkeypatch.setattr(time, "time", lambda: fake_now["t"])

    rl.try_consume("ip")
    rl.try_consume("ip")
    assert rl.status("ip").used == 2

    # Advance past the window — both events age out.
    fake_now["t"] += 11
    s = rl.status("ip")
    assert s.used == 0
    assert s.allowed is True


def test_next_reset_matches_oldest_event(monkeypatch: pytest.MonkeyPatch) -> None:
    rl = RateLimiter(max_per_window=2, window_sec=100)
    fake_now = {"t": 5000.0}
    monkeypatch.setattr(time, "time", lambda: fake_now["t"])

    s1 = rl.try_consume("ip")
    fake_now["t"] += 30
    s2 = rl.try_consume("ip")

    # Oldest event was at t=5000; window=100 → reset at 5100.
    assert s1.next_reset == pytest.approx(5100.0)
    assert s2.next_reset == pytest.approx(5100.0)


# --------------------------------------------------------------------- #
# Pruning
# --------------------------------------------------------------------- #
def test_reset_drops_one_ip() -> None:
    rl = RateLimiter(max_per_window=2, window_sec=60)
    rl.try_consume("a")
    rl.try_consume("b")
    rl.reset("a")
    assert rl.status("a").used == 0
    assert rl.status("b").used == 1


def test_reset_all_clears_everything() -> None:
    rl = RateLimiter(max_per_window=2, window_sec=60)
    rl.try_consume("a")
    rl.try_consume("b")
    rl.reset()
    assert rl.status("a").used == 0
    assert rl.status("b").used == 0


def test_prune_stale_drops_empty_ips(monkeypatch: pytest.MonkeyPatch) -> None:
    rl = RateLimiter(max_per_window=2, window_sec=10)
    fake_now = {"t": 1_000_000.0}
    monkeypatch.setattr(time, "time", lambda: fake_now["t"])

    rl.try_consume("a")
    rl.try_consume("b")
    fake_now["t"] += 11        # both events age out
    pruned = rl.prune_stale()
    assert pruned == 2


# --------------------------------------------------------------------- #
# client_ip_from_headers
# --------------------------------------------------------------------- #
def test_client_ip_prefers_forwarded_for() -> None:
    headers = [("Host", "x"), ("X-Forwarded-For", "10.0.0.1, 192.168.1.1")]
    assert client_ip_from_headers(headers, fallback="2.2.2.2") == "10.0.0.1"


def test_client_ip_falls_back_to_real_ip_header() -> None:
    headers = [("X-Real-IP", " 192.168.1.5 ")]
    assert client_ip_from_headers(headers, fallback="2.2.2.2") == "192.168.1.5"


def test_client_ip_falls_back_to_argument_when_headers_silent() -> None:
    assert client_ip_from_headers([], fallback="9.9.9.9") == "9.9.9.9"
    assert client_ip_from_headers([], fallback=None) == "unknown"
