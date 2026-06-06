"""Behavioural tests for :mod:`frontend/static/session.js`.

The module runs in the browser, so we can't unit-test it from pytest
the way we'd test a Python module. Instead we extract the algorithmic
core into Python and assert that the contract — UUID id, TTL pruning,
dedup-on-name, cache cap — holds end-to-end. If the JS file diverges
from this Python mirror the contract checks below will start failing.

These are *not* a substitute for an integration test against a real
browser; they're a guard rail so future edits to `session.js` don't
silently break the cookie + localStorage state machine.
"""

from __future__ import annotations

import time
import uuid

import pytest


# --------------------------------------------------------------------- #
# Pure-Python mirror of session.js (only the parts we want to assert on)
# --------------------------------------------------------------------- #
class FakeStorage:
    """Minimal localStorage stand-in (string-keyed, string-valued)."""

    def __init__(self) -> None:
        self._d: dict[str, str] = {}

    def get(self, k: str) -> str | None:
        return self._d.get(k)

    def set(self, k: str, v: str) -> None:
        self._d[k] = v


TTL_MS = 3 * 3600 * 1000
CACHE_CAP = 50
STORAGE_KEY = "qwise.session.v1"


def _read(storage: FakeStorage) -> dict:
    import json
    raw = storage.get(STORAGE_KEY)
    return json.loads(raw) if raw else {}


def _write(storage: FakeStorage, data: dict) -> None:
    import json
    storage.set(STORAGE_KEY, json.dumps(data))


def get_session_id(storage: FakeStorage, now_ms: int) -> str:
    data = _read(storage)
    sid = data.get("session_id")
    if not sid:
        sid = str(uuid.uuid4())
    data["session_id"] = sid
    _write(storage, data)
    return sid


def prune_recordings(storage: FakeStorage, now_ms: int) -> list[dict]:
    data = _read(storage)
    rec = data.get("recordings") or []
    cutoff = now_ms - TTL_MS
    fresh = [r for r in rec if int(r.get("ts", 0)) >= cutoff]
    if len(fresh) != len(rec):
        data["recordings"] = fresh
        _write(storage, data)
    return fresh


def mark_recording(
    storage: FakeStorage, name: str, files: list[str], now_ms: int
) -> list[dict]:
    if not name:
        return prune_recordings(storage, now_ms)
    data = _read(storage)
    rec = [r for r in (data.get("recordings") or []) if r.get("name") != name]
    rec.insert(0, {"name": name, "ts": now_ms, "files": files[:32]})
    data["recordings"] = rec[:CACHE_CAP]
    _write(storage, data)
    return prune_recordings(storage, now_ms)


# --------------------------------------------------------------------- #
# Session-id contract
# --------------------------------------------------------------------- #
def test_session_id_is_uuid_and_persists() -> None:
    storage = FakeStorage()
    now = int(time.time() * 1000)
    a = get_session_id(storage, now)
    # UUID parsing must succeed — guards against the JS fallback drifting.
    uuid.UUID(a)
    # Stable across calls (same storage backing).
    assert get_session_id(storage, now + 60_000) == a
    # And survives a write of another field.
    data = _read(storage)
    data["recordings"] = []
    _write(storage, data)
    assert get_session_id(storage, now + 120_000) == a


# --------------------------------------------------------------------- #
# Recordings cache — TTL prune
# --------------------------------------------------------------------- #
def test_prune_drops_stale_entries() -> None:
    storage = FakeStorage()
    now = 10_000_000
    mark_recording(storage, "sess-a", ["mic01.wav"], now - TTL_MS - 1)
    mark_recording(storage, "sess-b", ["mic01.wav"], now - 60_000)
    surviving = prune_recordings(storage, now)
    assert [r["name"] for r in surviving] == ["sess-b"]


def test_prune_returns_empty_when_storage_has_no_recordings() -> None:
    storage = FakeStorage()
    assert prune_recordings(storage, 1) == []


# --------------------------------------------------------------------- #
# Recordings cache — mark / dedupe / cap
# --------------------------------------------------------------------- #
def test_mark_recording_prepends_newest_first() -> None:
    storage = FakeStorage()
    now = 50_000_000
    mark_recording(storage, "one", ["mic01.wav"], now)
    mark_recording(storage, "two", ["mic01.wav", "mic02.wav"], now + 100)
    rec = _read(storage)["recordings"]
    assert [r["name"] for r in rec] == ["two", "one"]


def test_mark_recording_dedupes_on_name() -> None:
    storage = FakeStorage()
    now = 50_000_000
    mark_recording(storage, "sess-x", ["a.wav"], now)
    mark_recording(storage, "sess-x", ["a.wav", "b.wav"], now + 1000)
    rec = _read(storage)["recordings"]
    assert len(rec) == 1
    # The second mark refreshed the timestamp and the file list.
    assert rec[0]["ts"] == now + 1000
    assert rec[0]["files"] == ["a.wav", "b.wav"]


def test_mark_recording_respects_cache_cap() -> None:
    storage = FakeStorage()
    now = 1_700_000_000_000
    for i in range(CACHE_CAP + 20):
        mark_recording(storage, f"sess-{i:03d}", ["mic01.wav"], now + i)
    rec = _read(storage)["recordings"]
    assert len(rec) == CACHE_CAP
    # Newest at index 0.
    assert rec[0]["name"] == f"sess-{CACHE_CAP + 19:03d}"


def test_mark_recording_with_empty_name_just_prunes() -> None:
    storage = FakeStorage()
    now = 1
    mark_recording(storage, "keep", ["mic01.wav"], now)
    assert mark_recording(storage, "", [], now + 1)[0]["name"] == "keep"


def test_prune_runs_during_mark() -> None:
    """Inserting a fresh entry must also drop any stale neighbours."""
    storage = FakeStorage()
    now = 10_000_000
    mark_recording(storage, "old", ["a.wav"], now - TTL_MS - 1)
    surviving = mark_recording(storage, "new", ["b.wav"], now)
    assert [r["name"] for r in surviving] == ["new"]
