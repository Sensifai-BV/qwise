"""Tests for :mod:`backend.api.cleanup`."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from backend.api.cleanup import cleanup_loop, purge_stale_sessions
from backend.api.ratelimit import RateLimiter


def _mk_session(root: Path, name: str, age_sec: float, files: int = 1) -> Path:
    """Create a fake session folder with the given mtime offset."""
    d = root / name
    d.mkdir(parents=True)
    for i in range(files):
        (d / f"mic{i:02d}.wav").write_bytes(b"RIFF.....")
    target = time.time() - float(age_sec)
    os.utime(d, (target, target))
    for p in d.iterdir():
        os.utime(p, (target, target))
    return d


def test_purge_stale_only_drops_old_folders(tmp_path: Path) -> None:
    fresh = _mk_session(tmp_path, "qwise_multi_fresh", age_sec=10)
    stale1 = _mk_session(tmp_path, "qwise_multi_stale1", age_sec=3_600)
    stale2 = _mk_session(tmp_path, "qwise_multi_stale2", age_sec=10_000)

    removed = purge_stale_sessions(tmp_path, ttl_sec=600)
    assert sorted(removed) == sorted([str(stale1), str(stale2)])
    assert fresh.is_dir()
    assert not stale1.exists()
    assert not stale2.exists()


def test_purge_stale_skips_files_and_missing_root(tmp_path: Path) -> None:
    # A stray file alongside session folders must not crash the cleanup.
    (tmp_path / "stray.txt").write_text("ignore me")
    fresh = _mk_session(tmp_path, "qwise_multi_now", age_sec=0)

    removed = purge_stale_sessions(tmp_path, ttl_sec=60)
    assert removed == []
    assert fresh.is_dir()
    assert (tmp_path / "stray.txt").exists()

    # Missing root → empty list, no error.
    missing_root = tmp_path / "does_not_exist"
    assert purge_stale_sessions(missing_root, ttl_sec=60) == []


def test_purge_stale_returns_empty_when_no_folders_match(tmp_path: Path) -> None:
    _mk_session(tmp_path, "young1", age_sec=5)
    _mk_session(tmp_path, "young2", age_sec=20)
    assert purge_stale_sessions(tmp_path, ttl_sec=600) == []


# --------------------------------------------------------------------- #
# Async loop wiring
# --------------------------------------------------------------------- #
def test_cleanup_loop_runs_at_least_once(tmp_path: Path) -> None:
    """Drive the cleanup loop briefly, then cancel — the stale folder
    must be gone and the rate limiter pruned."""
    rl = RateLimiter(max_per_window=2, window_sec=1)
    # Simulate an event from a long time ago so prune_stale removes it.
    rl.try_consume("203.0.113.7")
    time.sleep(1.05)

    stale = _mk_session(tmp_path, "qwise_multi_stale", age_sec=10)

    async def runner() -> int:
        task = asyncio.create_task(
            cleanup_loop(tmp_path, rl, ttl_sec=1, interval_sec=0.05)
        )
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return 0

    asyncio.run(runner())
    assert not stale.exists()
    # The expired event should have been pruned.
    assert rl.status("203.0.113.7").used == 0
