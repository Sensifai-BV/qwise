"""Background task: purge recording folders + stale rate-limit entries.

Runs as an asyncio task started by the FastAPI ``lifespan`` hook. Walks
the configured recordings directory and removes any sub-folder whose
mtime is older than ``ttl_sec``. Also prunes IP entries from the rate
limiter to keep memory bounded.

The cleanup is best-effort: any per-folder failure is logged and the
loop continues.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path

from .ratelimit import RateLimiter

log = logging.getLogger(__name__)


def purge_stale_sessions(records_root: Path, ttl_sec: float) -> list[str]:
    """Remove session folders older than ``ttl_sec``.

    Returns the list of removed folder paths. Missing root + non-dir
    entries are skipped silently — the goal is to never crash the
    cleanup loop, no matter what's on disk.
    """
    removed: list[str] = []
    root = Path(records_root)
    if not root.is_dir():
        return removed
    now = time.time()
    cutoff = now - float(ttl_sec)
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError as ex:                       # pragma: no cover
            log.warning("[Q-WiSE] cleanup: stat failed on %s (%s)", entry, ex)
            continue
        if mtime > cutoff:
            continue
        try:
            shutil.rmtree(entry)
            removed.append(str(entry))
            log.info("[Q-WiSE] cleanup: purged stale session %s", entry)
        except OSError as ex:                       # pragma: no cover
            log.warning("[Q-WiSE] cleanup: rmtree failed on %s (%s)", entry, ex)
    return removed


async def cleanup_loop(
    records_root: Path,
    rate_limiter: RateLimiter,
    ttl_sec: float,
    interval_sec: float,
) -> None:
    """Run :func:`purge_stale_sessions` + ``rate_limiter.prune_stale``
    every ``interval_sec`` until cancelled."""
    log.info(
        "[Q-WiSE] cleanup loop online "
        "(records_root=%s, ttl=%.0fs, interval=%.0fs)",
        records_root, ttl_sec, interval_sec,
    )
    try:
        while True:
            try:
                purged = purge_stale_sessions(records_root, ttl_sec)
                ip_pruned = rate_limiter.prune_stale()
                if purged or ip_pruned:
                    log.info(
                        "[Q-WiSE] cleanup pass: %d folders, %d IPs pruned",
                        len(purged), ip_pruned,
                    )
            except Exception as ex:                 # pragma: no cover
                log.warning("[Q-WiSE] cleanup pass errored: %s", ex)
            await asyncio.sleep(float(interval_sec))
    except asyncio.CancelledError:
        log.info("[Q-WiSE] cleanup loop cancelled")
        raise


__all__ = ["purge_stale_sessions", "cleanup_loop"]
