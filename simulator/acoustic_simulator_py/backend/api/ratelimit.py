"""Per-IP rate limit for the ``recording_start`` action.

A sliding-window counter. Each rec-start adds a timestamp under the
caller's IP; the counter prunes entries older than ``window_sec`` at
every check, so storage stays bounded to ``≤ max_per_window`` entries
per active IP. Thread-safe via a single :class:`threading.Lock`.

Use:
    rl = RateLimiter(max_per_window=10, window_sec=3 * 3600)
    status = rl.try_consume("203.0.113.7")
    if not status.allowed: ...                          # rate-limited
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any, Iterable


# --------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------- #
@dataclass
class QuotaStatus:
    """Snapshot returned by :meth:`RateLimiter.status` / ``try_consume``."""

    allowed: bool        # would a new attempt right now be allowed?
    used: int            # events inside the current window
    remaining: int       # ``max(0, limit - used)``
    limit: int           # ``max_per_window``
    window_sec: float    # window length in seconds
    next_reset: float    # epoch seconds when the oldest event ages out;
                         # equals current time when no events are recorded

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def client_ip_from_headers(
    headers: Iterable[tuple[str, str] | tuple[bytes, bytes]],
    fallback: str | None = None,
) -> str:
    """Best-effort client IP — respects ``X-Forwarded-For`` for proxied
    deployments (HuggingFace Spaces sits behind a CDN). Returns
    ``fallback`` when nothing usable is present.
    """
    norm: dict[str, str] = {}
    for k, v in headers:
        ks = k.decode() if isinstance(k, (bytes, bytearray)) else k
        vs = v.decode() if isinstance(v, (bytes, bytearray)) else v
        norm[ks.lower()] = vs

    xff = norm.get("x-forwarded-for")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    real = norm.get("x-real-ip")
    if real:
        return real.strip()
    return fallback or "unknown"


# --------------------------------------------------------------------- #
# Rate limiter
# --------------------------------------------------------------------- #
class RateLimiter:
    """Sliding-window per-IP counter."""

    def __init__(self, max_per_window: int = 10, window_sec: float = 3 * 3600):
        if max_per_window < 1:
            raise ValueError("max_per_window must be >= 1")
        if window_sec <= 0:
            raise ValueError("window_sec must be positive")
        self.max_per_window = int(max_per_window)
        self.window_sec = float(window_sec)
        self._events: dict[str, deque[float]] = {}
        self._lock = Lock()

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    def status(self, ip: str) -> QuotaStatus:
        """Return current quota without consuming an event."""
        now = time.time()
        with self._lock:
            events = self._prune_locked(ip, now)
            return self._build_status(events, now, allowed=len(events) < self.max_per_window)

    def try_consume(self, ip: str) -> QuotaStatus:
        """Attempt to register a new event under ``ip``.

        If the post-prune count is already at the limit the call is
        rejected (no event added) and the returned status has
        ``allowed=False``. Otherwise the event lands and the status
        reflects the post-consume counts.
        """
        now = time.time()
        with self._lock:
            events = self._prune_locked(ip, now)
            if len(events) >= self.max_per_window:
                return self._build_status(events, now, allowed=False)
            events.append(now)
            return self._build_status(events, now, allowed=True)

    def reset(self, ip: str | None = None) -> None:
        """Drop a single IP's history, or all IPs' if ``ip`` is None."""
        with self._lock:
            if ip is None:
                self._events.clear()
            else:
                self._events.pop(ip, None)

    def prune_stale(self) -> int:
        """Drop IPs whose entire window has expired. Returns count removed."""
        now = time.time()
        removed = 0
        with self._lock:
            for ip in list(self._events.keys()):
                events = self._prune_locked(ip, now)
                if not events:
                    del self._events[ip]
                    removed += 1
        return removed

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _prune_locked(self, ip: str, now: float) -> deque[float]:
        events = self._events.setdefault(ip, deque())
        cutoff = now - self.window_sec
        while events and events[0] < cutoff:
            events.popleft()
        return events

    def _build_status(
        self, events: deque[float], now: float, *, allowed: bool
    ) -> QuotaStatus:
        used = len(events)
        remaining = max(0, self.max_per_window - used)
        next_reset = events[0] + self.window_sec if events else now
        return QuotaStatus(
            allowed=allowed,
            used=used,
            remaining=remaining,
            limit=self.max_per_window,
            window_sec=self.window_sec,
            next_reset=next_reset,
        )


__all__ = ["RateLimiter", "QuotaStatus", "client_ip_from_headers"]
