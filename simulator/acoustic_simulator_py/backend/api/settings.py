"""Runtime knobs read from environment variables.

Centralises the env-var contract so :mod:`backend.api.app`,
:mod:`backend.api.ratelimit` and :mod:`backend.api.cleanup` can share it.

Defaults match what the task brief asked for (10 recordings per IP per
3 hours, cleanup every 5 minutes); HF Spaces / production overrides
land via ``QWISE_RATE_LIMIT_MAX``, ``QWISE_RATE_LIMIT_WINDOW_SEC``,
``QWISE_CLEANUP_INTERVAL_SEC``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeSettings:
    rate_limit_max: int           # max recording_start per window per IP
    rate_limit_window_sec: float  # window length (also the recording TTL)
    cleanup_interval_sec: float   # seconds between purge passes


def load_runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        rate_limit_max=int(os.getenv("QWISE_RATE_LIMIT_MAX", "10")),
        rate_limit_window_sec=float(
            os.getenv("QWISE_RATE_LIMIT_WINDOW_SEC", str(3 * 3600))
        ),
        cleanup_interval_sec=float(
            os.getenv("QWISE_CLEANUP_INTERVAL_SEC", "300")
        ),
    )


__all__ = ["RuntimeSettings", "load_runtime_settings"]
