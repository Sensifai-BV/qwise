"""FastAPI surface — no MATLAB counterpart.

* :mod:`backend.api.app`        — application factory + ``uvicorn`` entry
* :mod:`backend.api.rest`       — REST routes (health + config)
* :mod:`backend.api.ws_stream`  — bidirectional ``/ws/stream`` handler
* :mod:`backend.api.pipeline`   — per-WebSocket :class:`PipelineSession`
"""

from .app import app, create_app
from .pipeline import ConfigUpdateError, PipelineSession

__all__ = ["app", "create_app", "PipelineSession", "ConfigUpdateError"]
