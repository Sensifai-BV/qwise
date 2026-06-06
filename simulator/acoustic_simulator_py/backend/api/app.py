"""FastAPI application factory.

Wires together the REST + WebSocket routers, the per-IP
:class:`RateLimiter`, and the background cleanup loop. The latter two
hang off ``app.state`` so request handlers and tests can reach them
through ``request.app.state`` / ``ws.app.state``.

Lifecycle (via FastAPI ``lifespan``):

    on startup → build :class:`RateLimiter`, start :func:`cleanup_loop`
                 as an asyncio task scoped to the app.
    on shutdown → cancel the cleanup task and await its tear-down.

Tests that use ``TestClient(app)`` *without* a ``with`` context skip
the lifespan; the WebSocket route therefore looks up the limiter via
:func:`get_rate_limiter`, which lazy-builds one if startup didn't run.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import Scope

from ..audio.io import DEFAULT_DATA_DIR
from ..config import default
from .cleanup import cleanup_loop
from .ratelimit import RateLimiter
from .recordings import router as recordings_router
from .rest import router as rest_router
from .settings import RuntimeSettings, load_runtime_settings
from .uploads import router as uploads_router
from .ws_stream import router as ws_router

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "frontend"

log = logging.getLogger(__name__)


class NoCacheStaticFiles(StaticFiles):
    """``StaticFiles`` variant that forces browsers to revalidate every
    static asset.

    Without this, Chrome / Firefox happily serve stale JS / CSS bundles
    across deploys: switching the spectrogram colormap from green to the
    MATLAB-style hot palette only takes effect after the user does a hard
    reload. The asset bundle is small, so the perf cost is negligible.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:    # type: ignore[override]
        resp = await super().get_response(path, scope)
        # Apply only to successful asset hits; 404s keep their default headers.
        if 200 <= resp.status_code < 300:
            resp.headers["Cache-Control"] = (
                "no-cache, no-store, must-revalidate"
            )
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
        return resp


# --------------------------------------------------------------------- #
# Lifespan + state helpers
# --------------------------------------------------------------------- #
def _records_root(settings_data_dir: Path | None = None) -> Path:
    """Resolve the recordings directory the same way :class:`AudioIO` does."""
    cfg = default()
    base = settings_data_dir or DEFAULT_DATA_DIR
    p = Path(cfg.record.dir)
    return p if p.is_absolute() else (Path(base) / p)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build per-app singletons + start the cleanup task."""
    settings: RuntimeSettings = load_runtime_settings()
    rl = RateLimiter(
        max_per_window=settings.rate_limit_max,
        window_sec=settings.rate_limit_window_sec,
    )
    app.state.settings = settings
    app.state.rate_limiter = rl
    app.state.records_root = _records_root()

    loop = asyncio.get_running_loop()
    app.state.cleanup_task = loop.create_task(
        cleanup_loop(
            app.state.records_root,
            rl,
            ttl_sec=settings.rate_limit_window_sec,
            interval_sec=settings.cleanup_interval_sec,
        )
    )
    log.info(
        "[Q-WiSE] app online (rate_limit %d / %.0fs, cleanup every %.0fs)",
        settings.rate_limit_max,
        settings.rate_limit_window_sec,
        settings.cleanup_interval_sec,
    )

    try:
        yield
    finally:
        task = getattr(app.state, "cleanup_task", None)
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


def get_rate_limiter(app: FastAPI) -> RateLimiter:
    """Fetch the app's :class:`RateLimiter`, building one lazily.

    Lazy-build covers the ``TestClient(app)``-without-``with`` path so
    older tests still work after lifespan landed.
    """
    rl = getattr(app.state, "rate_limiter", None)
    if rl is None:
        settings = load_runtime_settings()
        rl = RateLimiter(
            max_per_window=settings.rate_limit_max,
            window_sec=settings.rate_limit_window_sec,
        )
        app.state.rate_limiter = rl
        app.state.settings = settings
    return rl


# --------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------- #
def create_app() -> FastAPI:
    """Build a fresh FastAPI app — no global side effects."""
    app = FastAPI(
        title="Q-WiSE Acoustic Simulator",
        version="0.1.0",
        description=(
            "Python port of the MATLAB Q-WiSE simulator, served as a "
            "FastAPI + WebSocket demo."
        ),
        lifespan=lifespan,
    )

    app.include_router(rest_router)
    app.include_router(uploads_router)
    app.include_router(recordings_router)
    app.include_router(ws_router)

    static_dir = FRONTEND_DIR / "static"
    if static_dir.is_dir():
        app.mount("/static", NoCacheStaticFiles(directory=static_dir), name="static")

    index_html = FRONTEND_DIR / "index.html"
    if index_html.is_file():
        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(index_html)

    return app


# Uvicorn entry point — ``uvicorn backend.api.app:app``.
app = create_app()


def main() -> None:
    """CLI entry exposed via ``[project.scripts]`` in ``pyproject.toml``."""
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("backend.api.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
