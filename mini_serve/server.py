"""FastAPI application for mini-serve.

Exposes:

* ``POST /v1/completions`` -- batched, optionally SSE-streamed completions.
* ``GET /health`` -- liveness probe.
* ``GET /ready`` -- readiness probe (backend loaded).
* ``GET /metrics`` -- Prometheus scrape endpoint.

Configuration is resolved from environment variables (see
:func:`config_from_env`) with sensible defaults so the service runs unchanged in
Docker, CI or locally.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sse_starlette.sse import EventSourceResponse

from mini_serve import __version__
from mini_serve.backends import Backend, DummyBackend
from mini_serve.batching import Batcher
from mini_serve.metrics import (
    QUEUE_LENGTH,
    REQUEST_COUNT,
    record_tokens,
)
from mini_serve.schemas import CompletionRequest

logger = logging.getLogger("mini_serve")

DEFAULT_CONFIG: dict[str, Any] = {
    "backend": "dummy",
    "max_batch_size": 8,
    "max_wait_ms": 20,
    "port": 8000,
}


def config_from_env() -> dict[str, Any]:
    """Build runtime config from environment variables.

    Recognised variables: ``MINI_SERVE_BACKEND``, ``MINI_SERVE_MAX_BATCH_SIZE``,
    ``MINI_SERVE_MAX_WAIT_MS``, ``MINI_SERVE_PORT``.
    """
    return {
        "backend": os.getenv("MINI_SERVE_BACKEND", DEFAULT_CONFIG["backend"]),
        "max_batch_size": int(
            os.getenv("MINI_SERVE_MAX_BATCH_SIZE", str(DEFAULT_CONFIG["max_batch_size"]))
        ),
        "max_wait_ms": int(
            os.getenv("MINI_SERVE_MAX_WAIT_MS", str(DEFAULT_CONFIG["max_wait_ms"]))
        ),
        "port": int(os.getenv("MINI_SERVE_PORT", str(DEFAULT_CONFIG["port"]))),
    }


def build_backend(name: str) -> Backend:
    """Instantiate a backend by name.

    Args:
        name: Backend identifier. Currently only ``"dummy"`` ships; extend this
            branch to wire a real HuggingFace / PyTorch backend.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    if name == "dummy":
        return DummyBackend()
    raise ValueError(f"Unknown backend: {name!r}")


def init_runtime(app: FastAPI, config: dict[str, Any]) -> None:
    """Create and attach backend + batcher to ``app.state``."""
    backend = build_backend(config["backend"])
    batcher = Batcher(backend, config["max_batch_size"], config["max_wait_ms"])
    app.state.backend = backend
    app.state.batcher = batcher
    app.state.config = config
    QUEUE_LENGTH.set(0)
    logger.info(
        "runtime initialised: backend=%s batch_size=%d wait_ms=%d",
        config["backend"],
        config["max_batch_size"],
        config["max_wait_ms"],
    )


def shutdown_runtime(app: FastAPI) -> None:
    """Detach runtime resources from ``app.state``."""
    batcher: Optional[Batcher] = getattr(app.state, "batcher", None)
    if batcher is not None:
        batcher.close()
    app.state.backend = None
    app.state.batcher = None


def create_app(config: Optional[dict[str, Any]] = None) -> FastAPI:
    """Factory that builds the FastAPI app.

    Args:
        config: Optional runtime config. When omitted, config is read from the
            environment (useful for production / Docker). Tests pass an explicit
            config and then call :func:`init_runtime` directly.
    """
    if config is None:
        config = config_from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        init_runtime(app, config)
        yield
        shutdown_runtime(app)

    app = FastAPI(
        title="mini-serve",
        version=__version__,
        description="A tiny model inference server with dynamic batching, "
        "SSE streaming and Prometheus metrics.",
        lifespan=lifespan,
    )
    app.state._config = config
    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness probe. Always returns ok while the process is up."""
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request) -> dict[str, str]:
        """Readiness probe. Returns ok only when the backend is ready."""
        backend: Optional[Backend] = getattr(request.app.state, "backend", None)
        if backend is None:
            raise HTTPException(status_code=503, detail="backend not initialised")
        if not await backend.ready():
            raise HTTPException(status_code=503, detail="backend not ready")
        return {"status": "ready"}

    @app.post("/v1/completions")
    async def completions(request: Request, payload: CompletionRequest):
        """Run a (batched) completion, streamed via SSE if requested."""
        batcher: Optional[Batcher] = getattr(request.app.state, "batcher", None)
        if batcher is None:
            raise HTTPException(status_code=503, detail="server not ready")

        REQUEST_COUNT.labels(endpoint="/v1/completions", status="accepted").inc()

        if payload.stream:
            return EventSourceResponse(
                _stream(batcher, payload), ping=15.0, media_type="text/event-stream"
            )

        output = await batcher.submit(payload.prompt, payload.max_tokens)
        return JSONResponse(
            {"id": output.request_id, "text": "".join(output.tokens), "tokens": output.tokens}
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        """Prometheus scrape endpoint."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def _stream(batcher: Batcher, payload: CompletionRequest):
    """Async generator yielding SSE chunks for a streamed completion."""
    try:
        output = await batcher.submit(payload.prompt, payload.max_tokens)
        start = time.perf_counter()
        emitted = 0
        for token in output.tokens:
            yield {"event": "token", "data": token}
            emitted += 1
            await asyncio.sleep(0.01)  # emulate per-token decode latency
        elapsed = time.perf_counter() - start
        if emitted:
            record_tokens(emitted, elapsed)
        yield {"event": "done", "data": "[DONE]"}
    except Exception as exc:  # noqa: BLE001 - report over the stream
        logger.exception("streaming completion failed")
        yield {"event": "error", "data": str(exc)}


# Module-level app used by `uvicorn mini_serve.server:app` (Docker / prod).
app = create_app()


if __name__ == "__main__":
    import uvicorn

    cfg = config_from_env()
    uvicorn.run(
        "mini_serve.server:app",
        host="0.0.0.0",
        port=cfg["port"],
        log_level="info",
    )
