"""Shared pytest fixtures for mini-serve tests."""

from __future__ import annotations

import httpx
import pytest

from mini_serve.server import create_app, init_runtime, shutdown_runtime

TEST_CONFIG: dict = {
    "backend": "dummy",
    "max_batch_size": 8,
    "max_wait_ms": 50,
    "port": 8000,
}


@pytest.fixture
async def client():
    """An httpx AsyncClient bound to a freshly initialised mini-serve app.

    We call :func:`init_runtime` directly because ``httpx.ASGITransport`` does
    not trigger the FastAPI lifespan, keeping each test fully isolated.
    """
    app = create_app(TEST_CONFIG)
    init_runtime(app, TEST_CONFIG)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    shutdown_runtime(app)
