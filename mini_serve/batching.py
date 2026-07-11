"""Dynamic (asyncio) batching for mini-serve.

The :class:`Batcher` collects incoming requests for a configurable window and
dispatches them to :meth:`Backend.forward` exactly once per window. This is the
core throughput lever of an inference server: many concurrent requests are
coalesced into a single batched forward pass.

Two knobs control the window:

* ``max_batch_size`` -- flush as soon as this many requests are queued.
* ``max_wait_ms`` -- flush after this long even if the batch is not full.

The implementation is single-flight and safe to call concurrently from many
async tasks on the same event loop (no external lock required because asyncio
is cooperatively scheduled and all queue mutations happen between ``await``
points).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Optional

from mini_serve.backends import Backend
from mini_serve.metrics import QUEUE_LENGTH, record_batch
from mini_serve.schemas import Output, Request

logger = logging.getLogger(__name__)


class Batcher:
    """Async batcher that coalesces requests into batched forward calls."""

    def __init__(self, backend: Backend, max_batch_size: int, max_wait_ms: int) -> None:
        self._backend = backend
        self._max_batch_size = max(1, int(max_batch_size))
        self._max_wait_s = max(0.0, float(max_wait_ms) / 1000.0)
        self._pending: list[tuple[Request, "asyncio.Future[Output]"]] = []
        self._timer: Optional[asyncio.TimerHandle] = None
        self._closed = False

    @property
    def queue_length(self) -> int:
        """Number of requests currently awaiting a batch flush."""
        return len(self._pending)

    async def submit(self, prompt: str, max_tokens: int) -> Output:
        """Submit a request and await its batched :class:`Output`.

        Args:
            prompt: Prompt text.
            max_tokens: Tokens requested for this request.

        Returns:
            The :class:`Output` produced by the backend for this request.

        Raises:
            RuntimeError: If the batcher has been closed.
        """
        if self._closed:
            raise RuntimeError("batcher is closed")

        req = Request(id=uuid.uuid4().hex, prompt=prompt, max_tokens=max_tokens)
        loop = asyncio.get_event_loop()
        future: asyncio.Future[Output] = loop.create_future()
        self._pending.append((req, future))
        QUEUE_LENGTH.set(len(self._pending))

        if len(self._pending) >= self._max_batch_size:
            self._schedule_flush(loop)
        elif self._timer is None:
            self._timer = loop.call_later(self._max_wait_s, self._schedule_flush, loop)

        return await future

    # -- internals --------------------------------------------------------- #
    def _schedule_flush(self, loop: asyncio.AbstractEventLoop) -> None:
        asyncio.ensure_future(self._flush())

    async def _flush(self) -> None:
        if self._closed:
            return
        # Cancel any pending timer; we are now taking ownership of the batch.
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

        batch = self._pending
        if not batch:
            return
        self._pending = []
        QUEUE_LENGTH.set(len(self._pending))

        requests = [req for req, _ in batch]
        start = time.perf_counter()
        try:
            outputs = await self._backend.forward(requests)
        except Exception as exc:  # noqa: BLE001 - surface to all waiters
            logger.exception("backend.forward failed")
            for _, future in batch:
                if not future.done():
                    future.set_exception(exc)
            return
        latency = time.perf_counter() - start
        record_batch(len(batch), latency)

        out_by_id = {out.request_id: out for out in outputs}
        for req, future in batch:
            if future.done():
                continue
            out = out_by_id.get(req.id)
            if out is None:
                future.set_exception(
                    RuntimeError("backend returned no output for a request")
                )
            else:
                future.set_result(out)

        # Requests that arrived while we awaited the backend need a new timer.
        if self._pending and self._timer is None:
            loop = asyncio.get_event_loop()
            self._timer = loop.call_later(self._max_wait_s, self._schedule_flush, loop)

    def close(self) -> None:
        """Mark the batcher closed (primarily for clean shutdown)."""
        self._closed = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
