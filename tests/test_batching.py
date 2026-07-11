"""Tests for the dynamic :class:`Batcher`.

These verify the core throughput guarantee: many concurrent requests are
coalesced into a small number of backend ``forward`` calls, and every request
still receives its own routed output.
"""

from __future__ import annotations

import asyncio

from mini_serve.backends import DummyBackend
from mini_serve.batching import Batcher
from mini_serve.schemas import Request


class CountingBackend(DummyBackend):
    """DummyBackend that counts how many times ``forward`` is invoked."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    async def forward(self, batch):
        self.forward_calls += 1
        return await super().forward(batch)


async def test_batches_n_requests_into_single_forward():
    backend = CountingBackend()
    batcher = Batcher(backend, max_batch_size=5, max_wait_ms=50)

    results = await asyncio.gather(
        *(batcher.submit("prompt", 4) for _ in range(5))
    )

    assert backend.forward_calls == 1
    assert len(results) == 5
    for out in results:
        assert len(out.tokens) == 4


async def test_multiple_waves_produce_multiple_batches():
    backend = CountingBackend()
    batcher = Batcher(backend, max_batch_size=5, max_wait_ms=50)

    wave1 = await asyncio.gather(*(batcher.submit("p", 2) for _ in range(5)))
    assert backend.forward_calls == 1
    assert len(wave1) == 5

    wave2 = await asyncio.gather(*(batcher.submit("p", 2) for _ in range(5)))
    assert backend.forward_calls == 2
    assert len(wave2) == 5


async def test_concurrent_requests_all_resolve_and_are_batched():
    backend = CountingBackend()
    batcher = Batcher(backend, max_batch_size=4, max_wait_ms=200)

    async def producer(i: int):
        out = await batcher.submit(f"req-{i}", 3)
        assert out.tokens, "each request must produce tokens"
        return out

    tasks = [asyncio.create_task(producer(i)) for i in range(20)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 20
    # 20 requests with batch size 4 must collapse into <= 5 forward calls,
    # proving they were batched rather than served one-by-one.
    assert 1 <= backend.forward_calls <= 5


async def test_outputs_routed_back_to_correct_requester():
    backend = CountingBackend()
    batcher = Batcher(backend, max_batch_size=10, max_wait_ms=50)

    prompts = [f"unique-prompt-{i}" for i in range(6)]
    results = await asyncio.gather(
        *(batcher.submit(p, 3) for p in prompts)
    )

    expected = DummyBackend()
    for prompt, out in zip(prompts, results):
        direct = (
            await expected.forward([Request(id="x", prompt=prompt, max_tokens=3)])
        )[0]
        assert out.tokens == direct.tokens
