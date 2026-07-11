"""Backend abstraction for mini-serve.

A :class:`Backend` exposes a single async ``forward`` method that turns a batch
of :class:`~mini_serve.schemas.Request` objects into a batch of
:class:`~mini_serve.schemas.Output` objects.

:class:`DummyBackend` is a deterministic, dependency-free implementation so the
whole project runs and is testable without a GPU or large model weights. The
README explains how to plug in a real HuggingFace / PyTorch backend.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod

from mini_serve.schemas import Output, Request

logger = logging.getLogger(__name__)


class Backend(ABC):
    """Abstract inference backend.

    Subclasses implement :meth:`forward` (batched inference) and
    :meth:`ready` (model-load / liveness probe hook).
    """

    @abstractmethod
    async def forward(self, batch: list[Request]) -> list[Output]:
        """Run batched inference.

        Args:
            batch: The requests collected by the batcher for this window.

        Returns:
            One :class:`~mini_serve.schemas.Output` per input request, in the
            same order as ``batch``.
        """
        raise NotImplementedError

    @abstractmethod
    async def ready(self) -> bool:
        """Return ``True`` when the backend can serve requests."""
        raise NotImplementedError


class DummyBackend(Backend):
    """Deterministic, dependency-free backend for demos and tests.

    Output tokens are derived from a SHA-256 hash of the prompt and token index,
    so results are stable across runs and need no model weights.
    """

    def __init__(self) -> None:
        self._ready = True

    async def ready(self) -> bool:
        return self._ready

    async def forward(self, batch: list[Request]) -> list[Output]:
        logger.debug("DummyBackend.forward called with batch of %d", len(batch))
        return [self._generate(req) for req in batch]

    def _generate(self, req: Request) -> Output:
        words = req.prompt.split()
        base = words or ["empty"]
        tokens: list[str] = []
        for i in range(req.max_tokens):
            word = base[i % len(base)]
            digest = hashlib.sha256(f"{req.prompt}:{i}".encode()).hexdigest()[:8]
            tokens.append(f"{word}_{digest}")
        return Output(request_id=req.id, tokens=tokens)
