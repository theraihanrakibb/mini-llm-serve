"""Request/response schemas for mini-serve.

Two layers of schemas live here:

* Public API models (Pydantic ``BaseModel`` subclasses) used by the HTTP
  layer in :mod:`mini_serve.server`.
* Internal dataclasses (``Request`` / ``Output``) that flow through the
  :class:`~mini_serve.batching.Batcher` and the backend ``forward`` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Public HTTP API schemas
# --------------------------------------------------------------------------- #
class CompletionRequest(BaseModel):
    """Payload for ``POST /v1/completions``.

    Attributes:
        prompt: The input prompt to run through the backend.
        max_tokens: Maximum number of tokens to generate (1-512).
        stream: When ``True`` the response is streamed via Server-Sent Events.
    """

    prompt: str = Field(..., min_length=1, description="Input prompt.")
    max_tokens: int = Field(
        default=16, ge=1, le=512, description="Maximum tokens to generate."
    )
    stream: bool = Field(
        default=False, description="Stream tokens via SSE when True."
    )


class CompletionResponse(BaseModel):
    """Non-streaming completion result.

    Attributes:
        id: Backend-generated request identifier.
        text: Concatenated generated text.
        tokens: The individual generated tokens, in order.
    """

    id: str
    text: str
    tokens: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Internal batching types
# --------------------------------------------------------------------------- #
@dataclass
class Request:
    """A single inference request routed through the batcher.

    Attributes:
        id: Unique identifier used to route the backend ``Output`` back to the
            coroutine that submitted this request.
        prompt: The prompt text.
        max_tokens: Number of tokens requested.
    """

    id: str
    prompt: str
    max_tokens: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Output:
    """A backend result keyed to a submitted :class:`Request`."""

    request_id: str
    tokens: list[str]
    meta: dict[str, Any] = field(default_factory=dict)
