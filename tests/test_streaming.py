"""Tests for SSE token streaming on ``POST /v1/completions``."""

from __future__ import annotations

from mini_serve.backends import DummyBackend
from mini_serve.schemas import Request


async def test_streaming_yields_multiple_chunks_in_order(client):
    prompt = "hello streaming world"
    max_tokens = 5

    token_events: list[str] = []
    done = False
    current_event: str | None = None

    async with client.stream(
        "POST",
        "/v1/completions",
        json={"prompt": prompt, "max_tokens": max_tokens, "stream": True},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        async for line in resp.aiter_lines():
            if line == "":
                current_event = None
                continue
            if line.startswith("event:"):
                current_event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                payload = line[len("data:") :].strip()
                if current_event == "token":
                    token_events.append(payload)
                elif current_event == "done":
                    done = True

    # One SSE chunk per generated token, in order, plus a final done event.
    assert done is True

    backend = DummyBackend()
    expected = (
        await backend.forward(
            [Request(id="x", prompt=prompt, max_tokens=max_tokens)]
        )
    )[0].tokens
    assert token_events == expected
    assert len(token_events) == max_tokens


async def test_non_streaming_returns_full_text(client):
    prompt = "batch me"
    resp = await client.post(
        "/v1/completions", json={"prompt": prompt, "max_tokens": 3, "stream": False}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["tokens"]
    assert body["text"] == "".join(body["tokens"])
