"""Tests for the Prometheus ``/metrics`` endpoint."""

from __future__ import annotations

EXPECTED_SERIES = [
    "mini_serve_requests_total",
    "mini_serve_queue_length",
    "mini_serve_batch_size",
    "mini_serve_batching_latency_seconds",
    "mini_serve_tokens_per_second",
]


async def test_metrics_exposes_expected_series(client):
    # Generate at least one batched forward so histogram series appear.
    resp = await client.post(
        "/v1/completions", json={"prompt": "metrics please", "max_tokens": 2}
    )
    assert resp.status_code == 200

    metrics = await client.get("/metrics")
    assert metrics.status_code == 200
    body = metrics.text
    for name in EXPECTED_SERIES:
        assert name in body


async def test_metrics_counts_requests(client):
    before = _read_request_count(await client.get("/metrics"))

    await client.post("/v1/completions", json={"prompt": "a", "max_tokens": 1})
    await client.post("/v1/completions", json={"prompt": "b", "max_tokens": 1})

    metrics = await client.get("/metrics")
    after = _read_request_count(metrics)

    # The counter must increase by at least the two requests we just made.
    assert after - before >= 2


def _read_request_count(metrics_response) -> float:
    label = 'endpoint="/v1/completions",status="accepted"'
    for line in metrics_response.text.splitlines():
        if line.startswith("mini_serve_requests_total") and label in line:
            return float(line.rsplit(" ", 1)[-1])
    return 0.0
