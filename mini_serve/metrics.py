"""Prometheus metrics for mini-serve.

All metrics are module-level singletons. They are exposed at ``GET /metrics``
by the server. Gauges are seeded with an initial value so their series are
always present, even before the first request.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "mini_serve_requests_total",
    "Total completion requests received, labelled by endpoint and status.",
    ["endpoint", "status"],
)

QUEUE_LENGTH = Gauge(
    "mini_serve_queue_length",
    "Current number of requests waiting in the batching queue.",
)
QUEUE_LENGTH.set(0)

BATCH_SIZE = Histogram(
    "mini_serve_batch_size",
    "Number of requests processed per backend forward call.",
    buckets=[1, 2, 4, 8, 16, 32, 64, 128],
)

BATCHING_LATENCY = Histogram(
    "mini_serve_batching_latency_seconds",
    "Latency of a single backend forward call (batching + inference).",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

TOKENS_PER_SEC = Gauge(
    "mini_serve_tokens_per_second",
    "Observed token throughput (tokens / second) for the last stream.",
)
TOKENS_PER_SEC.set(0)


def record_batch(size: int, latency_seconds: float) -> None:
    """Record one backend ``forward`` invocation."""
    BATCH_SIZE.observe(size)
    BATCHING_LATENCY.observe(latency_seconds)


def record_tokens(total_tokens: int, elapsed_seconds: float) -> None:
    """Record streaming throughput for a completed stream."""
    if elapsed_seconds > 0:
        TOKENS_PER_SEC.set(total_tokens / elapsed_seconds)
