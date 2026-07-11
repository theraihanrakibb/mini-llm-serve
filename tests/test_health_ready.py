"""Tests for health and readiness probes."""

from __future__ import annotations


async def test_health_liveness(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_ready_when_backend_loaded(client):
    resp = await client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
