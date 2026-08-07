from __future__ import annotations

from fastapi.testclient import TestClient

from fraud_detection.api import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

