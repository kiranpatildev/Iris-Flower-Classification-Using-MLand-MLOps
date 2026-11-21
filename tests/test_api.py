import asyncio

import httpx
import pytest

from iris_flow.api import app

def request(method, path, **kwargs):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(run())


def test_health():
    response = request("GET", "/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_known_setosa():
    response = request("POST", "/predict", json={
        "sepal_length": 5.1, "sepal_width": 3.5,
        "petal_length": 1.4, "petal_width": 0.2,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["species"] == "setosa"
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=0.0002)


def test_rejects_invalid_measurement():
    response = request("POST", "/predict", json={
        "sepal_length": -1, "sepal_width": 3.5,
        "petal_length": 1.4, "petal_width": 0.2,
    })
    assert response.status_code == 422

