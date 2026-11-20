import pandas as pd
import pytest
from fastapi.testclient import TestClient

from carbon_forecast import api
from carbon_forecast.data import UpstreamError


@pytest.fixture
def client(fitted_model, raw):
    api.state.model = fitted_model
    api.state.cache.clear()
    api.state.history_fetcher = lambda: raw.iloc[-9 * 48 :]
    api.state.official_fetcher = lambda start: raw["forecast"].loc[pd.Timestamp(start) :]
    with TestClient(api.app) as c:
        yield c
    api.state.cache.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_forecast_payload(client, raw):
    r = client.get("/forecast", params={"hours": 12})
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecast"]) == 24
    first = body["forecast"][0]
    assert first["lower"] <= first["predicted"] <= first["upper"]
    assert first["horizon_hours"] == 0.5
    assert pd.Timestamp(body["latest_observation"]) == raw.index[-1]
    assert body["greenest_window"]["mean_predicted"] > 0


def test_forecast_is_cached(client):
    calls = {"n": 0}
    original = api.state.history_fetcher

    def counting():
        calls["n"] += 1
        return original()

    api.state.history_fetcher = counting
    client.get("/forecast")
    client.get("/forecast")
    assert calls["n"] == 1


def test_upstream_failure_returns_503(client):
    def broken():
        raise UpstreamError("down")

    api.state.history_fetcher = broken
    assert client.get("/forecast").status_code == 503


def test_hours_validation(client):
    assert client.get("/forecast", params={"hours": 30}).status_code == 422


def test_no_model_returns_503(client):
    api.state.model = None
    assert client.get("/forecast").status_code == 503
    assert client.get("/health").json()["status"] == "degraded"
