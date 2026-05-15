"""HTTP-level smoke tests for the FastAPI app."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tests.test_engine import FakeFinnhub


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ASTRA_TICK_SECONDS", "3600")
    # Re-import so settings pick up new env
    import importlib

    import astra.config as cfg_mod
    importlib.reload(cfg_mod)
    import astra.app as app_mod
    importlib.reload(app_mod)

    with patch("astra.data.finnhub.FinnhubClient", FakeFinnhub), \
         patch("astra.engine.runner.FinnhubClient", FakeFinnhub), \
         patch("astra.engine.backtest.FinnhubClient", FakeFinnhub), \
         patch("astra.signals.aggregator.FinnhubClient", FakeFinnhub), \
         patch("astra.llm.client.LMStudioClient.health", new=_no_health):
        with TestClient(app_mod.app) as c:
            yield c


async def _no_health(self):
    return False


def test_index_shows_setup_when_no_account(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Welcome" in r.text


def test_setup_creates_account_and_dashboard(client):
    r = client.post("/api/setup", data={"starting_balance": "10000"})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    r = client.get("/")
    assert r.status_code == 200
    assert "Equity" in r.text


def test_force_tick_and_status(client):
    client.post("/api/setup", data={"starting_balance": "10000"})
    r = client.post("/api/engine/tick")
    assert r.status_code == 200
    r = client.get("/api/status")
    assert r.status_code == 200
    assert "Engine" in r.text


def test_reset_keeps_memory_via_api(client):
    client.post("/api/setup", data={"starting_balance": "5000"})
    client.post("/api/engine/tick")
    r = client.post("/api/reset", data={"starting_balance": "8000"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["memory_kept"] is True
    r = client.get("/")
    assert "Equity" in r.text


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_memory_page(client):
    client.post("/api/setup", data={"starting_balance": "5000"})
    r = client.get("/memory")
    assert r.status_code == 200
    assert "Strategy lessons" in r.text


def test_signals_endpoint(client):
    client.post("/api/setup", data={"starting_balance": "5000"})
    r = client.get("/api/signals/AAPL")
    assert r.status_code == 200
    d = r.json()
    assert d["symbol"] == "AAPL"
    assert "pattern_hash" in d


def test_backtest_via_api(client):
    client.post("/api/setup", data={"starting_balance": "5000"})
    r = client.post("/api/backtest", data={
        "symbols": "AAPL,MSFT",
        "starting_balance": "10000",
        "days": "90",
        "use_llm": "false",
    })
    assert r.status_code == 200
    d = r.json()
    assert "summary" in d
    assert "equity_curve" in d
