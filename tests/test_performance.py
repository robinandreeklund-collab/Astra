"""Performance page smoke tests."""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tests.test_engine import FakeFinnhub


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ASTRA_TICK_SECONDS", "3600")
    import astra.config as cfg_mod
    importlib.reload(cfg_mod)
    import astra.app as app_mod
    importlib.reload(app_mod)

    async def _no_health(self):
        return False

    with patch("astra.data.finnhub.FinnhubClient", FakeFinnhub), \
         patch("astra.engine.runner.FinnhubClient", FakeFinnhub), \
         patch("astra.signals.aggregator.FinnhubClient", FakeFinnhub), \
         patch("astra.llm.client.LMStudioClient.health", new=_no_health):
        with TestClient(app_mod.app) as c:
            yield c


def test_performance_page_no_account(client):
    r = client.get("/performance")
    assert r.status_code == 200
    assert "No account" in r.text


def test_performance_page_with_account(client):
    client.post("/api/setup", data={"starting_balance": "50000"})
    r = client.get("/performance")
    assert r.status_code == 200
    assert "Scorecard" in r.text
    assert "Sharpe" in r.text
    assert "R-multiple distribution" in r.text


def test_experts_page(client):
    client.post("/api/setup", data={"starting_balance": "50000"})
    r = client.get("/experts")
    assert r.status_code == 200
    assert "Expert ensemble" in r.text


def test_training_page(client):
    r = client.get("/training")
    assert r.status_code == 200
    assert "Walk-forward training" in r.text
