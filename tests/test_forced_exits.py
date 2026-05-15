"""Verify hard stop-loss and take-profit fire before the LLM/heuristic."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from astra.config import settings
from astra.engine.runner import EngineState, TradingEngine
from tests.test_engine import FakeFinnhub

pytestmark = pytest.mark.asyncio


@pytest.fixture
def patched(monkeypatch):
    with patch("astra.data.finnhub.FinnhubClient", FakeFinnhub), \
         patch("astra.engine.runner.FinnhubClient", FakeFinnhub), \
         patch("astra.signals.aggregator.FinnhubClient", FakeFinnhub):
        async def _no_health(self):
            return False
        with patch("astra.llm.client.LMStudioClient.health", _no_health):
            yield


async def test_stop_loss_force_sells_position(portfolio_db, memory_db, cache_db, patched):
    settings.watchlist_size = 1
    settings.stop_loss_pct = 0.05
    await portfolio_db.create_account(10_000)
    await memory_db.start_generation(10_000)
    # Open AAPL position at high price so any reasonable last_close yields a loss
    await portfolio_db.upsert_position("AAPL", qty=10, avg_price=10_000.0)
    await portfolio_db.update_cash(0)

    state = EngineState()
    engine = TradingEngine(portfolio_db, memory_db, cache_db, state)
    result = await engine.tick()

    actions = result.get("actions", [])
    sells = [a for a in actions if a["action"] == "SELL"]
    assert sells, f"Expected forced stop-loss SELL, got {actions}"
    assert sells[0].get("forced") is True
    # Position should now be closed
    assert await portfolio_db.get_position("AAPL") is None


async def test_take_profit_force_sells(portfolio_db, memory_db, cache_db, patched):
    settings.watchlist_size = 1
    settings.take_profit_pct = 0.05
    settings.stop_loss_pct = 0.99  # disabled
    await portfolio_db.create_account(10_000)
    await memory_db.start_generation(10_000)
    # Avg price extremely low so any reasonable last_close → big gain
    await portfolio_db.upsert_position("AAPL", qty=10, avg_price=0.01)
    await portfolio_db.update_cash(0)

    state = EngineState()
    engine = TradingEngine(portfolio_db, memory_db, cache_db, state)
    result = await engine.tick()

    actions = result.get("actions", [])
    sells = [a for a in actions if a["action"] == "SELL" and a.get("forced")]
    assert sells, f"Expected forced take-profit SELL, got {actions}"
    assert await portfolio_db.get_position("AAPL") is None
