"""Engine + memory integration tests with a fake Finnhub."""

from __future__ import annotations

import time
import random
from typing import Any
from unittest.mock import patch

import pytest

from astra.config import settings
from astra.engine.runner import EngineState, TradingEngine

pytestmark = pytest.mark.asyncio


def _make_history(days=120, drift=0.002, seed=1):
    rng = random.Random(seed)
    t0 = int(time.time()) - days * 86400
    ts, op, hi, lo, cl, vo = [], [], [], [], [], []
    p = 100.0
    for i in range(days):
        ret = rng.gauss(drift, 0.012)
        p = max(1.0, p * (1 + ret))
        ts.append(t0 + i * 86400)
        op.append(p * (1 - 0.001)); hi.append(p * 1.01); lo.append(p * 0.99); cl.append(p)
        vo.append(100000)
    return {"s": "ok", "t": ts, "o": op, "h": hi, "l": lo, "c": cl, "v": vo}


class FakeFinnhub:
    """Stand-in for FinnhubClient used during tests."""

    def __init__(self, *_, **__):
        self.api_key = "test"
        self.cache = None
        self.bucket = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def close(self):
        pass

    async def quote(self, symbol):
        # Use last close from the synthetic history
        h = _make_history(seed=hash(symbol) & 0xFFFF)
        return {"c": h["c"][-1], "h": h["h"][-1], "l": h["l"][-1], "o": h["o"][-1]}

    async def candles(self, symbol, resolution, from_ts, to_ts):
        return _make_history(seed=hash(symbol) & 0xFFFF)

    async def recommendation(self, symbol):
        return [{"period": "2026-05-01", "strongBuy": 5, "buy": 8, "hold": 4, "sell": 1, "strongSell": 0}]

    async def insider_transactions(self, symbol):
        return {"data": [{"transactionDate": "2026-05-01", "change": 1000, "transactionPrice": 100}]}

    async def company_news(self, symbol, fr, to):
        return [{"headline": "Solid quarter", "summary": "ok", "source": "x", "datetime": 1}]

    async def news_sentiment(self, symbol):
        return {"sentiment": {"bullishPercent": 0.7, "bearishPercent": 0.3},
                "buzz": {"buzz": 1.2, "weeklyAverage": 8},
                "companyNewsScore": 0.65, "sectorAverageNewsScore": 0.5}

    async def earnings_calendar(self, fr, to, symbol=None):
        return {"earningsCalendar": []}

    async def earnings(self, symbol):
        return [{"period": "2026-Q1", "actual": 1.5, "estimate": 1.3, "surprise": 0.2, "surprisePercent": 15}]

    async def social_sentiment(self, symbol):
        return {"reddit": [{"score": 0.6, "mention": 50}], "twitter": [{"score": 0.55, "mention": 100}]}

    async def index_constituents(self, symbol="^GSPC"):
        return {"constituents": ["AAPL", "MSFT", "NVDA"]}


@pytest.fixture
def patch_finnhub():
    with patch("astra.data.finnhub.FinnhubClient", FakeFinnhub), \
         patch("astra.engine.runner.FinnhubClient", FakeFinnhub), \
         patch("astra.engine.backtest.FinnhubClient", FakeFinnhub), \
         patch("astra.signals.aggregator.FinnhubClient", FakeFinnhub):
        yield


@pytest.fixture
def patch_llm_unreachable():
    """Make the LM Studio health probe always return False."""
    from astra.llm import client as llm_client

    async def _no(self):
        return False

    with patch.object(llm_client.LMStudioClient, "health", _no):
        yield


async def test_full_tick_runs_and_records_state(
    portfolio_db, memory_db, cache_db, patch_finnhub, patch_llm_unreachable
):
    settings.watchlist_size = 3
    await portfolio_db.create_account(100_000)
    await memory_db.start_generation(100_000)
    state = EngineState()
    engine = TradingEngine(portfolio_db, memory_db, cache_db, state)
    result = await engine.tick()
    assert "actions" in result
    assert result["equity"] > 0
    # Equity curve should now have at least one point
    hist = await portfolio_db.equity_history(limit=10)
    assert hist


async def test_reset_keeps_memory(
    portfolio_db, memory_db, cache_db, patch_finnhub, patch_llm_unreachable
):
    await portfolio_db.create_account(50_000)
    await memory_db.add_lesson("Keep me across resets", generation=1)
    await memory_db.record_pattern_outcome("h1", "DESC", win=True, pnl=100, hold_minutes=60)
    await portfolio_db.wipe()
    assert await portfolio_db.get_account() is None
    lessons = await memory_db.all_lessons()
    patterns = await memory_db.all_patterns()
    assert len(lessons) == 1
    assert len(patterns) == 1


async def test_backtest_runs(patch_finnhub, patch_llm_unreachable, cache_db):
    from astra.engine.backtest import run_backtest
    result = await run_backtest(["AAPL", "MSFT"], 100_000, days=90, cache=cache_db, use_llm=False)
    s = result.summary()
    assert s["start_equity"] == 100_000
    assert isinstance(s["return_pct"], float)
    assert isinstance(s["trade_count"], int)
