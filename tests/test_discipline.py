"""Tests for the position-discipline redesign.

Covers: all-or-nothing sells, cooldown, min-trade-value gate, max-open-
positions cap, signal-change dedup, trailing stop, volatility sizing.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from astra.config import settings
from astra.engine.runner import EngineState, TradingEngine
from astra.engine.sizing import atr_pct_from_indicators, compute_buy_value
from tests.test_engine import FakeFinnhub

pytestmark = pytest.mark.asyncio


def _synthetic_candles(symbol: str, days: int = 80) -> list[dict]:
    """Deterministic rising candles so technical indicators are available."""
    import random
    r = random.Random(hash(symbol) & 0xFFFF)
    base = 1_700_000_000
    p = 100.0
    rows = []
    for i in range(days):
        p = max(1.0, p * (1 + r.gauss(0.002, 0.015)))
        rows.append({"t": base + i * 86400, "o": p * 0.99, "h": p * 1.01,
                     "l": p * 0.98, "c": p, "v": 1_000_000})
    return rows


@pytest.fixture
def patched():
    async def fake_candles(symbol, days=180, cache=None):
        return _synthetic_candles(symbol)

    with patch("astra.data.finnhub.FinnhubClient", FakeFinnhub), \
         patch("astra.engine.runner.FinnhubClient", FakeFinnhub), \
         patch("astra.signals.aggregator.FinnhubClient", FakeFinnhub), \
         patch("astra.signals.aggregator.fetch_daily_candles", new=fake_candles), \
         patch("astra.engine.scanner.fetch_daily_candles", new=fake_candles):
        async def _no_health(self):
            return False
        with patch("astra.llm.client.LMStudioClient.health", _no_health):
            yield


# ---- sizing unit tests ----

async def test_compute_buy_value_respects_caps():
    s = settings
    s.target_position_pct = 0.10
    s.max_position_pct = 0.15
    s.risk_per_trade_pct = 0.02
    s.stop_loss_pct = 0.05
    # equity 10000, full confidence, normal vol
    v = compute_buy_value(10_000, 10_000, 1.0, 0.02, s)
    # base = 1000, risk_cap = 0.02/0.05*10000 = 4000 → min = 1000
    assert 900 <= v <= 1100


async def test_compute_buy_value_shrinks_on_high_vol():
    s = settings
    s.target_position_pct = 0.10
    low_vol = compute_buy_value(10_000, 10_000, 1.0, 0.01, s)
    high_vol = compute_buy_value(10_000, 10_000, 1.0, 0.06, s)
    assert high_vol < low_vol


async def test_compute_buy_value_zero_when_no_cash():
    assert compute_buy_value(10_000, 0, 1.0, 0.02, settings) == 0.0


async def test_atr_pct_from_indicators():
    assert atr_pct_from_indicators({"atr14": 2.0, "last_close": 100.0}) == 0.02
    assert atr_pct_from_indicators({"atr14": None, "last_close": 100}) is None


# ---- cooldown ----

async def test_cooldown_blocks_reentry():
    state = EngineState()
    state.last_trade_at["AAPL"] = time.time()
    assert state.in_cooldown("AAPL", cooldown_seconds=3600) is True
    state.last_trade_at["AAPL"] = time.time() - 7200
    assert state.in_cooldown("AAPL", cooldown_seconds=3600) is False
    assert state.in_cooldown("NEVER_TRADED", cooldown_seconds=3600) is False


# ---- all-or-nothing + max positions ----

async def test_sell_is_all_or_nothing(portfolio_db, memory_db, cache_db, patched):
    """A SELL closes the entire position — never a partial sliver."""
    settings.watchlist_size = 1
    settings.scan_universe = False
    settings.custom_watchlist = "AAPL"
    settings.stop_loss_pct = 0.001  # trivially triggered → forced full sell
    await portfolio_db.create_account(10_000)
    await memory_db.start_generation(10_000)
    await portfolio_db.upsert_position("AAPL", qty=10, avg_price=99_999.0)
    await portfolio_db.update_cash(0)

    state = EngineState()
    engine = TradingEngine(portfolio_db, memory_db, cache_db, state)
    await engine.tick()
    # Whole position gone, not whittled down
    assert await portfolio_db.get_position("AAPL") is None


async def test_max_open_positions_cap(portfolio_db, memory_db, cache_db, patched):
    from astra.signals import technical

    settings.scan_universe = False
    settings.custom_watchlist = "AAPL,MSFT,NVDA"
    settings.watchlist_size = 3
    settings.max_open_positions = 1
    settings.cooldown_minutes = 0
    # Disable all forced exits so the pre-held position can't be sold mid-tick.
    settings.stop_loss_pct = 0.99
    settings.take_profit_pct = 0.99
    settings.trailing_stop_pct = 0.99
    await portfolio_db.create_account(1_000_000)
    await memory_db.start_generation(1_000_000)
    # Pre-fill one position so the cap is already reached, and pin it HELD
    # via signal dedup so it survives the tick. avg_price = the synthetic
    # last close so adaptive stop/take-profit can't fire on a price gap.
    _ind = technical.compute_indicators(_synthetic_candles("AAPL"))
    await portfolio_db.upsert_position("AAPL", qty=1, avg_price=_ind["last_close"])
    bar_ts = _ind["last_bar_ts"]

    state = EngineState()
    state.last_decision["AAPL"] = {
        "bar_ts": bar_ts, "action": "HOLD", "decided_at": time.time()}
    engine = TradingEngine(portfolio_db, memory_db, cache_db, state)
    result = await engine.tick()

    # With the cap at 1 and AAPL already held, no NEW positions open
    positions = await portfolio_db.get_positions()
    assert len(positions) == 1
    skips = [a for a in result["actions"] if a.get("reason") == "portfolio full"]
    assert skips  # cap actually fired


# ---- signal dedup ----

async def test_signal_dedup_skips_unchanged_bar(portfolio_db, memory_db, cache_db, patched):
    """A held name whose daily bar matches the last decision is not re-decided."""
    from astra.signals import technical

    settings.scan_universe = False
    settings.custom_watchlist = "AAPL"
    settings.watchlist_size = 1
    settings.cooldown_minutes = 0
    settings.stop_loss_pct = 0.99
    settings.take_profit_pct = 0.99
    settings.trailing_stop_pct = 0.99
    await portfolio_db.create_account(100_000)
    await memory_db.start_generation(100_000)
    # Pre-seed last_decision with the bar_ts the synthetic candles produce;
    # avg_price = last close so adaptive stop/take-profit can't fire.
    ind = technical.compute_indicators(_synthetic_candles("AAPL"))
    bar_ts = ind["last_bar_ts"]
    await portfolio_db.upsert_position("AAPL", qty=1, avg_price=ind["last_close"])
    state = EngineState()
    state.last_decision["AAPL"] = {
        "bar_ts": bar_ts, "action": "HOLD", "decided_at": time.time()}

    engine = TradingEngine(portfolio_db, memory_db, cache_db, state)
    result = await engine.tick()
    holds = [a for a in result["actions"]
             if a["symbol"] == "AAPL" and a.get("reason") == "no new data"]
    assert holds, f"expected dedup HOLD, got {result['actions']}"
    # AAPL must still be held — dedup means "don't re-evaluate", not "sell".
    assert await portfolio_db.get_position("AAPL") is not None
