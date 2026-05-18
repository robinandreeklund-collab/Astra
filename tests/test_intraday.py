"""Intraday replay — monitor-tick exit checks fire against the live price."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import astra.data.simulator as sim
from astra.config import settings
from astra.data.simulator import HistoricalMarket
from astra.engine.runner import EngineState, TradingEngine

pytestmark = pytest.mark.asyncio

_BASE = 1_700_000_000 - (1_700_000_000 % 86400)
_HOURS = (14, 15, 16, 17, 18, 19, 20)


def _hourly_falling(n_days: int = 300) -> list[dict]:
    """Hourly bars where the price steadily falls — so a held position
    will breach its stop intraday."""
    rows = []
    price = 200.0
    for d in range(n_days):
        for h in _HOURS:
            price = max(1.0, price * 0.998)
            t = _BASE + d * 86400 + h * 3600
            rows.append({"t": t, "o": price, "h": price * 1.001,
                         "l": price * 0.999, "c": price, "v": 1000})
    return rows


async def test_monitor_tick_fires_intraday_stop(
    portfolio_db, memory_db, cache_db
):
    """A held position underwater past its stop is sold on a monitor tick —
    no scan, no LLM, just the intraday price crossing the stop."""
    settings.simulate_data = True
    settings.stop_loss_pct = 0.05
    settings.trailing_stop_pct = 0.99
    settings.take_profit_pct = 0.99
    await portfolio_db.create_account(100_000)
    await memory_db.start_generation(100_000)

    # Arm the replay market with a steadily falling stock.
    sim._MARKET = HistoricalMarket({"AAA": _hourly_falling(300)}, replay_days=50)
    market = sim._MARKET
    # Open AAA at the current price, then let the market fall well past 5%.
    entry = market.current_price("AAA")
    await portfolio_db.upsert_position("AAA", qty=10, avg_price=entry)
    for _ in range(60):  # ~0.998^60 ≈ -11% — past the 5% stop
        market.advance()

    state = EngineState()
    engine = TradingEngine(portfolio_db, memory_db, cache_db, state)
    result = await engine._monitor_tick()

    sells = [a for a in result["actions"]
             if a["action"] == "SELL" and a.get("forced")]
    assert sells, f"expected an intraday stop-loss sell, got {result['actions']}"
    # The position is closed.
    assert await portfolio_db.get_position("AAA") is None
    sim.reset_market()
    settings.simulate_data = False


async def test_monitor_tick_holds_when_no_stop_breached(
    portfolio_db, memory_db, cache_db
):
    settings.simulate_data = True
    settings.stop_loss_pct = 0.50  # very loose — won't trigger
    settings.trailing_stop_pct = 0.99
    settings.take_profit_pct = 0.99
    await portfolio_db.create_account(100_000)
    await memory_db.start_generation(100_000)

    sim._MARKET = HistoricalMarket({"AAA": _hourly_falling(300)}, replay_days=50)
    market = sim._MARKET
    entry = market.current_price("AAA")
    await portfolio_db.upsert_position("AAA", qty=10, avg_price=entry)
    market.advance()

    state = EngineState()
    engine = TradingEngine(portfolio_db, memory_db, cache_db, state)
    result = await engine._monitor_tick()

    assert not [a for a in result["actions"] if a["action"] == "SELL"]
    assert await portfolio_db.get_position("AAA") is not None
    sim.reset_market()
    settings.simulate_data = False


async def test_daily_loss_guard_keys_on_replay_day(
    portfolio_db, memory_db, cache_db
):
    """The daily-loss guard must reset per REPLAY day — never stay blocked
    for the whole real session after one bad replay day."""
    import asyncio as _aio

    settings.simulate_data = True
    settings.scan_top_n = 3
    await portfolio_db.create_account(25_000)
    await memory_db.start_generation(25_000)
    sim.reset_market()

    state = EngineState()
    engine = TradingEngine(portfolio_db, memory_db, cache_db, state)
    engine.ensure_universe = lambda: _aio.sleep(
        0, result=["AAA", "BBB", "CCC"])
    # Feed the loader synthetic hourly data so no network is needed.
    async def fake_load(symbols, cache, replay_days=252):
        sim._MARKET = HistoricalMarket(
            {s: _hourly_falling(400) for s in symbols}, replay_days=60)
        return sim._MARKET
    with patch("astra.data.simulator.load_market", new=fake_load):
        seen_days = set()
        for _ in range(60):
            await engine.tick()
            if state.day_key is not None:
                seen_days.add(state.day_key)

    # The guard's day key advanced through several distinct REPLAY days
    # (an int YYYYMMDD), not a single frozen wall-clock date.
    assert len(seen_days) >= 3
    assert all(isinstance(d, int) for d in seen_days)
    sim.reset_market()
    settings.simulate_data = False

