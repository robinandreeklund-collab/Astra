"""DB layer tests."""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.asyncio


async def test_create_account_and_update_cash(portfolio_db):
    acc = await portfolio_db.create_account(100_000)
    assert acc["starting_balance"] == 100_000
    assert acc["cash"] == 100_000
    assert acc["mode"] == "paper"
    await portfolio_db.update_cash(95_000)
    acc = await portfolio_db.get_account()
    assert acc["cash"] == 95_000


async def test_position_upsert_and_delete(portfolio_db):
    await portfolio_db.create_account(50_000)
    await portfolio_db.upsert_position("AAPL", 10, 150.0)
    p = await portfolio_db.get_position("AAPL")
    assert p["qty"] == 10
    await portfolio_db.upsert_position("AAPL", 5, 160.0)
    p = await portfolio_db.get_position("AAPL")
    assert p["qty"] == 5
    await portfolio_db.delete_position("AAPL")
    assert await portfolio_db.get_position("AAPL") is None


async def test_trade_insert_and_list(portfolio_db):
    await portfolio_db.create_account(10_000)
    tid = await portfolio_db.insert_trade(
        symbol="AAPL", side="BUY", qty=1, price=100, fees=1,
        signal_snapshot={"technical": {"rsi14": 30}},
        llm_reasoning="test", pattern_hash="abc",
    )
    assert tid > 0
    trades = await portfolio_db.list_trades()
    assert len(trades) == 1
    assert trades[0]["signal_snapshot"]["technical"]["rsi14"] == 30


async def test_open_buy_trade_for(portfolio_db):
    await portfolio_db.create_account(10_000)
    buy_id = await portfolio_db.insert_trade(
        symbol="MSFT", side="BUY", qty=2, price=200, fees=1,
        signal_snapshot={}, llm_reasoning=None,
    )
    open_t = await portfolio_db.open_buy_trade_for("MSFT")
    assert open_t["id"] == buy_id
    await portfolio_db.insert_trade(
        symbol="MSFT", side="SELL", qty=2, price=210, fees=1,
        signal_snapshot={}, llm_reasoning=None, closed_trade_id=buy_id, pnl=18,
    )
    assert await portfolio_db.open_buy_trade_for("MSFT") is None


async def test_equity_curve_and_wipe(portfolio_db):
    await portfolio_db.create_account(1000)
    await portfolio_db.append_equity(1000, 1000, ts="2026-01-01T00:00:00+00:00")
    await portfolio_db.append_equity(1100, 600, ts="2026-01-02T00:00:00+00:00")
    hist = await portfolio_db.equity_history(limit=10)
    assert len(hist) == 2
    assert hist[-1]["equity"] == 1100
    await portfolio_db.wipe()
    assert await portfolio_db.get_account() is None
    assert await portfolio_db.equity_history(limit=10) == []


async def test_memory_pattern_outcome_persists(memory_db):
    await memory_db.record_pattern_outcome("hash1", "RSI<30|MACD:bull", win=True, pnl=50, hold_minutes=120)
    await memory_db.record_pattern_outcome("hash1", "RSI<30|MACD:bull", win=False, pnl=-20, hold_minutes=60)
    stats = await memory_db.all_patterns()
    assert len(stats) == 1
    p = stats[0]
    assert p["wins"] == 1 and p["losses"] == 1
    assert abs(p["total_pnl"] - 30) < 0.01
    assert 50 < p["avg_hold_minutes"] < 130
    assert 0 < p["confidence"] < 1


async def test_lessons_and_generations(memory_db):
    gen_id = await memory_db.start_generation(1000)
    lid = await memory_db.add_lesson("test lesson", generation=gen_id)
    top = await memory_db.top_lessons()
    assert top and top[0]["id"] == lid
    await memory_db.close_generation(gen_id, 1500, 10, "ok")
    gens = await memory_db.all_generations()
    assert gens[0]["ending_equity"] == 1500


async def test_cache_ttl(cache_db):
    await cache_db.set("k", {"v": 1}, ttl_seconds=3600)
    hit = await cache_db.get("k")
    assert hit == {"v": 1}
    await cache_db.set("expired", {"v": 2}, ttl_seconds=-1)
    assert await cache_db.get("expired") is None
