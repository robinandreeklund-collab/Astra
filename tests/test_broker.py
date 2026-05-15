"""Paper broker tests."""

from __future__ import annotations

import pytest

from astra.engine.broker import InsufficientCash, NoPosition, PaperBroker

pytestmark = pytest.mark.asyncio


async def test_buy_then_sell_records_pnl(portfolio_db):
    await portfolio_db.create_account(10_000)
    broker = PaperBroker(portfolio_db)
    fill_b = await broker.buy("AAPL", 10, 100.0, {"_pattern_desc": "p1"}, "buy", "h1")
    assert fill_b.side == "BUY"
    acc = await portfolio_db.get_account()
    # 10 * (100 * 1.0005) + 1 fee = 1001.5
    assert acc["cash"] < 10_000
    assert (await portfolio_db.get_position("AAPL"))["qty"] == 10

    fill_s = await broker.sell("AAPL", 10, 110.0, {"_pattern_desc": "p1"}, "sell", "h1")
    assert fill_s.side == "SELL"
    assert fill_s.pnl is not None and fill_s.pnl > 0
    assert await portfolio_db.get_position("AAPL") is None
    assert fill_s.closed_trade_id is not None


async def test_insufficient_cash(portfolio_db):
    await portfolio_db.create_account(50)
    broker = PaperBroker(portfolio_db)
    with pytest.raises(InsufficientCash):
        await broker.buy("AAPL", 10, 100, {}, "x")


async def test_sell_without_position(portfolio_db):
    await portfolio_db.create_account(10_000)
    broker = PaperBroker(portfolio_db)
    with pytest.raises(NoPosition):
        await broker.sell("AAPL", 1, 100, {}, "x")


async def test_mark_to_market(portfolio_db):
    await portfolio_db.create_account(10_000)
    broker = PaperBroker(portfolio_db)
    await broker.buy("AAPL", 5, 100.0, {}, "x")
    equity, cash = await broker.mark_to_market({"AAPL": 120.0})
    assert equity > cash
    assert equity > 10_000 - 1000  # ballpark
