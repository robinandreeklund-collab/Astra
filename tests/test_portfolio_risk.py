"""Portfolio-risk and R-multiple tests."""

from __future__ import annotations

import pytest

from astra.engine.portfolio_risk import (
    check_new_position,
    portfolio_heat,
    sector_exposure,
    sector_of,
)
from astra.profiles.profile import StockProfile

pytestmark = pytest.mark.asyncio


# ---- sector map ----

async def test_sector_of():
    assert sector_of("AAPL") == "Tech"
    assert sector_of("JPM") == "Financials"
    assert sector_of("XOM") == "Energy"
    assert sector_of("ZZZZ") == "Other"


async def test_sector_exposure():
    positions = [
        {"symbol": "AAPL", "qty": 10, "avg_price": 100},
        {"symbol": "MSFT", "qty": 5, "avg_price": 200},
        {"symbol": "JPM", "qty": 4, "avg_price": 150},
    ]
    prices = {"AAPL": 100, "MSFT": 200, "JPM": 150}
    exp = sector_exposure(positions, prices, equity=10_000)
    # Tech: 1000 + 1000 = 2000 / 10000 = 0.20; Financials 600/10000 = 0.06
    assert exp["Tech"] == pytest.approx(0.20)
    assert exp["Financials"] == pytest.approx(0.06)


# ---- portfolio heat ----

async def test_portfolio_heat():
    positions = [{"symbol": "AAPL", "qty": 10, "avg_price": 100}]
    prices = {"AAPL": 100}
    # value 1000, stop 5% → risk 50 → heat 50/10000 = 0.5%
    heat = portfolio_heat(positions, prices, {"AAPL": 0.05}, 10_000, 0.05)
    assert heat == pytest.approx(0.005)


# ---- check_new_position ----

async def test_sector_cap_blocks():
    # Already 30% in Tech, adding 10% more would breach a 35% cap.
    positions = [{"symbol": "AAPL", "qty": 30, "avg_price": 100}]  # $3000
    prices = {"AAPL": 100}
    ok, why = check_new_position(
        "MSFT", add_value=1000, positions=positions, prices=prices,
        stop_pcts={"AAPL": 0.05}, equity=10_000, new_stop_pct=0.05,
        max_sector_pct=0.35, max_portfolio_heat=0.50, default_stop=0.05)
    assert ok is False and "sector cap" in why


async def test_heat_cap_blocks():
    positions = [{"symbol": "AAPL", "qty": 80, "avg_price": 100}]  # $8000
    prices = {"AAPL": 100}
    # current heat = 8000*0.05/10000 = 4%; add 1000*0.05/10000 = 0.5%.
    ok, why = check_new_position(
        "XOM", add_value=1000, positions=positions, prices=prices,
        stop_pcts={"AAPL": 0.05}, equity=10_000, new_stop_pct=0.05,
        max_sector_pct=0.90, max_portfolio_heat=0.04, default_stop=0.05)
    assert ok is False and "heat" in why


async def test_new_position_allowed_within_limits():
    ok, why = check_new_position(
        "XOM", add_value=500, positions=[], prices={}, stop_pcts={},
        equity=10_000, new_stop_pct=0.05, max_sector_pct=0.35,
        max_portfolio_heat=0.25, default_stop=0.05)
    assert ok is True


# ---- R-multiples in the profile ----

async def test_r_multiple_drives_expectancy():
    p = StockProfile(symbol="AAA")
    p.record_trade_outcome(True, 100, 60, ["macd_bull"], r_multiple=2.0)
    p.record_trade_outcome(False, -50, 60, ["macd_bull"], r_multiple=-1.0)
    assert p.sum_r == 1.0
    assert p.expectancy_r() == 0.5


async def test_edge_score_uses_r_expectancy():
    good = StockProfile(symbol="G")
    bad = StockProfile(symbol="B")
    for _ in range(8):
        good.record_trade_outcome(True, 100, 60, [], r_multiple=2.0)
        bad.record_trade_outcome(False, -100, 60, [], r_multiple=-1.0)
    assert good.edge_score() > 0.3
    assert bad.edge_score() < -0.3
