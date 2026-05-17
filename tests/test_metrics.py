"""Performance-metric tests."""

from __future__ import annotations

import pytest

from astra.engine import metrics

pytestmark = pytest.mark.asyncio


async def test_max_drawdown():
    assert metrics.max_drawdown([100, 120, 90, 110]) == pytest.approx(0.25)
    assert metrics.max_drawdown([100, 110, 120]) == 0.0
    assert metrics.max_drawdown([]) == 0.0


async def test_sharpe_positive_for_steady_growth():
    eq = [100 * (1.01 ** i) for i in range(60)]
    assert metrics.sharpe(eq) > 0


async def test_sharpe_zero_for_flat():
    assert metrics.sharpe([100, 100, 100, 100]) == 0.0


async def test_sortino_ignores_upside_vol():
    eq = [100, 110, 105, 130, 120, 150]
    assert metrics.sortino(eq) != 0.0


async def test_cagr():
    # doubles over exactly one year
    eq = [100.0] * 1 + [200.0]
    eq = [100.0 + i for i in range(2)]  # trivial
    eq = [100.0, 200.0]
    # 1 period; cagr needs years>0 — use 252 points doubling
    eq = [100.0 * (2 ** (i / 252)) for i in range(253)]
    assert metrics.cagr(eq) == pytest.approx(1.0, abs=0.05)


async def test_trade_stats():
    trades = [
        {"pnl": 100}, {"pnl": -40}, {"pnl": 60}, {"pnl": -20}, {"side": "BUY"},
    ]
    s = metrics.trade_stats(trades)
    assert s["closed_trades"] == 4
    assert s["wins"] == 2 and s["losses"] == 2
    assert s["win_rate"] == 0.5
    assert s["profit_factor"] == pytest.approx(160 / 60)


async def test_full_report():
    eq = [100, 105, 103, 110]
    trades = [{"pnl": 8}, {"pnl": -3}]
    rep = metrics.full_report(eq, trades, 100.0, benchmark_return_pct=5.0)
    assert rep["return_pct"] == pytest.approx(10.0)
    assert rep["alpha_pct"] == pytest.approx(5.0)
    assert "sharpe" in rep and "max_drawdown_pct" in rep
