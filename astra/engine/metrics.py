"""Performance metrics — the honest scorecard.

Used by the trainer, the backtester and the evaluation dashboard. All
return-based metrics work off an equity curve (a list of portfolio values
sampled over time); trade-based metrics work off closed trades.
"""

from __future__ import annotations

import math
from typing import Any

TRADING_DAYS = 252


def _returns(equity: list[float]) -> list[float]:
    out: list[float] = []
    for i in range(1, len(equity)):
        prev = equity[i - 1]
        if prev > 0:
            out.append(equity[i] / prev - 1.0)
    return out


def max_drawdown(equity: list[float]) -> float:
    """Largest peak-to-trough decline as a positive fraction (0.2 = -20%)."""
    if not equity:
        return 0.0
    peak = equity[0]
    worst = 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            dd = (peak - v) / peak
            worst = max(worst, dd)
    return worst


def sharpe(equity: list[float], periods_per_year: int = TRADING_DAYS) -> float:
    rets = _returns(equity)
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(periods_per_year)


def sortino(equity: list[float], periods_per_year: int = TRADING_DAYS) -> float:
    rets = _returns(equity)
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    downside = [r for r in rets if r < 0]
    if not downside:
        return 0.0
    dvar = sum(r * r for r in downside) / len(downside)
    dstd = math.sqrt(dvar)
    if dstd == 0:
        return 0.0
    return (mean / dstd) * math.sqrt(periods_per_year)


def cagr(equity: list[float], periods_per_year: int = TRADING_DAYS) -> float:
    if len(equity) < 2 or equity[0] <= 0:
        return 0.0
    total = equity[-1] / equity[0]
    years = (len(equity) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    if total <= 0:
        return -1.0
    return total ** (1.0 / years) - 1.0


def calmar(equity: list[float], periods_per_year: int = TRADING_DAYS) -> float:
    dd = max_drawdown(equity)
    if dd == 0:
        return 0.0
    return cagr(equity, periods_per_year) / dd


def trade_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Win rate, profit factor, expectancy from a list of closed trades."""
    closed = [t for t in trades if t.get("pnl") is not None]
    if not closed:
        return {
            "closed_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "profit_factor": 0.0, "expectancy": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
        }
    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    avg_win = (gross_win / len(wins)) if wins else 0.0
    avg_loss = (gross_loss / len(losses)) if losses else 0.0
    win_rate = len(wins) / len(closed)
    pf = (gross_win / gross_loss) if gross_loss > 0 else (gross_win and 99.0 or 0.0)
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
    return {
        "closed_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "profit_factor": min(99.0, pf),
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def full_report(
    equity: list[float],
    trades: list[dict[str, Any]],
    starting_balance: float,
    benchmark_return_pct: float = 0.0,
) -> dict[str, Any]:
    """The complete scorecard for a run."""
    end_equity = equity[-1] if equity else starting_balance
    ret_pct = ((end_equity - starting_balance) / starting_balance * 100
               if starting_balance > 0 else 0.0)
    ts = trade_stats(trades)
    return {
        "starting_balance": starting_balance,
        "end_equity": end_equity,
        "return_pct": ret_pct,
        "cagr_pct": cagr(equity) * 100,
        "max_drawdown_pct": max_drawdown(equity) * 100,
        "sharpe": sharpe(equity),
        "sortino": sortino(equity),
        "calmar": calmar(equity),
        "benchmark_return_pct": benchmark_return_pct,
        "alpha_pct": ret_pct - benchmark_return_pct,
        **ts,
    }
