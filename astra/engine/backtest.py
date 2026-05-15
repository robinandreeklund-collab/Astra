"""Historical replay backtester.

Given a set of symbols and a date range, replays daily candles from Finnhub,
feeds the heuristic decision engine (LLM optional), and computes performance.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from astra.data.finnhub import FinnhubClient
from astra.db.cache import CacheDB
from astra.db.memory import MemoryDB
from astra.db.portfolio import PortfolioDB
from astra.engine.broker import InsufficientCash, NoPosition, PaperBroker
from astra.engine.risk import RiskManager
from astra.llm.client import LMStudioClient
from astra.llm.decision import DecisionEngine
from astra.memory.recorder import MemoryRecorder
from astra.memory.retriever import MemoryRetriever
from astra.signals import technical
from astra.signals.aggregator import SignalBundle

log = logging.getLogger(__name__)


class BacktestResult:
    def __init__(self) -> None:
        self.trades: list[dict[str, Any]] = []
        self.equity_curve: list[dict[str, Any]] = []
        self.start_equity: float = 0.0
        self.end_equity: float = 0.0
        self.benchmark_return_pct: float = 0.0

    def summary(self) -> dict[str, Any]:
        ret = (self.end_equity - self.start_equity) / self.start_equity if self.start_equity > 0 else 0.0
        wins = [t for t in self.trades if (t.get("pnl") or 0) > 0]
        losses = [t for t in self.trades if (t.get("pnl") or 0) <= 0 and t.get("side") == "SELL"]
        return {
            "start_equity": self.start_equity,
            "end_equity": self.end_equity,
            "return_pct": ret * 100,
            "trade_count": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / max(1, len(wins) + len(losses)),
            "benchmark_return_pct": self.benchmark_return_pct,
            "alpha_pct": ret * 100 - self.benchmark_return_pct,
        }


async def run_backtest(
    symbols: list[str],
    starting_balance: float,
    days: int,
    cache: CacheDB,
    use_llm: bool = False,
    benchmark_symbol: str = "SPY",
) -> BacktestResult:
    # Use isolated portfolio/memory DBs so backtest doesn't taint paper state
    import tempfile
    from pathlib import Path

    tmp = tempfile.mkdtemp(prefix="astra-bt-")
    bt_portfolio = PortfolioDB(Path(tmp) / "portfolio.db")
    bt_memory = MemoryDB(Path(tmp) / "memory.db")
    await bt_portfolio.init()
    await bt_memory.init()
    await bt_portfolio.create_account(starting_balance, mode="backtest")
    await bt_memory.start_generation(starting_balance)

    result = BacktestResult()
    result.start_equity = starting_balance

    async with FinnhubClient(cache=cache) as fc:
        now = int(time.time())
        from_ts = now - 60 * 60 * 24 * days

        # Fetch all candles up-front
        history: dict[str, list[dict[str, Any]]] = {}
        for sym in symbols:
            try:
                data = await fc.candles(sym, "D", from_ts, now)
                if data.get("s") != "ok":
                    continue
                rows = []
                for i in range(len(data.get("t", []))):
                    rows.append({
                        "t": data["t"][i],
                        "o": data["o"][i], "h": data["h"][i], "l": data["l"][i],
                        "c": data["c"][i], "v": data.get("v", [0]*len(data["t"]))[i],
                    })
                history[sym] = rows
            except Exception as e:
                log.warning("backtest fetch %s failed: %s", sym, e)

        # Benchmark
        bench_return = 0.0
        try:
            bdata = await fc.candles(benchmark_symbol, "D", from_ts, now)
            if bdata.get("s") == "ok" and bdata.get("c"):
                c = bdata["c"]
                if c[0] > 0:
                    bench_return = (c[-1] - c[0]) / c[0] * 100
        except Exception:
            pass
        result.benchmark_return_pct = bench_return

        # Build a sorted union of dates
        all_ts = sorted(set(int(r["t"]) for rows in history.values() for r in rows))
        if not all_ts:
            return result

        llm = LMStudioClient() if use_llm else None
        llm_ok = await llm.health() if llm else False
        decider = DecisionEngine(llm if llm_ok else None)
        broker = PaperBroker(bt_portfolio)
        risk = RiskManager(bt_portfolio)
        retriever = MemoryRetriever(bt_memory)
        recorder = MemoryRecorder(bt_portfolio, bt_memory)

        for ts in all_ts:
            iso_ts = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            current_prices: dict[str, float] = {}
            for sym, rows in history.items():
                bars_until_now = [r for r in rows if int(r["t"]) <= ts]
                if not bars_until_now:
                    continue
                last_close = float(bars_until_now[-1]["c"])
                current_prices[sym] = last_close
                if len(bars_until_now) < 30:
                    continue

                tech_ind = technical.compute_indicators(bars_until_now)
                bundle = SignalBundle(symbol=sym, fetched_at=float(ts))
                bundle.technical = tech_ind
                bundle.quote = {"c": last_close}
                pattern_hash, pattern_desc = bundle.pattern_hash()
                snapshot = bundle.to_dict()
                snapshot["_pattern_desc"] = pattern_desc

                pos = await bt_portfolio.get_position(sym)
                lessons, patterns = await retriever.for_decision(pattern_hash)
                acc = await bt_portfolio.get_account()
                cash = acc["cash"] if acc else 0.0

                decision = await decider.decide(bundle.to_dict(), pos, cash, lessons, patterns)

                if decision.action == "BUY" and pos is None:
                    qty = await risk.size_buy(last_close, decision.confidence * decision.size_pct)
                    if qty > 0:
                        ok, _ = await risk.validate_buy(sym, qty, last_close)
                        if ok:
                            try:
                                fill = await broker.buy(sym, qty, last_close, snapshot,
                                                        decision.reasoning, pattern_hash,
                                                        executed_at=iso_ts)
                                result.trades.append({"side": "BUY", "symbol": sym,
                                                      "qty": fill.qty, "price": fill.price,
                                                      "ts": iso_ts})
                            except InsufficientCash:
                                pass
                elif decision.action == "SELL" and pos is not None:
                    qty = pos["qty"]
                    try:
                        fill = await broker.sell(sym, qty, last_close, snapshot,
                                                 decision.reasoning, pattern_hash,
                                                 executed_at=iso_ts)
                        result.trades.append({"side": "SELL", "symbol": sym,
                                              "qty": fill.qty, "price": fill.price,
                                              "pnl": fill.pnl, "ts": iso_ts})
                        last_sell = await bt_portfolio.list_trades(limit=1)
                        if last_sell:
                            await recorder.record_close(last_sell[0])
                    except NoPosition:
                        pass

            # Equity at end of day
            equity, cash = await broker.mark_to_market(current_prices)
            await bt_portfolio.append_equity(equity, cash, ts=iso_ts)
            result.equity_curve.append({"ts": iso_ts, "equity": equity, "cash": cash})

        if llm:
            await llm.close()

        # Close out remaining positions at the last price
        last_prices = current_prices
        for p in await bt_portfolio.get_positions():
            sym = p["symbol"]
            px = last_prices.get(sym, p["avg_price"])
            try:
                await broker.sell(sym, p["qty"], px, {"_pattern_desc": "EOT-close"},
                                  "End of backtest", None,
                                  executed_at=datetime.fromtimestamp(all_ts[-1], tz=timezone.utc).isoformat())
            except Exception:
                pass

        acc = await bt_portfolio.get_account()
        result.end_equity = acc["cash"] if acc else starting_balance

    return result
