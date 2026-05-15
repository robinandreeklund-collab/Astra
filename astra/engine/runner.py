"""Main trading loop — single tick + scheduler-driven loop."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from astra.config import settings
from astra.data.finnhub import FinnhubClient
from astra.data.universe import load_universe, watchlist_from_universe
from astra.db.cache import CacheDB
from astra.db.memory import MemoryDB
from astra.db.portfolio import PortfolioDB
from astra.engine.broker import InsufficientCash, NoPosition, PaperBroker
from astra.engine.risk import RiskManager
from astra.llm.client import LMStudioClient
from astra.llm.decision import Decision, DecisionEngine, HeuristicDecisionEngine
from astra.memory.recorder import MemoryRecorder
from astra.memory.retriever import MemoryRetriever
from astra.memory.reflector import Reflector
from astra.signals.aggregator import SignalAggregator, SignalBundle

log = logging.getLogger(__name__)


class TickEvent:
    """Used by SSE to broadcast tick activity."""

    def __init__(self, kind: str, payload: dict[str, Any]) -> None:
        self.kind = kind
        self.payload = payload
        self.ts = time.time()


class EngineState:
    def __init__(self) -> None:
        self.running: bool = False
        self.last_tick: float | None = None
        self.last_error: str | None = None
        self.tick_count: int = 0
        # Cache last seen prices by symbol so the UI can show live P&L
        # without re-fetching per request. Updated at the end of each tick.
        self.last_prices: dict[str, float] = {}
        self.subscribers: list[asyncio.Queue[TickEvent]] = []

    async def broadcast(self, evt: TickEvent) -> None:
        dead: list[asyncio.Queue[TickEvent]] = []
        for q in self.subscribers:
            try:
                q.put_nowait(evt)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            try:
                self.subscribers.remove(q)
            except ValueError:
                pass

    def subscribe(self) -> asyncio.Queue[TickEvent]:
        q: asyncio.Queue[TickEvent] = asyncio.Queue(maxsize=200)
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[TickEvent]) -> None:
        try:
            self.subscribers.remove(q)
        except ValueError:
            pass


class TradingEngine:
    def __init__(
        self,
        portfolio: PortfolioDB,
        memory: MemoryDB,
        cache: CacheDB,
        state: EngineState,
    ) -> None:
        self.portfolio = portfolio
        self.memory = memory
        self.cache = cache
        self.state = state
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._universe: list[str] = []

    async def ensure_universe(self) -> list[str]:
        if self._universe:
            return self._universe
        async with FinnhubClient(cache=self.cache) as fc:
            self._universe = await load_universe(fc)
        return self._universe

    async def watchlist(self) -> list[str]:
        u = await self.ensure_universe()
        return watchlist_from_universe(u, settings.watchlist_size)

    async def tick(self) -> dict[str, Any]:
        """Run one decision cycle across the watchlist."""
        acc = await self.portfolio.get_account()
        if not acc:
            return {"skipped": True, "reason": "no account"}

        watch = await self.watchlist()
        async with FinnhubClient(cache=self.cache) as fc:
            agg = SignalAggregator(fc)

            llm = LMStudioClient()
            llm_ok = await llm.health()
            decider = DecisionEngine(llm if llm_ok else None)
            if not llm_ok:
                await llm.close()

            broker = PaperBroker(self.portfolio)
            risk = RiskManager(self.portfolio)
            retriever = MemoryRetriever(self.memory)
            recorder = MemoryRecorder(self.portfolio, self.memory)

            # Always evaluate held positions, plus top of watchlist
            held = {p["symbol"] for p in await self.portfolio.get_positions()}
            symbols: list[str] = list(dict.fromkeys(list(held) + watch))[: settings.watchlist_size + len(held)]

            actions: list[dict[str, Any]] = []
            prices: dict[str, float] = {}

            for symbol in symbols:
                try:
                    bundle = await agg.fetch(symbol)
                except Exception as e:
                    log.warning("Signal fetch failed for %s: %s", symbol, e)
                    continue

                quote = bundle.quote or {}
                ref_price = float(quote.get("c") or 0)
                if ref_price <= 0:
                    tech = bundle.technical or {}
                    ref_price = float(tech.get("last_close") or 0)
                if ref_price <= 0:
                    continue
                prices[symbol] = ref_price

                pos = await self.portfolio.get_position(symbol)
                lessons, patterns = await retriever.for_decision(
                    bundle.pattern_hash()[0]
                )
                acc = await self.portfolio.get_account()
                cash = acc["cash"] if acc else 0.0

                decision = await decider.decide(
                    bundle.to_dict(), pos, cash, lessons, patterns
                )

                pattern_hash, pattern_desc = bundle.pattern_hash()
                snapshot = bundle.to_dict()
                snapshot["_pattern_desc"] = pattern_desc
                snapshot["_decision"] = decision.to_dict()

                await self.portfolio.log_thought(
                    symbol=symbol,
                    action=decision.action,
                    confidence=decision.confidence,
                    reasoning=decision.reasoning[:1000],
                )

                await self.state.broadcast(TickEvent("thought", {
                    "symbol": symbol,
                    "action": decision.action,
                    "confidence": decision.confidence,
                    "reasoning": decision.reasoning,
                    "source": decision.source,
                    "ref_price": ref_price,
                }))

                if decision.action == "BUY" and pos is None:
                    qty = await risk.size_buy(ref_price, decision.confidence * decision.size_pct)
                    if qty <= 0:
                        actions.append({"symbol": symbol, "action": "SKIP", "reason": "size=0"})
                        continue
                    ok, why = await risk.validate_buy(symbol, qty, ref_price)
                    if not ok:
                        actions.append({"symbol": symbol, "action": "SKIP", "reason": why})
                        await self.portfolio.log_thought(symbol, "SKIP", decision.confidence,
                                                         f"risk-blocked: {why}", accepted=False)
                        continue
                    try:
                        fill = await broker.buy(symbol, qty, ref_price, snapshot,
                                                decision.reasoning, pattern_hash)
                    except InsufficientCash as e:
                        actions.append({"symbol": symbol, "action": "SKIP", "reason": str(e)})
                        continue
                    actions.append({"symbol": symbol, "action": "BUY",
                                    "qty": fill.qty, "price": fill.price})
                    await self.state.broadcast(TickEvent("trade", {
                        "symbol": symbol, "side": "BUY", "qty": fill.qty,
                        "price": fill.price, "reasoning": decision.reasoning,
                    }))

                elif decision.action == "SELL" and pos is not None:
                    qty = pos["qty"] * decision.size_pct if 0 < decision.size_pct <= 1 else pos["qty"]
                    qty = round(qty, 4)
                    if qty <= 0:
                        qty = pos["qty"]
                    ok, why = await risk.validate_sell(symbol, qty)
                    if not ok:
                        actions.append({"symbol": symbol, "action": "SKIP", "reason": why})
                        continue
                    try:
                        fill = await broker.sell(symbol, qty, ref_price, snapshot,
                                                 decision.reasoning, pattern_hash)
                    except NoPosition as e:
                        actions.append({"symbol": symbol, "action": "SKIP", "reason": str(e)})
                        continue
                    actions.append({"symbol": symbol, "action": "SELL",
                                    "qty": fill.qty, "price": fill.price, "pnl": fill.pnl})
                    await self.state.broadcast(TickEvent("trade", {
                        "symbol": symbol, "side": "SELL", "qty": fill.qty,
                        "price": fill.price, "pnl": fill.pnl,
                        "reasoning": decision.reasoning,
                    }))
                    # Record pattern outcome
                    sell_trade = await self.portfolio.list_trades(limit=1)
                    if sell_trade:
                        await recorder.record_close(sell_trade[0])
                else:
                    actions.append({"symbol": symbol, "action": "HOLD"})

            if llm_ok:
                await llm.close()

            # Mark-to-market
            mtm_prices = {s: prices.get(s, p["avg_price"])
                          for p in await self.portfolio.get_positions()
                          for s in [p["symbol"]]}
            mtm_prices.update(prices)
            equity, cash = await broker.mark_to_market(mtm_prices)
            await self.portfolio.append_equity(equity, cash)

            # Cache prices on engine state so the UI can show live P&L
            # without each request re-hitting the data APIs.
            self.state.last_prices.update(mtm_prices)
            await self.state.broadcast(TickEvent("equity", {"equity": equity, "cash": cash}))

        self.state.last_tick = time.time()
        self.state.tick_count += 1
        self.state.last_error = None
        return {"actions": actions, "equity": equity, "cash": cash}

    async def reflect(self) -> list[str]:
        llm = LMStudioClient()
        ok = await llm.health()
        try:
            r = Reflector(self.portfolio, self.memory, llm if ok else None)
            return await r.reflect()
        finally:
            await llm.close()

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self.state.running = True
        self._task = asyncio.create_task(self._run_loop())
        log.info("Engine started (tick=%ds)", settings.tick_seconds)

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        self.state.running = False
        log.info("Engine stopped")

    async def _run_loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    await self.tick()
                except Exception as e:
                    log.exception("tick error")
                    self.state.last_error = str(e)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=settings.tick_seconds)
                except asyncio.TimeoutError:
                    pass
        finally:
            self.state.running = False
