"""HTTP routes — JSON API + HTMX-rendered HTML."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from astra.config import settings
from astra.engine.backtest import run_backtest

log = logging.getLogger(__name__)


def build_router(app: FastAPI) -> APIRouter:
    r = APIRouter()
    portfolio = app.state.portfolio
    memory = app.state.memory
    cache = app.state.cache
    engine = app.state.engine
    state = app.state.engine_state
    templates = app.state.templates

    # ---- Pages ----

    @r.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        acc = await portfolio.get_account()
        if not acc:
            return templates.TemplateResponse(
                request, "setup.html", {"settings": settings}
            )
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "account": acc,
                "settings": settings,
                "engine_running": state.running,
            },
        )

    @r.get("/trades", response_class=HTMLResponse)
    async def trades_page(request: Request):
        trades = await portfolio.list_trades(limit=500)
        return templates.TemplateResponse(
            request, "trades.html", {"trades": trades}
        )

    @r.get("/memory", response_class=HTMLResponse)
    async def memory_page(request: Request):
        lessons = await memory.all_lessons()
        patterns = await memory.all_patterns()
        generations = await memory.all_generations()
        return templates.TemplateResponse(
            request,
            "memory.html",
            {
                "lessons": lessons,
                "patterns": patterns,
                "generations": generations,
            },
        )

    @r.get("/backtest", response_class=HTMLResponse)
    async def backtest_page(request: Request):
        return templates.TemplateResponse(request, "backtest.html", {})

    @r.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request):
        acc = await portfolio.get_account()
        return templates.TemplateResponse(
            request,
            "settings.html",
            {
                "account": acc,
                "settings": settings,
                "engine_running": state.running,
            },
        )

    # ---- Setup / Account ----

    @r.post("/api/setup")
    async def setup(starting_balance: float = Form(...)):
        if starting_balance <= 0:
            raise HTTPException(400, "starting_balance must be > 0")
        acc = await portfolio.create_account(starting_balance, mode="paper")
        await memory.start_generation(starting_balance)
        await portfolio.append_equity(starting_balance, starting_balance)
        await engine.start()
        return {"ok": True, "account": acc}

    @r.post("/api/reset")
    async def reset(starting_balance: float = Form(...)):
        if starting_balance <= 0:
            raise HTTPException(400, "starting_balance must be > 0")
        # Close current generation
        gen = await memory.current_generation()
        if gen:
            acc = await portfolio.get_account()
            trades = await portfolio.list_trades(limit=100000)
            ending = (await portfolio.equity_history(limit=1) or [{}])[-1].get(
                "equity", acc["cash"] if acc else starting_balance
            )
            await memory.close_generation(
                gen["id"], float(ending), len(trades), "reset by user"
            )
        await engine.stop()
        await portfolio.wipe()
        new_acc = await portfolio.create_account(starting_balance, mode="paper")
        await memory.start_generation(starting_balance)
        await portfolio.append_equity(starting_balance, starting_balance)
        await engine.start()
        return {"ok": True, "account": new_acc, "memory_kept": True}

    # ---- Engine control ----

    @r.post("/api/engine/start")
    async def engine_start():
        await engine.start()
        return {"ok": True, "running": state.running}

    @r.post("/api/engine/stop")
    async def engine_stop():
        await engine.stop()
        return {"ok": True, "running": state.running}

    @r.post("/api/engine/tick")
    async def engine_tick():
        result = await engine.tick()
        return {"ok": True, "result": result}

    @r.post("/api/engine/reflect")
    async def engine_reflect():
        lessons = await engine.reflect()
        return {"ok": True, "lessons": lessons}

    @r.post("/api/engine/tick-interval")
    async def engine_tick_interval(seconds: int = Form(...)):
        if seconds < 5 or seconds > 24 * 3600:
            raise HTTPException(400, "seconds must be between 5 and 86400")
        settings.tick_seconds = seconds
        # Restart the engine loop so the new interval is picked up immediately.
        was_running = state.running
        if was_running:
            await engine.stop()
            await engine.start()
        return {"ok": True, "tick_seconds": seconds, "engine_running": state.running}

    # ---- Status fragments (HTMX) ----

    @r.get("/api/status", response_class=HTMLResponse)
    async def status_fragment(request: Request):
        acc = await portfolio.get_account()
        positions = await portfolio.get_positions()
        equity_hist = await portfolio.equity_history(limit=1)
        equity = equity_hist[-1]["equity"] if equity_hist else (acc["cash"] if acc else 0)
        return templates.TemplateResponse(
            request,
            "_status.html",
            {
                "account": acc,
                "equity": equity,
                "positions": positions,
                "engine_running": state.running,
                "last_tick": state.last_tick,
                "tick_count": state.tick_count,
                "last_error": state.last_error,
                "tick_seconds": settings.tick_seconds,
            },
        )

    @r.get("/api/positions", response_class=HTMLResponse)
    async def positions_fragment(request: Request):
        positions = await portfolio.get_positions()
        return templates.TemplateResponse(
            request, "_positions.html", {"positions": positions}
        )

    @r.get("/api/thoughts", response_class=HTMLResponse)
    async def thoughts_fragment(request: Request):
        thoughts = await portfolio.recent_thoughts(limit=30)
        return templates.TemplateResponse(
            request, "_thoughts.html", {"thoughts": thoughts}
        )

    @r.get("/api/trades-table", response_class=HTMLResponse)
    async def trades_fragment(request: Request):
        trades = await portfolio.list_trades(limit=50)
        return templates.TemplateResponse(
            request, "_trades_table.html", {"trades": trades}
        )

    # ---- JSON API ----

    @r.get("/api/equity")
    async def api_equity(limit: int = 500):
        hist = await portfolio.equity_history(limit=limit)
        return {"points": hist}

    @r.get("/api/signals/{symbol}")
    async def api_signals(symbol: str):
        from astra.data.finnhub import FinnhubClient
        from astra.signals.aggregator import SignalAggregator

        async with FinnhubClient(cache=cache) as fc:
            agg = SignalAggregator(fc)
            bundle = await agg.fetch(symbol.upper())
            ph, pd = bundle.pattern_hash()
            d = bundle.to_dict()
            d["pattern_hash"] = ph
            d["pattern_desc"] = pd
            return d

    @r.get("/api/lessons")
    async def api_lessons():
        return {"lessons": await memory.all_lessons()}

    @r.get("/api/patterns")
    async def api_patterns():
        return {"patterns": await memory.all_patterns()}

    @r.post("/api/backtest")
    async def api_backtest(
        symbols: str = Form(...),
        starting_balance: float = Form(...),
        days: int = Form(60),
        use_llm: bool = Form(False),
    ):
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if not sym_list:
            raise HTTPException(400, "symbols required")
        if starting_balance <= 0:
            raise HTTPException(400, "starting_balance must be > 0")
        if days <= 0 or days > 365 * 5:
            raise HTTPException(400, "days out of range")
        try:
            result = await run_backtest(
                sym_list, starting_balance, days, cache, use_llm=use_llm
            )
        except Exception as e:
            log.exception("backtest failed")
            raise HTTPException(500, f"backtest failed: {e}")
        return {
            "summary": result.summary(),
            "equity_curve": result.equity_curve,
            "trades": result.trades,
        }

    # ---- SSE ----

    @r.get("/sse/events")
    async def sse(request: Request):
        q = state.subscribe()

        async def event_generator():
            try:
                # Send hello so the client knows it's connected
                yield {"event": "hello", "data": json.dumps({"running": state.running})}
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        evt = await asyncio.wait_for(q.get(), timeout=15.0)
                        yield {
                            "event": evt.kind,
                            "data": json.dumps(evt.payload, default=str),
                        }
                    except asyncio.TimeoutError:
                        yield {"event": "ping", "data": "{}"}
            finally:
                state.unsubscribe(q)

        return EventSourceResponse(event_generator())

    @r.get("/health")
    async def health():
        return {"ok": True, "running": state.running, "tick_count": state.tick_count}

    return r
