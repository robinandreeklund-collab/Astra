"""HTTP routes — JSON API + HTMX-rendered HTML."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
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
        critique = await memory.get_model("critique")
        return templates.TemplateResponse(
            request,
            "memory.html",
            {
                "lessons": lessons,
                "patterns": patterns,
                "generations": generations,
                "critique": critique,
            },
        )

    @r.get("/backtest", response_class=HTMLResponse)
    async def backtest_page(request: Request):
        return templates.TemplateResponse(request, "backtest.html", {})

    @r.get("/training", response_class=HTMLResponse)
    async def training_page(request: Request):
        return templates.TemplateResponse(
            request, "training.html",
            {"status": app.state.training_status},
        )

    @r.post("/api/train")
    async def api_train(symbols: str = Form(""), starting_balance: float = Form(100000)):
        st = app.state.training_status
        if st.get("running"):
            return {"ok": False, "error": "training already running"}
        from astra.data.universe import FALLBACK_SP500, parse_custom_watchlist
        syms = parse_custom_watchlist(symbols) or list(FALLBACK_SP500)
        st.clear()
        st.update({"running": True, "message": "starting…", "result": None})

        async def _run() -> None:
            from astra.engine.trainer import run_training
            try:
                def _progress(m: str) -> None:
                    st["message"] = m
                result = await run_training(
                    syms, starting_balance, memory, cache, progress=_progress)
                st["result"] = result
                st["message"] = "complete" if result.get("ok") else result.get("error", "failed")
            except Exception as e:
                log.exception("training failed")
                st["message"] = f"error: {e}"
                st["result"] = {"ok": False, "error": str(e)}
            finally:
                st["running"] = False

        asyncio.create_task(_run())
        return {"ok": True, "started": True, "symbols": len(syms)}

    @r.get("/api/train/status")
    async def api_train_status():
        return app.state.training_status

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

    @r.get("/profiles", response_class=HTMLResponse)
    async def profiles_page(request: Request):
        from astra.profiles import GLOBAL_SYMBOL, load_all_profiles
        all_profiles = await load_all_profiles(memory)
        rows = []
        for sym, p in all_profiles.items():
            if sym == GLOBAL_SYMBOL:
                continue
            ch = p.character or {}
            rows.append({
                "symbol": sym,
                "archetype": ch.get("archetype", "—"),
                "vol_tier": ch.get("vol_tier", "—"),
                "atr_pct": ch.get("atr_pct"),
                "trades": p.trades,
                "win_rate": p.win_rate(),
                "profit_factor": p.profit_factor(),
                "expectancy_r": p.expectancy_r(),
                "realized_pnl": p.realized_pnl,
                "edge_score": p.edge_score(),
                "state": p.state(),
                "conviction": p.conviction_multiplier(),
                "streak": p.current_streak,
                "playbook": p.playbook,
            })
        # Most-traded first, then by edge
        rows.sort(key=lambda r: (r["trades"], r["edge_score"]), reverse=True)
        return templates.TemplateResponse(
            request, "profiles.html", {"profiles": rows},
        )

    @r.get("/experts", response_class=HTMLResponse)
    async def experts_page(request: Request):
        from astra.engine.ensemble import load_ensemble
        ensemble = await load_ensemble(memory)
        return templates.TemplateResponse(
            request, "experts.html",
            {"weights": ensemble.regime_table(), "updates": ensemble.updates},
        )

    @r.get("/api/profiles")
    async def api_profiles():
        from astra.profiles import GLOBAL_SYMBOL, load_all_profiles
        all_profiles = await load_all_profiles(memory)
        return {
            "profiles": [
                p.to_dict() for s, p in all_profiles.items() if s != GLOBAL_SYMBOL
            ]
        }

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
        result = await engine.reflect()
        return {"ok": True, **result}

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

    @r.post("/api/engine/sim-mode")
    async def engine_sim_mode(enabled: bool = Form(...)):
        settings.simulate_data = bool(enabled)
        # Start a fresh synthetic market and clear stale per-symbol caches so
        # simulated and real data never cross-contaminate.
        if settings.simulate_data:
            from astra.data.simulator import reset_simulator
            from astra.data.universe import FALLBACK_SP500
            reset_simulator(list(FALLBACK_SP500))
        state.last_prices.clear()
        state.signal_history.clear()
        state.last_decision.clear()
        state.last_trade_at.clear()
        state.last_scan = {}
        return {"ok": True, "simulate_data": settings.simulate_data}

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
                "simulate_data": settings.simulate_data,
            },
        )

    @r.get("/api/positions", response_class=HTMLResponse)
    async def positions_fragment(request: Request):
        positions = await portfolio.get_positions()
        prices = state.last_prices
        enriched: list[dict[str, Any]] = []
        total_value = 0.0
        total_cost = 0.0
        for p in positions:
            last = float(prices.get(p["symbol"]) or p["avg_price"])
            qty = float(p["qty"])
            avg = float(p["avg_price"])
            value = qty * last
            cost = qty * avg
            pnl = value - cost
            pnl_pct = (pnl / cost) if cost > 0 else 0.0
            enriched.append({
                **p,
                "last_price": last,
                "market_value": value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "has_live_price": p["symbol"] in prices,
            })
            total_value += value
            total_cost += cost

        # Portfolio-level risk: total heat + sector concentration.
        from astra.engine.portfolio_risk import portfolio_heat, sector_exposure
        from astra.profiles import load_all_profiles
        profiles = await load_all_profiles(memory)
        acc = await portfolio.get_account()
        equity = (float(acc["cash"]) if acc else 0.0) + total_value
        risk_prices = {p["symbol"]: float(prices.get(p["symbol"]) or p["avg_price"])
                       for p in positions}
        stop_pcts = {}
        for p in positions:
            prof = profiles.get(p["symbol"])
            stop_pcts[p["symbol"]] = (prof.adaptive_stop_pct(settings.stop_loss_pct)
                                      if prof else settings.stop_loss_pct)
        heat = portfolio_heat(positions, risk_prices, stop_pcts, equity,
                              settings.stop_loss_pct)
        exposure = sector_exposure(positions, risk_prices, equity)
        top_sector = max(exposure.items(), key=lambda x: x[1]) if exposure else ("—", 0.0)
        return templates.TemplateResponse(
            request, "_positions.html",
            {
                "positions": enriched,
                "total_value": total_value,
                "total_cost": total_cost,
                "total_pnl": total_value - total_cost,
                "total_pnl_pct": ((total_value - total_cost) / total_cost) if total_cost > 0 else 0.0,
                "heat": heat,
                "max_heat": settings.max_portfolio_heat,
                "top_sector": top_sector[0],
                "top_sector_pct": top_sector[1],
            },
        )

    @r.get("/api/summary", response_class=HTMLResponse)
    async def summary_fragment(request: Request):
        acc = await portfolio.get_account()
        positions = await portfolio.get_positions()
        prices = state.last_prices
        market_value = sum(
            float(p["qty"]) * float(prices.get(p["symbol"]) or p["avg_price"])
            for p in positions
        )
        cost_basis = sum(float(p["qty"]) * float(p["avg_price"]) for p in positions)
        cash = float(acc["cash"]) if acc else 0.0
        equity = cash + market_value
        start = float(acc["starting_balance"]) if acc else 0.0
        # Today's change: oldest equity point with today's ISO date prefix
        hist = await portfolio.equity_history(limit=2000)
        today = datetime.now(timezone.utc).date().isoformat() if hist else ""
        today_points = [h for h in hist if h["ts"].startswith(today)] if hist else []
        day_open = today_points[0]["equity"] if today_points else (hist[0]["equity"] if hist else equity)

        # Decompose total P&L so the user can see why equity != starting + unrealized.
        # total_pnl = realized_pnl + unrealized_pnl - fees_on_open_positions
        all_trades = await portfolio.list_trades(limit=100000)
        fees_paid = sum(float(t.get("fees") or 0) for t in all_trades)
        realized_pnl = sum(
            float(t.get("pnl") or 0)
            for t in all_trades
            if t.get("side") == "SELL" and t.get("pnl") is not None
        )
        total_pnl = equity - start
        unrealized = market_value - cost_basis
        # The remainder of total P&L not explained by current MTM or closed
        # trades is the buy-fee + slippage drag on positions still open.
        open_position_drag = (unrealized + realized_pnl) - total_pnl
        return templates.TemplateResponse(
            request, "_summary.html",
            {
                "equity": equity,
                "cash": cash,
                "market_value": market_value,
                "starting_balance": start,
                "total_pnl": total_pnl,
                "total_pnl_pct": (total_pnl / start) if start > 0 else 0.0,
                "unrealized": unrealized,
                "unrealized_pct": (unrealized / cost_basis) if cost_basis > 0 else 0.0,
                "realized_pnl": realized_pnl,
                "fees_paid": fees_paid,
                "open_position_drag": open_position_drag,
                "trade_count": len(all_trades),
                "day_change": equity - day_open,
                "day_change_pct": ((equity - day_open) / day_open) if day_open > 0 else 0.0,
                "position_count": len(positions),
            },
        )

    @r.get("/api/scan", response_class=HTMLResponse)
    async def scan_fragment(request: Request):
        scan = state.last_scan or {}
        return templates.TemplateResponse(
            request, "_scan.html", {"scan": scan}
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

    @r.get("/api/trade/{trade_id}")
    async def api_trade(trade_id: int):
        """Trade detail + price chart with buy/sell markers."""
        trade = await portfolio.get_trade(trade_id)
        if not trade:
            raise HTTPException(404, "trade not found")
        symbol = trade["symbol"]

        # Resolve the buy/sell pair.
        buy = sell = None
        if trade["side"] == "SELL":
            sell = trade
            if trade.get("closed_trade_id"):
                buy = await portfolio.get_trade(int(trade["closed_trade_id"]))
        else:
            buy = trade
            sell = await portfolio.find_sell_for_buy(trade_id)

        # Candle history for the chart (yfinance live, simulator in sim mode).
        from astra.data.yahoo import fetch_daily_candles
        candles = await fetch_daily_candles(symbol, days=180, cache=cache)
        dates = [
            datetime.fromtimestamp(int(c["t"]), tz=timezone.utc).date().isoformat()
            for c in candles
        ]
        closes = [float(c["c"]) for c in candles]
        ts_to_index = {int(c["t"]): i for i, c in enumerate(candles)}

        def marker(t: dict[str, Any] | None) -> dict[str, Any] | None:
            if not t:
                return None
            snap = t.get("signal_snapshot") or {}
            bar_ts = (snap.get("technical") or {}).get("last_bar_ts")
            idx = ts_to_index.get(int(bar_ts)) if bar_ts else None
            return {
                "index": idx,
                "price": float(t["price"]),
                "executed_at": t["executed_at"],
                "qty": float(t["qty"]),
                "fees": float(t["fees"]),
                "reasoning": t.get("llm_reasoning") or "",
            }

        return {
            "symbol": symbol,
            "dates": dates,
            "closes": closes,
            "buy": marker(buy),
            "sell": marker(sell),
            "pnl": (float(sell["pnl"]) if sell and sell.get("pnl") is not None else None),
            "open": sell is None,
        }

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
