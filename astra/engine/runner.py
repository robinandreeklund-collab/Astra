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
        # Per-symbol short signal history (most recent first). Used to compute
        # tick-over-tick deltas for the LLM prompt.
        self.signal_history: dict[str, list[dict[str, Any]]] = {}
        # Stats from the latest universe scan (how many symbols, which top
        # candidates) — shown on the dashboard so the user can see what the
        # bot considered, not just what it traded.
        self.last_scan: dict[str, Any] = {}
        # Latest market regime classification.
        self.regime: dict[str, Any] = {"regime": "unknown"}
        # Historical-replay progress (when simulation mode is on).
        self.replay: dict[str, Any] = {}
        # Per-symbol last decision: {bar_ts, action, decided_at}. Used to skip
        # re-deciding a symbol when the underlying daily data hasn't changed.
        self.last_decision: dict[str, dict[str, Any]] = {}
        # Per-symbol last trade timestamp — enforces the cooldown window.
        self.last_trade_at: dict[str, float] = {}
        self.subscribers: list[asyncio.Queue[TickEvent]] = []

    def in_cooldown(self, symbol: str, cooldown_seconds: float,
                    now: float | None = None) -> bool:
        """Cooldown check against the supplied clock. In replay mode `now`
        is the replay timestamp, so the cooldown is measured in replay time
        — not wall-clock, which would lock symbols for tens of replay days."""
        last = self.last_trade_at.get(symbol)
        if last is None:
            return False
        now = now if now is not None else time.time()
        return (now - last) < cooldown_seconds

    def push_signal(self, symbol: str, snapshot: dict[str, Any], keep: int = 3) -> None:
        hist = self.signal_history.setdefault(symbol, [])
        hist.insert(0, snapshot)
        if len(hist) > keep:
            del hist[keep:]

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
        # Serialise ticks: the scheduler loop and a manual /api/engine/tick
        # must never run concurrently or they double-trade the same symbol.
        self._tick_lock = asyncio.Lock()

    async def ensure_universe(self) -> list[str]:
        if settings.simulate_data:
            # No network in sim mode — use the static large-cap list.
            from astra.data.universe import FALLBACK_SP500
            return list(FALLBACK_SP500)
        if self._universe:
            return self._universe
        async with FinnhubClient(cache=self.cache) as fc:
            self._universe = await load_universe(fc)
        return self._universe

    async def watchlist(self) -> list[str]:
        from astra.data.universe import parse_custom_watchlist
        u = await self.ensure_universe()
        priority = parse_custom_watchlist(settings.custom_watchlist)
        return watchlist_from_universe(u, settings.watchlist_size, priority=priority)

    async def tick(self) -> dict[str, Any]:
        """Run one decision cycle, serialised so two ticks never overlap."""
        if self._tick_lock.locked():
            # A tick is already running — don't queue a duplicate.
            return {"skipped": True, "reason": "tick already in progress"}
        async with self._tick_lock:
            return await self._tick_impl()

    async def _tick_impl(self) -> dict[str, Any]:
        """Run one decision cycle.

        When ASTRA_SCAN_UNIVERSE=true (the default), every tick:
          1. Scans the full S&P 500 with cached yfinance candles + TA.
          2. Picks the top `scan_top_n` symbols by signal strength + momentum.
          3. Always includes held positions + custom-watchlist priorities.
          4. Deep-dives (Finnhub + LLM) only on that union.

        When scan is off, falls back to the static watchlist behaviour.
        """
        acc = await self.portfolio.get_account()
        if not acc:
            return {"skipped": True, "reason": "no account"}

        from astra.data.universe import parse_custom_watchlist
        priority = parse_custom_watchlist(settings.custom_watchlist)

        # Build the deep-dive symbol list
        held = {p["symbol"] for p in await self.portfolio.get_positions()}
        universe = await self.ensure_universe()

        # Simulation mode: intraday historical replay. Advance one HOURLY
        # bar. On a new trading day → full decision cycle; within a day →
        # a light monitor tick that only checks held positions' stops.
        # The very first replay tick loads the data and runs a FULL cycle
        # on the starting day (no advance yet) so the bot trades from tick 1.
        monitor_only = False
        market = None
        if settings.simulate_data:
            from astra.data.simulator import get_market, load_market
            market = get_market()
            if market is None:
                self.state.replay = {"loading": True}
                market = await load_market(universe, self.cache)
                monitor_only = False  # first tick = full cycle
            else:
                market.advance()
                monitor_only = not market.is_new_day
            self.state.replay = market.status()
            if market.at_end:
                log.info("Historical replay complete (%d days)",
                         market.total_replay_days)
                await self.state.broadcast(TickEvent("replay_done", {
                    "datetime": market.current_datetime(),
                    "days": market.total_replay_days,
                }))
                self._stop.set()

        if monitor_only:
            return await self._monitor_tick()

        # Cooldown clock: replay time in sim mode, wall-clock otherwise.
        clock = (float(market.current_ts)
                 if (settings.simulate_data and market) else time.time())

        # Load per-stock adaptive profiles + the global signal prior.
        from astra.profiles import load_all_profiles, load_global
        from astra.profiles.profile import all_global_rates
        profiles = await load_all_profiles(self.memory)
        global_profile = await load_global(self.memory)
        global_rates = all_global_rates(global_profile)

        # Load the contextual bandit (the cross-universe policy model).
        from astra.engine.contextual_bandit import extract_features, load_bandit
        bandit = await load_bandit(self.memory)

        # Load the expert ensemble (regime-weighted strategy votes).
        from astra.engine.ensemble import load_ensemble
        from astra.engine.experts import run_experts
        ensemble = await load_ensemble(self.memory)

        if settings.scan_universe and universe:
            from astra.engine.scanner import scan_universe
            # Pass the prior snapshots so the scanner can weight momentum,
            # and the profiles so it can apply the Thompson-sampling bandit.
            priors = {sym: hist[0] for sym, hist in self.state.signal_history.items() if hist}
            scan = await scan_universe(
                universe, self.cache, settings.scan_top_n,
                priors=priors, profiles=profiles,
            )
            scored = scan["candidates"]
            regime = scan["regime"]
            cross_section = {s["symbol"]: s for s in scored}
            top_candidates = [s["symbol"] for s in scored]
            symbols = list(dict.fromkeys(list(held) + priority + top_candidates))
            self.state.last_scan = {
                "universe_size": len(universe),
                "scanned": scan["scanned"],
                "deep_dive": len(symbols),
                "regime": regime,
                "top_candidates": [
                    {"symbol": s["symbol"], "score": round(s["score"], 2),
                     "bull": s["bull"], "bear": s["bear"],
                     "rs_rank": s.get("rs_rank")}
                    for s in scored[:10]
                ],
            }
        else:
            watch = await self.watchlist()
            symbols = list(dict.fromkeys(list(held) + watch))[
                : settings.watchlist_size + len(held)
            ]
            regime = {"regime": "unknown"}
            cross_section = {}

        from astra.signals.regime import regime_risk_multiplier
        regime_mult = regime_risk_multiplier(regime.get("regime", "unknown"))
        self.state.regime = regime

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

            from astra.engine.sizing import atr_pct_from_indicators, compute_buy_value
            from astra.signals.deltas import compute_deltas

            actions: list[dict[str, Any]] = []
            prices: dict[str, float] = {}
            cooldown_sec = settings.cooldown_minutes * 60
            open_count = len(await self.portfolio.get_positions())

            # Daily loss guard: if we're down past the limit today, block new
            # entries for the rest of the day. Exits still run normally.
            entries_blocked, block_reason = await risk.can_trade_today()
            entries_blocked = not entries_blocked
            if entries_blocked:
                log.info("Entries blocked today: %s", block_reason)

            async def _do_sell(symbol, pos, ref_price, bundle, reasoning, source):
                """Close 100% of a position. All-or-nothing — no partial sells."""
                ph, pdesc = bundle.pattern_hash()
                snap = bundle.to_dict()
                snap["_pattern_desc"] = pdesc
                snap["_decision"] = {
                    "action": "SELL", "size_pct": 1.0, "confidence": 1.0,
                    "reasoning": reasoning, "source": source,
                }
                try:
                    fill = await broker.sell(symbol, float(pos["qty"]),
                                             ref_price, snap, reasoning, ph)
                except NoPosition:
                    return None
                self.state.last_trade_at[symbol] = clock
                await self.portfolio.log_thought(symbol, "SELL", 1.0, reasoning)
                await self.state.broadcast(TickEvent("trade", {
                    "symbol": symbol, "side": "SELL", "qty": fill.qty,
                    "price": fill.price, "pnl": fill.pnl, "reasoning": reasoning,
                }))
                sell_trade = await self.portfolio.list_trades(limit=1)
                if sell_trade:
                    await recorder.record_close(sell_trade[0])
                return fill

            for symbol in symbols:
                try:
                    bundle = await agg.fetch(symbol)
                except Exception as e:
                    log.warning("Signal fetch failed for %s: %s", symbol, e)
                    continue

                quote = bundle.quote or {}
                ref_price = float(quote.get("c") or 0)
                tech = bundle.technical or {}
                if ref_price <= 0:
                    ref_price = float(tech.get("last_close") or 0)
                if ref_price <= 0:
                    continue
                prices[symbol] = ref_price
                bar_ts = int(tech.get("last_bar_ts") or 0)

                # Per-stock adaptive profile. Refresh its Layer-1 character
                # from the freshly fetched candles and persist it.
                from astra.profiles import StockProfile, save_profile
                profile = profiles.get(symbol) or StockProfile(symbol=symbol)
                if bundle.character and bundle.character.get("classified"):
                    profile.character = bundle.character
                    await save_profile(self.memory, profile)
                    profiles[symbol] = profile

                # Volatility-scaled, per-stock stop / take-profit / trailing.
                stop_pct = profile.adaptive_stop_pct(settings.stop_loss_pct)
                take_pct = profile.adaptive_target_pct(settings.take_profit_pct)

                pos = await self.portfolio.get_position(symbol)

                # ============ HELD POSITION ============
                if pos is not None:
                    avg = float(pos["avg_price"])
                    # Track the high-water mark for the trailing stop.
                    prev_hwm = float(pos.get("high_water_mark") or avg)
                    hwm = max(prev_hwm, ref_price)
                    if hwm > prev_hwm:
                        await self.portfolio.update_high_water_mark(symbol, hwm)

                    pct = (ref_price - avg) / avg if avg > 0 else 0.0
                    drop_from_high = (ref_price - hwm) / hwm if hwm > 0 else 0.0

                    # --- Forced exits: always run, bypass cooldown/LLM ---
                    forced_reason = None
                    if pct <= -stop_pct:
                        forced_reason = (f"STOP-LOSS: {pct*100:.1f}% "
                                         f"(adaptive limit -{stop_pct*100:.1f}%)")
                    elif pct >= take_pct:
                        forced_reason = (f"TAKE-PROFIT: {pct*100:+.1f}% "
                                         f"(adaptive target +{take_pct*100:.1f}%)")
                    elif (drop_from_high <= -settings.trailing_stop_pct and pct > 0):
                        forced_reason = (
                            f"TRAILING-STOP: {drop_from_high*100:.1f}% off the high "
                            f"(still +{pct*100:.1f}% vs entry)"
                        )
                    if forced_reason:
                        fill = await _do_sell(symbol, pos, ref_price, bundle,
                                              forced_reason, "forced")
                        if fill:
                            open_count -= 1
                            actions.append({"symbol": symbol, "action": "SELL",
                                            "qty": fill.qty, "price": fill.price,
                                            "pnl": fill.pnl, "forced": True})
                        continue

                    # --- Cooldown: just hold, don't re-evaluate ---
                    if self.state.in_cooldown(symbol, cooldown_sec, clock):
                        actions.append({"symbol": symbol, "action": "HOLD",
                                        "reason": "cooldown"})
                        continue

                    # --- Signal dedup: same daily bar as last decision → hold ---
                    prev_dec = self.state.last_decision.get(symbol)
                    if prev_dec and prev_dec.get("bar_ts") == bar_ts and bar_ts > 0:
                        actions.append({"symbol": symbol, "action": "HOLD",
                                        "reason": "no new data"})
                        continue

                    # --- Ask the model: EXIT or HOLD? ---
                    prior = self.state.signal_history.get(symbol, [])
                    bundle_dict = bundle.to_dict()
                    bundle_dict["_deltas"] = compute_deltas(
                        bundle_dict, prior[0] if prior else None)
                    bundle_dict["_regime"] = regime
                    bundle_dict["_cross_section"] = cross_section.get(symbol)
                    lessons, patterns = await retriever.for_decision(
                        bundle.pattern_hash()[0])
                    decision = await decider.decide(
                        bundle_dict, pos, 0.0, lessons, patterns, mode="exit",
                        profile_card=profile.card(global_rates))
                    self.state.push_signal(symbol, bundle_dict)
                    self.state.last_decision[symbol] = {
                        "bar_ts": bar_ts, "action": decision.action,
                        "decided_at": time.time()}
                    await self.portfolio.log_thought(
                        symbol, decision.action, decision.confidence,
                        decision.reasoning[:1000])
                    await self.state.broadcast(TickEvent("thought", {
                        "symbol": symbol, "action": decision.action,
                        "confidence": decision.confidence,
                        "reasoning": decision.reasoning, "source": decision.source,
                        "ref_price": ref_price}))

                    if decision.action == "SELL":
                        fill = await _do_sell(symbol, pos, ref_price, bundle,
                                              decision.reasoning, decision.source)
                        if fill:
                            open_count -= 1
                            actions.append({"symbol": symbol, "action": "SELL",
                                            "qty": fill.qty, "price": fill.price,
                                            "pnl": fill.pnl})
                    else:
                        actions.append({"symbol": symbol, "action": "HOLD"})
                    continue

                # ============ FLAT (no position) ============
                # --- Daily loss limit: no new entries ---
                if entries_blocked:
                    continue
                # --- Cooldown ---
                if self.state.in_cooldown(symbol, cooldown_sec, clock):
                    continue
                # --- Concentration cap ---
                if open_count >= settings.max_open_positions:
                    actions.append({"symbol": symbol, "action": "SKIP",
                                    "reason": "portfolio full"})
                    continue

                # --- Whole-share affordability (Avanza: no fractional shares) ---
                # Skip unbuyable names BEFORE spending an LLM call on them.
                acc = await self.portfolio.get_account()
                cash = float(acc["cash"]) if acc else 0.0
                equity_pts = await self.portfolio.equity_history(limit=1)
                equity = equity_pts[-1]["equity"] if equity_pts else cash
                max_position_value = equity * settings.max_position_pct
                if ref_price > max_position_value:
                    actions.append({"symbol": symbol, "action": "SKIP",
                                    "reason": f"1 share ${ref_price:.0f} > position "
                                              f"cap ${max_position_value:.0f}"})
                    continue
                if ref_price + settings.min_fee > cash:
                    actions.append({"symbol": symbol, "action": "SKIP",
                                    "reason": f"1 share ${ref_price:.0f} unaffordable "
                                              f"(cash ${cash:.0f})"})
                    continue

                # --- Signal dedup: already passed on this bar → skip ---
                prev_dec = self.state.last_decision.get(symbol)
                if (prev_dec and prev_dec.get("bar_ts") == bar_ts and bar_ts > 0
                        and prev_dec.get("action") != "BUY"):
                    continue

                prior = self.state.signal_history.get(symbol, [])
                bundle_dict = bundle.to_dict()
                bundle_dict["_deltas"] = compute_deltas(
                    bundle_dict, prior[0] if prior else None)
                bundle_dict["_regime"] = regime
                bundle_dict["_cross_section"] = cross_section.get(symbol)
                lessons, patterns = await retriever.for_decision(
                    bundle.pattern_hash()[0])

                # Contextual bandit: the cross-universe policy model's
                # Thompson-sampled estimate of this context's reward (in R).
                features = extract_features(bundle_dict, profile, regime)
                policy_score = bandit.sample_predict(features)
                bundle_dict["_policy_score"] = policy_score
                bundle_dict["_bandit_features"] = features

                # Expert ensemble: regime-weighted vote of the strategy experts.
                regime_name = regime.get("regime", "unknown")
                expert_votes = run_experts(bundle_dict)
                ensemble_vote = ensemble.combine(expert_votes, regime_name)
                bundle_dict["_ensemble"] = {
                    **ensemble_vote,
                    "experts": {n: v.to_dict() for n, v in expert_votes.items()},
                }
                bundle_dict["_expert_votes"] = {
                    n: v.action for n, v in expert_votes.items()}
                bundle_dict["_entry_regime"] = regime_name

                decision = await decider.decide(
                    bundle_dict, None, cash, lessons, patterns, mode="entry",
                    profile_card=profile.card(global_rates))
                pattern_hash, pattern_desc = bundle.pattern_hash()
                bundle_dict["_pattern_desc"] = pattern_desc
                bundle_dict["_decision"] = decision.to_dict()
                self.state.push_signal(symbol, bundle_dict)
                self.state.last_decision[symbol] = {
                    "bar_ts": bar_ts, "action": decision.action,
                    "decided_at": time.time()}
                await self.portfolio.log_thought(
                    symbol, decision.action, decision.confidence,
                    decision.reasoning[:1000])
                await self.state.broadcast(TickEvent("thought", {
                    "symbol": symbol, "action": decision.action,
                    "confidence": decision.confidence,
                    "reasoning": decision.reasoning, "source": decision.source,
                    "ref_price": ref_price}))

                # --- Entry requires real conviction ---
                if decision.action != "BUY":
                    actions.append({"symbol": symbol, "action": "HOLD"})
                    continue
                if decision.confidence < settings.entry_min_confidence:
                    actions.append({"symbol": symbol, "action": "SKIP",
                                    "reason": f"confidence {decision.confidence:.2f} "
                                              f"< {settings.entry_min_confidence}"})
                    continue

                # --- Benched stocks: the bot keeps losing here, skip entry ---
                if profile.state() == "BENCHED":
                    actions.append({"symbol": symbol, "action": "SKIP",
                                    "reason": "stock benched (poor track record)"})
                    continue

                # --- Policy model: skip contexts it has learned to avoid ---
                if policy_score < -0.8:
                    actions.append({"symbol": symbol, "action": "SKIP",
                                    "reason": f"policy model: expected "
                                              f"{policy_score:+.2f}R in this context"})
                    continue

                # --- Sizing: volatility × conviction × regime × policy × ensemble ---
                # The expert ensemble advises via sizing, not a hard veto —
                # the LLM may see what the rule-experts miss. When few experts
                # back the entry the position is sized right down (to ~40%).
                consensus = ensemble_vote["consensus"]
                bandit_factor = max(0.3, min(1.4, 0.6 + 0.5 * policy_score))
                ensemble_factor = max(0.4, min(1.3, 0.85 + 0.3 * consensus))
                atr_pct = atr_pct_from_indicators(tech)
                value = compute_buy_value(
                    equity, cash, decision.confidence * max(0.5, decision.size_pct),
                    atr_pct, settings,
                    conviction_multiplier=(profile.conviction_multiplier()
                                           * regime_mult * bandit_factor
                                           * ensemble_factor))

                # Whole shares only — floor the budget to an integer share count.
                qty = int(value // ref_price)
                if qty < 1:
                    actions.append({"symbol": symbol, "action": "SKIP",
                                    "reason": f"budget ${value:.0f} < 1 share "
                                              f"(${ref_price:.0f})"})
                    continue
                notional = qty * ref_price
                if notional < settings.min_trade_value:
                    actions.append({"symbol": symbol, "action": "SKIP",
                                    "reason": f"order ${notional:.0f} < "
                                              f"min ${settings.min_trade_value:.0f}"})
                    continue

                # --- Portfolio-level risk: sector cap + total heat ---
                from astra.engine.portfolio_risk import check_new_position
                held_positions = await self.portfolio.get_positions()
                risk_prices = dict(self.state.last_prices)
                risk_prices.update(prices)
                stop_pcts = {}
                for hp in held_positions:
                    hprof = profiles.get(hp["symbol"])
                    stop_pcts[hp["symbol"]] = (
                        hprof.adaptive_stop_pct(settings.stop_loss_pct)
                        if hprof else settings.stop_loss_pct)
                ok, why = check_new_position(
                    symbol, notional, held_positions, risk_prices, stop_pcts,
                    equity, stop_pct, settings.max_sector_pct,
                    settings.max_portfolio_heat, settings.stop_loss_pct)
                if not ok:
                    actions.append({"symbol": symbol, "action": "SKIP", "reason": why})
                    continue

                # Record the stop distance so R-multiples can be computed
                # when this trade later closes.
                bundle_dict["_stop_pct"] = stop_pct
                try:
                    fill = await broker.buy(symbol, float(qty), ref_price, bundle_dict,
                                            decision.reasoning, pattern_hash)
                except InsufficientCash as e:
                    actions.append({"symbol": symbol, "action": "SKIP",
                                    "reason": str(e)})
                    continue
                self.state.last_trade_at[symbol] = clock
                open_count += 1
                actions.append({"symbol": symbol, "action": "BUY",
                                "qty": fill.qty, "price": fill.price})
                await self.state.broadcast(TickEvent("trade", {
                    "symbol": symbol, "side": "BUY", "qty": fill.qty,
                    "price": fill.price, "reasoning": decision.reasoning}))

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

    async def _monitor_tick(self) -> dict[str, Any]:
        """Intraday replay tick — no scan, no LLM. Only checks held
        positions' stop-loss / take-profit / trailing-stop against the live
        intraday price, so exits fire when the price actually crosses the
        level, not just at the daily close."""
        from astra.data.simulator import get_market
        from astra.profiles import StockProfile, load_all_profiles
        from astra.signals.aggregator import SignalBundle

        market = get_market()
        broker = PaperBroker(self.portfolio)
        recorder = MemoryRecorder(self.portfolio, self.memory)
        positions = await self.portfolio.get_positions()
        profiles = await load_all_profiles(self.memory)
        actions: list[dict[str, Any]] = []
        prices: dict[str, float] = {}
        monitored: list[tuple[str, float]] = []

        for p in positions:
            symbol = p["symbol"]
            price = market.current_price(symbol) if market else None
            if not price or price <= 0:
                continue
            prices[symbol] = price
            avg = float(p["avg_price"])
            prof = profiles.get(symbol) or StockProfile(symbol=symbol)
            stop_pct = prof.adaptive_stop_pct(settings.stop_loss_pct)
            take_pct = prof.adaptive_target_pct(settings.take_profit_pct)

            prev_hwm = float(p.get("high_water_mark") or avg)
            hwm = max(prev_hwm, price)
            if hwm > prev_hwm:
                await self.portfolio.update_high_water_mark(symbol, hwm)

            pct = (price - avg) / avg if avg > 0 else 0.0
            drop = (price - hwm) / hwm if hwm > 0 else 0.0
            monitored.append((symbol, pct))
            reason = None
            if pct <= -stop_pct:
                reason = (f"STOP-LOSS: {pct*100:.1f}% intraday "
                          f"(adaptive limit -{stop_pct*100:.1f}%)")
            elif pct >= take_pct:
                reason = (f"TAKE-PROFIT: {pct*100:+.1f}% intraday "
                          f"(adaptive target +{take_pct*100:.1f}%)")
            elif drop <= -settings.trailing_stop_pct and pct > 0:
                reason = (f"TRAILING-STOP: {drop*100:.1f}% off the high "
                          f"(still +{pct*100:.1f}%)")
            if not reason:
                continue

            bundle = SignalBundle(symbol=symbol, fetched_at=time.time())
            ph, pdesc = bundle.pattern_hash()
            snap = bundle.to_dict()
            snap["_pattern_desc"] = pdesc
            snap["_decision"] = {"action": "SELL", "size_pct": 1.0,
                                 "confidence": 1.0, "reasoning": reason,
                                 "source": "forced"}
            try:
                fill = await broker.sell(symbol, float(p["qty"]), price,
                                         snap, reason, ph)
            except NoPosition:
                continue
            self.state.last_trade_at[symbol] = (
                float(market.current_ts) if market else time.time())
            await self.portfolio.log_thought(symbol, "SELL", 1.0, reason)
            await self.state.broadcast(TickEvent("trade", {
                "symbol": symbol, "side": "SELL", "qty": fill.qty,
                "price": fill.price, "pnl": fill.pnl, "reasoning": reason}))
            sell_trade = await self.portfolio.list_trades(limit=1)
            if sell_trade:
                await recorder.record_close(sell_trade[0])
            actions.append({"symbol": symbol, "action": "SELL",
                            "qty": fill.qty, "price": fill.price,
                            "pnl": fill.pnl, "forced": True})

        mtm = {p["symbol"]: prices.get(p["symbol"], p["avg_price"])
               for p in await self.portfolio.get_positions()}
        mtm.update(prices)
        equity, cash = await broker.mark_to_market(mtm)
        await self.portfolio.append_equity(equity, cash)
        self.state.last_prices.update(mtm)
        await self.state.broadcast(TickEvent("equity", {"equity": equity, "cash": cash}))

        # Heartbeat — show the bot IS watching every intraday hour, even
        # when no exit triggered. New ENTRY decisions wait for the next
        # trading day (the daily signals don't change intraday).
        when = (self.state.replay.get("datetime") or "")[-5:]
        if monitored:
            ranked = sorted(monitored, key=lambda x: x[1], reverse=True)
            best, worst = ranked[0], ranked[-1]
            fired = sum(1 for a in actions if a["action"] == "SELL")
            msg = (f"Monitoring {len(monitored)} positions @ {when} — "
                   f"best {best[0]} {best[1]*100:+.1f}%, "
                   f"worst {worst[0]} {worst[1]*100:+.1f}%")
            msg += (f"; {fired} exit(s) triggered" if fired
                    else "; no exit triggers")
        else:
            msg = (f"No open positions @ {when} — "
                   f"next entries at the start of the next trading day")
        await self.portfolio.log_thought(None, "MONITOR", None, msg)
        await self.state.broadcast(TickEvent("thought", {
            "symbol": None, "action": "MONITOR", "confidence": None,
            "reasoning": msg, "source": "monitor"}))

        self.state.last_tick = time.time()
        self.state.tick_count += 1
        return {"monitor": True, "actions": actions, "equity": equity, "cash": cash}

    async def reflect(self) -> dict[str, Any]:
        """Maintenance cycle: distil lessons, refresh per-stock playbooks,
        and run the strategy critic. All LLM-backed with heuristic fallbacks."""
        llm = LMStudioClient()
        ok = await llm.health()
        try:
            client = llm if ok else None
            r = Reflector(self.portfolio, self.memory, client)
            lessons = await r.reflect()
            playbooks = await self._refresh_playbooks(client)
            critique = await self._run_critic(client)
            return {"lessons": lessons, "playbooks_updated": playbooks,
                    "critique": critique}
        finally:
            await llm.close()

    async def _refresh_playbooks(self, client: Any) -> int:
        """Regenerate stale playbooks for stocks with enough trade history."""
        from astra.llm.roles import generate_playbook
        from astra.profiles import (
            GLOBAL_SYMBOL, StockProfile, load_all_profiles, save_profile)
        from astra.profiles.profile import all_global_rates

        profiles = await load_all_profiles(self.memory)
        global_rates = all_global_rates(
            profiles.get(GLOBAL_SYMBOL) or StockProfile(symbol=GLOBAL_SYMBOL))
        updated = 0
        # Refresh at most 8 per cycle to bound LLM cost.
        stale = [
            p for s, p in profiles.items()
            if s != GLOBAL_SYMBOL and p.trades >= 5
            and p.trades - p.playbook_at_trade >= 5
        ]
        stale.sort(key=lambda p: p.trades - p.playbook_at_trade, reverse=True)
        for prof in stale[:8]:
            recent = await self.portfolio.closed_trades(limit=12)
            recent = [t for t in recent if t.get("symbol") == prof.symbol]
            prof.playbook = await generate_playbook(
                prof, recent, global_rates, client)
            prof.playbook_at_trade = prof.trades
            await save_profile(self.memory, prof)
            updated += 1
        return updated

    async def _run_critic(self, client: Any) -> list[str]:
        from astra.llm.roles import generate_critique
        from astra.engine.metrics import full_report
        from astra.profiles import GLOBAL_SYMBOL, load_all_profiles

        acc = await self.portfolio.get_account()
        if not acc:
            return []
        equity = [h["equity"] for h in await self.portfolio.equity_history(limit=5000)]
        trades = await self.portfolio.list_trades(limit=100000)
        report = full_report(equity, trades, float(acc["starting_balance"]))
        profiles = await load_all_profiles(self.memory)
        summaries = [
            {"symbol": s, "edge": p.edge_score()}
            for s, p in profiles.items() if s != GLOBAL_SYMBOL and p.trades >= 3
        ]
        critique = await generate_critique(report, summaries, client)
        await self.memory.save_model("critique", {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "suggestions": critique,
            "report": report,
        })
        return critique

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
