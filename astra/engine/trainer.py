"""Walk-forward offline training.

Replays years of real daily data through a streamlined version of the
engine so the per-stock adaptive profiles accumulate experience *before*
the bot trades live. The trained profiles are persisted to memory.db, so
a freshly-started bot is no longer cold — it already "knows" each stock.

Walk-forward discipline: history is split into a TRAIN window (profiles
learn and are persisted) and a VALIDATE window (profiles frozen, perf
measured on unseen data). The train-vs-validate gap exposes overfitting.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from astra.config import settings
from astra.data.yahoo import fetch_daily_candles
from astra.db.cache import CacheDB
from astra.db.memory import MemoryDB
from astra.engine.metrics import full_report
from astra.engine.sizing import atr_pct_from_indicators, compute_buy_value
from astra.profiles.character import compute_character
from astra.profiles.profile import (
    GLOBAL_SYMBOL,
    StockProfile,
    extract_signals,
    save_profile,
)
from astra.signals import technical

log = logging.getLogger(__name__)

_FEE_MIN = 0.10
_FEE_PCT = 0.0025


def _commission(notional: float) -> float:
    return max(_FEE_MIN, _FEE_PCT * abs(notional))


class _Portfolio:
    """Lightweight in-memory portfolio for fast replay."""

    def __init__(self, starting: float) -> None:
        self.cash = starting
        self.starting = starting
        # symbol -> {qty, avg, hwm, entry_signals, entry_day}
        self.positions: dict[str, dict[str, Any]] = {}
        self.trades: list[dict[str, Any]] = []
        self.equity_curve: list[float] = [starting]

    def equity(self, prices: dict[str, float]) -> float:
        mv = sum(p["qty"] * prices.get(s, p["avg"])
                 for s, p in self.positions.items())
        return self.cash + mv

    def buy(self, symbol: str, qty: int, price: float,
            signals: list[str], day: int, stop_pct: float) -> None:
        cost = qty * price + _commission(qty * price)
        if cost > self.cash or qty < 1:
            return
        self.cash -= cost
        self.positions[symbol] = {
            "qty": qty, "avg": price, "hwm": price,
            "entry_signals": signals, "entry_day": day, "stop_pct": stop_pct,
        }
        self.trades.append({"side": "BUY", "symbol": symbol, "qty": qty,
                            "price": price, "pnl": None, "day": day})

    def sell(self, symbol: str, price: float, day: int) -> dict[str, Any] | None:
        pos = self.positions.pop(symbol, None)
        if not pos:
            return None
        proceeds = pos["qty"] * price - _commission(pos["qty"] * price)
        self.cash += proceeds
        pnl = (price - pos["avg"]) * pos["qty"] - _commission(pos["qty"] * price)
        initial_risk = pos["qty"] * pos["avg"] * pos.get("stop_pct", 0.05)
        r_multiple = (pnl / initial_risk) if initial_risk > 0 else 0.0
        rec = {"side": "SELL", "symbol": symbol, "qty": pos["qty"],
               "price": price, "pnl": pnl, "day": day,
               "entry_signals": pos["entry_signals"],
               "hold_days": day - pos["entry_day"],
               "r_multiple": round(r_multiple, 3)}
        self.trades.append(rec)
        return rec


async def _load_history(
    symbols: list[str], cache: CacheDB,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for sym in symbols:
        try:
            rows = await fetch_daily_candles(sym, days=1825, cache=cache)
            if len(rows) >= 80:
                out[sym] = rows
        except Exception as e:
            log.debug("training history fetch failed %s: %s", sym, e)
    return out


def _replay_window(
    history: dict[str, list[dict[str, Any]]],
    date_index: list[int],
    profiles: dict[str, StockProfile],
    starting: float,
    learn: bool,
) -> _Portfolio:
    """Replay one window. When learn=True, profiles are updated by outcomes."""
    pf = _Portfolio(starting)
    # Pre-index each symbol's candles by timestamp for trailing lookups.
    by_ts: dict[str, dict[int, int]] = {
        s: {int(c["t"]): i for i, c in enumerate(rows)}
        for s, rows in history.items()
    }

    for day, ts in enumerate(date_index):
        prices: dict[str, float] = {}
        for sym, rows in history.items():
            idx = by_ts[sym].get(ts)
            if idx is None or idx < 60:
                continue
            window = rows[: idx + 1]
            last_close = float(window[-1]["c"])
            prices[sym] = last_close
            tech = technical.compute_indicators(window[-220:])
            if not tech.get("available"):
                continue

            prof = profiles.setdefault(sym, StockProfile(symbol=sym))
            stop_pct = prof.adaptive_stop_pct(settings.stop_loss_pct)
            take_pct = prof.adaptive_target_pct(settings.take_profit_pct)

            pos = pf.positions.get(sym)
            if pos:
                pos["hwm"] = max(pos["hwm"], last_close)
                pct = (last_close - pos["avg"]) / pos["avg"]
                drop = (last_close - pos["hwm"]) / pos["hwm"]
                exit_now = (
                    pct <= -stop_pct or pct >= take_pct
                    or (drop <= -settings.trailing_stop_pct and pct > 0)
                )
                if not exit_now:
                    exit_now = _heuristic_exit(tech, pos, last_close)
                if exit_now:
                    rec = pf.sell(sym, last_close, day)
                    if rec and learn:
                        _learn(profiles, sym, rec)
                continue

            # Flat — entry decision
            if len(pf.positions) >= settings.max_open_positions:
                continue
            if not _heuristic_entry(tech):
                continue
            equity = pf.equity(prices)
            max_pos = equity * settings.max_position_pct
            if last_close > max_pos or last_close + _FEE_MIN > pf.cash:
                continue
            atr_pct = atr_pct_from_indicators(tech)
            value = compute_buy_value(
                equity, pf.cash, 0.7, atr_pct, settings,
                conviction_multiplier=prof.conviction_multiplier())
            qty = int(value // last_close)
            if qty < 1:
                continue
            signals = extract_signals({"technical": tech})
            pf.buy(sym, qty, last_close, signals, day, stop_pct)

        pf.equity_curve.append(pf.equity(prices))

    return pf


def _heuristic_entry(tech: dict[str, Any]) -> bool:
    """Lightweight bullish-stack check (mirrors HeuristicDecisionEngine)."""
    bull = bear = 0
    rsi = tech.get("rsi14")
    if isinstance(rsi, (int, float)):
        if rsi < 35:
            bull += 1
        elif rsi > 70:
            bear += 1
    if tech.get("macd_state") == "bullish":
        bull += 1
    elif tech.get("macd_state") == "bearish":
        bear += 1
    bb = tech.get("bb_position")
    if isinstance(bb, (int, float)):
        if bb < 0.25:
            bull += 1
        elif bb > 0.85:
            bear += 1
    if tech.get("sma_cross") == "golden":
        bull += 1
    elif tech.get("sma_cross") == "death":
        bear += 1
    return bull >= bear + 1 and (bull + bear) >= 2


def _heuristic_exit(tech: dict[str, Any], pos: dict[str, Any],
                    last_close: float) -> bool:
    bear = 0
    if tech.get("macd_state") == "bearish":
        bear += 1
    rsi = tech.get("rsi14")
    if isinstance(rsi, (int, float)) and rsi > 72:
        bear += 1
    bb = tech.get("bb_position")
    if isinstance(bb, (int, float)) and bb > 0.9:
        bear += 1
    gain = (last_close - pos["avg"]) / pos["avg"]
    return bear >= 2 or (gain > 0.08 and bear >= 1)


def _learn(profiles: dict[str, StockProfile], symbol: str,
           rec: dict[str, Any]) -> None:
    pnl = rec["pnl"]
    win = pnl > 0
    hold_minutes = rec.get("hold_days", 0) * 24 * 60
    signals = rec.get("entry_signals", [])
    r = rec.get("r_multiple", 0.0)
    profiles.setdefault(symbol, StockProfile(symbol=symbol)).record_trade_outcome(
        win, pnl, hold_minutes, signals, r)
    if signals:
        g = profiles.setdefault(GLOBAL_SYMBOL, StockProfile(symbol=GLOBAL_SYMBOL))
        g.record_trade_outcome(win, pnl, hold_minutes, signals, r)


async def run_training(
    symbols: list[str],
    starting_balance: float,
    memory: MemoryDB,
    cache: CacheDB,
    validate_fraction: float = 0.30,
    persist: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Walk-forward training run. Returns train/validate reports."""
    def _say(m: str) -> None:
        log.info("training: %s", m)
        if progress:
            progress(m)

    _say(f"loading history for {len(symbols)} symbols…")
    history = await _load_history(symbols, cache)
    if not history:
        return {"ok": False, "error": "no history available"}

    all_ts = sorted({int(c["t"]) for rows in history.values() for c in rows})
    if len(all_ts) < 150:
        return {"ok": False, "error": "insufficient history"}

    split = int(len(all_ts) * (1.0 - validate_fraction))
    train_ts = all_ts[:split]
    validate_ts = all_ts[split:]
    _say(f"{len(train_ts)} train days, {len(validate_ts)} validate days")

    profiles: dict[str, StockProfile] = {}

    # --- TRAIN: profiles learn ---
    _say("replaying train window (profiles learning)…")
    train_pf = _replay_window(history, train_ts, profiles, starting_balance, learn=True)

    # Character from full history.
    for sym, rows in history.items():
        ch = compute_character(rows)
        if ch.get("classified"):
            profiles.setdefault(sym, StockProfile(symbol=sym)).character = ch

    # --- VALIDATE: profiles frozen ---
    _say("replaying validate window (profiles frozen)…")
    frozen = {s: StockProfile.from_dict(p.to_dict()) for s, p in profiles.items()}
    validate_pf = _replay_window(
        history, validate_ts, frozen, starting_balance, learn=False)

    if persist:
        _say("persisting trained profiles to memory.db…")
        for prof in profiles.values():
            await save_profile(memory, prof)

    train_report = full_report(
        train_pf.equity_curve, train_pf.trades, starting_balance)
    validate_report = full_report(
        validate_pf.equity_curve, validate_pf.trades, starting_balance)

    _say("done")
    return {
        "ok": True,
        "symbols": len(history),
        "profiles_built": len([p for p in profiles if p != GLOBAL_SYMBOL]),
        "train": train_report,
        "validate": validate_report,
        "overfitting_gap_pct": train_report["return_pct"] - validate_report["return_pct"],
    }
