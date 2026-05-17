"""Autonomous evolution loop — walk-forward generational optimisation.

Each generation replays the NEXT, never-traded, time-ordered window of
history. The bot starts every generation with $10k and the strategy
knowledge (per-stock profiles, contextual bandit) accumulated from
EARLIER windows — but it has never seen the prices of the window it is
about to trade. That guarantee is structural: windows are strictly
non-overlapping and walk forward in time, so the bot can carry general
per-stock technique forward but never the price path.

Between generations a bounded parameter tuner nudges the risk knobs;
because the next window is unseen, every change is validated out-of-sample
automatically. The loop stops when the return target is hit, the windows
run out, or performance plateaus.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from astra.config import settings
from astra.db.cache import CacheDB
from astra.db.memory import MemoryDB
from astra.engine.contextual_bandit import load_bandit, save_bandit
from astra.engine.metrics import full_report
from astra.engine.trainer import _load_history, _replay_window
from astra.profiles.character import compute_character
from astra.profiles.profile import StockProfile, load_all_profiles, save_profile

log = logging.getLogger(__name__)

_WARMUP_DAYS = 250          # reserved as indicator warm-up; never traded
_MIN_WINDOW = 20            # drop a trailing window shorter than this

# Bounded ranges for the parameter tuner — nothing can drift out of these.
_BOUNDS = {
    "stop_loss_pct": (0.02, 0.12),
    "take_profit_pct": (0.06, 0.40),
    "entry_min_confidence": (0.30, 0.80),
    "risk_per_trade_pct": (0.005, 0.04),
}


def _clamp(name: str, value: float) -> float:
    lo, hi = _BOUNDS[name]
    return max(lo, min(hi, value))


def propose_parameter_changes(report: dict[str, Any]) -> dict[str, tuple[float, str]]:
    """Heuristic, bounded parameter nudges based on a generation's scorecard.

    Deterministic on purpose — an LLM in a tight autonomous loop drifts.
    Returns {param: (new_value, reason)}."""
    changes: dict[str, tuple[float, str]] = {}
    dd = report.get("max_drawdown_pct", 0)
    wr = report.get("win_rate", 0)
    pf = report.get("profit_factor", 0)
    sharpe = report.get("sharpe", 0)
    trades = report.get("closed_trades", 0)

    if dd > 25:
        new = round(_clamp("stop_loss_pct", settings.stop_loss_pct * 0.85), 4)
        if new != settings.stop_loss_pct:
            changes["stop_loss_pct"] = (new, f"max drawdown {dd:.0f}% too high — tighten stop")

    if trades < 8:
        new = round(_clamp("entry_min_confidence", settings.entry_min_confidence - 0.05), 2)
        if new != settings.entry_min_confidence:
            changes["entry_min_confidence"] = (new, f"only {trades} trades — lower the entry bar")
    elif wr < 0.40 and trades >= 20:
        new = round(_clamp("entry_min_confidence", settings.entry_min_confidence + 0.05), 2)
        if new != settings.entry_min_confidence:
            changes["entry_min_confidence"] = (new, f"win rate {wr*100:.0f}% low — raise the entry bar")

    if pf and pf < 1.0:
        new = round(_clamp("take_profit_pct", settings.take_profit_pct * 1.12), 4)
        if new != settings.take_profit_pct:
            changes["take_profit_pct"] = (new, f"profit factor {pf:.2f} < 1 — let winners run")

    if sharpe < 0.3:
        new = round(_clamp("risk_per_trade_pct", settings.risk_per_trade_pct * 0.9), 4)
        if new != settings.risk_per_trade_pct:
            changes["risk_per_trade_pct"] = (new, f"Sharpe {sharpe:.2f} weak — cut risk per trade")

    return changes


def _params_snapshot() -> dict[str, float]:
    return {k: getattr(settings, k) for k in _BOUNDS}


async def run_evolution(
    symbols: list[str],
    memory: MemoryDB,
    cache: CacheDB,
    target_return_pct: float = 15.0,
    starting_balance: float = 10_000.0,
    window_days: int = 63,
    max_generations: int = 20,
    plateau_patience: int = 4,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the walk-forward evolution loop. Returns the generation history."""
    started = time.monotonic()

    def _report(phase: str, gen: int = 0, total: int = 0, detail: str = "") -> None:
        log.info("evolution: %s gen %d/%d %s", phase, gen, total, detail)
        if progress:
            progress({"phase": phase, "generation": gen, "total": total,
                      "detail": detail,
                      "elapsed_s": round(time.monotonic() - started, 1)})

    _report("Loading 5y history", 0, 0, f"{len(symbols)} symbols")
    history = await _load_history(symbols, cache)
    if not history:
        return {"ok": False, "error": "no history available"}

    all_ts = sorted({int(c["t"]) for rows in history.values() for c in rows})
    if len(all_ts) < _WARMUP_DAYS + window_days * 2:
        return {"ok": False, "error": "insufficient history for walk-forward windows"}

    # Strictly non-overlapping, time-ordered windows after the warm-up reserve.
    window_ts = all_ts[_WARMUP_DAYS:]
    windows: list[list[int]] = []
    for i in range(0, len(window_ts), window_days):
        chunk = window_ts[i:i + window_days]
        if len(chunk) >= _MIN_WINDOW:
            windows.append(chunk)
    windows = windows[:max_generations]
    if not windows:
        return {"ok": False, "error": "no walk-forward windows"}

    # Strategy knowledge carried forward across generations.
    profiles = await load_all_profiles(memory)
    bandit = await load_bandit(memory)

    generations: list[dict[str, Any]] = []
    best_return: float | None = None
    no_improve = 0
    status = "completed all walk-forward windows"

    from datetime import datetime, timezone

    def _d(ts: int) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()

    for gen_i, window in enumerate(windows):
        win_start = window[0]
        # Honest character: classified ONLY from data before this window.
        for sym, rows in history.items():
            pre = [c for c in rows if int(c["t"]) < win_start]
            if len(pre) >= 40:
                ch = compute_character(pre)
                if ch.get("classified"):
                    profiles.setdefault(sym, StockProfile(symbol=sym)).character = ch

        _report("Replaying generation", gen_i + 1, len(windows),
                f"{_d(window[0])} → {_d(window[-1])}")
        pf = _replay_window(history, window, profiles, starting_balance,
                            learn=True, bandit=bandit)
        report = full_report(pf.equity_curve, pf.trades, starting_balance)

        # Persist the strategy knowledge so it carries to the next window.
        for prof in profiles.values():
            await save_profile(memory, prof)
        await save_bandit(memory, bandit)

        gen_rec: dict[str, Any] = {
            "n": gen_i + 1,
            "window": f"{_d(window[0])} → {_d(window[-1])}",
            "report": report,
            "params": _params_snapshot(),
            "changes": [],
        }
        generations.append(gen_rec)

        if report["return_pct"] >= target_return_pct:
            status = (f"target +{target_return_pct:.0f}% reached at "
                      f"generation {gen_i + 1}")
            break

        if best_return is None or report["return_pct"] > best_return:
            best_return = report["return_pct"]
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= plateau_patience:
            status = (f"plateau — no improvement for {plateau_patience} "
                      f"generations")
            break

        # Tune the risk knobs for the NEXT (unseen) window.
        changes = propose_parameter_changes(report)
        for param, (value, reason) in changes.items():
            setattr(settings, param, value)
            gen_rec["changes"].append(
                {"param": param, "value": value, "reason": reason})

    await memory.save_model("evolution", {
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "target_return_pct": target_return_pct,
        "generations": generations,
        "final_params": _params_snapshot(),
    })

    return {
        "ok": True,
        "status": status,
        "generations": generations,
        "target_return_pct": target_return_pct,
        "final_params": _params_snapshot(),
    }
