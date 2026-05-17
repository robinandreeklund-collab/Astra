"""Universe scanner — picks the most actionable symbols each tick.

The free Finnhub tier (60 req/min) and the LLM's per-call cost both rule out
deep-diving every S&P 500 name on every tick. Instead, we do a cheap scan
across the whole universe using only cached yfinance candles, score each
symbol by signal strength + tick-over-tick momentum, and return the top N
candidates. The engine then deep-dives only those plus any held positions.

The first cold run warms the cache and is slower; subsequent runs are nearly
instant because yfinance daily bars only change once per day and we cache
them on disk.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

from astra.data.yahoo import fetch_daily_candles
from astra.db.cache import CacheDB
from astra.signals import technical
from astra.signals.aggregator import SignalBundle
from astra.signals.deltas import compute_deltas

log = logging.getLogger(__name__)


def _ta_score(tech: dict[str, Any]) -> tuple[float, int, int]:
    """Return (abs_strength, bull, bear) so we can sort by interest level."""
    if not tech.get("available"):
        return 0.0, 0, 0
    bull = 0
    bear = 0
    rsi = tech.get("rsi14")
    if isinstance(rsi, (int, float)):
        if rsi < 30:
            bull += 1
        elif rsi > 70:
            bear += 1
    if tech.get("macd_state") == "bullish":
        bull += 1
    elif tech.get("macd_state") == "bearish":
        bear += 1
    bb = tech.get("bb_position")
    if isinstance(bb, (int, float)):
        if bb < 0.2:
            bull += 1
        elif bb > 0.8:
            bear += 1
    sma = tech.get("sma_cross")
    if sma == "golden":
        bull += 1
    elif sma == "death":
        bear += 1
    # Volume spike is a tiebreaker rather than a direction
    vr = tech.get("volume_ratio")
    boost = 0.5 if isinstance(vr, (int, float)) and vr >= 1.5 else 0.0
    return float(abs(bull - bear)) + boost, bull, bear


async def _fetch_and_score(
    symbol: str,
    cache: CacheDB,
    prior: dict[str, Any] | None,
) -> dict[str, Any] | None:
    rows = await fetch_daily_candles(symbol, days=180, cache=cache)
    if not rows:
        return None
    ind = technical.compute_indicators(rows)
    if not ind.get("available"):
        return None
    score, bull, bear = _ta_score(ind)
    # Pseudo-bundle just for delta computation against the prior tick.
    cur = {"technical": ind, "symbol": symbol}
    delta = compute_deltas(cur, prior)
    # Momentum bonus: real changes (not the "no notable" sentinel) lift score.
    changes = delta.get("changes", []) if delta.get("available") else []
    momentum_bonus = 0.0
    if changes and "no notable changes" not in changes[0]:
        momentum_bonus = 0.4 * len(changes)
    return {
        "symbol": symbol,
        "score": score + momentum_bonus,
        "bull": bull,
        "bear": bear,
        "indicators": ind,
        "deltas": delta,
        "last_close": ind.get("last_close"),
    }


def _attach_cross_section(scored: list[dict[str, Any]]) -> None:
    """Compute cross-sectional relative strength + percentile rank.

    Relative strength = a stock's 5-day return minus the universe mean.
    A documented, robust edge (the momentum factor): leaders tend to keep
    leading. Each result gets `rel_strength` and `rs_rank` (0..1 percentile).
    """
    moms: list[tuple[dict[str, Any], float]] = []
    for r in scored:
        m = (r.get("indicators") or {}).get("pct_change_5d")
        if isinstance(m, (int, float)):
            moms.append((r, float(m)))
    if not moms:
        return
    universe_mean = sum(m for _, m in moms) / len(moms)
    ordered = sorted(moms, key=lambda x: x[1])
    n = len(ordered)
    for rank, (r, m) in enumerate(ordered):
        r["rel_strength"] = round(m - universe_mean, 3)
        r["rs_rank"] = round(rank / max(1, n - 1), 3)


async def scan_universe(
    symbols: list[str],
    cache: CacheDB,
    top_n: int,
    priors: dict[str, dict[str, Any]] | None = None,
    profiles: dict[str, Any] | None = None,
    concurrency: int = 16,
) -> dict[str, Any]:
    """Score the universe; return top candidates + the market regime.

    Returns {"candidates": [...top_n...], "regime": {...}, "scanned": N}.

    Each candidate's raw TA score is multiplied by a Thompson-sampling
    bandit weight (proven winners up, losers down) and a cross-sectional
    relative-strength boost (leaders up). The regime is classified from the
    breadth and momentum of the whole scanned set.
    """
    from astra.signals.regime import classify_regime

    if not symbols:
        return {"candidates": [], "regime": {"regime": "unknown"}, "scanned": 0}
    t0 = time.monotonic()
    sem = asyncio.Semaphore(concurrency)
    rng = random.Random()

    async def _one(sym: str) -> dict[str, Any] | None:
        async with sem:
            try:
                prior = (priors or {}).get(sym)
                return await _fetch_and_score(sym, cache, prior)
            except Exception as e:
                log.debug("scan failed for %s: %s", sym, e)
                return None

    results = await asyncio.gather(*[_one(s) for s in symbols])
    scored = [r for r in results if r is not None and r["score"] > 0]

    regime = classify_regime(scored)
    _attach_cross_section(scored)

    for r in scored:
        r["ta_score"] = r["score"]
        weight = 1.0
        prof = (profiles or {}).get(r["symbol"])
        if prof is not None:
            # Bandit sample ≈ a win-rate draw; map to a 0.4..1.6 multiplier.
            weight = 0.4 + 1.2 * prof.bandit_sample(rng)
        # Cross-sectional momentum boost: top-decile RS gets up to +30%.
        rs_rank = r.get("rs_rank", 0.5)
        rs_boost = 0.85 + 0.45 * rs_rank
        r["bandit_weight"] = round(weight, 3)
        r["score"] = r["ta_score"] * weight * rs_boost

    scored.sort(key=lambda r: r["score"], reverse=True)
    top = scored[: max(1, top_n)]
    elapsed = time.monotonic() - t0
    log.info(
        "Universe scan: %d/%d scored, regime=%s, top %d in %.2fs",
        len(scored), len(symbols), regime.get("regime"), len(top), elapsed,
    )
    return {"candidates": top, "regime": regime, "scanned": len(scored)}
