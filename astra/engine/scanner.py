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


async def scan_universe(
    symbols: list[str],
    cache: CacheDB,
    top_n: int,
    priors: dict[str, dict[str, Any]] | None = None,
    concurrency: int = 16,
) -> list[dict[str, Any]]:
    """Score every symbol in `symbols` and return the top `top_n` by score."""
    if not symbols:
        return []
    t0 = time.monotonic()
    sem = asyncio.Semaphore(concurrency)

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
    scored.sort(key=lambda r: r["score"], reverse=True)
    top = scored[: max(1, top_n)]
    elapsed = time.monotonic() - t0
    log.info(
        "Universe scan: %d/%d symbols scored, top %d in %.2fs (best score=%.2f)",
        len(scored), len(symbols), len(top), elapsed,
        top[0]["score"] if top else 0.0,
    )
    return top
