"""Yahoo Finance daily candles via yfinance, cached on disk.

Used as the historical-candle source because Finnhub's /stock/candle endpoint
is no longer on the free tier. yfinance scrapes Yahoo (slow + rate-limited
client-side) so we cache aggressively: daily bars only change once per day,
and we serve up to `MAX_CACHE_SECONDS` from SQLite without hitting Yahoo at
all.

The optional `cache` parameter is a CacheDB; if given, results are persisted
across process restarts. A small in-memory layer covers within-process repeat
calls.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import pandas as pd
import yfinance as yf

from astra.db.cache import CacheDB

log = logging.getLogger(__name__)

# Daily candles don't change intraday — cache for a few hours so the engine
# tick loop doesn't re-fetch every cycle.
MAX_CACHE_SECONDS = 4 * 60 * 60

# Per-process in-memory cache (avoids touching SQLite on hot path).
_MEM_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def _fetch_sync(symbol: str, period: str) -> list[dict[str, Any]]:
    """Blocking yfinance fetch, executed in a worker thread."""
    try:
        ticker = yf.Ticker(symbol)
        df: pd.DataFrame = ticker.history(period=period, interval="1d", auto_adjust=False)
        if df is None or df.empty:
            return []
        df = df.reset_index()
        rows: list[dict[str, Any]] = []
        for _, r in df.iterrows():
            ts = r["Date"]
            try:
                t = int(pd.Timestamp(ts).timestamp())
            except Exception:
                continue
            rows.append({
                "t": t,
                "o": float(r["Open"]),
                "h": float(r["High"]),
                "l": float(r["Low"]),
                "c": float(r["Close"]),
                "v": float(r["Volume"]) if "Volume" in r else 0.0,
            })
        return rows
    except Exception as e:
        log.warning("yfinance fetch %s failed: %s", symbol, e)
        return []


def _period_for(days: int) -> str:
    if days <= 30:
        return "1mo"
    if days <= 90:
        return "3mo"
    if days <= 180:
        return "6mo"
    if days <= 365:
        return "1y"
    if days <= 730:
        return "2y"
    return "5y"


async def fetch_daily_candles(
    symbol: str,
    days: int = 180,
    cache: CacheDB | None = None,
    force_real: bool = False,
) -> list[dict[str, Any]]:
    """Return daily candles for ~`days` back, served from cache when possible.

    In simulation mode the candles come from the historical-replay market —
    only days at or before the replay cursor, never the future. `force_real`
    bypasses that to fetch actual Yahoo data (used to LOAD the replay)."""
    from astra.config import settings as _settings
    if _settings.simulate_data and not force_real:
        from astra.data.simulator import get_market
        market = get_market()
        return market.candles(symbol, days) if market else []

    sym = symbol.replace(".", "-")  # BRK.B -> BRK-B for yfinance
    period = _period_for(days)
    cache_key = f"yf:{sym}:{period}"

    # In-memory hot cache
    hit = _MEM_CACHE.get(cache_key)
    now = time.time()
    if hit and now - hit[0] < MAX_CACHE_SECONDS:
        return hit[1]

    # Persistent disk cache
    if cache is not None:
        disk = await cache.get(cache_key)
        if disk and isinstance(disk, list) and disk:
            _MEM_CACHE[cache_key] = (now, disk)
            return disk

    rows = await asyncio.to_thread(_fetch_sync, sym, period)
    if rows:
        _MEM_CACHE[cache_key] = (now, rows)
        if cache is not None:
            await cache.set(cache_key, rows, ttl_seconds=MAX_CACHE_SECONDS)
            await cache.store_candles(sym, "D", rows)
    return rows


async def fetch_last_close(symbol: str, cache: CacheDB | None = None) -> float | None:
    rows = await fetch_daily_candles(symbol, days=30, cache=cache)
    if not rows:
        return None
    return float(rows[-1]["c"])
