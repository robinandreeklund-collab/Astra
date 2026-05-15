"""Yahoo Finance daily candles via yfinance.

Used as the primary candle source because Finnhub's /stock/candle endpoint
is no longer on the free tier. yfinance scrapes Yahoo and is rate-limited
client-side; we call it from a thread pool so we don't block the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


def _fetch_sync(symbol: str, period: str) -> list[dict[str, Any]]:
    """Blocking yfinance fetch, executed in a worker thread."""
    try:
        # auto_adjust=False keeps raw OHLC; progress=False silences yfinance
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


async def fetch_daily_candles(symbol: str, days: int = 180) -> list[dict[str, Any]]:
    """Async wrapper that returns daily candles for ~`days` days back."""
    if days <= 30:
        period = "1mo"
    elif days <= 90:
        period = "3mo"
    elif days <= 180:
        period = "6mo"
    elif days <= 365:
        period = "1y"
    elif days <= 730:
        period = "2y"
    else:
        period = "5y"
    sym = symbol.replace(".", "-")  # BRK.B -> BRK-B for yfinance
    return await asyncio.to_thread(_fetch_sync, sym, period)


async def fetch_last_close(symbol: str) -> float | None:
    rows = await fetch_daily_candles(symbol, days=10)
    if not rows:
        return None
    return float(rows[-1]["c"])
