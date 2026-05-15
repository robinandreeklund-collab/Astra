"""S&P 500 universe management with offline fallback."""

from __future__ import annotations

import logging
from pathlib import Path

from astra.data.finnhub import FinnhubClient
from astra.db.cache import CacheDB

log = logging.getLogger(__name__)

# Static fallback list of large-cap S&P 500 names — used if Finnhub is unavailable
# or no API key is configured. Not exhaustive; runner picks from here for safety.
FALLBACK_SP500: list[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK.B", "TSLA", "AVGO",
    "LLY", "JPM", "V", "WMT", "UNH", "XOM", "MA", "PG", "ORCL", "JNJ", "HD",
    "COST", "ABBV", "BAC", "MRK", "CVX", "KO", "NFLX", "AMD", "PEP", "CRM",
    "ADBE", "TMO", "LIN", "ACN", "MCD", "CSCO", "ABT", "WFC", "DHR", "INTC",
    "TXN", "DIS", "NOW", "VZ", "INTU", "CAT", "QCOM", "PFE", "AMGN", "IBM",
    "GS", "MS", "AXP", "PM", "T", "RTX", "NEE", "BLK", "HON", "SPGI", "LOW",
    "ELV", "BKNG", "AMAT", "BA", "C", "PLD", "DE", "SCHW", "GILD", "ADI",
    "MDT", "TJX", "VRTX", "SBUX", "MMC", "REGN", "ETN", "ISRG", "ADP", "BMY",
    "PANW", "KLAC", "LRCX", "MU", "SYK", "CB", "PGR", "ZTS", "BX", "FI",
    "MO", "BSX", "DUK", "SO", "USB", "EQIX", "CI", "CSX", "ICE", "CL",
]


async def load_universe(client: FinnhubClient | None = None) -> list[str]:
    """Best-effort load of S&P 500 tickers; falls back to static list."""
    if client is not None and client.api_key:
        try:
            data = await client.index_constituents("^GSPC")
            const = data.get("constituents") or []
            if const:
                return [s.replace(".", "-") for s in const]
        except Exception as e:
            log.warning("Failed to load S&P 500 from Finnhub: %s", e)
    return list(FALLBACK_SP500)


def watchlist_from_universe(universe: list[str], size: int) -> list[str]:
    return universe[: max(1, size)]
