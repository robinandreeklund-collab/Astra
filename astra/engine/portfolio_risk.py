"""Portfolio-level risk control.

Per-position sizing isn't enough — ten tech stocks each at 10% all crash
together. This module enforces portfolio-wide limits:

  * Sector cap     — no more than `max_sector_pct` of equity in one sector.
  * Portfolio heat — total risk across open positions (each position's
                     value × its stop distance) capped at `max_portfolio_heat`.
                     This is the real measure of "how much can the book lose".

Sector membership uses a static map for the large-cap universe; unmapped
symbols fall into "Other".
"""

from __future__ import annotations

from typing import Any

# Compact GICS-style sector map for the fallback universe.
SECTOR_MAP: dict[str, str] = {
    # Information Technology
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AVGO": "Tech",
    "ORCL": "Tech", "CRM": "Tech", "ADBE": "Tech", "CSCO": "Tech",
    "ACN": "Tech", "AMD": "Tech", "TXN": "Tech", "QCOM": "Tech",
    "INTC": "Tech", "IBM": "Tech", "NOW": "Tech", "INTU": "Tech",
    "AMAT": "Tech", "ADI": "Tech", "KLAC": "Tech", "LRCX": "Tech",
    "MU": "Tech", "PANW": "Tech", "SMCI": "Tech", "MRVL": "Tech",
    "ANET": "Tech", "CRWD": "Tech", "SNOW": "Tech", "DDOG": "Tech",
    "NET": "Tech", "TEAM": "Tech", "WDAY": "Tech", "FTNT": "Tech",
    "ASML": "Tech", "TSM": "Tech", "ARM": "Tech", "PLTR": "Tech",
    "MDB": "Tech",
    # Communication Services
    "GOOGL": "Communication", "META": "Communication", "NFLX": "Communication",
    "DIS": "Communication", "VZ": "Communication", "T": "Communication",
    # Consumer Discretionary
    "AMZN": "ConsumerDisc", "TSLA": "ConsumerDisc", "HD": "ConsumerDisc",
    "MCD": "ConsumerDisc", "LOW": "ConsumerDisc", "BKNG": "ConsumerDisc",
    "TJX": "ConsumerDisc", "SBUX": "ConsumerDisc", "UBER": "ConsumerDisc",
    "ABNB": "ConsumerDisc", "SHOP": "ConsumerDisc", "DASH": "ConsumerDisc",
    "RIVN": "ConsumerDisc", "LCID": "ConsumerDisc", "NIO": "ConsumerDisc",
    "BABA": "ConsumerDisc", "JD": "ConsumerDisc",
    # Consumer Staples
    "WMT": "ConsumerStaples", "PG": "ConsumerStaples", "COST": "ConsumerStaples",
    "KO": "ConsumerStaples", "PEP": "ConsumerStaples", "PM": "ConsumerStaples",
    "MO": "ConsumerStaples", "CL": "ConsumerStaples",
    # Financials
    "JPM": "Financials", "V": "Financials", "MA": "Financials",
    "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
    "MS": "Financials", "AXP": "Financials", "BLK": "Financials",
    "SPGI": "Financials", "C": "Financials", "SCHW": "Financials",
    "BX": "Financials", "FI": "Financials", "ICE": "Financials",
    "CB": "Financials", "PGR": "Financials", "USB": "Financials",
    # Health Care
    "LLY": "Healthcare", "UNH": "Healthcare", "JNJ": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "TMO": "Healthcare",
    "ABT": "Healthcare", "DHR": "Healthcare", "PFE": "Healthcare",
    "AMGN": "Healthcare", "MDT": "Healthcare", "VRTX": "Healthcare",
    "REGN": "Healthcare", "ISRG": "Healthcare", "BMY": "Healthcare",
    "SYK": "Healthcare", "BSX": "Healthcare", "ELV": "Healthcare",
    "CI": "Healthcare", "GILD": "Healthcare", "ZTS": "Healthcare",
    # Energy
    "XOM": "Energy", "CVX": "Energy",
    # Industrials
    "CAT": "Industrials", "RTX": "Industrials", "HON": "Industrials",
    "BA": "Industrials", "DE": "Industrials", "ETN": "Industrials",
    "ADP": "Industrials", "CSX": "Industrials",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    # Materials
    "LIN": "Materials",
    # Real Estate
    "PLD": "RealEstate", "EQIX": "RealEstate",
}


def sector_of(symbol: str) -> str:
    return SECTOR_MAP.get(symbol.upper(), "Other")


def sector_exposure(
    positions: list[dict[str, Any]],
    prices: dict[str, float],
    equity: float,
) -> dict[str, float]:
    """Fraction of equity held in each sector."""
    if equity <= 0:
        return {}
    out: dict[str, float] = {}
    for p in positions:
        sym = p["symbol"]
        val = float(p["qty"]) * float(prices.get(sym, p["avg_price"]))
        sec = sector_of(sym)
        out[sec] = out.get(sec, 0.0) + val / equity
    return out


def portfolio_heat(
    positions: list[dict[str, Any]],
    prices: dict[str, float],
    stop_pcts: dict[str, float],
    equity: float,
    default_stop: float,
) -> float:
    """Total open risk as a fraction of equity.

    Each position can lose (value × its stop distance) before the stop
    fires; summed, that's the book's heat — what it can lose if every
    open trade hits its stop."""
    if equity <= 0:
        return 0.0
    risk = 0.0
    for p in positions:
        sym = p["symbol"]
        val = float(p["qty"]) * float(prices.get(sym, p["avg_price"]))
        risk += val * stop_pcts.get(sym, default_stop)
    return risk / equity


def check_new_position(
    symbol: str,
    add_value: float,
    positions: list[dict[str, Any]],
    prices: dict[str, float],
    stop_pcts: dict[str, float],
    equity: float,
    new_stop_pct: float,
    max_sector_pct: float,
    max_portfolio_heat: float,
    default_stop: float,
) -> tuple[bool, str]:
    """Return (allowed, reason) for opening `add_value` of `symbol`."""
    if equity <= 0:
        return False, "no equity"

    # Sector cap
    exposure = sector_exposure(positions, prices, equity)
    sec = sector_of(symbol)
    new_sec_pct = exposure.get(sec, 0.0) + add_value / equity
    if new_sec_pct > max_sector_pct:
        return False, (f"sector cap: {sec} would be "
                       f"{new_sec_pct*100:.0f}% > {max_sector_pct*100:.0f}%")

    # Portfolio heat cap
    current_heat = portfolio_heat(positions, prices, stop_pcts, equity, default_stop)
    added_heat = (add_value * new_stop_pct) / equity
    if current_heat + added_heat > max_portfolio_heat:
        return False, (f"portfolio heat: {(current_heat+added_heat)*100:.0f}% "
                       f"> {max_portfolio_heat*100:.0f}%")

    return True, "ok"
