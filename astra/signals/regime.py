"""Market regime detection.

The bot trades stock-by-stock but most stocks move with the market tide.
Knowing whether the whole universe is risk-on or risk-off lets the engine
press harder when conditions are favourable and pull back when they're not.

Regime is read straight off the universe scan — the scanner already
computes indicators for every symbol, so breadth and momentum dispersion
are nearly free.
"""

from __future__ import annotations

from typing import Any

REGIMES = ("risk-on", "neutral", "risk-off", "high-vol", "unknown")


def classify_regime(scored_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify the market regime from a full set of scored scan results."""
    inds = [r["indicators"] for r in scored_results if r.get("indicators")]
    if len(inds) < 5:
        return {"regime": "unknown", "sample": len(inds)}

    bullish = sum(1 for i in inds if i.get("macd_state") == "bullish")
    breadth = bullish / len(inds)

    moms = [float(i["pct_change_5d"]) for i in inds
            if isinstance(i.get("pct_change_5d"), (int, float))]
    avg_mom = sum(moms) / len(moms) if moms else 0.0
    up_fraction = (sum(1 for m in moms if m > 0) / len(moms)) if moms else 0.0

    dispersion = 0.0
    if len(moms) > 1:
        var = sum((m - avg_mom) ** 2 for m in moms) / len(moms)
        dispersion = var ** 0.5

    if dispersion > 6.0:
        regime = "high-vol"
    elif breadth > 0.58 and avg_mom > 0.3:
        regime = "risk-on"
    elif breadth < 0.38 or avg_mom < -1.5:
        regime = "risk-off"
    else:
        regime = "neutral"

    return {
        "regime": regime,
        "breadth": round(breadth, 3),
        "avg_momentum": round(avg_mom, 3),
        "up_fraction": round(up_fraction, 3),
        "dispersion": round(dispersion, 3),
        "sample": len(inds),
    }


def regime_risk_multiplier(regime: str) -> float:
    """How hard to press, given the regime. Scales conviction / size."""
    return {
        "risk-on": 1.15,
        "neutral": 0.90,
        "risk-off": 0.45,
        "high-vol": 0.60,
        "unknown": 0.85,
    }.get(regime, 0.85)


def regime_summary(regime: dict[str, Any]) -> str:
    """One-line description for the LLM prompt."""
    name = regime.get("regime", "unknown")
    if name == "unknown":
        return "MARKET REGIME: unknown (not enough data)"
    return (
        f"MARKET REGIME: {name} "
        f"(breadth {regime.get('breadth', 0)*100:.0f}% bullish, "
        f"avg 5d momentum {regime.get('avg_momentum', 0):+.1f}%)"
    )
