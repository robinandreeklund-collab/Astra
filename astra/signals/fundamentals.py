"""Fundamentals + earnings calendar + analyst recommendations."""

from __future__ import annotations

from typing import Any


def summarize_recommendations(recs: list[dict[str, Any]] | None) -> dict[str, Any]:
    if not recs:
        return {"available": False}
    latest = recs[0]
    total = sum(int(latest.get(k, 0) or 0) for k in ("strongBuy", "buy", "hold", "sell", "strongSell"))
    if total == 0:
        return {"available": False}
    bull = int(latest.get("strongBuy", 0) or 0) + int(latest.get("buy", 0) or 0)
    bear = int(latest.get("sell", 0) or 0) + int(latest.get("strongSell", 0) or 0)
    trend = None
    if len(recs) >= 2:
        prev = recs[1]
        prev_bull = int(prev.get("strongBuy", 0) or 0) + int(prev.get("buy", 0) or 0)
        if bull > prev_bull:
            trend = "upgrading"
        elif bull < prev_bull:
            trend = "downgrading"
        else:
            trend = "stable"
    return {
        "available": True,
        "period": latest.get("period"),
        "strong_buy": int(latest.get("strongBuy", 0) or 0),
        "buy": int(latest.get("buy", 0) or 0),
        "hold": int(latest.get("hold", 0) or 0),
        "sell": int(latest.get("sell", 0) or 0),
        "strong_sell": int(latest.get("strongSell", 0) or 0),
        "bull_pct": bull / total,
        "bear_pct": bear / total,
        "trend": trend,
    }


def summarize_earnings(earnings: list[dict[str, Any]] | None) -> dict[str, Any]:
    if not earnings:
        return {"available": False}
    latest = earnings[0]
    actual = latest.get("actual")
    estimate = latest.get("estimate")
    surprise = latest.get("surprise")
    surprise_pct = latest.get("surprisePercent")
    return {
        "available": True,
        "period": latest.get("period"),
        "actual": actual,
        "estimate": estimate,
        "surprise": surprise,
        "surprise_pct": surprise_pct,
        "beat": (actual is not None and estimate is not None and actual > estimate),
    }


def summarize_calendar(calendar: dict[str, Any] | None, symbol: str) -> dict[str, Any]:
    if not calendar:
        return {"upcoming": False}
    items = (calendar.get("earningsCalendar") or [])
    rel = [i for i in items if i.get("symbol") == symbol]
    if not rel:
        return {"upcoming": False}
    nxt = sorted(rel, key=lambda x: x.get("date", ""))[0]
    return {
        "upcoming": True,
        "date": nxt.get("date"),
        "hour": nxt.get("hour"),
        "estimate": nxt.get("epsEstimate"),
    }
