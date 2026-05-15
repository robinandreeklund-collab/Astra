"""News headlines + sentiment."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any


def summarize_headlines(news_items: list[dict[str, Any]], max_items: int = 5) -> dict[str, Any]:
    if not news_items:
        return {"count": 0, "headlines": []}
    items = sorted(news_items, key=lambda x: x.get("datetime", 0), reverse=True)[:max_items]
    return {
        "count": len(news_items),
        "headlines": [
            {
                "headline": (it.get("headline") or "")[:200],
                "source": it.get("source"),
                "summary": (it.get("summary") or "")[:300],
            }
            for it in items
        ],
    }


def summarize_sentiment(sent: dict[str, Any] | None) -> dict[str, Any]:
    if not sent:
        return {"available": False}
    s = sent.get("sentiment") or {}
    return {
        "available": True,
        "bullish_pct": s.get("bullishPercent"),
        "bearish_pct": s.get("bearishPercent"),
        "buzz": (sent.get("buzz") or {}).get("buzz"),
        "weekly_avg": (sent.get("buzz") or {}).get("weeklyAverage"),
        "company_news_score": sent.get("companyNewsScore"),
        "sector_avg": sent.get("sectorAverageNewsScore"),
    }


def date_range(days: int = 5) -> tuple[str, str]:
    today = date.today()
    return (today - timedelta(days=days)).isoformat(), today.isoformat()
