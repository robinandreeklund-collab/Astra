"""Insider transactions."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any


def summarize_insider(data: dict[str, Any] | None, days: int = 60) -> dict[str, Any]:
    if not data or not data.get("data"):
        return {"available": False}
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    rows = [r for r in data["data"] if (r.get("transactionDate") or "") >= cutoff]
    buys = sum(float(r.get("change", 0)) for r in rows if float(r.get("change", 0)) > 0)
    sells = sum(-float(r.get("change", 0)) for r in rows if float(r.get("change", 0)) < 0)
    net_value = sum(
        float(r.get("change", 0)) * float(r.get("transactionPrice", 0) or 0)
        for r in rows
    )
    return {
        "available": True,
        "window_days": days,
        "transactions": len(rows),
        "shares_bought": buys,
        "shares_sold": sells,
        "net_value_usd": net_value,
        "net_direction": "buying" if net_value > 0 else "selling" if net_value < 0 else "neutral",
    }
