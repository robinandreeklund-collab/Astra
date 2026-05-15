"""Tick-over-tick signal change detection.

Given the current SignalBundle and the prior snapshot (if any), produce a
compact list of human-readable changes plus a few numeric deltas the LLM
prompt can consume. The aim is to give the model fresh information beyond
the absolute signal levels — e.g. "RSI flipped 65→42 in one tick" is a
much stronger trade trigger than "RSI=42".
"""

from __future__ import annotations

from typing import Any


def _num(d: dict[str, Any] | None, key: str) -> float | None:
    if not d:
        return None
    v = d.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def compute_deltas(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a dict describing what changed since `previous`."""
    if not previous:
        return {"available": False, "reason": "no_prior_snapshot"}

    cur_tech = (current.get("technical") or {})
    prev_tech = (previous.get("technical") or {})

    out: dict[str, Any] = {"available": True, "changes": []}

    # RSI movement
    rsi_now = _num(cur_tech, "rsi14")
    rsi_was = _num(prev_tech, "rsi14")
    if rsi_now is not None and rsi_was is not None:
        diff = rsi_now - rsi_was
        out["rsi_delta"] = diff
        if abs(diff) >= 5:
            direction = "rose" if diff > 0 else "fell"
            out["changes"].append(f"RSI {direction} {rsi_was:.0f}→{rsi_now:.0f}")
        # Crossing key thresholds is more useful than absolute value
        if rsi_was >= 30 and rsi_now < 30:
            out["changes"].append("RSI crossed BELOW 30 (oversold)")
        elif rsi_was < 30 and rsi_now >= 30:
            out["changes"].append("RSI crossed ABOVE 30 (out of oversold)")
        if rsi_was <= 70 and rsi_now > 70:
            out["changes"].append("RSI crossed ABOVE 70 (overbought)")
        elif rsi_was > 70 and rsi_now <= 70:
            out["changes"].append("RSI crossed BELOW 70 (out of overbought)")

    # MACD state flip
    macd_now = cur_tech.get("macd_state")
    macd_was = prev_tech.get("macd_state")
    if macd_now and macd_was and macd_now != macd_was:
        out["changes"].append(f"MACD flipped {macd_was} → {macd_now}")
        out["macd_flip"] = f"{macd_was}->{macd_now}"

    # Bollinger position movement
    bb_now = _num(cur_tech, "bb_position")
    bb_was = _num(prev_tech, "bb_position")
    if bb_now is not None and bb_was is not None:
        diff = bb_now - bb_was
        out["bb_delta"] = diff
        if abs(diff) >= 0.2:
            out["changes"].append(f"BB position moved {bb_was:.2f}→{bb_now:.2f}")

    # Price momentum vs prior tick
    price_now = _num(cur_tech, "last_close")
    price_was = _num(prev_tech, "last_close")
    if price_now and price_was and price_was > 0:
        pct = (price_now - price_was) / price_was * 100
        out["price_pct_change"] = pct
        if abs(pct) >= 1.5:
            sign = "+" if pct > 0 else ""
            out["changes"].append(f"Price moved {sign}{pct:.1f}% since last tick")

    # News sentiment shift
    sent_now = _num(current.get("sentiment"), "company_news_score")
    sent_was = _num(previous.get("sentiment"), "company_news_score")
    if sent_now is not None and sent_was is not None:
        diff = sent_now - sent_was
        out["sentiment_delta"] = diff
        if abs(diff) >= 0.1:
            direction = "improved" if diff > 0 else "worsened"
            out["changes"].append(
                f"News sentiment {direction} {sent_was:.2f}→{sent_now:.2f}"
            )

    # Analyst trend flip
    rec_now = (current.get("recommendations") or {}).get("trend")
    rec_was = (previous.get("recommendations") or {}).get("trend")
    if rec_now and rec_was and rec_now != rec_was:
        out["changes"].append(f"Analyst trend: {rec_was} → {rec_now}")

    # Insider direction flip
    ins_now = (current.get("insider") or {}).get("net_direction")
    ins_was = (previous.get("insider") or {}).get("net_direction")
    if ins_now and ins_was and ins_now != ins_was and ins_now != "neutral":
        out["changes"].append(f"Insider activity: {ins_was} → {ins_now}")

    # Volume spike
    vol_now = _num(cur_tech, "volume_ratio")
    if vol_now and vol_now >= 1.5:
        out["changes"].append(f"Volume {vol_now:.1f}× the 20-day average")

    if not out["changes"]:
        out["changes"] = ["no notable changes since last tick"]
    return out
