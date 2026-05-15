"""Decision engine — uses LM Studio when available, heuristic fallback otherwise."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from astra.llm.client import DECISION_SCHEMA, LMStudioClient
from astra.llm.prompts import (
    SYSTEM_TRADER,
    build_decision_user_prompt,
)

log = logging.getLogger(__name__)


@dataclass
class Decision:
    action: str  # BUY | SELL | HOLD
    size_pct: float
    confidence: float
    reasoning: str
    source: str  # "llm" | "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "size_pct": self.size_pct,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "source": self.source,
        }


def clamp01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if v != v or v == float("inf") or v == float("-inf"):
        return default
    return max(0.0, min(1.0, v))


class DecisionEngine:
    """LLM-backed decision engine; falls back to heuristic on error."""

    def __init__(self, client: LMStudioClient | None = None) -> None:
        self.client = client
        self.fallback = HeuristicDecisionEngine()

    async def decide(
        self,
        bundle_dict: dict[str, Any],
        position: dict[str, Any] | None,
        cash: float,
        lessons: list[str],
        pattern_stats: list[dict[str, Any]],
    ) -> Decision:
        if self.client is None:
            return await self.fallback.decide(bundle_dict, position, cash, lessons, pattern_stats)
        try:
            user = build_decision_user_prompt(bundle_dict, position, cash, lessons, pattern_stats)
            raw = await self.client.chat_json(
                SYSTEM_TRADER, user,
                temperature=0.2, max_tokens=400,
                schema=DECISION_SCHEMA,
            )
            action = str(raw.get("action", "HOLD")).upper()
            if action not in {"BUY", "SELL", "HOLD"}:
                action = "HOLD"
            return Decision(
                action=action,
                size_pct=clamp01(raw.get("size_pct"), 0.0),
                confidence=clamp01(raw.get("confidence"), 0.0),
                reasoning=str(raw.get("reasoning") or "")[:500],
                source="llm",
            )
        except Exception as e:
            log.warning("LLM decision failed (%s); falling back to heuristic", e)
            d = await self.fallback.decide(bundle_dict, position, cash, lessons, pattern_stats)
            d.reasoning = f"[llm-fallback: {e.__class__.__name__}] {d.reasoning}"
            return d


class HeuristicDecisionEngine:
    """Rule-based fallback so the system is usable without an LLM running."""

    async def decide(
        self,
        bundle_dict: dict[str, Any],
        position: dict[str, Any] | None,
        cash: float,
        lessons: list[str],
        pattern_stats: list[dict[str, Any]],
    ) -> Decision:
        tech = bundle_dict.get("technical") or {}
        if not tech.get("available"):
            return Decision("HOLD", 0.0, 0.0, "no technical data", "heuristic")

        rsi_v = tech.get("rsi14")
        macd = tech.get("macd_state")
        bb_p = tech.get("bb_position")
        sma = tech.get("sma_cross")
        cal = bundle_dict.get("earnings_calendar") or {}

        sent = bundle_dict.get("sentiment") or {}
        sent_score = sent.get("company_news_score")

        ins = (bundle_dict.get("insider") or {}).get("net_direction")
        rec_trend = (bundle_dict.get("recommendations") or {}).get("trend")

        bull = 0
        bear = 0
        reasons: list[str] = []

        if rsi_v is not None:
            if rsi_v < 30:
                bull += 1; reasons.append(f"RSI {rsi_v:.0f} oversold")
            elif rsi_v > 70:
                bear += 1; reasons.append(f"RSI {rsi_v:.0f} overbought")
        if macd == "bullish":
            bull += 1; reasons.append("MACD bullish")
        elif macd == "bearish":
            bear += 1; reasons.append("MACD bearish")
        if bb_p is not None:
            if bb_p < 0.2:
                bull += 1; reasons.append("near lower BB")
            elif bb_p > 0.8:
                bear += 1; reasons.append("near upper BB")
        if sma == "golden":
            bull += 1; reasons.append("golden cross")
        elif sma == "death":
            bear += 1; reasons.append("death cross")
        if ins == "buying":
            bull += 1; reasons.append("insider buying")
        elif ins == "selling":
            bear += 1; reasons.append("insider selling")
        if rec_trend == "upgrading":
            bull += 1; reasons.append("analyst upgrades")
        elif rec_trend == "downgrading":
            bear += 1; reasons.append("analyst downgrades")
        if isinstance(sent_score, (int, float)):
            if sent_score > 0.6:
                bull += 1; reasons.append("positive news sentiment")
            elif sent_score < 0.4:
                bear += 1; reasons.append("negative news sentiment")

        # Earnings within 2d → never BUY
        block_buy = False
        if cal.get("upcoming") and cal.get("date"):
            # Allow rough day-based comparison; we only care about <=2 days
            from datetime import date
            try:
                d = date.fromisoformat(cal["date"])
                if (d - date.today()).days <= 2:
                    block_buy = True
                    reasons.append(f"earnings on {cal['date']} (no BUY)")
            except Exception:
                pass

        signal_strength = abs(bull - bear)
        total = bull + bear

        # Sell logic if we hold the position
        if position is not None:
            last = tech.get("last_close") or 0
            avg = position.get("avg_price") or 0
            gain = (last - avg) / avg if avg > 0 else 0
            if gain > 0.10 and bear >= 1:
                return Decision("SELL", 1.0, min(0.9, 0.5 + signal_strength * 0.1),
                                "Take profit + bearish signal(s): " + ", ".join(reasons),
                                "heuristic")
            if bear >= bull + 2 and total >= 3:
                return Decision("SELL", 1.0, 0.6,
                                "Bearish dominance: " + ", ".join(reasons),
                                "heuristic")
            if gain < -0.08:
                return Decision("SELL", 1.0, 0.7,
                                f"Stop-loss at {gain:.1%}", "heuristic")

        if block_buy:
            return Decision("HOLD", 0.0, 0.3,
                            "Earnings imminent; " + ", ".join(reasons),
                            "heuristic")
        if bull >= bear + 2 and total >= 3 and position is None:
            conf = min(0.85, 0.4 + signal_strength * 0.1)
            return Decision("BUY", min(1.0, 0.5 + signal_strength * 0.1), conf,
                            "Bullish signals: " + ", ".join(reasons),
                            "heuristic")

        return Decision("HOLD", 0.0, 0.3 + 0.05 * signal_strength,
                        "Mixed/weak: " + ", ".join(reasons or ["no clear edge"]),
                        "heuristic")
