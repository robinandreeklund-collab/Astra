"""LLM prompt templates."""

from __future__ import annotations

import json
from typing import Any

SYSTEM_TRADER = """You are Astra, a disciplined paper-money trading assistant.
You receive a snapshot of signals for one US large-cap stock plus past learned
patterns and lessons. You must decide whether to BUY, SELL, or HOLD.

RULES:
- Output ONLY a single valid JSON object. No prose outside JSON.
- Never go all-in: size_pct is fraction of MAX position size (0.0–1.0).
- HOLD if confidence < 0.4 or signals conflict strongly.
- Refuse to BUY if upcoming earnings are within 2 days (mark HOLD with reason).
- SELL when own position exists AND signals turn bearish OR strong gain captured.
- Use lessons from past trades: avoid patterns with poor win-rate.

OUTPUT SCHEMA:
{
  "action": "BUY" | "SELL" | "HOLD",
  "size_pct": 0.0,
  "confidence": 0.0,
  "reasoning": "1-3 short sentences"
}
"""

SYSTEM_REFLECTOR = """You are Astra's reflection module. You read recent closed
trades (winners and losers) and produce up to 3 short, generalizable lessons that
would have helped. Lessons must be specific patterns, NOT vague advice.

Output ONLY JSON of this shape:
{ "lessons": [ "lesson string", ... ] }

BAD lesson: "Be more careful"
GOOD lesson: "RSI<25 + earnings in <3d: bounces fail 70% of time, avoid BUY"
"""


def build_decision_user_prompt(
    bundle_dict: dict[str, Any],
    position: dict[str, Any] | None,
    cash: float,
    lessons: list[str],
    pattern_stats: list[dict[str, Any]],
) -> str:
    parts: list[str] = []
    parts.append(f"Symbol: {bundle_dict.get('symbol')}")
    parts.append(f"Cash available: ${cash:,.2f}")
    if position:
        parts.append(
            f"Current position: {position['qty']:.4f} shares @ ${position['avg_price']:.2f}"
        )
    else:
        parts.append("Current position: none")
    parts.append("")
    parts.append("SIGNALS:")
    parts.append(json.dumps({
        "quote": bundle_dict.get("quote"),
        "technical": bundle_dict.get("technical"),
        "news": bundle_dict.get("news"),
        "sentiment": bundle_dict.get("sentiment"),
        "insider": bundle_dict.get("insider"),
        "recommendations": bundle_dict.get("recommendations"),
        "earnings": bundle_dict.get("earnings"),
        "earnings_calendar": bundle_dict.get("earnings_calendar"),
        "social": bundle_dict.get("social"),
    }, indent=2, default=str))
    if lessons:
        parts.append("")
        parts.append("PAST LESSONS (highest scoring first):")
        for l in lessons:
            parts.append(f"- {l}")
    if pattern_stats:
        parts.append("")
        parts.append("PATTERN STATS (relevant):")
        for p in pattern_stats:
            n = p["wins"] + p["losses"]
            if n == 0:
                continue
            parts.append(
                f"- {p['pattern_desc']}: {p['wins']}W/{p['losses']}L "
                f"({100*p['wins']/n:.0f}% wr, ${p['total_pnl']:.0f} pnl, conf {p['confidence']:.2f})"
            )
    parts.append("")
    parts.append("Decide. JSON only.")
    return "\n".join(parts)


def build_reflection_user_prompt(closed_trades: list[dict[str, Any]]) -> str:
    parts: list[str] = ["Recent closed trades:"]
    for t in closed_trades:
        sig = t.get("signal_snapshot") or {}
        tech = (sig.get("technical") or {})
        parts.append(
            f"- {t['symbol']} {t['side']} qty={t['qty']:.2f} pnl=${(t.get('pnl') or 0):.2f} "
            f"RSI={tech.get('rsi14')} MACD={tech.get('macd_state')} "
            f"BB={tech.get('bb_position')} reasoning={t.get('llm_reasoning') or ''}"
        )
    parts.append("")
    parts.append("Produce up to 3 lessons. JSON only.")
    return "\n".join(parts)
