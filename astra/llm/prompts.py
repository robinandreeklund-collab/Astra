"""LLM prompt templates — compact to minimise context length for local models."""

from __future__ import annotations

from typing import Any

SYSTEM_TRADER = """You are Astra, a paper-money trading bot. Read the signal block
for ONE stock and decide BUY, SELL, or HOLD.

Output a single JSON object, nothing else:
{"action":"BUY|SELL|HOLD","size_pct":0.0,"confidence":0.0,"reasoning":"short"}

Decision principles:
- BIAS TOWARD ACTION when the CHANGES block shows momentum (RSI flip, MACD cross,
  volume spike, sentiment shift). Fresh changes are more actionable than static
  absolute values.
- A confidence of 0.3-0.5 with size_pct ~0.4 is a valid "exploratory" position —
  don't HOLD just because you're not 100% sure.
- size_pct = 1.0 is reserved for very high-conviction setups (3+ strong signals
  aligned + favourable CHANGES).
- HOLD only when signals are truly conflicted or there is no edge.
- NEVER BUY if earnings are within 2 days.
- SELL when holding AND signals turn bearish, momentum reverses, or stop/profit
  targets are hit by your own assessment.
"""

SYSTEM_REFLECTOR = """Read recent closed trades and write up to 3 short, concrete
lessons that would have improved decisions. Specific patterns only, no vague advice.

Output JSON only: {"lessons":["...","..."]}
"""


def _fmt_num(v: Any, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def build_decision_user_prompt(
    bundle_dict: dict[str, Any],
    position: dict[str, Any] | None,
    cash: float,
    lessons: list[str],
    pattern_stats: list[dict[str, Any]],
) -> str:
    parts: list[str] = []
    sym = bundle_dict.get("symbol")
    quote = bundle_dict.get("quote") or {}
    tech = bundle_dict.get("technical") or {}
    last_price = quote.get("c") or tech.get("last_close")

    parts.append(f"SYMBOL: {sym}   PRICE: ${_fmt_num(last_price)}   CASH: ${cash:,.0f}")
    if position:
        gain = ((last_price or 0) - position["avg_price"]) / position["avg_price"] if position.get("avg_price") else 0
        parts.append(
            f"HOLDING: {position['qty']:.2f} sh @ ${position['avg_price']:.2f} "
            f"(P&L {gain*100:+.1f}%)"
        )
    else:
        parts.append("HOLDING: none")

    # Technical — flat lines, not JSON
    if tech.get("available"):
        parts.append(
            f"TECH: RSI={_fmt_num(tech.get('rsi14'),0)} MACD={tech.get('macd_state') or 'na'} "
            f"BB={_fmt_num(tech.get('bb_position'),2)} SMA={tech.get('sma_cross') or 'na'} "
            f"1d={_fmt_num(tech.get('pct_change_1d'),1)}% 5d={_fmt_num(tech.get('pct_change_5d'),1)}% "
            f"volR={_fmt_num(tech.get('volume_ratio'),2)}"
        )
    else:
        parts.append("TECH: unavailable")

    rec = bundle_dict.get("recommendations") or {}
    if rec.get("available"):
        parts.append(
            f"ANALYSTS: bull={rec.get('bull_pct',0)*100:.0f}% bear={rec.get('bear_pct',0)*100:.0f}% "
            f"trend={rec.get('trend') or 'na'}"
        )

    ins = bundle_dict.get("insider") or {}
    if ins.get("available"):
        parts.append(
            f"INSIDER {ins.get('window_days',60)}d: net={ins.get('net_direction','na')} "
            f"value=${ins.get('net_value_usd',0):,.0f}"
        )

    earn = bundle_dict.get("earnings") or {}
    if earn.get("available"):
        parts.append(
            f"LAST EARNINGS: {'BEAT' if earn.get('beat') else 'miss'} "
            f"surprise={_fmt_num(earn.get('surprise_pct'),1)}%"
        )

    cal = bundle_dict.get("earnings_calendar") or {}
    if cal.get("upcoming") and cal.get("date"):
        parts.append(f"NEXT EARNINGS: {cal.get('date')}")

    sent = bundle_dict.get("sentiment") or {}
    if sent.get("available"):
        parts.append(
            f"NEWS SENT: score={_fmt_num(sent.get('company_news_score'),2)} "
            f"sector={_fmt_num(sent.get('sector_avg'),2)} buzz={_fmt_num(sent.get('buzz'),2)}"
        )

    news = bundle_dict.get("news") or {}
    if news.get("count"):
        heads = news.get("headlines") or []
        # Only top 3, short
        for h in heads[:3]:
            hl = (h.get("headline") or "").strip()
            if hl:
                parts.append(f"NEWS: {hl[:120]}")

    # Tick-over-tick changes — high-signal block for the model
    deltas = bundle_dict.get("_deltas") or {}
    if deltas.get("available"):
        parts.append("CHANGES since last tick:")
        for ch in deltas.get("changes", []):
            parts.append(f"  · {ch}")

    if lessons:
        parts.append("LESSONS:")
        for l in lessons[:6]:
            parts.append(f"- {l[:160]}")

    if pattern_stats:
        useful = [p for p in pattern_stats if (p["wins"] + p["losses"]) >= 2]
        if useful:
            parts.append("PATTERN HISTORY:")
            for p in useful[:5]:
                n = p["wins"] + p["losses"]
                parts.append(
                    f"- {p['pattern_desc'][:60]}: "
                    f"{p['wins']}W/{p['losses']}L "
                    f"({100*p['wins']/n:.0f}% wr, ${p['total_pnl']:.0f})"
                )

    parts.append("\nReturn JSON only.")
    return "\n".join(parts)


def build_reflection_user_prompt(closed_trades: list[dict[str, Any]]) -> str:
    parts: list[str] = ["Recent closed trades:"]
    for t in closed_trades:
        sig = t.get("signal_snapshot") or {}
        tech = (sig.get("technical") or {})
        parts.append(
            f"- {t['symbol']} {t['side']} qty={t['qty']:.2f} pnl=${(t.get('pnl') or 0):.2f} "
            f"RSI={tech.get('rsi14')} MACD={tech.get('macd_state')} "
            f"BB={tech.get('bb_position')}"
        )
    parts.append("")
    parts.append("Up to 3 concrete lessons. JSON only.")
    return "\n".join(parts)
