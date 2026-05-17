"""LLM prompt templates — compact to minimise context length for local models."""

from __future__ import annotations

from typing import Any

SYSTEM_TRADER = """You are Astra, a paper-money trading bot. Read the signal block
for ONE stock and decide BUY, SELL, or HOLD.

Output a single JSON object, nothing else:
{"action":"BUY|SELL|HOLD","size_pct":0.0,"confidence":0.0,"reasoning":"..."}

REASONING REQUIREMENTS (mandatory):
- 1-3 sentences, never just one word.
- Name the SPECIFIC signals you weighted. Examples:
  GOOD: "RSI 28 + MACD bullish flip + analyst upgrades; mild headwind from
         insider selling but momentum dominates."
  BAD:  "conflicted" / "mixed signals" / "no edge"
- For HOLD, identify exactly which signals conflict (e.g. "bullish RSI vs
  bearish MACD with no momentum to break the tie").

CONFIDENCE REQUIREMENTS:
- Confidence is your conviction in the *decision*, not in the price direction.
  A high-confidence HOLD ("nothing to do here, signals truly cancel out")
  should be 0.6-0.8. A low-confidence HOLD ("I'm unsure either way") is 0.3.
- Never return confidence 0.0 unless data is missing — it tells the user nothing.

DECISION PRINCIPLES:
- BIAS TOWARD ACTION when the CHANGES block shows momentum (RSI flip, MACD
  cross, volume spike, sentiment shift). Fresh changes beat static levels.
- Confidence 0.3-0.5 with size_pct ~0.4 is a valid exploratory position —
  don't HOLD just because you're not 100% sure.
- size_pct = 1.0 only for high-conviction setups (3+ aligned signals plus
  favourable CHANGES).
- NEVER BUY if earnings are within 2 days.
- SELL when holding AND signals turn bearish, momentum reverses, or your
  own profit/loss assessment says exit.
- The {SYMBOL} PROFILE block (if present) is THIS bot's real track record on
  this exact stock — weight it heavily. If a signal has failed on this name
  before, distrust it now. If the stock is a trender, don't fade dips.
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
    mode: str = "entry",
    profile_card: str | None = None,
) -> str:
    parts: list[str] = []
    sym = bundle_dict.get("symbol")
    quote = bundle_dict.get("quote") or {}
    tech = bundle_dict.get("technical") or {}
    last_price = quote.get("c") or tech.get("last_close")

    # Mode header tells the model exactly what choice it's making.
    if mode == "exit":
        parts.append(
            "TASK: You HOLD this position. Decide SELL (close 100%) or HOLD. "
            "A BUY is not possible. Choose SELL if the thesis has weakened, "
            "momentum reversed, or the position looks exhausted."
        )
    else:
        parts.append(
            "TASK: You are FLAT on this stock. Decide BUY (open a new position) "
            "or HOLD (pass). Only BUY on a genuinely attractive setup — you do "
            "not have to trade. A SELL is not possible."
        )

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

    # Market regime — the tide most stocks move with.
    regime = bundle_dict.get("_regime") or {}
    if regime.get("regime") and regime["regime"] != "unknown":
        from astra.signals.regime import regime_summary
        parts.append(regime_summary(regime))

    # Cross-sectional relative strength — is this stock a leader or laggard?
    cs = bundle_dict.get("_cross_section") or {}
    rs_rank = cs.get("rs_rank")
    if isinstance(rs_rank, (int, float)):
        tier = ("top decile" if rs_rank > 0.9 else "leader" if rs_rank > 0.7
                else "laggard" if rs_rank < 0.3 else "middle")
        parts.append(
            f"REL STRENGTH: {tier} (rank {rs_rank*100:.0f}/100 vs universe, "
            f"{cs.get('rel_strength', 0):+.1f}% vs market)"
        )

    # Contextual bandit — the policy model's learned read of this context.
    ps = bundle_dict.get("_policy_score")
    if isinstance(ps, (int, float)):
        verdict = ("favourable" if ps > 0.2 else
                   "unfavourable" if ps < -0.2 else "neutral")
        parts.append(
            f"POLICY MODEL: {verdict} — expected {ps:+.2f}R in this "
            f"signal/regime context (learned across all stocks)"
        )

    # Expert ensemble — regime-weighted vote of the strategy experts.
    ens = bundle_dict.get("_ensemble") or {}
    if ens:
        buyers = [n for n, v in (ens.get("experts") or {}).items()
                  if v.get("action") == "BUY"]
        parts.append(
            f"EXPERT ENSEMBLE: {ens.get('action')} "
            f"(consensus {ens.get('consensus', 0):+.2f}, "
            f"{len(buyers)}/{len(ens.get('experts') or {})} experts buy"
            + (f": {', '.join(buyers)}" if buyers else "") + ")"
        )

    # Tick-over-tick changes — high-signal block for the model
    deltas = bundle_dict.get("_deltas") or {}
    if deltas.get("available"):
        parts.append("CHANGES since last tick:")
        for ch in deltas.get("changes", []):
            parts.append(f"  · {ch}")

    # Per-stock adaptive profile — the bot's own track record on THIS name.
    # Weight it heavily: it reflects what has actually worked here.
    if profile_card:
        parts.append("")
        parts.append(profile_card)

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
