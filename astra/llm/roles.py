"""The LLM's expanded roles beyond per-tick deciding.

The local LLM is a reasoning layer over structured memory — good at
synthesis and narrative, not at being a signal source. Two roles here:

  * Playbook author — per stock, distil the profile + recent trade history
    into a concise "how to trade this name" playbook, fed back into the
    decision prompt.
  * Critic — review aggregate performance and propose concrete changes
    (advisory only; a human decides whether to apply them).

Both have deterministic heuristic fallbacks so the system still produces
something useful when no LLM is running.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

SYSTEM_PLAYBOOK = """You write a trading PLAYBOOK for one specific stock,
based only on this bot's own track record and the stock's character.

Output JSON only: {"playbook": ["bullet", "bullet", ...]}

3-6 short, concrete bullets. Cover: the stock's character (trender vs
mean-reverter), which signals have worked vs failed on THIS name, entry
patterns, exit discipline, and any gotcha. Be specific — cite the numbers.
No generic advice ("be careful"). If history is thin, say so plainly.
"""

SYSTEM_CRITIC = """You are the strategy critic. Review the bot's aggregate
performance and propose concrete, actionable changes.

Output JSON only: {"suggestions": ["suggestion", ...]}

2-4 suggestions. Each must name a specific lever (a setting, a signal, a
behaviour) and the direction to change it, with the evidence. No vague
advice. These are advisory — a human decides whether to apply them.
"""


def _trade_narrative(trades: list[dict[str, Any]]) -> str:
    lines = []
    for t in trades[:12]:
        sig = t.get("signal_snapshot") or {}
        tech = sig.get("technical") or {}
        lines.append(
            f"- {t.get('side')} pnl=${(t.get('pnl') or 0):.0f} "
            f"RSI={tech.get('rsi14')} MACD={tech.get('macd_state')} "
            f"reason={(t.get('llm_reasoning') or '')[:80]}"
        )
    return "\n".join(lines) if lines else "(no closed trades yet)"


async def generate_playbook(
    profile: Any,
    recent_trades: list[dict[str, Any]],
    global_rates: dict[str, float],
    client: Any | None,
) -> str:
    """Return a freshly written playbook for one stock."""
    if client is not None:
        try:
            user = (
                profile.card(global_rates)
                + "\n\nRecent trades on this stock:\n"
                + _trade_narrative(recent_trades)
                + "\n\nWrite the playbook. JSON only."
            )
            raw = await client.chat_json(
                SYSTEM_PLAYBOOK, user, temperature=0.3, max_tokens=400)
            bullets = [str(b).strip() for b in (raw.get("playbook") or []) if b]
            if bullets:
                return "\n".join(f"• {b}" for b in bullets[:6])
        except Exception as e:
            log.warning("playbook LLM failed for %s: %s", profile.symbol, e)
    return _heuristic_playbook(profile, global_rates)


def _heuristic_playbook(profile: Any, global_rates: dict[str, float]) -> str:
    """Deterministic playbook when no LLM is available."""
    ch = profile.character or {}
    bullets: list[str] = []
    arche = ch.get("archetype")
    if arche == "trender":
        bullets.append("Trender — ride momentum, don't fade pullbacks.")
    elif arche == "mean-reverter":
        bullets.append("Mean-reverter — fade extremes, distrust breakouts.")
    elif arche == "choppy":
        bullets.append("Choppy — no persistent edge; trade only strong setups.")

    if profile.trades == 0:
        bullets.append("No trade history yet — rely on character + global priors.")
    else:
        bullets.append(
            f"Record: {profile.trades} trades, "
            f"{(profile.win_rate() or 0)*100:.0f}% win, "
            f"{profile.expectancy_r():+.2f}R/trade.")
        rated = []
        for tag in profile.signal_efficacy:
            n = profile.signal_sample_size(tag)
            if n >= 2:
                gr = global_rates.get(tag, 0.5)
                rated.append((tag, profile.signal_win_rate(tag, gr), n))
        rated.sort(key=lambda x: x[1], reverse=True)
        if rated:
            best = rated[0]
            bullets.append(f"Best signal here: {best[0]} ({best[1]*100:.0f}% win).")
            worst = rated[-1]
            if worst[0] != best[0] and worst[1] < 0.45:
                bullets.append(f"Avoid: {worst[0]} ({worst[1]*100:.0f}% win).")
    state = profile.state()
    if state == "PROVEN":
        bullets.append("State PROVEN — size up with conviction.")
    elif state in ("PROBATION", "BENCHED"):
        bullets.append(f"State {state} — minimal size or skip until it recovers.")
    return "\n".join(f"• {b}" for b in bullets)


async def generate_critique(
    report: dict[str, Any],
    profile_summaries: list[dict[str, Any]],
    client: Any | None,
) -> list[str]:
    """Return advisory suggestions reviewing aggregate performance."""
    if client is not None:
        try:
            worst = sorted(profile_summaries, key=lambda p: p.get("edge", 0))[:5]
            best = sorted(profile_summaries, key=lambda p: -p.get("edge", 0))[:5]
            user = (
                f"Aggregate: return {report.get('return_pct', 0):.1f}%, "
                f"Sharpe {report.get('sharpe', 0):.2f}, "
                f"max DD {report.get('max_drawdown_pct', 0):.1f}%, "
                f"win rate {report.get('win_rate', 0)*100:.0f}%, "
                f"{report.get('closed_trades', 0)} closed trades.\n"
                f"Best stocks: {[p['symbol'] for p in best]}\n"
                f"Worst stocks: {[p['symbol'] for p in worst]}\n\n"
                "Propose changes. JSON only."
            )
            raw = await client.chat_json(
                SYSTEM_CRITIC, user, temperature=0.4, max_tokens=400)
            sugg = [str(s).strip() for s in (raw.get("suggestions") or []) if s]
            if sugg:
                return sugg[:4]
        except Exception as e:
            log.warning("critic LLM failed: %s", e)
    return _heuristic_critique(report)


def _heuristic_critique(report: dict[str, Any]) -> list[str]:
    out: list[str] = []
    wr = report.get("win_rate", 0)
    pf = report.get("profit_factor", 0)
    dd = report.get("max_drawdown_pct", 0)
    sharpe = report.get("sharpe", 0)
    if dd > 20:
        out.append(f"Max drawdown {dd:.0f}% is high — tighten stops or lower "
                   f"max_portfolio_heat.")
    if wr < 0.4 and report.get("closed_trades", 0) > 10:
        out.append(f"Win rate {wr*100:.0f}% is low — raise entry_min_confidence "
                   f"so only stronger setups trade.")
    if pf and pf < 1.0:
        out.append(f"Profit factor {pf:.2f} below 1.0 — losers outweigh winners; "
                   f"widen take-profit or cut losers faster.")
    if sharpe < 0.5:
        out.append(f"Sharpe {sharpe:.2f} is weak — risk-adjusted return is poor; "
                   f"favour proven stocks (raise the bandit's influence).")
    if not out:
        out.append("No glaring issues — metrics are within healthy ranges.")
    return out
