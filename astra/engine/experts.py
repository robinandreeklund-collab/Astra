"""Trading experts — distinct entry strategies that vote.

No single strategy works in every regime. Each expert is a small, focused
rule set with a clear thesis. They vote independently; the ensemble
(astra/engine/ensemble.py) weights the votes by each expert's learned
track record in the current regime.

Each expert votes BUY or HOLD for an entry, with a confidence 0..1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExpertVote:
    action: str          # "BUY" or "HOLD"
    confidence: float     # 0..1

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "confidence": round(self.confidence, 3)}


def _num(d: dict[str, Any], k: str) -> float | None:
    v = d.get(k)
    return float(v) if isinstance(v, (int, float)) else None


def momentum_expert(bundle: dict[str, Any]) -> ExpertVote:
    """Thesis: strong, confirmed momentum keeps running."""
    tech = bundle.get("technical") or {}
    cs = bundle.get("_cross_section") or {}
    mom = _num(tech, "pct_change_5d") or 0.0
    rs = cs.get("rs_rank", 0.5)
    macd = tech.get("macd_state")
    score = 0.0
    if mom > 2.0:
        score += 0.4
    if macd == "bullish":
        score += 0.3
    if isinstance(rs, (int, float)) and rs > 0.6:
        score += 0.3
    return ExpertVote("BUY", min(1.0, score)) if score >= 0.6 else ExpertVote("HOLD", 0.0)


def mean_reversion_expert(bundle: dict[str, Any]) -> ExpertVote:
    """Thesis: oversold extremes in a non-collapsing stock bounce."""
    tech = bundle.get("technical") or {}
    rsi = _num(tech, "rsi14")
    bb = _num(tech, "bb_position")
    sma = tech.get("sma_cross")
    score = 0.0
    if rsi is not None and rsi < 35:
        score += 0.45
    if bb is not None and bb < 0.25:
        score += 0.35
    if sma != "death":
        score += 0.20
    return ExpertVote("BUY", min(1.0, score)) if score >= 0.6 else ExpertVote("HOLD", 0.0)


def trend_expert(bundle: dict[str, Any]) -> ExpertVote:
    """Thesis: an established uptrend (golden cross + MACD) persists."""
    tech = bundle.get("technical") or {}
    sma = tech.get("sma_cross")
    macd = tech.get("macd_state")
    mom = _num(tech, "pct_change_5d") or 0.0
    score = 0.0
    if sma == "golden":
        score += 0.45
    if macd == "bullish":
        score += 0.35
    if mom > 0:
        score += 0.20
    return ExpertVote("BUY", min(1.0, score)) if score >= 0.6 else ExpertVote("HOLD", 0.0)


def breakout_expert(bundle: dict[str, Any]) -> ExpertVote:
    """Thesis: a volume-backed push through the upper band runs further."""
    tech = bundle.get("technical") or {}
    vr = _num(tech, "volume_ratio") or 0.0
    mom = _num(tech, "pct_change_5d") or 0.0
    bb = _num(tech, "bb_position")
    score = 0.0
    if vr >= 1.5:
        score += 0.4
    if mom > 3.0:
        score += 0.35
    if bb is not None and bb > 0.6:
        score += 0.25
    return ExpertVote("BUY", min(1.0, score)) if score >= 0.6 else ExpertVote("HOLD", 0.0)


EXPERTS = {
    "momentum": momentum_expert,
    "mean_reversion": mean_reversion_expert,
    "trend": trend_expert,
    "breakout": breakout_expert,
}
EXPERT_NAMES = tuple(EXPERTS.keys())


def run_experts(bundle: dict[str, Any]) -> dict[str, ExpertVote]:
    """Collect every expert's vote for an entry context."""
    return {name: fn(bundle) for name, fn in EXPERTS.items()}
