"""Expert ensemble — regime-aware multiplicative-weights meta-learner.

Each expert (astra/engine/experts.py) votes; this combines the votes,
weighted by how well each expert has performed *in the current regime*.

Learning is the Hedge / multiplicative-weights algorithm: after a trade
closes, each expert that voted BUY on the entry has its weight scaled by
exp(eta · reward), where reward is the trade's R-multiple. Good experts
in a regime accumulate weight, bad ones decay toward irrelevance — this
*is* the champion-challenger mechanism: a new expert starts even and
either earns its place or fades.
"""

from __future__ import annotations

import math
from typing import Any

from astra.engine.experts import EXPERT_NAMES
from astra.signals.regime import REGIMES

_ETA = 0.15          # learning rate
_MIN_WEIGHT = 0.05   # floor so a beaten expert can still recover


class ExpertEnsemble:
    def __init__(self) -> None:
        # weights[regime][expert] — start uniform.
        self.weights: dict[str, dict[str, float]] = {
            reg: {name: 1.0 for name in EXPERT_NAMES} for reg in REGIMES
        }
        self.updates = 0

    def _regime_weights(self, regime: str) -> dict[str, float]:
        return self.weights.setdefault(
            regime, {name: 1.0 for name in EXPERT_NAMES})

    def combine(
        self, votes: dict[str, Any], regime: str,
    ) -> dict[str, Any]:
        """Return {consensus, action, confidence} for the weighted vote.

        consensus is −1..+1: +1 = all weight behind BUY, −1 = all behind HOLD.
        """
        w = self._regime_weights(regime)
        total = sum(w.get(n, 1.0) for n in votes) or 1.0
        buy_weight = 0.0
        conf_acc = 0.0
        for name, vote in votes.items():
            wi = w.get(name, 1.0)
            if vote.action == "BUY":
                buy_weight += wi
                conf_acc += wi * vote.confidence
        buy_frac = buy_weight / total
        consensus = round(2.0 * buy_frac - 1.0, 3)
        confidence = round((conf_acc / buy_weight) if buy_weight > 0 else 0.0, 3)
        action = "BUY" if buy_frac > 0.5 else "HOLD"
        return {"consensus": consensus, "action": action,
                "confidence": confidence, "buy_fraction": round(buy_frac, 3)}

    def update(
        self, expert_votes: dict[str, str], regime: str, reward_r: float,
    ) -> None:
        """Hedge update: reward experts that voted BUY by the realized R."""
        w = self._regime_weights(regime)
        r = max(-3.0, min(3.0, float(reward_r)))
        for name in EXPERT_NAMES:
            voted_buy = expert_votes.get(name) == "BUY"
            if not voted_buy:
                continue  # an abstaining expert is neither rewarded nor blamed
            w[name] = max(_MIN_WEIGHT, w[name] * math.exp(_ETA * r))
        # Renormalise so weights stay comparable across regimes.
        total = sum(w.values())
        if total > 0:
            for name in w:
                w[name] = w[name] / total * len(w)
        self.updates += 1

    def regime_table(self) -> dict[str, dict[str, float]]:
        """Rounded weights for display."""
        return {
            reg: {n: round(v, 3) for n, v in w.items()}
            for reg, w in self.weights.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {"weights": self.weights, "updates": self.updates}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpertEnsemble":
        e = cls()
        w = data.get("weights")
        if isinstance(w, dict):
            for reg, ew in w.items():
                if isinstance(ew, dict):
                    e.weights[reg] = {n: float(ew.get(n, 1.0)) for n in EXPERT_NAMES}
        e.updates = int(data.get("updates", 0))
        return e


_MODEL_KEY = "expert_ensemble_v1"


async def load_ensemble(memory: Any) -> ExpertEnsemble:
    data = await memory.get_model(_MODEL_KEY)
    return ExpertEnsemble.from_dict(data) if data else ExpertEnsemble()


async def save_ensemble(memory: Any, ensemble: ExpertEnsemble) -> None:
    await memory.save_model(_MODEL_KEY, ensemble.to_dict())
