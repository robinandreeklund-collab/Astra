"""Contextual bandit — policy learning across the whole universe.

The per-stock profiles *specialise* (what works on NVDA). This is the
*generalisation* layer: one model that learns, from every trade across
every stock, a policy — "given THIS context (signals + regime + relative
strength + profile state), what risk-adjusted reward does entering have?"

Algorithm: linear Thompson sampling — Bayesian linear regression over the
feature vector with a Thompson draw for exploration. Light (a 13×13
matrix), no deep learning, robust on the modest data a daily-bar system
produces. Reward is the trade's R-multiple.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FEATURE_NAMES = (
    "bias", "rsi", "macd", "bb_pos", "sma_cross", "volume_ratio",
    "momentum_5d", "rs_rank", "regime", "profile_edge", "profile_state",
    "delta_count", "atr_pct",
)
N_FEATURES = len(FEATURE_NAMES)

_REGIME_VALUE = {
    "risk-on": 1.0, "neutral": 0.0, "risk-off": -1.0,
    "high-vol": -0.5, "unknown": 0.0,
}


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def extract_features(
    bundle_dict: dict[str, Any],
    profile: Any,
    regime: dict[str, Any] | None,
) -> list[float]:
    """Build the normalised feature vector for one decision context."""
    tech = bundle_dict.get("technical") or {}
    cs = bundle_dict.get("_cross_section") or {}
    deltas = bundle_dict.get("_deltas") or {}

    rsi = tech.get("rsi14")
    f_rsi = ((float(rsi) - 50.0) / 50.0) if isinstance(rsi, (int, float)) else 0.0

    macd = tech.get("macd_state")
    f_macd = 1.0 if macd == "bullish" else (-1.0 if macd == "bearish" else 0.0)

    bb = tech.get("bb_position")
    f_bb = ((float(bb) - 0.5) * 2.0) if isinstance(bb, (int, float)) else 0.0

    sma = tech.get("sma_cross")
    f_sma = 1.0 if sma == "golden" else (-1.0 if sma == "death" else 0.0)

    vr = tech.get("volume_ratio")
    f_vol = _clip(float(vr) - 1.0, -1.0, 2.0) if isinstance(vr, (int, float)) else 0.0

    ch = tech.get("pct_change_5d")
    f_mom = _clip(float(ch) / 10.0, -1.0, 1.0) if isinstance(ch, (int, float)) else 0.0

    rs = cs.get("rs_rank")
    f_rs = ((float(rs) - 0.5) * 2.0) if isinstance(rs, (int, float)) else 0.0

    f_regime = _REGIME_VALUE.get((regime or {}).get("regime", "unknown"), 0.0)

    f_edge = profile.edge_score() if profile is not None else 0.0
    state = profile.state() if profile is not None else "NEUTRAL"
    f_state = {"PROVEN": 1.0, "NEUTRAL": 0.0,
               "PROBATION": -0.7, "BENCHED": -1.0}.get(state, 0.0)

    changes = deltas.get("changes", []) if deltas.get("available") else []
    real_changes = [c for c in changes if "no notable" not in c]
    f_delta = _clip(len(real_changes) / 5.0, 0.0, 1.0)

    atr = tech.get("atr14")
    last = tech.get("last_close")
    atr_pct = (float(atr) / float(last)) if (atr and last and last > 0) else 0.0
    f_atr = _clip(atr_pct * 20.0, 0.0, 2.0)

    return [1.0, f_rsi, f_macd, f_bb, f_sma, f_vol, f_mom, f_rs,
            f_regime, f_edge, f_state, f_delta, f_atr]


class LinTS:
    """Linear Thompson sampling — Bayesian linear regression bandit."""

    def __init__(self, d: int = N_FEATURES, ridge: float = 1.0,
                 explore: float = 0.35) -> None:
        self.d = d
        self.ridge = ridge
        self.explore = explore
        self.A = np.eye(d) * ridge       # precision matrix
        self.b = np.zeros(d)             # x·reward accumulator
        self.updates = 0

    def update(self, x: list[float], reward: float) -> None:
        xv = np.asarray(x, dtype=float)
        # Clip reward (R-multiple) so a single outlier can't dominate.
        r = max(-3.0, min(3.0, float(reward)))
        self.A += np.outer(xv, xv)
        self.b += r * xv
        self.updates += 1

    def _mean(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    def predict(self, x: list[float]) -> float:
        """Greedy expected reward (posterior mean)."""
        return float(np.dot(self._mean(), np.asarray(x, dtype=float)))

    def sample_predict(self, x: list[float]) -> float:
        """Thompson draw — expected reward under a sampled policy."""
        xv = np.asarray(x, dtype=float)
        a_inv = np.linalg.inv(self.A)
        mean = a_inv @ self.b
        cov = (self.explore ** 2) * a_inv
        try:
            theta = np.random.multivariate_normal(mean, cov)
        except Exception:
            theta = mean
        return float(np.dot(theta, xv))

    def to_dict(self) -> dict[str, Any]:
        return {
            "d": self.d, "ridge": self.ridge, "explore": self.explore,
            "A": self.A.tolist(), "b": self.b.tolist(), "updates": self.updates,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LinTS":
        m = cls(d=int(data.get("d", N_FEATURES)),
                ridge=float(data.get("ridge", 1.0)),
                explore=float(data.get("explore", 0.35)))
        try:
            A = np.asarray(data["A"], dtype=float)
            b = np.asarray(data["b"], dtype=float)
            if A.shape == (m.d, m.d) and b.shape == (m.d,):
                m.A, m.b = A, b
        except Exception:
            pass
        m.updates = int(data.get("updates", 0))
        return m


_MODEL_KEY = "contextual_bandit_v1"


async def load_bandit(memory: Any) -> LinTS:
    data = await memory.get_model(_MODEL_KEY)
    return LinTS.from_dict(data) if data else LinTS()


async def save_bandit(memory: Any, bandit: LinTS) -> None:
    await memory.save_model(_MODEL_KEY, bandit.to_dict())
