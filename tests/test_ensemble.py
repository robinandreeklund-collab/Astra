"""Expert ensemble + experts tests."""

from __future__ import annotations

import pytest

from astra.engine.ensemble import ExpertEnsemble, load_ensemble, save_ensemble
from astra.engine.experts import EXPERT_NAMES, run_experts

pytestmark = pytest.mark.asyncio


# ---- experts ----

async def test_momentum_expert_votes_buy_on_strong_momentum():
    bundle = {
        "technical": {"pct_change_5d": 5.0, "macd_state": "bullish"},
        "_cross_section": {"rs_rank": 0.85},
    }
    votes = run_experts(bundle)
    assert votes["momentum"].action == "BUY"


async def test_mean_reversion_expert_votes_buy_when_oversold():
    bundle = {"technical": {"rsi14": 25, "bb_position": 0.1, "sma_cross": "golden"}}
    votes = run_experts(bundle)
    assert votes["mean_reversion"].action == "BUY"


async def test_experts_hold_on_flat_data():
    votes = run_experts({"technical": {"rsi14": 50, "macd_state": "bullish",
                                       "bb_position": 0.5, "pct_change_5d": 0.1}})
    # No strong setup → most experts abstain
    assert sum(1 for v in votes.values() if v.action == "BUY") <= 1


async def test_run_experts_returns_all():
    votes = run_experts({"technical": {}})
    assert set(votes.keys()) == set(EXPERT_NAMES)


# ---- ensemble ----

async def test_ensemble_combine_consensus():
    from astra.engine.experts import ExpertVote
    votes = {n: ExpertVote("BUY", 0.8) for n in EXPERT_NAMES}
    e = ExpertEnsemble()
    out = e.combine(votes, "risk-on")
    assert out["consensus"] == 1.0  # everyone buys
    assert out["action"] == "BUY"

    votes = {n: ExpertVote("HOLD", 0.0) for n in EXPERT_NAMES}
    out = e.combine(votes, "risk-on")
    assert out["consensus"] == -1.0
    assert out["action"] == "HOLD"


async def test_ensemble_hedge_rewards_correct_experts():
    e = ExpertEnsemble()
    # 'momentum' voted BUY and the trade made +2R; others abstained.
    for _ in range(20):
        e.update({"momentum": "BUY", "trend": "HOLD"}, "risk-on", reward_r=2.0)
    w = e.weights["risk-on"]
    assert w["momentum"] > w["trend"]  # winner gained weight


async def test_ensemble_hedge_punishes_losing_experts():
    e = ExpertEnsemble()
    for _ in range(20):
        e.update({"breakout": "BUY"}, "high-vol", reward_r=-1.5)
    w = e.weights["high-vol"]
    assert w["breakout"] < 1.0  # loser lost weight


async def test_ensemble_weights_are_regime_specific():
    e = ExpertEnsemble()
    for _ in range(15):
        e.update({"momentum": "BUY"}, "risk-on", reward_r=2.0)
    # risk-off untouched
    assert e.weights["risk-on"]["momentum"] > e.weights["risk-off"]["momentum"]


async def test_ensemble_serialisation_round_trip():
    e = ExpertEnsemble()
    for _ in range(10):
        e.update({"trend": "BUY"}, "neutral", reward_r=1.0)
    e2 = ExpertEnsemble.from_dict(e.to_dict())
    assert e2.updates == 10
    assert e2.weights["neutral"]["trend"] == pytest.approx(
        e.weights["neutral"]["trend"])


async def test_ensemble_persistence(memory_db):
    e = ExpertEnsemble()
    for _ in range(5):
        e.update({"momentum": "BUY"}, "risk-on", reward_r=1.5)
    await save_ensemble(memory_db, e)
    loaded = await load_ensemble(memory_db)
    assert loaded.updates == 5
