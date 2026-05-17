"""Contextual bandit (linear Thompson sampling) tests."""

from __future__ import annotations

import random

import pytest

from astra.engine.contextual_bandit import (
    N_FEATURES,
    LinTS,
    extract_features,
    load_bandit,
    save_bandit,
)
from astra.profiles.profile import StockProfile

pytestmark = pytest.mark.asyncio


async def test_extract_features_shape():
    bundle = {
        "technical": {"rsi14": 30, "macd_state": "bullish", "bb_position": 0.2,
                      "sma_cross": "golden", "volume_ratio": 1.8,
                      "pct_change_5d": 4.0, "atr14": 2.0, "last_close": 100},
        "_cross_section": {"rs_rank": 0.8},
        "_deltas": {"available": True, "changes": ["RSI rose 40→55", "MACD flipped"]},
    }
    feats = extract_features(bundle, StockProfile(symbol="X"), {"regime": "risk-on"})
    assert len(feats) == N_FEATURES
    assert feats[0] == 1.0  # bias
    assert all(isinstance(f, float) for f in feats)


async def test_extract_features_handles_missing():
    feats = extract_features({"technical": {}}, None, None)
    assert len(feats) == N_FEATURES
    assert feats[0] == 1.0


async def test_lints_learns_a_positive_context():
    """Feed a context that always rewards +2R; the model should predict high."""
    b = LinTS()
    good = [1.0, 0.5, 1.0, -0.5, 1.0, 0.5, 0.4, 0.6, 1.0, 0.0, 0.0, 0.4, 0.4]
    for _ in range(40):
        b.update(good, reward=2.0)
    assert b.predict(good) > 0.8
    assert b.updates == 40


async def test_lints_separates_good_from_bad_contexts():
    b = LinTS()
    good = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0]
    bad = [1.0, -1.0, -1.0, 0.0, -1.0, 0.0, -0.5, -0.5, -1.0, 0.0, 0.0, 0.0, 0.0]
    for _ in range(50):
        b.update(good, reward=2.0)
        b.update(bad, reward=-1.5)
    assert b.predict(good) > b.predict(bad)


async def test_lints_reward_is_clipped():
    b = LinTS()
    x = [1.0] + [0.0] * (N_FEATURES - 1)
    b.update(x, reward=999.0)  # absurd outlier
    # Clipped to 3 → prediction can't blow up
    assert b.predict(x) <= 3.5


async def test_lints_serialisation_round_trip():
    b = LinTS()
    ctx = [1.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
    for _ in range(10):
        b.update(ctx, reward=1.0)
    d = b.to_dict()
    b2 = LinTS.from_dict(d)
    assert b2.updates == 10
    assert abs(b2.predict(ctx) - b.predict(ctx)) < 1e-9


async def test_bandit_persistence(memory_db):
    b = LinTS()
    ctx = [1.0] + [0.2] * (N_FEATURES - 1)
    for _ in range(5):
        b.update(ctx, reward=1.5)
    await save_bandit(memory_db, b)
    loaded = await load_bandit(memory_db)
    assert loaded.updates == 5


async def test_sample_predict_varies():
    """Thompson sampling should produce a spread of estimates."""
    b = LinTS()
    x = [1.0] + [0.1] * (N_FEATURES - 1)
    samples = [b.sample_predict(x) for _ in range(30)]
    assert len(set(round(s, 4) for s in samples)) > 1
