"""Synthetic market simulator tests."""

from __future__ import annotations

import pytest

from astra.data.simulator import SimulatedMarket, get_simulator, reset_simulator

pytestmark = pytest.mark.asyncio


async def test_seeds_history_for_all_symbols():
    sim = SimulatedMarket(["AAA", "BBB", "CCC"], seed=1)
    for s in ("AAA", "BBB", "CCC"):
        rows = sim.candles(s, days=500)
        assert len(rows) >= 200
        # candles are well-formed
        for r in rows[:5]:
            assert r["h"] >= r["l"]
            assert r["c"] > 0
            assert r["v"] > 0


async def test_advance_appends_one_bar():
    sim = SimulatedMarket(["AAA"], seed=2)
    before = len(sim.candles("AAA", days=9999))
    sim.advance()
    after = len(sim.candles("AAA", days=9999))
    assert after == before + 1
    # new bar timestamp is one day later
    rows = sim.candles("AAA", days=9999)
    assert rows[-1]["t"] - rows[-2]["t"] == 86400


async def test_history_is_capped():
    sim = SimulatedMarket(["AAA"], seed=3)
    for _ in range(200):
        sim.advance()
    # should not grow unbounded
    assert len(sim.candles("AAA", days=9999)) <= 280


async def test_candles_respects_days_arg():
    sim = SimulatedMarket(["AAA"], seed=4)
    assert len(sim.candles("AAA", days=30)) == 30
    assert len(sim.candles("AAA", days=60)) == 60


async def test_advance_produces_price_movement():
    sim = SimulatedMarket(["AAA"], seed=5)
    start = sim.last_close("AAA")
    for _ in range(50):
        sim.advance()
    end = sim.last_close("AAA")
    # 50 days of a random walk should move the price
    assert end != start


async def test_get_and_reset_simulator():
    s1 = get_simulator(["AAA", "BBB"])
    s2 = get_simulator()
    assert s1 is s2  # singleton
    s3 = reset_simulator(["XXX"])
    assert s3 is not s1
    assert "XXX" in s3.history


async def test_indicators_available_from_sim_candles():
    """Sim candles must produce usable technical indicators."""
    from astra.signals import technical
    sim = SimulatedMarket(["AAA"], seed=7)
    ind = technical.compute_indicators(sim.candles("AAA", days=180))
    assert ind["available"] is True
    assert ind["last_bar_ts"] > 0
    assert 0 <= ind["rsi14"] <= 100
