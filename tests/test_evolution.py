"""Evolution loop tests — walk-forward, no lookahead, bounded tuning."""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from astra.config import settings
from astra.engine.evolution import (
    _BOUNDS,
    propose_parameter_changes,
    run_evolution,
)
from astra.profiles import load_all_profiles

pytestmark = pytest.mark.asyncio


def _history(symbol: str, days: int = 1400) -> list[dict]:
    r = random.Random(hash(symbol) & 0xFFFF)
    p = 50.0 + r.random() * 150
    base = 1_500_000_000
    rows = []
    for i in range(days):
        p = max(1.0, p * (1 + r.gauss(0.0005, 0.018)))
        rows.append({"t": base + i * 86400, "o": p * 0.99, "h": p * 1.02,
                     "l": p * 0.98, "c": p, "v": 1_000_000})
    return rows


@pytest.fixture
def patched_history():
    async def fake(symbol, days=1825, cache=None, force_real=False):
        return _history(symbol)
    with patch("astra.engine.trainer.fetch_daily_candles", new=fake):
        yield


# ---- parameter tuner ----

async def test_tuner_tightens_stop_on_high_drawdown():
    settings.stop_loss_pct = 0.08
    changes = propose_parameter_changes(
        {"max_drawdown_pct": 35, "win_rate": 0.5, "profit_factor": 1.2,
         "sharpe": 1.0, "closed_trades": 20})
    assert "stop_loss_pct" in changes
    assert changes["stop_loss_pct"][0] < 0.08


async def test_tuner_respects_bounds():
    settings.stop_loss_pct = _BOUNDS["stop_loss_pct"][0]  # already at the floor
    changes = propose_parameter_changes(
        {"max_drawdown_pct": 50, "win_rate": 0.5, "profit_factor": 1.0,
         "sharpe": 1.0, "closed_trades": 20})
    # Can't go below the floor → no change proposed.
    assert "stop_loss_pct" not in changes


async def test_tuner_raises_entry_bar_on_low_win_rate():
    settings.entry_min_confidence = 0.5
    changes = propose_parameter_changes(
        {"max_drawdown_pct": 5, "win_rate": 0.30, "profit_factor": 1.1,
         "sharpe": 1.0, "closed_trades": 40})
    assert changes["entry_min_confidence"][0] > 0.5


# ---- the loop ----

async def test_evolution_runs_walk_forward(memory_db, cache_db, patched_history):
    result = await run_evolution(
        ["AAA", "BBB", "CCC"], memory_db, cache_db,
        target_return_pct=999.0,  # unreachable → runs all windows
        window_days=80, max_generations=6,
    )
    assert result["ok"] is True
    gens = result["generations"]
    assert len(gens) >= 2
    # Windows are time-ordered and non-overlapping — each starts after the
    # previous one ends.
    for a, b in zip(gens, gens[1:]):
        a_end = a["window"].split(" → ")[1]
        b_start = b["window"].split(" → ")[0]
        assert b_start > a_end, "windows must not overlap"


async def test_evolution_persists_learning(memory_db, cache_db, patched_history):
    await run_evolution(
        ["AAA", "BBB"], memory_db, cache_db,
        target_return_pct=999.0, window_days=80, max_generations=4,
    )
    # Per-stock profiles were built and persisted across generations.
    profiles = await load_all_profiles(memory_db)
    assert len(profiles) >= 1


async def test_evolution_stops_on_target(memory_db, cache_db, patched_history):
    result = await run_evolution(
        ["AAA", "BBB", "CCC"], memory_db, cache_db,
        target_return_pct=-999.0,  # trivially reachable → stops at gen 1
        window_days=80, max_generations=10,
    )
    assert result["ok"] is True
    assert len(result["generations"]) == 1
    assert "target" in result["status"]


async def test_evolution_records_final_params(memory_db, cache_db, patched_history):
    result = await run_evolution(
        ["AAA", "BBB"], memory_db, cache_db,
        target_return_pct=999.0, window_days=80, max_generations=4,
    )
    fp = result["final_params"]
    for key in _BOUNDS:
        assert key in fp
        lo, hi = _BOUNDS[key]
        assert lo <= fp[key] <= hi  # tuning never escaped the bounds
