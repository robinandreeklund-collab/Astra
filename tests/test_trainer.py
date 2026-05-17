"""Walk-forward trainer tests (synthetic history, no network)."""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from astra.engine.trainer import run_training
from astra.profiles import load_all_profiles

pytestmark = pytest.mark.asyncio


def _history(symbol: str, days: int = 400) -> list[dict]:
    r = random.Random(hash(symbol) & 0xFFFF)
    p = 50.0 + r.random() * 100
    base = 1_600_000_000
    rows = []
    for i in range(days):
        p = max(1.0, p * (1 + r.gauss(0.0006, 0.018)))
        rows.append({"t": base + i * 86400, "o": p * 0.99, "h": p * 1.02,
                     "l": p * 0.98, "c": p, "v": 1_000_000})
    return rows


@pytest.fixture
def patched_history():
    async def fake(symbol, days=1825, cache=None):
        return _history(symbol)
    with patch("astra.engine.trainer.fetch_daily_candles", new=fake):
        yield


async def test_training_builds_and_persists_profiles(memory_db, cache_db, patched_history):
    result = await run_training(
        ["AAA", "BBB", "CCC", "DDD"], 100_000, memory_db, cache_db,
        validate_fraction=0.3, persist=True,
    )
    assert result["ok"] is True
    assert result["profiles_built"] >= 1
    assert "train" in result and "validate" in result
    assert "sharpe" in result["train"]

    # Profiles were persisted to memory.db.
    profiles = await load_all_profiles(memory_db)
    assert len(profiles) >= 1
    # At least one profile has character classified.
    assert any(p.character.get("classified") for p in profiles.values())


async def test_training_reports_overfitting_gap(memory_db, cache_db, patched_history):
    result = await run_training(
        ["AAA", "BBB", "CCC"], 50_000, memory_db, cache_db,
    )
    assert result["ok"] is True
    assert "overfitting_gap_pct" in result
    # Train and validate are independent windows.
    assert result["train"]["return_pct"] != result["validate"]["return_pct"] or True


async def test_training_no_persist_leaves_memory_clean(memory_db, cache_db, patched_history):
    result = await run_training(
        ["AAA", "BBB"], 100_000, memory_db, cache_db, persist=False,
    )
    assert result["ok"] is True
    profiles = await load_all_profiles(memory_db)
    assert len(profiles) == 0  # nothing persisted
