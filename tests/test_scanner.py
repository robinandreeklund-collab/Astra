"""Universe scanner tests using the fake yfinance fetch path."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from astra.engine.scanner import scan_universe, _ta_score

pytestmark = pytest.mark.asyncio


async def test_ta_score_oversold_bullish():
    tech = {
        "available": True,
        "rsi14": 25, "macd_state": "bullish", "bb_position": 0.1,
        "sma_cross": "golden", "volume_ratio": 1.0,
    }
    score, bull, bear = _ta_score(tech)
    assert bull == 4 and bear == 0
    assert score == 4.0


async def test_ta_score_unavailable():
    score, bull, bear = _ta_score({"available": False})
    assert score == 0.0 and bull == 0 and bear == 0


async def test_ta_score_volume_spike_boost():
    base = {"available": True, "rsi14": 50, "macd_state": "bullish",
            "bb_position": 0.5, "sma_cross": "golden", "volume_ratio": 1.0}
    spiking = {**base, "volume_ratio": 2.0}
    s1, _, _ = _ta_score(base)
    s2, _, _ = _ta_score(spiking)
    assert s2 > s1


async def test_scan_respects_top_n_and_orders_desc(cache_db):
    """Scanner returns at most top_n results, ordered by score descending."""
    import random
    rng = random.Random(0)

    def synth(seed: int, drift: float) -> list[dict]:
        r = random.Random(seed)
        p = 100.0
        out = []
        for i in range(80):
            p = max(1.0, p * (1 + r.gauss(drift, 0.02)))
            out.append({"t": 1700000000 + i*86400,
                        "o": p*0.99, "h": p*1.01, "l": p*0.98, "c": p,
                        "v": 1000000 + r.randint(0, 500000)})
        return out

    candles = {f"SYM{i}": synth(i, rng.gauss(0, 0.005)) for i in range(20)}

    async def fake_fetch(symbol, days=180, cache=None):
        return candles.get(symbol, [])

    with patch("astra.engine.scanner.fetch_daily_candles", new=fake_fetch):
        scan = await scan_universe(list(candles.keys()), cache_db, top_n=5)

    top = scan["candidates"]
    assert len(top) <= 5
    # Strictly non-increasing scores
    scores = [t["score"] for t in top]
    assert scores == sorted(scores, reverse=True)
    assert all(t["score"] > 0 for t in top)
    # Regime is classified and cross-sectional RS attached.
    assert scan["regime"]["regime"] in (
        "risk-on", "neutral", "risk-off", "high-vol", "unknown")
    assert all("rs_rank" in t for t in top)


async def test_scan_filters_out_empty_data(cache_db):
    async def fake_fetch(symbol, days=180, cache=None):
        return []  # no data for any symbol

    with patch("astra.engine.scanner.fetch_daily_candles", new=fake_fetch):
        scan = await scan_universe(["A", "B", "C"], cache_db, top_n=10)
    assert scan["candidates"] == []
