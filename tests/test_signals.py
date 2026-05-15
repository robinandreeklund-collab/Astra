"""Signal computation tests."""

from __future__ import annotations

import math
import random
import time

import pytest

from astra.signals import technical
from astra.signals.aggregator import SignalBundle


def make_candles(n=120, start=100.0, drift=0.001, vol=0.01, seed=42):
    rng = random.Random(seed)
    out = []
    p = start
    t = int(time.time()) - n * 86400
    for i in range(n):
        ret = rng.gauss(drift, vol)
        p = max(1.0, p * (1 + ret))
        h = p * (1 + abs(rng.gauss(0, vol / 2)))
        l = p * (1 - abs(rng.gauss(0, vol / 2)))
        o = (h + l) / 2
        out.append({"t": t + i * 86400, "o": o, "h": h, "l": l, "c": p, "v": 100000})
    return out


def test_indicators_from_synthetic_candles():
    cs = make_candles(120, drift=0.002)
    ind = technical.compute_indicators(cs)
    assert ind["available"] is True
    assert 0 < ind["rsi14"] < 100
    assert ind["macd_state"] in ("bullish", "bearish")
    assert ind["bb_position"] is not None


def test_indicators_insufficient_data():
    cs = make_candles(5)
    ind = technical.compute_indicators(cs)
    assert ind["available"] is False


def test_pattern_hash_stable():
    cs = make_candles(120, drift=0.002, seed=1)
    ind = technical.compute_indicators(cs)
    b = SignalBundle(symbol="X", fetched_at=0)
    b.technical = ind
    h1, d1 = b.pattern_hash()
    h2, d2 = b.pattern_hash()
    assert h1 == h2 and d1 == d2

    # Changing a feature changes the hash
    b.technical["macd_state"] = "bullish" if ind["macd_state"] == "bearish" else "bearish"
    h3, _ = b.pattern_hash()
    assert h3 != h1


def test_summarize_news_and_insider():
    from astra.signals import insider, news

    items = [
        {"headline": "Good news", "summary": "...", "source": "x", "datetime": 1},
        {"headline": "Better news", "summary": "...", "source": "y", "datetime": 2},
    ]
    s = news.summarize_headlines(items)
    assert s["count"] == 2 and len(s["headlines"]) == 2

    ins_data = {"data": [
        {"transactionDate": "2026-05-01", "change": 1000, "transactionPrice": 100},
        {"transactionDate": "2026-05-02", "change": -500, "transactionPrice": 105},
    ]}
    summary = insider.summarize_insider(ins_data, days=60)
    assert summary["available"] is True
    assert summary["shares_bought"] == 1000
    assert summary["shares_sold"] == 500
