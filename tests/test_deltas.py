"""Signal-delta computation tests."""

from __future__ import annotations

import pytest

from astra.signals.deltas import compute_deltas


def test_no_prior_snapshot():
    d = compute_deltas({"technical": {"rsi14": 50}}, None)
    assert d["available"] is False


def test_rsi_threshold_cross_detected():
    prev = {"technical": {"rsi14": 32, "macd_state": "bullish", "last_close": 100}}
    cur = {"technical": {"rsi14": 28, "macd_state": "bullish", "last_close": 100}}
    d = compute_deltas(cur, prev)
    assert any("crossed BELOW 30" in c for c in d["changes"])


def test_macd_flip_detected():
    prev = {"technical": {"macd_state": "bullish"}}
    cur = {"technical": {"macd_state": "bearish"}}
    d = compute_deltas(cur, prev)
    assert d["macd_flip"] == "bullish->bearish"
    assert any("MACD flipped" in c for c in d["changes"])


def test_volume_spike_detected():
    prev = {"technical": {"volume_ratio": 1.0}}
    cur = {"technical": {"volume_ratio": 2.5}}
    d = compute_deltas(cur, prev)
    assert any("Volume 2.5" in c for c in d["changes"])


def test_sentiment_shift_detected():
    prev = {"sentiment": {"company_news_score": 0.3}}
    cur = {"sentiment": {"company_news_score": 0.65}}
    d = compute_deltas(cur, prev)
    assert any("sentiment improved" in c for c in d["changes"])


def test_no_notable_changes():
    prev = {"technical": {"rsi14": 50.1, "macd_state": "bullish", "last_close": 100}}
    cur = {"technical": {"rsi14": 50.2, "macd_state": "bullish", "last_close": 100.1}}
    d = compute_deltas(cur, prev)
    assert d["changes"] == ["no notable changes since last tick"]
