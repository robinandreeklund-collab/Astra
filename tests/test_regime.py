"""Market-regime classification tests."""

from __future__ import annotations

import pytest

from astra.signals.regime import classify_regime, regime_risk_multiplier, regime_summary

pytestmark = pytest.mark.asyncio


def _results(macd: str, mom: float, n: int = 20):
    return [
        {"indicators": {"macd_state": macd, "pct_change_5d": mom}}
        for _ in range(n)
    ]


async def test_risk_on_when_broad_and_rising():
    r = classify_regime(_results("bullish", 1.5))
    assert r["regime"] == "risk-on"
    assert r["breadth"] == 1.0


async def test_risk_off_when_weak():
    r = classify_regime(_results("bearish", -3.0))
    assert r["regime"] == "risk-off"


async def test_high_vol_on_wide_dispersion():
    # Half the universe up huge, half down huge → big dispersion.
    res = ([{"indicators": {"macd_state": "bullish", "pct_change_5d": 15}}] * 10
           + [{"indicators": {"macd_state": "bearish", "pct_change_5d": -15}}] * 10)
    r = classify_regime(res)
    assert r["regime"] == "high-vol"


async def test_unknown_on_tiny_sample():
    r = classify_regime(_results("bullish", 1.0, n=3))
    assert r["regime"] == "unknown"


async def test_risk_multiplier_ordering():
    assert regime_risk_multiplier("risk-on") > regime_risk_multiplier("neutral")
    assert regime_risk_multiplier("neutral") > regime_risk_multiplier("risk-off")


async def test_regime_summary_text():
    r = classify_regime(_results("bullish", 1.5))
    s = regime_summary(r)
    assert "MARKET REGIME" in s and "risk-on" in s
