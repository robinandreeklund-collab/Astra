"""Decision-engine tests (heuristic + LLM fallback)."""

from __future__ import annotations

import pytest

from astra.llm.decision import DecisionEngine, HeuristicDecisionEngine

pytestmark = pytest.mark.asyncio


async def _decide(bundle, position=None, cash=10000):
    eng = HeuristicDecisionEngine()
    return await eng.decide(bundle, position, cash, [], [])


async def test_heuristic_buy_on_bullish_stack():
    bundle = {
        "symbol": "AAPL",
        "technical": {"available": True, "rsi14": 28, "macd_state": "bullish",
                      "bb_position": 0.15, "sma_cross": "golden", "last_close": 150},
        "recommendations": {"trend": "upgrading"},
        "insider": {"net_direction": "buying"},
        "sentiment": {"company_news_score": 0.7},
        "earnings_calendar": {"upcoming": False},
    }
    d = await _decide(bundle)
    assert d.action == "BUY"
    assert d.confidence > 0.5


async def test_heuristic_block_buy_on_earnings():
    from datetime import date, timedelta
    soon = (date.today() + timedelta(days=1)).isoformat()
    bundle = {
        "symbol": "AAPL",
        "technical": {"available": True, "rsi14": 28, "macd_state": "bullish",
                      "bb_position": 0.15, "sma_cross": "golden", "last_close": 150},
        "recommendations": {"trend": "upgrading"},
        "insider": {"net_direction": "buying"},
        "sentiment": {"company_news_score": 0.7},
        "earnings_calendar": {"upcoming": True, "date": soon},
    }
    d = await _decide(bundle)
    assert d.action == "HOLD"


async def test_heuristic_sell_on_stop_loss():
    bundle = {
        "symbol": "AAPL",
        "technical": {"available": True, "rsi14": 50, "macd_state": "bearish",
                      "bb_position": 0.5, "sma_cross": "death", "last_close": 80},
    }
    d = await _decide(bundle, position={"qty": 1, "avg_price": 100, "symbol": "AAPL"})
    assert d.action == "SELL"


async def test_decision_engine_falls_back_when_llm_unreachable():
    """When LLM is None (or unreachable), DecisionEngine uses heuristic."""
    eng = DecisionEngine(client=None)
    bundle = {
        "symbol": "X",
        "technical": {"available": True, "rsi14": 50, "macd_state": "bullish",
                      "bb_position": 0.5, "sma_cross": "golden", "last_close": 100},
    }
    d = await eng.decide(bundle, None, 10000, [], [])
    assert d.source == "heuristic"


async def test_clamp01():
    from astra.llm.decision import clamp01
    assert clamp01("0.5") == 0.5
    assert clamp01(-1) == 0.0
    assert clamp01(2) == 1.0
    assert clamp01("nope") == 0.0
    assert clamp01(None) == 0.0
