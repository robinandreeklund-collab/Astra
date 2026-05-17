"""LLM multi-role tests — playbook + critic (heuristic fallbacks)."""

from __future__ import annotations

import pytest

from astra.llm.roles import generate_critique, generate_playbook
from astra.profiles.profile import StockProfile

pytestmark = pytest.mark.asyncio


async def test_heuristic_playbook_for_new_stock():
    p = StockProfile(symbol="AAA", character={"archetype": "trender",
                                              "classified": True})
    pb = await generate_playbook(p, [], {}, client=None)
    assert "Trender" in pb
    assert "No trade history" in pb


async def test_heuristic_playbook_for_traded_stock():
    p = StockProfile(symbol="BBB", character={"archetype": "mean-reverter"})
    for _ in range(6):
        p.record_trade_outcome(True, 100, 60, ["macd_bull"], r_multiple=1.5)
    pb = await generate_playbook(p, [], {"macd_bull": 0.5}, client=None)
    assert "Record:" in pb
    assert "macd_bull" in pb
    assert "PROVEN" in pb


async def test_playbook_uses_llm_when_available():
    class FakeLLM:
        async def chat_json(self, system, user, **kw):
            return {"playbook": ["Buy dips on this trender", "Avoid RSI>70"]}
    p = StockProfile(symbol="CCC")
    pb = await generate_playbook(p, [], {}, client=FakeLLM())
    assert "Buy dips" in pb
    assert pb.startswith("•")


async def test_playbook_falls_back_on_llm_error():
    class BrokenLLM:
        async def chat_json(self, system, user, **kw):
            raise RuntimeError("llm down")
    p = StockProfile(symbol="DDD", character={"archetype": "choppy"})
    pb = await generate_playbook(p, [], {}, client=BrokenLLM())
    assert "Choppy" in pb  # heuristic fallback ran


async def test_heuristic_critique_flags_drawdown():
    report = {"max_drawdown_pct": 30, "win_rate": 0.5, "profit_factor": 1.2,
              "sharpe": 1.0, "closed_trades": 20}
    sugg = await generate_critique(report, [], client=None)
    assert any("drawdown" in s.lower() for s in sugg)


async def test_heuristic_critique_flags_low_win_rate():
    report = {"max_drawdown_pct": 5, "win_rate": 0.3, "profit_factor": 1.1,
              "sharpe": 1.0, "closed_trades": 30}
    sugg = await generate_critique(report, [], client=None)
    assert any("win rate" in s.lower() for s in sugg)


async def test_critique_uses_llm_when_available():
    class FakeLLM:
        async def chat_json(self, system, user, **kw):
            return {"suggestions": ["Raise entry_min_confidence to 0.65"]}
    sugg = await generate_critique(
        {"return_pct": 5}, [{"symbol": "AAA", "edge": 0.5}], client=FakeLLM())
    assert "entry_min_confidence" in sugg[0]


async def test_critique_healthy_when_metrics_fine():
    report = {"max_drawdown_pct": 5, "win_rate": 0.55, "profit_factor": 1.8,
              "sharpe": 1.5, "closed_trades": 40}
    sugg = await generate_critique(report, [], client=None)
    assert sugg  # always returns at least one line
