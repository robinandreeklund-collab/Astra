"""Nightly reflection — turn recent trades into compact lessons."""

from __future__ import annotations

import logging
from typing import Any

from astra.db.memory import MemoryDB
from astra.db.portfolio import PortfolioDB
from astra.llm.client import LMStudioClient
from astra.llm.prompts import SYSTEM_REFLECTOR, build_reflection_user_prompt

log = logging.getLogger(__name__)


class Reflector:
    def __init__(
        self,
        portfolio: PortfolioDB,
        memory: MemoryDB,
        client: LMStudioClient | None,
    ) -> None:
        self.portfolio = portfolio
        self.memory = memory
        self.client = client

    async def reflect(self, max_trades: int = 30) -> list[str]:
        closed = await self.portfolio.closed_trades(limit=max_trades)
        if len(closed) < 3:
            return []

        gen = await self.memory.current_generation()
        gen_id = gen["id"] if gen else 0

        lessons: list[str] = []
        if self.client is not None:
            try:
                user = build_reflection_user_prompt(closed)
                raw = await self.client.chat_json(
                    SYSTEM_REFLECTOR, user, temperature=0.3, max_tokens=400
                )
                lessons = [str(l).strip()[:240] for l in (raw.get("lessons") or []) if l]
            except Exception as e:
                log.warning("Reflection LLM failed: %s — using heuristic lessons", e)

        if not lessons:
            lessons = self._heuristic_lessons(closed)

        evidence_ids = [t["id"] for t in closed[:10] if t.get("id")]
        for l in lessons:
            await self.memory.add_lesson(l, generation=gen_id, evidence_trade_ids=evidence_ids)
        log.info("Reflection produced %d lessons", len(lessons))
        return lessons

    def _heuristic_lessons(self, closed: list[dict[str, Any]]) -> list[str]:
        if not closed:
            return []
        winners = [t for t in closed if (t.get("pnl") or 0) > 0]
        losers = [t for t in closed if (t.get("pnl") or 0) <= 0]
        out: list[str] = []
        wr = len(winners) / len(closed) if closed else 0
        out.append(f"Recent batch: {len(winners)}W/{len(losers)}L ({wr:.0%} win-rate)")

        def avg(items: list[dict[str, Any]], path: list[str]) -> float | None:
            vals: list[float] = []
            for t in items:
                v: Any = t.get("signal_snapshot") or {}
                for p in path:
                    if isinstance(v, dict):
                        v = v.get(p)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            return sum(vals) / len(vals) if vals else None

        w_rsi = avg(winners, ["technical", "rsi14"])
        l_rsi = avg(losers, ["technical", "rsi14"])
        if w_rsi is not None and l_rsi is not None and abs(w_rsi - l_rsi) > 5:
            out.append(f"Winners avg RSI={w_rsi:.0f}, losers avg RSI={l_rsi:.0f} — favor that range")
        return out
