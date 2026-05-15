"""Record pattern outcomes when trades close."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from astra.db.memory import MemoryDB
from astra.db.portfolio import PortfolioDB

log = logging.getLogger(__name__)


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


class MemoryRecorder:
    def __init__(self, portfolio: PortfolioDB, memory: MemoryDB) -> None:
        self.portfolio = portfolio
        self.memory = memory

    async def record_close(self, sell_trade: dict[str, Any]) -> None:
        if sell_trade.get("side") != "SELL":
            return
        pattern_hash = sell_trade.get("pattern_hash")
        if not pattern_hash:
            return
        pnl = sell_trade.get("pnl") or 0.0
        win = pnl > 0
        hold_minutes = 0.0
        closed_id = sell_trade.get("closed_trade_id")
        if closed_id:
            opening = await self.portfolio.get_trade(int(closed_id))
            if opening:
                a = _parse_iso(opening.get("executed_at"))
                b = _parse_iso(sell_trade.get("executed_at"))
                if a and b:
                    hold_minutes = max(0.0, (b - a).total_seconds() / 60.0)

        snapshot = sell_trade.get("signal_snapshot") or {}
        pattern_desc = (snapshot.get("_pattern_desc") or pattern_hash)
        await self.memory.record_pattern_outcome(
            pattern_hash=pattern_hash,
            pattern_desc=pattern_desc,
            win=win,
            pnl=float(pnl),
            hold_minutes=hold_minutes,
        )
        log.info("Recorded pattern outcome %s win=%s pnl=%.2f", pattern_hash, win, pnl)
