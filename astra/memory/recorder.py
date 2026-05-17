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
        pnl = float(sell_trade.get("pnl") or 0.0)
        win = pnl > 0
        symbol = sell_trade.get("symbol")

        hold_minutes = 0.0
        opening: dict[str, Any] | None = None
        closed_id = sell_trade.get("closed_trade_id")
        if closed_id:
            opening = await self.portfolio.get_trade(int(closed_id))
            if opening:
                a = _parse_iso(opening.get("executed_at"))
                b = _parse_iso(sell_trade.get("executed_at"))
                if a and b:
                    hold_minutes = max(0.0, (b - a).total_seconds() / 60.0)

        snapshot = sell_trade.get("signal_snapshot") or {}

        # --- Global pattern stats ---
        pattern_hash = sell_trade.get("pattern_hash")
        if pattern_hash:
            pattern_desc = snapshot.get("_pattern_desc") or pattern_hash
            await self.memory.record_pattern_outcome(
                pattern_hash=pattern_hash,
                pattern_desc=pattern_desc,
                win=win,
                pnl=pnl,
                hold_minutes=hold_minutes,
            )

        # --- Per-stock adaptive profile (Layers 2 & 3 + bandit) ---
        if symbol:
            await self._update_profile(symbol, win, pnl, hold_minutes, opening)

        log.info("Recorded close %s win=%s pnl=%.2f", symbol, win, pnl)

    async def _update_profile(
        self,
        symbol: str,
        win: bool,
        pnl: float,
        hold_minutes: float,
        opening: dict[str, Any] | None,
    ) -> None:
        from astra.profiles import (
            GLOBAL_SYMBOL,
            extract_signals,
            load_profile,
            save_profile,
        )

        # Entry signals come from the BUY trade's stored snapshot.
        entry_snapshot = (opening or {}).get("signal_snapshot") or {}
        entry_signals = extract_signals(entry_snapshot)

        # R-multiple: P&L expressed as a multiple of the dollar risk taken
        # (initial stop distance). The size-independent learning unit.
        r_multiple = self._r_multiple(opening, entry_snapshot, pnl)

        profile = await load_profile(self.memory, symbol)
        profile.record_trade_outcome(win, pnl, hold_minutes, entry_signals, r_multiple)
        await save_profile(self.memory, profile)

        # The global aggregate profile collects signal outcomes across all
        # stocks; it's the shrinkage prior for per-stock signal rates.
        if entry_signals:
            global_profile = await load_profile(self.memory, GLOBAL_SYMBOL)
            global_profile.record_trade_outcome(
                win, pnl, hold_minutes, entry_signals, r_multiple)
            await save_profile(self.memory, global_profile)

        # Contextual bandit: update the policy model with the entry context
        # features and the realized R-multiple reward.
        features = entry_snapshot.get("_bandit_features")
        if isinstance(features, list) and features:
            from astra.engine.contextual_bandit import load_bandit, save_bandit
            bandit = await load_bandit(self.memory)
            bandit.update(features, r_multiple)
            await save_bandit(self.memory, bandit)

        # Expert ensemble: Hedge update — reward experts that voted BUY on
        # this entry by the realized R, within the entry's regime.
        expert_votes = entry_snapshot.get("_expert_votes")
        entry_regime = entry_snapshot.get("_entry_regime", "unknown")
        if isinstance(expert_votes, dict) and expert_votes:
            from astra.engine.ensemble import load_ensemble, save_ensemble
            ensemble = await load_ensemble(self.memory)
            ensemble.update(expert_votes, str(entry_regime), r_multiple)
            await save_ensemble(self.memory, ensemble)

    @staticmethod
    def _r_multiple(
        opening: dict[str, Any] | None,
        entry_snapshot: dict[str, Any],
        pnl: float,
    ) -> float:
        """pnl / initial dollar risk (qty × entry price × stop %)."""
        if not opening:
            return 0.0
        stop_pct = entry_snapshot.get("_stop_pct")
        try:
            qty = float(opening.get("qty") or 0)
            price = float(opening.get("price") or 0)
            stop_pct = float(stop_pct) if stop_pct is not None else 0.0
        except (TypeError, ValueError):
            return 0.0
        initial_risk = qty * price * stop_pct
        if initial_risk <= 0:
            return 0.0
        return round(pnl / initial_risk, 3)
