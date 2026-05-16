"""Risk management: position sizing, exposure caps, daily loss limit."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from astra.config import settings
from astra.db.portfolio import PortfolioDB


class RiskManager:
    def __init__(self, portfolio: PortfolioDB) -> None:
        self.portfolio = portfolio

    async def can_trade_today(self) -> tuple[bool, str]:
        acc = await self.portfolio.get_account()
        if not acc:
            return False, "no account"
        today = datetime.now(timezone.utc).date().isoformat()
        history = await self.portfolio.equity_history(limit=2000)
        today_points = [h for h in history if h["ts"].startswith(today)]
        if not today_points:
            return True, "no equity points yet today"
        start_today = today_points[0]["equity"]
        latest = today_points[-1]["equity"]
        loss_pct = (latest - start_today) / start_today if start_today > 0 else 0.0
        if loss_pct < -settings.daily_loss_limit_pct:
            return False, f"daily loss {loss_pct:.2%} exceeds limit"
        return True, "ok"

    async def size_buy(self, ref_price: float, confidence: float) -> float:
        """Compute qty for a BUY given current cash + position cap + confidence."""
        acc = await self.portfolio.get_account()
        if not acc:
            return 0.0
        positions = await self.portfolio.get_positions()
        # Approx total equity using avg cost (good enough for sizing)
        equity_est = acc["cash"] + sum(p["qty"] * p["avg_price"] for p in positions)
        max_position_value = equity_est * settings.max_position_pct
        confidence = max(0.0, min(1.0, float(confidence)))
        target_value = max_position_value * confidence
        if target_value < 1.0 or ref_price <= 0:
            return 0.0
        # Don't spend more than 95% of available cash on a single buy
        target_value = min(target_value, acc["cash"] * 0.95)
        qty = target_value / ref_price
        return max(0.0, round(qty, 4))

    async def validate_buy(self, symbol: str, qty: float, ref_price: float) -> tuple[bool, str]:
        ok, why = await self.can_trade_today()
        if not ok:
            return False, why
        acc = await self.portfolio.get_account()
        if not acc:
            return False, "no account"
        notional = qty * ref_price
        commission = max(settings.min_fee, settings.fee_pct * notional)
        cost = notional + commission
        if cost > acc["cash"]:
            return False, "insufficient cash"
        existing = await self.portfolio.get_position(symbol)
        positions = await self.portfolio.get_positions()
        equity_est = acc["cash"] + sum(p["qty"] * p["avg_price"] for p in positions)
        new_pos_value = ((existing["qty"] if existing else 0) + qty) * ref_price
        if new_pos_value > equity_est * settings.max_position_pct * 1.1:
            return False, "position cap"
        return True, "ok"

    async def validate_sell(self, symbol: str, qty: float) -> tuple[bool, str]:
        existing = await self.portfolio.get_position(symbol)
        if not existing or existing["qty"] < qty - 1e-9:
            return False, "no position / qty too high"
        return True, "ok"
