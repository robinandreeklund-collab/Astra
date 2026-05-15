"""Paper-money broker. Simulates fills with slippage + flat fee."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from astra.config import settings
from astra.db.portfolio import PortfolioDB

log = logging.getLogger(__name__)


@dataclass
class Fill:
    side: str
    symbol: str
    qty: float
    price: float
    fees: float
    cash_delta: float
    pnl: float | None = None
    closed_trade_id: int | None = None


class InsufficientCash(Exception):
    pass


class NoPosition(Exception):
    pass


class PaperBroker:
    def __init__(self, portfolio: PortfolioDB) -> None:
        self.portfolio = portfolio

    def _slippage_price(self, side: str, ref_price: float) -> float:
        bps = settings.slippage_bps / 10000.0
        return ref_price * (1 + bps) if side == "BUY" else ref_price * (1 - bps)

    async def buy(
        self,
        symbol: str,
        qty: float,
        ref_price: float,
        signal_snapshot: dict[str, Any],
        reasoning: str | None,
        pattern_hash: str | None = None,
        executed_at: str | None = None,
    ) -> Fill:
        if qty <= 0:
            raise ValueError("qty must be > 0")
        price = self._slippage_price("BUY", ref_price)
        fees = settings.fee_per_trade
        cost = price * qty + fees

        acc = await self.portfolio.get_account()
        if not acc:
            raise RuntimeError("no account")
        if acc["cash"] < cost:
            raise InsufficientCash(f"cash {acc['cash']:.2f} < cost {cost:.2f}")

        new_cash = acc["cash"] - cost
        await self.portfolio.update_cash(new_cash)

        existing = await self.portfolio.get_position(symbol)
        if existing:
            new_qty = existing["qty"] + qty
            new_avg = (existing["qty"] * existing["avg_price"] + qty * price) / new_qty
        else:
            new_qty = qty
            new_avg = price
        await self.portfolio.upsert_position(symbol, new_qty, new_avg)

        await self.portfolio.insert_trade(
            symbol=symbol, side="BUY", qty=qty, price=price, fees=fees,
            signal_snapshot=signal_snapshot, llm_reasoning=reasoning,
            pattern_hash=pattern_hash, executed_at=executed_at,
        )
        log.info("BUY %s %.4f @ %.4f (cash %.2f→%.2f)",
                 symbol, qty, price, acc["cash"], new_cash)
        return Fill("BUY", symbol, qty, price, fees, -cost)

    async def sell(
        self,
        symbol: str,
        qty: float,
        ref_price: float,
        signal_snapshot: dict[str, Any],
        reasoning: str | None,
        pattern_hash: str | None = None,
        executed_at: str | None = None,
    ) -> Fill:
        if qty <= 0:
            raise ValueError("qty must be > 0")
        existing = await self.portfolio.get_position(symbol)
        if not existing or existing["qty"] < qty - 1e-9:
            raise NoPosition(f"cannot SELL {qty} {symbol} (have {existing['qty'] if existing else 0})")

        price = self._slippage_price("SELL", ref_price)
        fees = settings.fee_per_trade
        proceeds = price * qty - fees

        acc = await self.portfolio.get_account()
        if not acc:
            raise RuntimeError("no account")
        new_cash = acc["cash"] + proceeds
        await self.portfolio.update_cash(new_cash)

        remaining = existing["qty"] - qty
        avg = existing["avg_price"]
        pnl = (price - avg) * qty - fees

        if remaining <= 1e-9:
            await self.portfolio.delete_position(symbol)
        else:
            await self.portfolio.upsert_position(symbol, remaining, avg)

        opening = await self.portfolio.open_buy_trade_for(symbol)
        closed_id = opening["id"] if opening else None

        await self.portfolio.insert_trade(
            symbol=symbol, side="SELL", qty=qty, price=price, fees=fees,
            signal_snapshot=signal_snapshot, llm_reasoning=reasoning,
            pnl=pnl, pattern_hash=pattern_hash, closed_trade_id=closed_id,
            executed_at=executed_at,
        )
        log.info("SELL %s %.4f @ %.4f (pnl %.2f, cash %.2f→%.2f)",
                 symbol, qty, price, pnl, acc["cash"], new_cash)
        return Fill("SELL", symbol, qty, price, fees, proceeds, pnl, closed_id)

    async def mark_to_market(self, prices: dict[str, float]) -> tuple[float, float]:
        """Returns (equity, cash) given a dict of {symbol: price}."""
        acc = await self.portfolio.get_account()
        if not acc:
            return 0.0, 0.0
        positions = await self.portfolio.get_positions()
        market_value = sum(p["qty"] * prices.get(p["symbol"], p["avg_price"]) for p in positions)
        return acc["cash"] + market_value, acc["cash"]
