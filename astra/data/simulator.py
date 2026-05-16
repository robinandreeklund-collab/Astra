"""Synthetic market simulator.

Lets the platform run with no live data — useful when the market is closed
or there's no API key. Each symbol follows a regime-switching random walk
(trending up / trending down / choppy) so the bot sees genuine momentum,
RSI extremes, MACD crosses and volume spikes to act on.

One `advance()` call appends a fresh daily bar to every symbol; the engine
calls it once per tick, so simulated time runs at one trading day per tick.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

log = logging.getLogger(__name__)

_SEED_DAYS = 220
_MAX_HISTORY = 280


class SimulatedMarket:
    def __init__(self, symbols: list[str], seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.symbols = list(symbols)
        self.history: dict[str, list[dict[str, Any]]] = {}
        self.regime: dict[str, dict[str, Any]] = {}
        self.created_at = time.time()
        self.days_advanced = 0
        self._seed_series()

    # ---- internals ----

    def _new_regime(self) -> dict[str, Any]:
        kind = self.rng.choice(["up", "up", "down", "chop", "chop"])
        drift = {"up": 0.006, "down": -0.006, "chop": 0.0}[kind]
        return {
            "kind": kind,
            "drift": drift,
            "vol": self.rng.uniform(0.012, 0.038),
            "days_left": self.rng.randint(6, 28),
        }

    def _step(self, symbol: str, price: float, ts: int) -> tuple[float, dict[str, Any]]:
        reg = self.regime.get(symbol)
        if reg is None or reg["days_left"] <= 0:
            reg = self._new_regime()
            self.regime[symbol] = reg
        reg["days_left"] -= 1

        ret = self.rng.gauss(reg["drift"], reg["vol"])
        new_price = max(1.0, price * (1 + ret))
        spread = abs(self.rng.gauss(0, reg["vol"] / 2))
        high = new_price * (1 + spread)
        low = new_price * (1 - spread)
        open_ = low + (high - low) * self.rng.random()
        vol = self.rng.uniform(5e5, 5e6)
        if self.rng.random() < 0.06:  # occasional volume spike
            vol *= self.rng.uniform(2.0, 5.0)
        candle = {
            "t": ts,
            "o": round(open_, 4),
            "h": round(high, 4),
            "l": round(low, 4),
            "c": round(new_price, 4),
            "v": round(vol, 0),
        }
        return new_price, candle

    def _seed_series(self) -> None:
        base_ts = int(time.time()) - _SEED_DAYS * 86400
        for sym in self.symbols:
            price = self.rng.uniform(20.0, 480.0)
            rows: list[dict[str, Any]] = []
            for i in range(_SEED_DAYS):
                price, candle = self._step(sym, price, base_ts + i * 86400)
                rows.append(candle)
            self.history[sym] = rows
        log.info("SimulatedMarket seeded %d symbols × %d days",
                 len(self.symbols), _SEED_DAYS)

    # ---- public API ----

    def advance(self) -> None:
        """Append one new daily bar to every symbol."""
        self.days_advanced += 1
        for sym in self.symbols:
            rows = self.history.get(sym)
            if not rows:
                continue
            last = rows[-1]
            price, candle = self._step(sym, float(last["c"]), int(last["t"]) + 86400)
            rows.append(candle)
            if len(rows) > _MAX_HISTORY:
                del rows[: len(rows) - _MAX_HISTORY]

    def candles(self, symbol: str, days: int = 180) -> list[dict[str, Any]]:
        rows = self.history.get(symbol, [])
        if days and len(rows) > days:
            return rows[-days:]
        return list(rows)

    def last_close(self, symbol: str) -> float | None:
        rows = self.history.get(symbol)
        return float(rows[-1]["c"]) if rows else None


# ---- module-level singleton ----

_SIM: SimulatedMarket | None = None


def get_simulator(symbols: list[str] | None = None) -> SimulatedMarket:
    """Return the live simulator, creating it on first use."""
    global _SIM
    if _SIM is None:
        if not symbols:
            from astra.data.universe import FALLBACK_SP500
            symbols = list(FALLBACK_SP500)
        _SIM = SimulatedMarket(symbols)
    return _SIM


def reset_simulator(symbols: list[str] | None = None) -> SimulatedMarket:
    """Discard the current simulated market and start a fresh one."""
    global _SIM
    _SIM = None
    return get_simulator(symbols)
