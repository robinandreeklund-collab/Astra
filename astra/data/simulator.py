"""Historical-replay market.

Simulation mode replays the last year of REAL daily price data, one
trading day per engine tick. The bot trades against actual market history
exactly as it would live — with one hard guarantee:

    THE BOT NEVER SEES THE FUTURE.

A cursor marks the current replay day. `candles()` returns only bars at or
before the cursor; `advance()` reveals the next day. There is no lookahead
anywhere. This makes the replay an honest out-of-sample test: train the
models on older data, then watch the trained bot trade a year it has never
seen and cannot peek into.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

# How many trading days of history are replayed (≈ one year).
DEFAULT_REPLAY_DAYS = 252
# Minimum bars a symbol needs to be included (warm-up for indicators).
_MIN_BARS = 150


class HistoricalMarket:
    """Replays real daily candles forward, revealing one day at a time."""

    def __init__(
        self,
        history: dict[str, list[dict[str, Any]]],
        replay_days: int = DEFAULT_REPLAY_DAYS,
    ) -> None:
        # Each symbol's full real history, sorted by timestamp.
        self.history: dict[str, list[dict[str, Any]]] = {
            s: sorted(rows, key=lambda c: int(c["t"]))
            for s, rows in history.items()
        }
        # A shared timeline of every distinct trading day.
        self.timeline: list[int] = sorted(
            {int(c["t"]) for rows in self.history.values() for c in rows}
        )
        self.replay_days = replay_days
        # The cursor starts `replay_days` from the end; everything before it
        # is warm-up history the bot may use, everything after is the future
        # it must NOT see.
        self.start_cursor = max(0, len(self.timeline) - replay_days - 1)
        self.cursor = self.start_cursor

    @property
    def current_ts(self) -> int:
        if not self.timeline:
            return 0
        return self.timeline[min(self.cursor, len(self.timeline) - 1)]

    @property
    def at_end(self) -> bool:
        return self.cursor >= len(self.timeline) - 1

    @property
    def total_replay_days(self) -> int:
        return max(1, len(self.timeline) - 1 - self.start_cursor)

    @property
    def day_number(self) -> int:
        return self.cursor - self.start_cursor

    @property
    def progress(self) -> float:
        return min(1.0, self.day_number / self.total_replay_days)

    def current_date(self) -> str:
        if not self.timeline:
            return ""
        return datetime.fromtimestamp(
            self.current_ts, tz=timezone.utc).date().isoformat()

    def advance(self) -> None:
        """Reveal the next trading day."""
        if self.cursor < len(self.timeline) - 1:
            self.cursor += 1

    def candles(self, symbol: str, days: int = 180) -> list[dict[str, Any]]:
        """Analysis history — only days STRICTLY BEFORE the cursor.

        The bot computes every indicator and signal off completed trading
        days. The current (cursor) day's full candle is deliberately NOT
        here — the bot must not see today's high/low/close as if the day
        were already over. The only thing it knows about 'today' is the
        current price (see current_price)."""
        ts = self.current_ts
        rows = [c for c in self.history.get(symbol, []) if int(c["t"]) < ts]
        if days and len(rows) > days:
            return rows[-days:]
        return rows

    def current_price(self, symbol: str) -> float | None:
        """The price RIGHT NOW — the close of the cursor day (or the last
        bar at or before it). This is all the bot knows about today; it is
        used to fill orders and mark positions, never to compute signals."""
        ts = self.current_ts
        last = None
        for c in self.history.get(symbol, []):
            if int(c["t"]) <= ts:
                last = c
            else:
                break
        return float(last["c"]) if last else None

    # Backwards-compatible alias.
    def last_close(self, symbol: str) -> float | None:
        return self.current_price(symbol)

    def status(self) -> dict[str, Any]:
        return {
            "active": True,
            "day": self.day_number,
            "total": self.total_replay_days,
            "date": self.current_date(),
            "progress": round(self.progress, 3),
            "at_end": self.at_end,
            "symbols": len(self.history),
        }


# ---- module-level singleton ----

_MARKET: HistoricalMarket | None = None


def get_market() -> HistoricalMarket | None:
    return _MARKET


def reset_market() -> None:
    global _MARKET
    _MARKET = None


def is_ready() -> bool:
    return _MARKET is not None and bool(_MARKET.timeline)


async def load_market(
    symbols: list[str],
    cache: Any,
    replay_days: int = DEFAULT_REPLAY_DAYS,
) -> HistoricalMarket:
    """Load real 5y history for the universe and arm the replay.

    Uses force_real so it fetches actual market data even though sim mode
    is on. After a training run the 5y candles are already cached, so this
    is fast."""
    global _MARKET
    from astra.data.yahoo import fetch_daily_candles

    history: dict[str, list[dict[str, Any]]] = {}
    for sym in symbols:
        try:
            rows = await fetch_daily_candles(
                sym, days=1825, cache=cache, force_real=True)
            if len(rows) >= _MIN_BARS:
                history[sym] = rows
        except Exception as e:
            log.debug("replay history fetch failed %s: %s", sym, e)

    _MARKET = HistoricalMarket(history, replay_days)
    log.info(
        "Historical replay armed: %d symbols, replaying %d days from %s",
        len(history), _MARKET.total_replay_days, _MARKET.current_date(),
    )
    return _MARKET
