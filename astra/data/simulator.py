"""Historical-replay market — intraday.

Simulation mode replays the last year of REAL market data at HOURLY
resolution (~7 ticks per trading day). The bot trades against actual
history exactly as it would live, with one hard guarantee:

    THE BOT NEVER SEES THE FUTURE.

Two clocks:
  * The cursor moves one HOURLY bar per engine tick — the price path.
  * Analysis (RSI, MACD, character, profiles) runs on COMPLETED DAILY
    bars only — days strictly before the cursor's date.

So the bot analyses finished trading days, knows only the current
intraday price for "today", and exits (stop-loss, take-profit, trailing)
are checked every hour against the live price — realistic, not just at
the daily close. No bar dated at or after the cursor is ever visible.
"""

from __future__ import annotations

import bisect
import logging
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_REPLAY_DAYS = 252        # ≈ one trading year
_MIN_HOURLY_BARS = 400           # a symbol needs enough intraday history


def _day_key(ts: int) -> int:
    """UTC date as YYYYMMDD — US market hours all fall on one UTC date."""
    d = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    return d.year * 10000 + d.month * 100 + d.day


def _day_start_ts(day_key: int) -> int:
    y, m, d = day_key // 10000, (day_key // 100) % 100, day_key % 100
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp())


def _resample_daily(hourly: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate hourly bars into completed daily bars."""
    by_day: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
    for b in hourly:
        by_day.setdefault(_day_key(int(b["t"])), []).append(b)
    daily: list[dict[str, Any]] = []
    for day, bars in by_day.items():
        daily.append({
            "t": _day_start_ts(day),
            "day": day,
            "o": float(bars[0]["o"]),
            "h": max(float(b["h"]) for b in bars),
            "l": min(float(b["l"]) for b in bars),
            "c": float(bars[-1]["c"]),
            "v": sum(float(b["v"]) for b in bars),
        })
    return daily


class HistoricalMarket:
    """Replays real hourly data forward, one hour per tick."""

    def __init__(
        self,
        hourly: dict[str, list[dict[str, Any]]],
        replay_days: int = DEFAULT_REPLAY_DAYS,
    ) -> None:
        self.hourly: dict[str, list[dict[str, Any]]] = {
            s: sorted(rows, key=lambda c: int(c["t"]))
            for s, rows in hourly.items()
        }
        # Per-symbol timestamp index for fast price lookups.
        self._hts: dict[str, list[int]] = {
            s: [int(c["t"]) for c in rows] for s, rows in self.hourly.items()
        }
        # Daily bars (resampled) per symbol — the analysis history.
        self.daily: dict[str, list[dict[str, Any]]] = {
            s: _resample_daily(rows) for s, rows in self.hourly.items()
        }
        # The shared hourly timeline + each bar's trading date.
        self.timeline: list[int] = sorted(
            {int(c["t"]) for rows in self.hourly.values() for c in rows})
        self.dates: list[int] = [_day_key(t) for t in self.timeline]
        self.distinct_days: list[int] = sorted(set(self.dates))

        self.replay_days = replay_days
        # Start the cursor `replay_days` trading days from the end.
        if len(self.distinct_days) > replay_days + 1:
            start_day = self.distinct_days[-(replay_days + 1)]
            self.start_cursor = bisect.bisect_left(self.dates, start_day)
        else:
            self.start_cursor = 0
        self.cursor = self.start_cursor
        self._is_new_day = True

    # ---- clock ----

    @property
    def current_ts(self) -> int:
        if not self.timeline:
            return 0
        return self.timeline[min(self.cursor, len(self.timeline) - 1)]

    @property
    def current_day(self) -> int:
        if not self.dates:
            return 0
        return self.dates[min(self.cursor, len(self.dates) - 1)]

    @property
    def is_new_day(self) -> bool:
        """True when the last advance() crossed into a new trading day."""
        return self._is_new_day

    @property
    def at_end(self) -> bool:
        return self.cursor >= len(self.timeline) - 1

    @property
    def total_replay_days(self) -> int:
        start_day = self.dates[self.start_cursor] if self.dates else 0
        return max(1, sum(1 for d in self.distinct_days if d >= start_day) - 1)

    @property
    def day_number(self) -> int:
        start_day = self.dates[self.start_cursor] if self.dates else 0
        return sum(1 for d in self.distinct_days
                   if start_day <= d <= self.current_day) - 1

    @property
    def progress(self) -> float:
        return min(1.0, self.day_number / self.total_replay_days)

    def current_datetime(self) -> str:
        if not self.timeline:
            return ""
        return datetime.fromtimestamp(
            self.current_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

    def advance(self) -> None:
        """Reveal the next hourly bar."""
        if self.cursor < len(self.timeline) - 1:
            prev_day = self.dates[self.cursor]
            self.cursor += 1
            self._is_new_day = self.dates[self.cursor] != prev_day
        else:
            self._is_new_day = False

    # ---- data access (no lookahead) ----

    def candles(self, symbol: str, days: int = 180) -> list[dict[str, Any]]:
        """Completed DAILY bars strictly before the cursor's date.

        Today's bar is never here — the bot analyses finished days only."""
        today = self.current_day
        rows = [c for c in self.daily.get(symbol, []) if c["day"] < today]
        if days and len(rows) > days:
            return rows[-days:]
        return rows

    def current_price(self, symbol: str) -> float | None:
        """The live intraday price — the close of the symbol's most recent
        hourly bar at or before the cursor. All the bot knows about 'today'."""
        ts = self.current_ts
        idx = bisect.bisect_right(self._hts.get(symbol, []), ts) - 1
        if idx < 0:
            return None
        return float(self.hourly[symbol][idx]["c"])

    # Alias kept for callers that used the daily-replay name.
    def last_close(self, symbol: str) -> float | None:
        return self.current_price(symbol)

    def status(self) -> dict[str, Any]:
        return {
            "active": True,
            "day": self.day_number + 1,          # 1-indexed for display
            "total": self.total_replay_days,
            "datetime": self.current_datetime(),
            "progress": round(self.progress, 3),
            "at_end": self.at_end,
            "new_day": self._is_new_day,
            "symbols": len(self.hourly),
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
    """Load real hourly history for the universe and arm the replay."""
    global _MARKET
    from astra.data.yahoo import fetch_intraday_candles

    hourly: dict[str, list[dict[str, Any]]] = {}
    for sym in symbols:
        try:
            rows = await fetch_intraday_candles(sym, cache=cache)
            if len(rows) >= _MIN_HOURLY_BARS:
                hourly[sym] = rows
        except Exception as e:
            log.debug("replay intraday fetch failed %s: %s", sym, e)

    _MARKET = HistoricalMarket(hourly, replay_days)
    log.info(
        "Intraday replay armed: %d symbols, %d hourly bars, replaying %d days",
        len(hourly), len(_MARKET.timeline), _MARKET.total_replay_days,
    )
    return _MARKET
