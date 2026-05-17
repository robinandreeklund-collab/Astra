"""Intraday historical-replay tests — the critical guarantee: NO LOOKAHEAD."""

from __future__ import annotations

import pytest

from astra.data.simulator import HistoricalMarket, get_market, reset_market

pytestmark = pytest.mark.asyncio

# A fixed Monday 00:00 UTC.
_BASE = 1_700_000_000 - (1_700_000_000 % 86400)
_HOURS = (14, 15, 16, 17, 18, 19, 20)  # ~7 market hours per day


def _hourly(n_days: int = 400) -> list[dict]:
    """Hourly bars; every bar's close == its day index, so a lookahead
    leak (a future day's close) is trivially detectable."""
    rows = []
    for d in range(n_days):
        for h in _HOURS:
            t = _BASE + d * 86400 + h * 3600
            rows.append({"t": t, "o": float(d), "h": float(d) + 0.5,
                         "l": float(d) - 0.5, "c": float(d), "v": 1000})
    return rows


async def test_resamples_hourly_to_daily():
    m = HistoricalMarket({"AAA": _hourly(300)}, replay_days=100)
    # 300 days of hourly → 300 daily bars.
    assert len(m.daily["AAA"]) == 300
    # Each daily close == that day's index (last hourly bar of the day).
    assert m.daily["AAA"][5]["c"] == 5.0


async def test_analysis_uses_only_completed_days():
    """candles() must return only days STRICTLY BEFORE the cursor's day."""
    m = HistoricalMarket({"AAA": _hourly(400)}, replay_days=100)
    for _ in range(200):  # advance many hourly bars
        m.advance()
        daily = m.candles("AAA", days=9999)
        cur = m.current_price("AAA")
        # The newest analysis day is YESTERDAY; today is never in candles().
        assert daily[-1]["c"] == cur - 1
        # No analysis candle may equal or exceed the current day.
        assert all(c["c"] < cur for c in daily)


async def test_advance_one_hour_and_new_day_flag():
    m = HistoricalMarket({"AAA": _hourly(400)}, replay_days=100)
    # Advancing within market hours of one day → not a new day.
    m.advance()
    first_day_flag = m.is_new_day
    # Step until the day flips.
    flipped = False
    for _ in range(20):
        m.advance()
        if m.is_new_day:
            flipped = True
            break
    assert flipped  # crossing into the next day is detected


async def test_intraday_ticks_per_day():
    """A trading day should contain multiple (hourly) replay ticks."""
    m = HistoricalMarket({"AAA": _hourly(400)}, replay_days=100)
    day0 = m.current_day
    ticks_in_day = 0
    for _ in range(20):
        m.advance()
        if m.current_day == day0:
            ticks_in_day += 1
        else:
            break
    assert ticks_in_day >= 3  # several ticks before the day rolls over


async def test_reaches_end():
    m = HistoricalMarket({"AAA": _hourly(200)}, replay_days=50)
    for _ in range(5000):
        m.advance()
    assert m.at_end is True
    assert m.progress == 1.0


async def test_current_price_is_intraday():
    m = HistoricalMarket({"AAA": _hourly(400)}, replay_days=100)
    m.advance()
    # current_price == the cursor day's close (day index).
    assert m.current_price("AAA") == float(m.candles("AAA")[-1]["c"] + 1)


async def test_status_shape():
    m = HistoricalMarket({"AAA": _hourly(400)}, replay_days=80)
    m.advance()
    s = m.status()
    assert s["active"] is True
    assert s["total"] == 80
    assert "datetime" in s
    assert "new_day" in s


async def test_get_and_reset_market():
    reset_market()
    assert get_market() is None
