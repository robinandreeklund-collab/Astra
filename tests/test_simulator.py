"""Historical-replay market tests — the critical guarantee is NO LOOKAHEAD."""

from __future__ import annotations

import pytest

from astra.data.simulator import HistoricalMarket, get_market, reset_market

pytestmark = pytest.mark.asyncio


def _history(symbol: str, n: int = 600) -> list[dict]:
    """Deterministic candle history; close == day index so lookahead is
    trivially detectable (a future close would exceed the cursor index)."""
    base = 1_500_000_000
    return [
        {"t": base + i * 86400, "o": i, "h": i, "l": i, "c": float(i), "v": 1000}
        for i in range(n)
    ]


async def test_replay_never_reveals_the_future():
    """The core guarantee: analysis candles are STRICTLY before the cursor —
    today's full candle is not visible, only the current price is."""
    m = HistoricalMarket({"AAA": _history("AAA", 600)}, replay_days=100)
    for _ in range(40):
        m.advance()
        visible = m.candles("AAA", days=9999)
        # close == day index; the newest analysis candle is YESTERDAY.
        assert visible[-1]["c"] == float(m.cursor - 1)
        # No analysis candle may be dated at or after the cursor day.
        assert all(int(c["t"]) < m.current_ts for c in visible)
        # The current price IS the cursor day's close.
        assert m.current_price("AAA") == float(m.cursor)


async def test_cursor_starts_a_year_back():
    m = HistoricalMarket({"AAA": _history("AAA", 600)}, replay_days=252)
    # 600 bars, replay 252 → cursor starts at 600-253 = 347.
    assert m.start_cursor == 347
    assert m.total_replay_days == 252


async def test_advance_moves_one_day():
    m = HistoricalMarket({"AAA": _history("AAA", 400)}, replay_days=100)
    before = m.current_ts
    m.advance()
    assert m.current_ts - before == 86400
    assert m.day_number == 1


async def test_reaches_end():
    m = HistoricalMarket({"AAA": _history("AAA", 300)}, replay_days=50)
    for _ in range(100):
        m.advance()
    assert m.at_end is True
    assert m.progress == 1.0


async def test_warmup_history_available_before_cursor():
    """The bot can see plenty of trailing history for indicators."""
    m = HistoricalMarket({"AAA": _history("AAA", 600)}, replay_days=100)
    visible = m.candles("AAA", days=9999)
    # At the start cursor there should be ~500 warm-up bars.
    assert len(visible) > 400


async def test_candles_respects_days_limit():
    m = HistoricalMarket({"AAA": _history("AAA", 600)}, replay_days=100)
    assert len(m.candles("AAA", days=60)) == 60


async def test_status_shape():
    m = HistoricalMarket({"AAA": _history("AAA", 400)}, replay_days=80)
    m.advance()
    s = m.status()
    assert s["active"] is True
    assert s["total"] == 80
    assert s["day"] == 1
    assert "date" in s


async def test_get_and_reset_market():
    reset_market()
    assert get_market() is None
