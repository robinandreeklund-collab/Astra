"""Per-stock adaptive profile tests."""

from __future__ import annotations

import random

import pytest

from astra.profiles.character import compute_character
from astra.profiles.profile import (
    GLOBAL_SYMBOL,
    StockProfile,
    all_global_rates,
    extract_signals,
    global_signal_rate,
    load_all_profiles,
    load_global,
    load_profile,
    save_profile,
)

pytestmark = pytest.mark.asyncio


def _candles(n=120, drift=0.002, vol=0.015, seed=1):
    r = random.Random(seed)
    p = 100.0
    base = 1_700_000_000
    rows = []
    for i in range(n):
        p = max(1.0, p * (1 + r.gauss(drift, vol)))
        rows.append({"t": base + i * 86400, "o": p * 0.99, "h": p * 1.02,
                     "l": p * 0.98, "c": p, "v": 1_000_000})
    return rows


# ---- Layer 1: character ----

async def test_character_classifies_with_enough_data():
    ch = compute_character(_candles(120, drift=0.004))
    assert ch["classified"] is True
    assert ch["archetype"] in ("trender", "mean-reverter", "choppy")
    assert ch["vol_tier"] in ("low", "medium", "high")
    assert ch["atr_pct"] is not None


async def test_character_insufficient_data():
    ch = compute_character(_candles(20))
    assert ch["classified"] is False


# ---- extract_signals ----

async def test_extract_signals_reads_snapshot():
    snap = {
        "technical": {"rsi14": 25, "macd_state": "bullish", "bb_position": 0.1,
                      "sma_cross": "golden", "volume_ratio": 2.0, "pct_change_5d": 5},
        "insider": {"net_direction": "buying"},
        "recommendations": {"trend": "upgrading"},
        "sentiment": {"company_news_score": 0.7},
    }
    tags = extract_signals(snap)
    for expected in ("rsi_oversold", "macd_bull", "bb_lower", "golden_cross",
                     "volume_spike", "momentum_up", "insider_buying",
                     "analyst_upgrading", "news_positive"):
        assert expected in tags


# ---- Layer 3: ledger + derived ----

async def test_record_trade_outcome_updates_ledger():
    p = StockProfile(symbol="AAA")
    p.record_trade_outcome(win=True, pnl=100, hold_minutes=120, entry_signals=["macd_bull"])
    p.record_trade_outcome(win=False, pnl=-40, hold_minutes=60, entry_signals=["macd_bull"])
    assert p.trades == 2
    assert p.wins == 1 and p.losses == 1
    assert p.realized_pnl == 60
    assert abs(p.profit_factor() - 2.5) < 0.01  # 100 / 40
    assert p.win_rate() == 0.5


async def test_streak_tracking():
    p = StockProfile(symbol="AAA")
    for _ in range(3):
        p.record_trade_outcome(True, 10, 1, [])
    assert p.current_streak == 3
    p.record_trade_outcome(False, -5, 1, [])
    assert p.current_streak == -1


async def test_edge_score_neutral_with_few_trades():
    p = StockProfile(symbol="AAA")
    p.record_trade_outcome(True, 100, 1, [])
    assert p.edge_score() == 0.0  # < 3 trades


async def test_state_benched_on_loss_streak():
    p = StockProfile(symbol="AAA")
    for _ in range(5):
        p.record_trade_outcome(False, -20, 1, [])
    assert p.state() == "BENCHED"
    assert p.conviction_multiplier() == 0.30


async def test_state_proven_on_strong_edge():
    p = StockProfile(symbol="AAA")
    for _ in range(8):
        p.record_trade_outcome(True, 100, 1, [])
    assert p.state() == "PROVEN"
    assert p.conviction_multiplier() > 1.0


# ---- adaptive stop ----

async def test_adaptive_stop_scales_with_atr():
    low = StockProfile(symbol="L", character={"atr_pct": 0.01})
    high = StockProfile(symbol="H", character={"atr_pct": 0.05})
    # default stop 5%
    assert low.adaptive_stop_pct(0.05) < high.adaptive_stop_pct(0.05)
    # no character → default unchanged
    assert StockProfile(symbol="N").adaptive_stop_pct(0.05) == 0.05


# ---- signal efficacy + shrinkage ----

async def test_signal_win_rate_shrinks_toward_global():
    p = StockProfile(symbol="AAA")
    # No stock data → returns the global prior
    assert abs(p.signal_win_rate("macd_bull", global_rate=0.7) - 0.7) < 0.01
    # Stock data pulls it away from the prior
    for _ in range(10):
        p.record_trade_outcome(True, 10, 1, ["macd_bull"])
    rate = p.signal_win_rate("macd_bull", global_rate=0.3)
    assert rate > 0.3  # stock evidence (all wins) lifts it above the prior


# ---- bandit ----

async def test_bandit_sample_in_range():
    rng = random.Random(0)
    p = StockProfile(symbol="AAA")
    for _ in range(20):
        v = p.bandit_sample(rng)
        assert 0.0 <= v <= 1.0


async def test_bandit_favours_winners():
    rng = random.Random(0)
    winner = StockProfile(symbol="W")
    loser = StockProfile(symbol="L")
    for _ in range(15):
        winner.record_trade_outcome(True, 10, 1, [])
        loser.record_trade_outcome(False, -10, 1, [])
    w_avg = sum(winner.bandit_sample(rng) for _ in range(200)) / 200
    l_avg = sum(loser.bandit_sample(rng) for _ in range(200)) / 200
    assert w_avg > l_avg


# ---- serialisation ----

async def test_profile_round_trip():
    p = StockProfile(symbol="AAA", character={"archetype": "trender"})
    p.record_trade_outcome(True, 50, 30, ["rsi_oversold", "macd_bull"])
    d = p.to_dict()
    p2 = StockProfile.from_dict(d)
    assert p2.symbol == "AAA"
    assert p2.trades == 1
    assert p2.signal_efficacy["macd_bull"]["w"] == 1
    assert p2.character["archetype"] == "trender"


# ---- persistence ----

async def test_persistence_round_trip(memory_db):
    p = StockProfile(symbol="NVDA")
    p.record_trade_outcome(True, 200, 90, ["momentum_up"])
    await save_profile(memory_db, p)

    loaded = await load_profile(memory_db, "NVDA")
    assert loaded.trades == 1
    assert loaded.realized_pnl == 200

    allp = await load_all_profiles(memory_db)
    assert "NVDA" in allp


async def test_global_rates(memory_db):
    g = await load_global(memory_db)
    assert g.symbol == GLOBAL_SYMBOL
    g.record_trade_outcome(True, 10, 1, ["macd_bull"])
    g.record_trade_outcome(True, 10, 1, ["macd_bull"])
    rates = all_global_rates(g)
    assert rates["macd_bull"] > 0.5  # two wins lift it above neutral


# ---- card ----

async def test_card_text_for_new_and_traded():
    fresh = StockProfile(symbol="AAA", character=compute_character(_candles()))
    card = fresh.card({})
    assert "AAA PROFILE" in card
    assert "No trade history" in card

    traded = StockProfile(symbol="BBB")
    for _ in range(6):
        traded.record_trade_outcome(True, 50, 60, ["macd_bull"], r_multiple=1.5)
    card2 = traded.card({"macd_bull": 0.5})
    assert "Record:" in card2
    assert "PROVEN" in card2
