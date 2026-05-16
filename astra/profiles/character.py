"""Layer 1 — stock character.

Classifies a stock from its price history alone (no trades needed), so every
symbol has a usable profile from day one. The high-value output is the
archetype: does this stock TREND (don't fade dips) or MEAN-REVERT (fade
extremes) or just chop sideways?

Metrics:
  atr_pct          — average true range as a fraction of price (volatility)
  efficiency_ratio — Kaufman ER: |net move| / |path length| over ~30 bars.
                     High = directional/trending, low = choppy.
  trend_score      — lag-1 autocorrelation of daily returns, scaled. Positive
                     means momentum persists; negative means it reverses.
  gap_freq         — fraction of sessions opening >2% away from prior close.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from astra.signals.technical import atr, candles_to_df


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _vol_tier(atr_pct: float | None) -> str:
    if atr_pct is None:
        return "unknown"
    if atr_pct < 0.015:
        return "low"
    if atr_pct < 0.030:
        return "medium"
    return "high"


def _archetype(trend_score: float, efficiency_ratio: float) -> str:
    if efficiency_ratio > 0.40 and trend_score > 0.03:
        return "trender"
    if trend_score < -0.03:
        return "mean-reverter"
    return "choppy"


def compute_character(candles: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the Layer-1 character dict for a stock from its daily candles."""
    df = candles_to_df(candles)
    if len(df) < 40:
        return {"classified": False, "reason": "insufficient_data", "bars": len(df)}

    close = df["close"].astype(float)
    rets = close.pct_change().dropna()

    # Lag-1 autocorrelation of returns → trend persistence.
    autocorr = float(rets.autocorr(lag=1)) if len(rets) > 5 else 0.0
    if np.isnan(autocorr):
        autocorr = 0.0
    trend_score = max(-1.0, min(1.0, autocorr * 5.0))

    # Kaufman efficiency ratio over the last ~30 bars.
    window = min(30, len(close) - 1)
    seg = close.iloc[-(window + 1):]
    net = abs(float(seg.iloc[-1]) - float(seg.iloc[0]))
    path = float(seg.diff().abs().sum())
    efficiency_ratio = (net / path) if path > 0 else 0.0

    # ATR as a fraction of price.
    atr_series = atr(df, 14)
    last_close = float(close.iloc[-1])
    atr_val = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else None
    atr_pct = (atr_val / last_close) if (atr_val and last_close > 0) else None

    # Gap frequency.
    prev_close = close.shift(1)
    gaps = (df["open"].astype(float) - prev_close).abs() / prev_close
    gap_freq = float((gaps.dropna() > 0.02).mean()) if len(gaps.dropna()) else 0.0

    return {
        "classified": True,
        "bars": int(len(df)),
        "atr_pct": round(atr_pct, 5) if atr_pct is not None else None,
        "vol_tier": _vol_tier(atr_pct),
        "efficiency_ratio": round(efficiency_ratio, 3),
        "trend_score": round(trend_score, 3),
        "archetype": _archetype(trend_score, efficiency_ratio),
        "gap_freq": round(gap_freq, 3),
        "updated_at": _now(),
    }
