"""Technical indicators computed from candle data.

We avoid external TA libs; numpy/pandas-only implementations keep deps light.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def candles_to_df(candles: list[dict[str, Any]]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(candles)
    if "t" in df.columns:
        df = df.rename(columns={"t": "ts", "o": "open", "h": "high",
                                "l": "low", "c": "close", "v": "volume"})
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast = ema(close, 12)
    slow = ema(close, 26)
    macd_line = fast - slow
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h_l = df["high"] - df["low"]
    h_c = (df["high"] - df["close"].shift()).abs()
    l_c = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_indicators(candles: list[dict[str, Any]]) -> dict[str, Any]:
    df = candles_to_df(candles)
    if len(df) < 30:
        return {"available": False, "reason": "insufficient_data", "bars": len(df)}

    close = df["close"]
    rsi14 = rsi(close, 14)
    macd_line, macd_sig, macd_hist = macd(close)
    bb_up, bb_mid, bb_lo = bollinger(close, 20, 2.0)
    sma50 = close.rolling(50).mean() if len(df) >= 50 else pd.Series(dtype=float)
    sma200 = close.rolling(200).mean() if len(df) >= 200 else pd.Series(dtype=float)
    atr14 = atr(df, 14)

    last = -1
    last_close = float(close.iloc[last])
    bb_up_last = float(bb_up.iloc[last]) if not pd.isna(bb_up.iloc[last]) else None
    bb_lo_last = float(bb_lo.iloc[last]) if not pd.isna(bb_lo.iloc[last]) else None

    bb_pos = None
    if bb_up_last is not None and bb_lo_last is not None and bb_up_last > bb_lo_last:
        bb_pos = (last_close - bb_lo_last) / (bb_up_last - bb_lo_last)

    sma_cross = None
    if len(df) >= 200:
        s50 = float(sma50.iloc[last])
        s200 = float(sma200.iloc[last])
        sma_cross = "golden" if s50 > s200 else "death"

    macd_state = None
    if len(df) >= 35:
        macd_state = "bullish" if float(macd_hist.iloc[last]) > 0 else "bearish"

    pct_change_1d = float((close.iloc[-1] / close.iloc[max(-2, -len(close))] - 1) * 100) if len(close) >= 2 else 0.0
    pct_change_5d = float((close.iloc[-1] / close.iloc[max(-6, -len(close))] - 1) * 100) if len(close) >= 6 else 0.0
    avg_vol = float(df["volume"].tail(20).mean()) if "volume" in df.columns and not df["volume"].tail(20).empty else 0.0
    last_vol = float(df["volume"].iloc[-1]) if "volume" in df.columns and len(df) > 0 else 0.0
    vol_ratio = (last_vol / avg_vol) if avg_vol > 0 else None

    try:
        last_bar_ts = int(df["ts"].iloc[-1].timestamp())
    except Exception:
        last_bar_ts = 0

    return {
        "available": True,
        "bars": int(len(df)),
        "last_bar_ts": last_bar_ts,
        "last_close": last_close,
        "rsi14": _safe_float(rsi14.iloc[last]),
        "macd_state": macd_state,
        "macd_hist": _safe_float(macd_hist.iloc[last]),
        "bb_position": _safe_float(bb_pos),
        "sma_cross": sma_cross,
        "atr14": _safe_float(atr14.iloc[last]),
        "pct_change_1d": pct_change_1d,
        "pct_change_5d": pct_change_5d,
        "volume_ratio": _safe_float(vol_ratio),
    }


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        if pd.isna(v) or not np.isfinite(v):
            return None
        return v
    except (TypeError, ValueError):
        return None
