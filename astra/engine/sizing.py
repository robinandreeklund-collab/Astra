"""Position sizing.

Combines three constraints into a single dollar target for a new BUY:

  1. Target weight   — `target_position_pct` of equity (concentration goal).
  2. Risk budget     — size so a stop-loss exit costs at most
                       `risk_per_trade_pct` of equity. With a 5% stop and a
                       2% risk budget, the risk cap is 0.02/0.05 = 40% of
                       equity; the target weight (10%) usually binds first,
                       but on a tight stop the risk cap protects the account.
  3. Volatility scale — shrink the position when the stock's ATR is high so
                        every holding carries comparable risk.

The result is also clamped by the hard `max_position_pct` cap and by
available cash, and multiplied by the decision confidence.
"""

from __future__ import annotations

from astra.config import Settings


def compute_buy_value(
    equity: float,
    cash: float,
    confidence: float,
    atr_pct: float | None,
    settings: Settings,
) -> float:
    """Return the dollar value to deploy into a new position (0 = skip)."""
    if equity <= 0 or cash <= 0:
        return 0.0
    confidence = max(0.0, min(1.0, confidence))

    base = equity * settings.target_position_pct
    risk_cap = (equity * settings.risk_per_trade_pct) / max(settings.stop_loss_pct, 0.01)

    # Volatility scaling: a "normal" daily ATR is ~2%. A 4%-ATR stock gets
    # half the size; a calm 1%-ATR stock is allowed up to the base (capped 1.0).
    vol_factor = 1.0
    if atr_pct and atr_pct > 0:
        vol_factor = min(1.0, 0.02 / atr_pct)

    target = min(base, risk_cap) * confidence * vol_factor
    target = min(target, equity * settings.max_position_pct)
    target = min(target, cash * 0.95)
    return max(0.0, target)


def atr_pct_from_indicators(tech: dict) -> float | None:
    """Derive ATR as a fraction of price from the technical-indicator block."""
    atr = tech.get("atr14")
    last = tech.get("last_close")
    if isinstance(atr, (int, float)) and isinstance(last, (int, float)) and last > 0:
        return float(atr) / float(last)
    return None
