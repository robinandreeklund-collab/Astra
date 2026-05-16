"""Per-stock adaptive profiles.

Each stock carries a living profile that adapts to its own trade history:

  Layer 1  character        — volatility / trend personality from price history
  Layer 2  signal_efficacy  — per-stock, per-signal Bayesian win rates
  Layer 3  ledger           — the realized trade record on this stock

Derived from those: an edge score, a state machine (NEUTRAL / PROVEN /
PROBATION / BENCHED), a conviction multiplier for sizing, adaptive
volatility-scaled stops, and a Thompson-sampling bandit weight.
"""

from astra.profiles.character import compute_character
from astra.profiles.profile import (
    GLOBAL_SYMBOL,
    StockProfile,
    extract_signals,
    global_signal_rate,
    load_all_profiles,
    load_global,
    load_profile,
    save_profile,
)

__all__ = [
    "compute_character",
    "StockProfile",
    "extract_signals",
    "global_signal_rate",
    "GLOBAL_SYMBOL",
    "load_profile",
    "save_profile",
    "load_all_profiles",
    "load_global",
]
