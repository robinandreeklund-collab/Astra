"""The StockProfile — Layers 2 & 3, derived metrics, bandit, persistence."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

# A synthetic symbol holding the cross-stock signal aggregate. Per-stock
# signal win rates shrink toward this; it in turn shrinks toward 0.5.
GLOBAL_SYMBOL = "__GLOBAL__"

# Per-trade decay so recent results dominate (regime change protection).
_SIGNAL_DECAY = 0.95
_BANDIT_DECAY = 0.97
# Pseudo-count strength for Bayesian shrinkage of per-stock signal rates.
_STOCK_PRIOR_STRENGTH = 4.0
_GLOBAL_PRIOR_STRENGTH = 6.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# All signal tags the profile tracks. Mirrors the heuristic's signal set.
SIGNAL_TAGS = (
    "rsi_oversold", "rsi_overbought", "macd_bull", "macd_bear",
    "bb_lower", "bb_upper", "golden_cross", "death_cross",
    "volume_spike", "momentum_up", "momentum_down",
    "insider_buying", "insider_selling",
    "analyst_upgrading", "analyst_downgrading",
    "news_positive", "news_negative",
)


def extract_signals(snapshot: dict[str, Any]) -> list[str]:
    """Pull the active signal tags from a stored signal snapshot."""
    tags: list[str] = []
    tech = snapshot.get("technical") or {}

    rsi = tech.get("rsi14")
    if isinstance(rsi, (int, float)):
        if rsi < 30:
            tags.append("rsi_oversold")
        elif rsi > 70:
            tags.append("rsi_overbought")

    macd = tech.get("macd_state")
    if macd == "bullish":
        tags.append("macd_bull")
    elif macd == "bearish":
        tags.append("macd_bear")

    bb = tech.get("bb_position")
    if isinstance(bb, (int, float)):
        if bb < 0.2:
            tags.append("bb_lower")
        elif bb > 0.8:
            tags.append("bb_upper")

    sma = tech.get("sma_cross")
    if sma == "golden":
        tags.append("golden_cross")
    elif sma == "death":
        tags.append("death_cross")

    vr = tech.get("volume_ratio")
    if isinstance(vr, (int, float)) and vr >= 1.5:
        tags.append("volume_spike")

    ch = tech.get("pct_change_5d")
    if isinstance(ch, (int, float)):
        if ch > 3:
            tags.append("momentum_up")
        elif ch < -3:
            tags.append("momentum_down")

    ins = (snapshot.get("insider") or {}).get("net_direction")
    if ins == "buying":
        tags.append("insider_buying")
    elif ins == "selling":
        tags.append("insider_selling")

    rec = (snapshot.get("recommendations") or {}).get("trend")
    if rec == "upgrading":
        tags.append("analyst_upgrading")
    elif rec == "downgrading":
        tags.append("analyst_downgrading")

    s = (snapshot.get("sentiment") or {}).get("company_news_score")
    if isinstance(s, (int, float)):
        if s > 0.6:
            tags.append("news_positive")
        elif s < 0.4:
            tags.append("news_negative")

    return tags


@dataclass
class StockProfile:
    symbol: str
    character: dict[str, Any] = field(default_factory=dict)
    # signal_efficacy[tag] = {"dw": float, "dl": float, "w": int, "l": int}
    signal_efficacy: dict[str, dict[str, float]] = field(default_factory=dict)
    trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    realized_pnl: float = 0.0
    sum_r: float = 0.0          # sum of trade R-multiples (pnl / initial risk)
    sum_hold_minutes: float = 0.0
    current_streak: int = 0       # +N win streak, -N loss streak
    last_traded: str | None = None
    beta_alpha: float = 1.0       # Thompson-sampling Beta(alpha, beta)
    beta_beta: float = 1.0
    # LLM-authored playbook — "how to trade this stock" — regenerated
    # periodically from the profile + recent trade history.
    playbook: str = ""
    playbook_at_trade: int = 0    # trade count when the playbook was written
    updated_at: str | None = None

    # ---- Layer 3: ledger stats ----

    def win_rate(self) -> float | None:
        return (self.wins / self.trades) if self.trades else None

    def profit_factor(self) -> float | None:
        if self.trades == 0:
            return None
        if self.gross_loss <= 0:
            return 5.0 if self.gross_profit > 0 else 1.0
        return min(5.0, self.gross_profit / self.gross_loss)

    def avg_hold_minutes(self) -> float:
        return (self.sum_hold_minutes / self.trades) if self.trades else 0.0

    def expectancy_r(self) -> float:
        """Average R-multiple per trade — the core risk-adjusted edge metric.

        R = trade P&L / the dollar risk taken (stop distance). +0.3R per
        trade is genuinely good; it's size-independent and stock-independent,
        so it's the right unit for learning across the universe."""
        return (self.sum_r / self.trades) if self.trades else 0.0

    def edge_score(self) -> float:
        """A signed −1..+1 measure of realized edge on this stock.

        Primarily driven by R-expectancy (risk-adjusted), with profit factor
        and win rate as supporting evidence."""
        if self.trades < 3:
            return 0.0
        pf = self.profit_factor() or 1.0
        wr = self.win_rate() or 0.5
        exp_r = self.expectancy_r()
        r_comp = max(-1.0, min(1.0, exp_r / 0.5))     # +0.5R expectancy → +1
        pf_comp = max(-1.0, min(1.0, (pf - 1.0) / 1.5))
        wr_comp = max(-1.0, min(1.0, (wr - 0.5) * 3.0))
        raw = 0.5 * r_comp + 0.3 * pf_comp + 0.2 * wr_comp
        confidence = min(1.0, self.trades / 12.0)
        return round(raw * confidence, 3)

    def state(self) -> str:
        """NEUTRAL / PROVEN / PROBATION / BENCHED."""
        if self.trades < 3:
            return "NEUTRAL"
        e = self.edge_score()
        if self.current_streak <= -4 or (e < -0.5 and self.trades >= 6):
            return "BENCHED"
        if e < -0.2:
            return "PROBATION"
        if e > 0.3 and self.trades >= 6:
            return "PROVEN"
        return "NEUTRAL"

    # ---- derived knobs for the engine ----

    def conviction_multiplier(self) -> float:
        """Scales position size: 0.3 (benched) … 1.5 (proven winner)."""
        st = self.state()
        if st == "BENCHED":
            return 0.30
        if st == "PROBATION":
            return 0.55
        return round(max(0.30, min(1.50, 1.0 + self.edge_score() * 0.6)), 3)

    def adaptive_stop_pct(self, default: float) -> float:
        """Volatility-scaled stop: ~2× the stock's ATR, clamped near default."""
        atr = self.character.get("atr_pct")
        if not isinstance(atr, (int, float)) or atr <= 0:
            return default
        return round(max(default * 0.5, min(default * 2.5, 2.0 * atr)), 4)

    def adaptive_target_pct(self, default: float) -> float:
        atr = self.character.get("atr_pct")
        if not isinstance(atr, (int, float)) or atr <= 0:
            return default
        return round(max(default * 0.5, min(default * 2.5, 3.0 * atr)), 4)

    # ---- Layer 2: signal efficacy ----

    def signal_win_rate(self, tag: str, global_rate: float) -> float:
        """Per-stock signal win rate, shrunk toward the global rate."""
        e = self.signal_efficacy.get(tag)
        dw = e["dw"] if e else 0.0
        dl = e["dl"] if e else 0.0
        eff_w = dw + _STOCK_PRIOR_STRENGTH * global_rate
        eff_l = dl + _STOCK_PRIOR_STRENGTH * (1.0 - global_rate)
        total = eff_w + eff_l
        return (eff_w / total) if total > 0 else 0.5

    def signal_sample_size(self, tag: str) -> int:
        e = self.signal_efficacy.get(tag)
        return (e["w"] + e["l"]) if e else 0  # type: ignore[return-value]

    # ---- bandit ----

    def bandit_sample(self, rng: random.Random) -> float:
        """Thompson sample — a draw from the stock's win-rate posterior."""
        a = max(0.05, self.beta_alpha)
        b = max(0.05, self.beta_beta)
        return rng.betavariate(a, b)

    # ---- mutation ----

    def record_trade_outcome(
        self,
        win: bool,
        pnl: float,
        hold_minutes: float,
        entry_signals: list[str],
        r_multiple: float = 0.0,
    ) -> None:
        self.trades += 1
        if win:
            self.wins += 1
            self.gross_profit += abs(pnl)
            self.current_streak = self.current_streak + 1 if self.current_streak >= 0 else 1
        else:
            self.losses += 1
            self.gross_loss += abs(pnl)
            self.current_streak = self.current_streak - 1 if self.current_streak <= 0 else -1
        self.realized_pnl += pnl
        self.sum_r += r_multiple
        self.sum_hold_minutes += max(0.0, hold_minutes)

        # Bandit update with mild decay.
        self.beta_alpha = self.beta_alpha * _BANDIT_DECAY + (1.0 if win else 0.0)
        self.beta_beta = self.beta_beta * _BANDIT_DECAY + (0.0 if win else 1.0)

        # Signal efficacy with per-trade decay.
        for tag in entry_signals:
            e = self.signal_efficacy.setdefault(tag, {"dw": 0.0, "dl": 0.0, "w": 0, "l": 0})
            e["dw"] *= _SIGNAL_DECAY
            e["dl"] *= _SIGNAL_DECAY
            if win:
                e["dw"] += 1.0
                e["w"] += 1
            else:
                e["dl"] += 1.0
                e["l"] += 1

        self.last_traded = _now()
        self.updated_at = _now()

    # ---- LLM card ----

    def card(self, global_rates: dict[str, float]) -> str:
        """A compact, human-readable profile block for the LLM prompt."""
        ch = self.character or {}
        lines: list[str] = [f"{self.symbol} PROFILE (this bot's own record on this stock):"]

        if ch.get("classified"):
            lines.append(
                f"- Character: {ch.get('archetype', '?')} "
                f"({ch.get('vol_tier', '?')} volatility, "
                f"ATR {(_pct(ch.get('atr_pct')))})"
            )
            arche = ch.get("archetype")
            if arche == "trender":
                lines.append("  → trends — do NOT fade dips, favour momentum")
            elif arche == "mean-reverter":
                lines.append("  → mean-reverts — fade extremes, distrust breakouts")

        if self.trades == 0:
            lines.append("- No trade history yet — running on character + global priors.")
        else:
            wr = (self.win_rate() or 0) * 100
            pf = self.profit_factor() or 0
            lines.append(
                f"- Record: {self.trades} trades, {wr:.0f}% win, "
                f"profit factor {pf:.2f}, {self.expectancy_r():+.2f}R/trade, "
                f"${self.realized_pnl:+.0f} realized"
            )
            streak = self.current_streak
            streak_note = ""
            if streak >= 2:
                streak_note = f", on a {streak}-win streak"
            elif streak <= -2:
                streak_note = f", on a {-streak}-loss streak"
            lines.append(f"- State: {self.state()}{streak_note}")

            # Best / worst signals by per-stock win rate (need a sample).
            rated: list[tuple[str, float, int]] = []
            for tag in self.signal_efficacy:
                n = self.signal_sample_size(tag)
                if n >= 2:
                    gr = global_rates.get(tag, 0.5)
                    rated.append((tag, self.signal_win_rate(tag, gr), n))
            rated.sort(key=lambda x: x[1], reverse=True)
            if rated:
                best = rated[0]
                lines.append(
                    f"- Works here: {best[0]} ({best[1]*100:.0f}% win, {best[2]} samples)"
                )
                worst = rated[-1]
                if worst[0] != best[0] and worst[1] < 0.45:
                    lines.append(
                        f"- Fails here: {worst[0]} ({worst[1]*100:.0f}% win, "
                        f"{worst[2]} samples) — avoid"
                    )

        if self.playbook:
            lines.append(f"PLAYBOOK for {self.symbol}:")
            lines.append(self.playbook)

        return "\n".join(lines)

    # ---- serialisation ----

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StockProfile":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def _pct(v: Any) -> str:
    if isinstance(v, (int, float)):
        return f"{v*100:.1f}%"
    return "n/a"


def global_signal_rate(global_profile: "StockProfile", tag: str) -> float:
    """Cross-stock win rate for a signal, shrunk toward 0.5."""
    e = global_profile.signal_efficacy.get(tag)
    dw = e["dw"] if e else 0.0
    dl = e["dl"] if e else 0.0
    eff_w = dw + _GLOBAL_PRIOR_STRENGTH * 0.5
    eff_l = dl + _GLOBAL_PRIOR_STRENGTH * 0.5
    return eff_w / (eff_w + eff_l)


def all_global_rates(global_profile: "StockProfile") -> dict[str, float]:
    return {tag: global_signal_rate(global_profile, tag) for tag in SIGNAL_TAGS}


# ---- persistence helpers (memory.db) ----

async def load_profile(memory: Any, symbol: str) -> StockProfile:
    raw = await memory.get_stock_profile(symbol)
    if raw:
        return StockProfile.from_dict(raw)
    return StockProfile(symbol=symbol)


async def save_profile(memory: Any, profile: StockProfile) -> None:
    profile.updated_at = _now()
    await memory.save_stock_profile(profile.symbol, profile.to_dict())


async def load_all_profiles(memory: Any) -> dict[str, StockProfile]:
    rows = await memory.all_stock_profiles()
    out: dict[str, StockProfile] = {}
    for r in rows:
        p = StockProfile.from_dict(r)
        out[p.symbol] = p
    return out


async def load_global(memory: Any) -> StockProfile:
    return await load_profile(memory, GLOBAL_SYMBOL)
