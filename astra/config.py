"""Configuration loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    finnhub_api_key: str = Field(default="", alias="FINNHUB_API_KEY")

    lm_studio_url: str = Field(default="http://localhost:1234/v1", alias="LM_STUDIO_URL")
    lm_studio_model: str = Field(default="local-model", alias="LM_STUDIO_MODEL")
    lm_studio_api_key: str = Field(default="lm-studio", alias="LM_STUDIO_API_KEY")

    host: str = Field(default="127.0.0.1", alias="ASTRA_HOST")
    port: int = Field(default=8765, alias="ASTRA_PORT")

    tick_seconds: int = Field(default=300, alias="ASTRA_TICK_SECONDS")
    watchlist_size: int = Field(default=25, alias="ASTRA_WATCHLIST_SIZE")
    max_position_pct: float = Field(default=0.15, alias="ASTRA_MAX_POSITION_PCT")
    daily_loss_limit_pct: float = Field(default=0.05, alias="ASTRA_DAILY_LOSS_LIMIT_PCT")
    # Commission model — Avanza-style: a percentage of the order value with
    # a per-trade minimum. fee = max(min_fee, fee_pct * order_value).
    # Avanza Mini: 0.25% (fee_pct=0.0025), minimum 1 SEK. Prices here are in
    # USD, so min_fee defaults to ~1 SEK converted (≈ $0.10).
    fee_pct: float = Field(default=0.0025, alias="ASTRA_FEE_PCT")
    min_fee: float = Field(default=0.10, alias="ASTRA_MIN_FEE")
    slippage_bps: float = Field(default=5.0, alias="ASTRA_SLIPPAGE_BPS")

    # Auto-exit rules — enforced before the LLM/heuristic even sees the symbol.
    stop_loss_pct: float = Field(default=0.05, alias="ASTRA_STOP_LOSS_PCT")
    take_profit_pct: float = Field(default=0.15, alias="ASTRA_TAKE_PROFIT_PCT")
    trailing_stop_pct: float = Field(default=0.08, alias="ASTRA_TRAILING_STOP_PCT")
    min_confidence: float = Field(default=0.30, alias="ASTRA_MIN_CONFIDENCE")

    # --- Position discipline (the redesign) ---
    # Concentrated portfolio: hold few, meaningful positions.
    max_open_positions: int = Field(default=10, alias="ASTRA_MAX_OPEN_POSITIONS")
    target_position_pct: float = Field(default=0.10, alias="ASTRA_TARGET_POSITION_PCT")
    # Never place a trade smaller than this dollar value — kills fee-bleed
    # from micro-trades.
    min_trade_value: float = Field(default=25.0, alias="ASTRA_MIN_TRADE_VALUE")
    # After trading a symbol, don't trade it again for this many minutes.
    cooldown_minutes: int = Field(default=60, alias="ASTRA_COOLDOWN_MINUTES")
    # Only ENTER a new position when conviction clears this bar.
    entry_min_confidence: float = Field(default=0.55, alias="ASTRA_ENTRY_MIN_CONFIDENCE")
    # Risk budget per trade for volatility-based sizing (fraction of equity
    # lost if the stop-loss is hit).
    risk_per_trade_pct: float = Field(default=0.02, alias="ASTRA_RISK_PER_TRADE_PCT")

    # User-supplied priority watchlist, comma-separated. These symbols always
    # get evaluated first; the rest of the slot count is filled from the
    # S&P 500 universe.
    custom_watchlist: str = Field(default="", alias="ASTRA_CUSTOM_WATCHLIST")

    # Universe scanner: when enabled, every tick scans the FULL S&P 500 using
    # cached yfinance candles only, ranks by signal strength + momentum, and
    # deep-dives (Finnhub + LLM) only on the top `scan_top_n` plus held
    # positions and custom priority tickers. Lets the bot effectively watch
    # the entire universe without hitting Finnhub free-tier rate limits.
    scan_universe: bool = Field(default=True, alias="ASTRA_SCAN_UNIVERSE")
    scan_top_n: int = Field(default=30, alias="ASTRA_SCAN_TOP_N")

    # Simulation mode: when on, all market data comes from a synthetic
    # regime-switching simulator instead of Finnhub/yfinance. Lets you watch
    # the bot trade when the market is closed or there's no API key.
    simulate_data: bool = Field(default=False, alias="ASTRA_SIMULATE_DATA")

    data_dir: Path = Field(default=Path("./data"), alias="ASTRA_DATA_DIR")

    @property
    def portfolio_db_path(self) -> Path:
        return self.data_dir / "portfolio.db"

    @property
    def memory_db_path(self) -> Path:
        return self.data_dir / "memory.db"

    @property
    def cache_db_path(self) -> Path:
        return self.data_dir / "cache.db"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
