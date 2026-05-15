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
    watchlist_size: int = Field(default=10, alias="ASTRA_WATCHLIST_SIZE")
    max_position_pct: float = Field(default=0.05, alias="ASTRA_MAX_POSITION_PCT")
    daily_loss_limit_pct: float = Field(default=0.03, alias="ASTRA_DAILY_LOSS_LIMIT_PCT")
    fee_per_trade: float = Field(default=1.0, alias="ASTRA_FEE_PER_TRADE")
    slippage_bps: float = Field(default=5.0, alias="ASTRA_SLIPPAGE_BPS")

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
