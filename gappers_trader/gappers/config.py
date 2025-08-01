"""Configuration management for the gap trading system."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """System configuration."""

    # Alpaca API
    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    alpaca_base_url: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    # Premium data feeds
    iex_cloud_api_key: Optional[str] = os.getenv("IEX_CLOUD_API_KEY")
    polygon_api_key: Optional[str] = os.getenv("POLYGON_API_KEY")

    # System
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    data_path: Path = Path(os.getenv("DATA_PATH", "./data"))
    cache_expiry_hours: int = int(os.getenv("CACHE_EXPIRY_HOURS", "24"))
    max_workers: int = int(os.getenv("MAX_WORKERS", "4"))

    # Trading
    default_position_size: float = float(os.getenv("DEFAULT_POSITION_SIZE", "10000"))
    default_commission: float = float(os.getenv("DEFAULT_COMMISSION", "0.005"))
    default_slippage_bps: float = float(os.getenv("DEFAULT_SLIPPAGE_BPS", "10"))

    # Database
    db_connection_string: str = os.getenv("DB_CONNECTION_STRING", "sqlite:///./data/trades.db")

    def __post_init__(self) -> None:
        """Validate configuration and create directories."""
        self.data_path = Path(self.data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        (self.data_path / "ohlcv").mkdir(exist_ok=True)
        (self.data_path / "universe").mkdir(exist_ok=True)
        (self.data_path / "cache").mkdir(exist_ok=True)

    @property
    def has_alpaca_credentials(self) -> bool:
        """Check if Alpaca credentials are configured."""
        return bool(self.alpaca_api_key and self.alpaca_secret_key)

    @property
    def has_premium_feeds(self) -> bool:
        """Check if any premium data feeds are configured."""
        return bool(self.iex_cloud_api_key or self.polygon_api_key)


config = Config()