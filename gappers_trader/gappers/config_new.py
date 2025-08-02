"""Production-grade configuration management with YAML support."""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DataSourcesConfig:
    """Data sources configuration."""
    primary: str = "yfinance"
    fallback: List[str] = field(default_factory=lambda: ["yfinance"])
    
    @dataclass
    class PolygonConfig:
        api_key: str = ""
        tier: str = "basic"
    
    @dataclass
    class TiingoConfig:
        api_key: str = ""
    
    @dataclass
    class AlpacaConfig:
        api_key: str = ""
        secret_key: str = ""
        paper: bool = True
    
    polygon: PolygonConfig = field(default_factory=PolygonConfig)
    tiingo: TiingoConfig = field(default_factory=TiingoConfig)
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)


@dataclass
class DataCollectionConfig:
    """Data collection configuration."""
    frequency_minutes: int = 30
    universe_size: int = 3000
    min_dollar_volume: float = 1000000
    min_price: float = 5.0
    max_price: float = 1000.0
    max_years_history: int = 5
    chunk_size_days: int = 30
    data_dir: str = "data"
    cache_dir: str = "data/cache"
    universe_dir: str = "data/universe"
    ohlcv_dir: str = "data/ohlcv"


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    top_k: int = 10
    min_gap_pct: float = 0.02
    max_gap_pct: float = 0.30
    profit_target_pct: float = 0.10
    trailing_stop_pct: float = 0.02
    hard_stop_pct: float = 0.04
    time_stop_hour: int = 15
    position_size_usd: float = 1000
    max_positions: int = 10
    max_portfolio_risk_pct: float = 0.20
    sector_diversification: bool = True
    max_per_sector: int = 3
    max_correlation: float = 0.7


@dataclass
class CostsConfig:
    """Trading costs configuration."""
    commission_per_share: float = 0.005
    slippage_bps: int = 10
    borrowing_rate: float = 0.02


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 100000
    benchmark: str = "SPY"
    min_sharpe_ratio: float = 1.0
    max_drawdown_pct: float = 0.15
    min_win_rate: float = 0.45


@dataclass
class MarketHoursConfig:
    """Market hours configuration."""
    premarket_start: str = "04:00"
    market_open: str = "09:30"
    market_close: str = "16:00"
    afterhours_end: str = "20:00"
    timezone: str = "America/New_York"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/gap_trader.log"
    max_bytes: int = 10485760
    backup_count: int = 5


@dataclass
class SecurityConfig:
    """Security configuration."""
    max_api_requests_per_minute: int = 60
    rate_limit_buffer: int = 5
    encrypt_sensitive_data: bool = True
    audit_trail: bool = True


@dataclass
class Config:
    """Main configuration class."""
    data_sources: DataSourcesConfig = field(default_factory=DataSourcesConfig)
    data_collection: DataCollectionConfig = field(default_factory=DataCollectionConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    market_hours: MarketHoursConfig = field(default_factory=MarketHoursConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'Config':
        """Load configuration from YAML file."""
        if config_path is None:
            # Look for config.yaml in current directory or parent directories
            current_dir = Path.cwd()
            possible_paths = [
                current_dir / "config.yaml",
                current_dir / "gappers_trader" / "config.yaml",
                current_dir.parent / "config.yaml",
            ]
            
            config_path = None
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Replace environment variables
                config_data = cls._replace_env_vars(config_data)
                
                # Create config object from loaded data
                config = cls._from_dict(config_data)
                logger.info(f"Loaded configuration from {config_path}")
                return config
                
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                logger.info("Using default configuration")
                return cls()
        else:
            logger.info("No config file found, using default configuration")
            return cls()
    
    @classmethod
    def _replace_env_vars(cls, data: Any) -> Any:
        """Replace environment variable placeholders in configuration."""
        if isinstance(data, dict):
            return {k: cls._replace_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls._replace_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            # Extract environment variable name
            env_var = data[2:-1]
            return os.getenv(env_var, "")
        else:
            return data
    
    @classmethod
    def _from_dict(cls, data: Dict) -> 'Config':
        """Create Config object from dictionary."""
        config = cls()
        
        # Update data sources
        if 'data_sources' in data:
            ds_data = data['data_sources']
            config.data_sources.primary = ds_data.get('primary', config.data_sources.primary)
            config.data_sources.fallback = ds_data.get('fallback', config.data_sources.fallback)
            
            if 'polygon' in ds_data:
                polygon_data = ds_data['polygon']
                config.data_sources.polygon.api_key = polygon_data.get('api_key', '')
                config.data_sources.polygon.tier = polygon_data.get('tier', 'basic')
            
            if 'tiingo' in ds_data:
                tiingo_data = ds_data['tiingo']
                config.data_sources.tiingo.api_key = tiingo_data.get('api_key', '')
            
            if 'alpaca' in ds_data:
                alpaca_data = ds_data['alpaca']
                config.data_sources.alpaca.api_key = alpaca_data.get('api_key', '')
                config.data_sources.alpaca.secret_key = alpaca_data.get('secret_key', '')
                config.data_sources.alpaca.paper = alpaca_data.get('paper', True)
        
        # Update data collection
        if 'data_collection' in data:
            dc_data = data['data_collection']
            for key, value in dc_data.items():
                if hasattr(config.data_collection, key):
                    setattr(config.data_collection, key, value)
        
        # Update strategy
        if 'strategy' in data:
            strategy_data = data['strategy']
            for key, value in strategy_data.items():
                if hasattr(config.strategy, key):
                    setattr(config.strategy, key, value)
        
        # Update costs
        if 'costs' in data:
            costs_data = data['costs']
            for key, value in costs_data.items():
                if hasattr(config.costs, key):
                    setattr(config.costs, key, value)
        
        # Update backtest
        if 'backtest' in data:
            backtest_data = data['backtest']
            for key, value in backtest_data.items():
                if hasattr(config.backtest, key):
                    setattr(config.backtest, key, value)
        
        # Update market hours
        if 'market_hours' in data:
            mh_data = data['market_hours']
            for key, value in mh_data.items():
                if hasattr(config.market_hours, key):
                    setattr(config.market_hours, key, value)
        
        # Update logging
        if 'logging' in data:
            logging_data = data['logging']
            for key, value in logging_data.items():
                if hasattr(config.logging, key):
                    setattr(config.logging, key, value)
        
        # Update security
        if 'security' in data:
            security_data = data['security']
            for key, value in security_data.items():
                if hasattr(config.security, key):
                    setattr(config.security, key, value)
        
        return config
    
    def save(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        try:
            config_dict = self._to_dict()
            
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise
    
    def _to_dict(self) -> Dict:
        """Convert Config object to dictionary."""
        return {
            'data_sources': {
                'primary': self.data_sources.primary,
                'fallback': self.data_sources.fallback,
                'polygon': {
                    'api_key': self.data_sources.polygon.api_key,
                    'tier': self.data_sources.polygon.tier
                },
                'tiingo': {
                    'api_key': self.data_sources.tiingo.api_key
                },
                'alpaca': {
                    'api_key': self.data_sources.alpaca.api_key,
                    'secret_key': self.data_sources.alpaca.secret_key,
                    'paper': self.data_sources.alpaca.paper
                }
            },
            'data_collection': {
                'frequency_minutes': self.data_collection.frequency_minutes,
                'universe_size': self.data_collection.universe_size,
                'min_dollar_volume': self.data_collection.min_dollar_volume,
                'min_price': self.data_collection.min_price,
                'max_price': self.data_collection.max_price,
                'max_years_history': self.data_collection.max_years_history,
                'chunk_size_days': self.data_collection.chunk_size_days,
                'data_dir': self.data_collection.data_dir,
                'cache_dir': self.data_collection.cache_dir,
                'universe_dir': self.data_collection.universe_dir,
                'ohlcv_dir': self.data_collection.ohlcv_dir
            },
            'strategy': {
                'top_k': self.strategy.top_k,
                'min_gap_pct': self.strategy.min_gap_pct,
                'max_gap_pct': self.strategy.max_gap_pct,
                'profit_target_pct': self.strategy.profit_target_pct,
                'trailing_stop_pct': self.strategy.trailing_stop_pct,
                'hard_stop_pct': self.strategy.hard_stop_pct,
                'time_stop_hour': self.strategy.time_stop_hour,
                'position_size_usd': self.strategy.position_size_usd,
                'max_positions': self.strategy.max_positions,
                'max_portfolio_risk_pct': self.strategy.max_portfolio_risk_pct,
                'sector_diversification': self.strategy.sector_diversification,
                'max_per_sector': self.strategy.max_per_sector,
                'max_correlation': self.strategy.max_correlation
            },
            'costs': {
                'commission_per_share': self.costs.commission_per_share,
                'slippage_bps': self.costs.slippage_bps,
                'borrowing_rate': self.costs.borrowing_rate
            },
            'backtest': {
                'start_date': self.backtest.start_date,
                'end_date': self.backtest.end_date,
                'initial_capital': self.backtest.initial_capital,
                'benchmark': self.backtest.benchmark,
                'min_sharpe_ratio': self.backtest.min_sharpe_ratio,
                'max_drawdown_pct': self.backtest.max_drawdown_pct,
                'min_win_rate': self.backtest.min_win_rate
            },
            'market_hours': {
                'premarket_start': self.market_hours.premarket_start,
                'market_open': self.market_hours.market_open,
                'market_close': self.market_hours.market_close,
                'afterhours_end': self.market_hours.afterhours_end,
                'timezone': self.market_hours.timezone
            },
            'logging': {
                'level': self.logging.level,
                'format': self.logging.format,
                'file': self.logging.file,
                'max_bytes': self.logging.max_bytes,
                'backup_count': self.logging.backup_count
            },
            'security': {
                'max_api_requests_per_minute': self.security.max_api_requests_per_minute,
                'rate_limit_buffer': self.security.rate_limit_buffer,
                'encrypt_sensitive_data': self.security.encrypt_sensitive_data,
                'audit_trail': self.security.audit_trail
            }
        }
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate data sources
        if self.data_sources.primary not in ['yfinance', 'polygon', 'tiingo', 'alpaca']:
            issues.append(f"Invalid primary data source: {self.data_sources.primary}")
        
        # Validate strategy parameters
        if self.strategy.min_gap_pct >= self.strategy.max_gap_pct:
            issues.append("min_gap_pct must be less than max_gap_pct")
        
        if self.strategy.profit_target_pct <= 0:
            issues.append("profit_target_pct must be positive")
        
        if self.strategy.hard_stop_pct <= 0:
            issues.append("hard_stop_pct must be positive")
        
        if self.strategy.top_k <= 0:
            issues.append("top_k must be positive")
        
        # Validate data collection
        if self.data_collection.min_price >= self.data_collection.max_price:
            issues.append("min_price must be less than max_price")
        
        if self.data_collection.min_dollar_volume <= 0:
            issues.append("min_dollar_volume must be positive")
        
        # Validate backtest dates
        try:
            start_date = datetime.strptime(self.backtest.start_date, '%Y-%m-%d')
            end_date = datetime.strptime(self.backtest.end_date, '%Y-%m-%d')
            if start_date >= end_date:
                issues.append("backtest start_date must be before end_date")
        except ValueError as e:
            issues.append(f"Invalid backtest date format: {e}")
        
        return issues
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        # Create logs directory if it doesn't exist
        log_file = Path(self.logging.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.logging.level.upper()),
            format=self.logging.format,
            handlers=[
                logging.FileHandler(self.logging.file),
                logging.StreamHandler()
            ]
        )
        
        logger.info("Logging configured successfully")


# Global config instance
config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global config
    if config is None:
        config = Config.load()
    return config


def set_config(new_config: Config) -> None:
    """Set global configuration instance."""
    global config
    config = new_config