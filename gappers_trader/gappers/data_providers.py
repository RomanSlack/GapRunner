"""Production-grade data provider abstraction layer supporting multiple sources."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Config

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """Abstract base class for data providers."""
    
    def __init__(self, config: Config):
        self.config = config
        self.name = self.__class__.__name__
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        logger.info(f"Initialized {self.name} data provider")

    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: datetime, 
                           end_date: datetime, interval: str = "1d") -> Optional[pd.DataFrame]:
        """Get historical price data for a symbol."""
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        pass

    @abstractmethod
    def get_universe_data(self, min_volume: float = 1000000) -> pd.DataFrame:
        """Get universe of tradeable symbols."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the data provider is available."""
        pass

    def _rate_limit(self) -> None:
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def _clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and validate price data."""
        if data.empty:
            return data
        
        original_len = len(data)
        
        # Remove timezone info to avoid issues
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        # Remove rows with missing critical data
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        existing_columns = [col for col in required_columns if col in data.columns]
        data = data.dropna(subset=existing_columns)
        
        # Remove rows with zero or negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                data = data[data[col] > 0]
        
        # Remove rows where High < Low (data errors)
        if 'High' in data.columns and 'Low' in data.columns:
            data = data[data['High'] >= data['Low']]
        
        # Remove rows where Close is outside High/Low range
        if all(col in data.columns for col in ['Close', 'High', 'Low']):
            data = data[(data['Close'] >= data['Low']) & (data['Close'] <= data['High'])]
        
        if all(col in data.columns for col in ['Open', 'High', 'Low']):
            data = data[(data['Open'] >= data['Low']) & (data['Open'] <= data['High'])]
        
        # Remove extreme outliers (price changes > 50% in one day)
        if len(data) > 1 and 'Close' in data.columns:
            pct_change = data['Close'].pct_change().abs()
            data = data[pct_change <= 0.5]
        
        # Log data quality
        removed_rows = original_len - len(data)
        if removed_rows > 0:
            logger.debug(f"Cleaned {symbol}: removed {removed_rows}/{original_len} invalid rows")
        
        return data


class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider using yfinance."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.min_request_interval = 0.1  # Yahoo allows ~10 requests per second

    def get_historical_data(self, symbol: str, start_date: datetime, 
                           end_date: datetime, interval: str = "1d") -> Optional[pd.DataFrame]:
        """Get historical data from Yahoo Finance."""
        try:
            self._rate_limit()
            
            ticker = yf.Ticker(symbol)
            
            # Add buffer to ensure we get all data
            buffer_start = start_date - timedelta(days=2)
            buffer_end = end_date + timedelta(days=1)
            
            data = ticker.history(
                start=buffer_start,
                end=buffer_end,
                interval=interval,
                auto_adjust=True,
                prepost=False,
                actions=False
            )
            
            if data.empty:
                logger.debug(f"No data returned for {symbol} from Yahoo Finance")
                return None
            
            # Clean the data
            data = self._clean_data(data, symbol)
            
            return data if not data.empty else None
            
        except Exception as e:
            logger.warning(f"Yahoo Finance error for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from Yahoo Finance."""
        try:
            self._rate_limit()
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Try different price fields
            price_fields = ['regularMarketPrice', 'currentPrice', 'ask', 'bid']
            
            for field in price_fields:
                if field in info and info[field] is not None:
                    return float(info[field])
            
            # Fallback to latest close price
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            
            return None
            
        except Exception as e:
            logger.warning(f"Yahoo Finance current price error for {symbol}: {e}")
            return None

    def get_universe_data(self, min_volume: float = 1000000) -> pd.DataFrame:
        """Get universe data from Yahoo Finance."""
        # Yahoo Finance doesn't have a direct universe endpoint
        # We'll use a predefined list of liquid stocks
        
        try:
            # S&P 500 + NASDAQ 100 + additional liquid stocks
            symbols = self._get_liquid_stock_list()
            
            universe_data = []
            
            for symbol in symbols[:1000]:  # Limit to avoid rate limits
                try:
                    self._rate_limit()
                    
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    # Get basic info
                    market_cap = info.get('marketCap', 0)
                    sector = info.get('sector', 'Unknown')
                    
                    # Get recent volume data
                    hist = ticker.history(period="10d")
                    if not hist.empty and 'Volume' in hist.columns:
                        avg_volume = hist['Volume'].mean()
                        avg_price = hist['Close'].mean()
                        
                        if avg_volume >= min_volume and avg_price >= 5.0:
                            universe_data.append({
                                'symbol': symbol,
                                'market_cap': market_cap,
                                'sector': sector,
                                'avg_volume': avg_volume,
                                'avg_price': avg_price
                            })
                
                except Exception as e:
                    logger.debug(f"Failed to get universe data for {symbol}: {e}")
                    continue
            
            return pd.DataFrame(universe_data)
            
        except Exception as e:
            logger.error(f"Yahoo Finance universe data error: {e}")
            return pd.DataFrame()

    def _get_liquid_stock_list(self) -> List[str]:
        """Get list of liquid stocks."""
        # Major liquid stocks across sectors
        return [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'ORCL',
            'CRM', 'ADBE', 'INTC', 'AMD', 'QCOM', 'NOW', 'TEAM', 'ZM', 'SNOW', 'DDOG',
            
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF',
            'AXP', 'BLK', 'SPGI', 'ICE', 'CME', 'MCO', 'SCHW', 'AON', 'MMC', 'TRV',
            
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY', 'BMY', 'AMGN',
            'MDT', 'DHR', 'SYK', 'BSX', 'EW', 'ISRG', 'VRTX', 'GILD', 'BIIB', 'REGN',
            
            # Consumer
            'WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'COST', 'LOW', 'TGT', 'SBUX',
            'NKE', 'LULU', 'TJX', 'DIS', 'CMCSA', 'VZ', 'T', 'NFLX', 'ROKU', 'SPOT',
            
            # Industrial
            'BA', 'CAT', 'DE', 'GE', 'MMM', 'HON', 'UPS', 'FDX', 'LMT', 'RTX',
            'NOC', 'GD', 'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'DOV', 'XYL', 'IEX',
            
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'PSX', 'VLO', 'MPC', 'HAL',
            'DVN', 'FANG', 'BKR', 'APA', 'EQT', 'CTRA', 'OIH', 'XLE', 'USO', 'UCO',
            
            # ETFs
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'AGG', 'BND', 'TLT',
            'GLD', 'SLV', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLB'
        ]

    def is_available(self) -> bool:
        """Check if Yahoo Finance is available."""
        try:
            # Test with a simple request
            ticker = yf.Ticker('AAPL')
            info = ticker.info
            return 'symbol' in info or 'shortName' in info
        except Exception:
            return False


class PolygonProvider(DataProvider):
    """Polygon.io data provider (premium)."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.api_key = config.data_sources.polygon.api_key
        self.base_url = "https://api.polygon.io"
        self.min_request_interval = 0.2  # Polygon basic plan: 5 requests per second

    def get_historical_data(self, symbol: str, start_date: datetime, 
                           end_date: datetime, interval: str = "1d") -> Optional[pd.DataFrame]:
        """Get historical data from Polygon."""
        if not self.api_key:
            logger.warning("Polygon API key not configured")
            return None
        
        try:
            self._rate_limit()
            
            # Convert interval to Polygon format
            polygon_interval = self._convert_interval(interval)
            
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{polygon_interval}"
            params = {
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'adjusted': 'true',
                'sort': 'asc',
                'apikey': self.api_key
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'OK' or not data.get('results'):
                logger.debug(f"No data returned for {symbol} from Polygon")
                return None
            
            # Convert to DataFrame
            results = data['results']
            df_data = []
            
            for result in results:
                df_data.append({
                    'Open': result['o'],
                    'High': result['h'],
                    'Low': result['l'],
                    'Close': result['c'],
                    'Volume': result['v']
                })
            
            if not df_data:
                return None
            
            df = pd.DataFrame(df_data)
            
            # Set date index
            timestamps = [result['t'] for result in results]
            df.index = pd.to_datetime(timestamps, unit='ms')
            
            # Clean the data
            df = self._clean_data(df, symbol)
            
            return df if not df.empty else None
            
        except Exception as e:
            logger.warning(f"Polygon error for {symbol}: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from Polygon."""
        if not self.api_key:
            return None
        
        try:
            self._rate_limit()
            
            url = f"{self.base_url}/v2/last/trade/{symbol}"
            params = {'apikey': self.api_key}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'OK' and 'results' in data:
                return float(data['results']['p'])
            
            return None
            
        except Exception as e:
            logger.warning(f"Polygon current price error for {symbol}: {e}")
            return None

    def get_universe_data(self, min_volume: float = 1000000) -> pd.DataFrame:
        """Get universe data from Polygon."""
        if not self.api_key:
            return pd.DataFrame()
        
        try:
            self._rate_limit()
            
            # Get list of active stocks
            url = f"{self.base_url}/v3/reference/tickers"
            params = {
                'market': 'stocks',
                'active': 'true',
                'sort': 'ticker',
                'limit': 1000,
                'apikey': self.api_key
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'OK' or not data.get('results'):
                return pd.DataFrame()
            
            universe_data = []
            
            for ticker_info in data['results']:
                try:
                    symbol = ticker_info['ticker']
                    market_cap = ticker_info.get('market_cap', 0)
                    
                    # Get recent volume data (simplified for demo)
                    universe_data.append({
                        'symbol': symbol,
                        'market_cap': market_cap,
                        'sector': ticker_info.get('sic_description', 'Unknown'),
                        'avg_volume': 0,  # Would need additional API call
                        'avg_price': 0    # Would need additional API call
                    })
                    
                except Exception as e:
                    logger.debug(f"Failed to process ticker {ticker_info}: {e}")
                    continue
            
            return pd.DataFrame(universe_data)
            
        except Exception as e:
            logger.error(f"Polygon universe data error: {e}")
            return pd.DataFrame()

    def _convert_interval(self, interval: str) -> str:
        """Convert standard interval to Polygon format."""
        interval_map = {
            '1m': '1/minute',
            '5m': '5/minute',
            '15m': '15/minute',
            '30m': '30/minute',
            '1h': '1/hour',
            '1d': '1/day',
            '1w': '1/week',
            '1M': '1/month'
        }
        return interval_map.get(interval, '1/day')

    def is_available(self) -> bool:
        """Check if Polygon is available."""
        if not self.api_key:
            return False
        
        try:
            url = f"{self.base_url}/v2/last/trade/AAPL"
            params = {'apikey': self.api_key}
            
            response = self.session.get(url, params=params, timeout=10)
            return response.status_code == 200
            
        except Exception:
            return False


class DataProviderManager:
    """Manages multiple data providers with fallback logic."""
    
    def __init__(self, config: Config):
        self.config = config
        self.providers = {}
        
        # Initialize providers based on configuration
        if config.data_sources.primary == 'yfinance' or 'yfinance' in config.data_sources.fallback:
            self.providers['yfinance'] = YahooFinanceProvider(config)
        
        if (config.data_sources.primary == 'polygon' or 'polygon' in config.data_sources.fallback) and \
           hasattr(config.data_sources, 'polygon') and config.data_sources.polygon.api_key:
            self.providers['polygon'] = PolygonProvider(config)
        
        # Set primary provider
        self.primary_provider_name = config.data_sources.primary
        self.fallback_providers = config.data_sources.fallback
        
        logger.info(f"DataProviderManager initialized with {len(self.providers)} providers")
        logger.info(f"Primary: {self.primary_provider_name}, Fallbacks: {self.fallback_providers}")

    def get_historical_data(self, symbol: str, start_date: datetime, 
                           end_date: datetime, interval: str = "1d") -> Optional[pd.DataFrame]:
        """Get historical data with fallback logic."""
        # Try primary provider first
        if self.primary_provider_name in self.providers:
            provider = self.providers[self.primary_provider_name]
            try:
                data = provider.get_historical_data(symbol, start_date, end_date, interval)
                if data is not None and not data.empty:
                    return data
            except Exception as e:
                logger.warning(f"Primary provider {self.primary_provider_name} failed for {symbol}: {e}")
        
        # Try fallback providers
        for provider_name in self.fallback_providers:
            if provider_name in self.providers and provider_name != self.primary_provider_name:
                provider = self.providers[provider_name]
                try:
                    data = provider.get_historical_data(symbol, start_date, end_date, interval)
                    if data is not None and not data.empty:
                        logger.info(f"Using fallback provider {provider_name} for {symbol}")
                        return data
                except Exception as e:
                    logger.warning(f"Fallback provider {provider_name} failed for {symbol}: {e}")
        
        logger.warning(f"All providers failed for {symbol}")
        return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price with fallback logic."""
        # Try primary provider first
        if self.primary_provider_name in self.providers:
            provider = self.providers[self.primary_provider_name]
            try:
                price = provider.get_current_price(symbol)
                if price is not None:
                    return price
            except Exception as e:
                logger.warning(f"Primary provider {self.primary_provider_name} failed for current price {symbol}: {e}")
        
        # Try fallback providers
        for provider_name in self.fallback_providers:
            if provider_name in self.providers and provider_name != self.primary_provider_name:
                provider = self.providers[provider_name]
                try:
                    price = provider.get_current_price(symbol)
                    if price is not None:
                        return price
                except Exception as e:
                    logger.warning(f"Fallback provider {provider_name} failed for current price {symbol}: {e}")
        
        return None

    def get_universe_data(self, min_volume: float = 1000000) -> pd.DataFrame:
        """Get universe data with fallback logic."""
        # Try primary provider first
        if self.primary_provider_name in self.providers:
            provider = self.providers[self.primary_provider_name]
            try:
                data = provider.get_universe_data(min_volume)
                if not data.empty:
                    return data
            except Exception as e:
                logger.warning(f"Primary provider {self.primary_provider_name} failed for universe data: {e}")
        
        # Try fallback providers
        for provider_name in self.fallback_providers:
            if provider_name in self.providers and provider_name != self.primary_provider_name:
                provider = self.providers[provider_name]
                try:
                    data = provider.get_universe_data(min_volume)
                    if not data.empty:
                        logger.info(f"Using fallback provider {provider_name} for universe data")
                        return data
                except Exception as e:
                    logger.warning(f"Fallback provider {provider_name} failed for universe data: {e}")
        
        logger.error("All providers failed for universe data")
        return pd.DataFrame()

    def check_provider_health(self) -> Dict[str, bool]:
        """Check health of all providers."""
        health_status = {}
        
        for name, provider in self.providers.items():
            try:
                health_status[name] = provider.is_available()
            except Exception as e:
                logger.warning(f"Health check failed for {name}: {e}")
                health_status[name] = False
        
        return health_status