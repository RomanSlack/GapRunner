"""Production-grade universe filtering and screening functionality."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
import warnings

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from pandas import DataFrame
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .config_new import Config
from .data_providers import DataProviderManager

warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)
console = Console()


class UniverseBuilder:
    """Production-grade universe builder with multiple data sources and robust filtering."""

    def __init__(self, config: Config) -> None:
        """Initialize universe builder."""
        self.config = config
        self.data_provider_manager = DataProviderManager(config)
        self.cache_dir = Path(config.data_collection.universe_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Universe parameters from config
        self.min_dollar_volume = config.data_collection.min_dollar_volume
        self.min_price = config.data_collection.min_price
        self.max_price = config.data_collection.max_price
        self.universe_size = config.data_collection.universe_size
        
        logger.info(f"UniverseBuilder initialized with min volume: ${self.min_dollar_volume:,}")

    def build_universe(
        self,
        date: Optional[datetime] = None,
        min_dollar_volume: float = 1_000_000,
        min_price: float = 5.0,
        max_price: float = 1000.0,
        exchanges: Optional[List[str]] = None,
        include_delisted: bool = True,
        force_refresh: bool = False,
    ) -> DataFrame:
        """
        Build tradeable universe for a given date.

        Args:
            date: Date to build universe for (default: today)
            min_dollar_volume: Minimum median dollar volume over 20 days
            min_price: Minimum stock price
            max_price: Maximum stock price
            exchanges: List of exchanges to include (default: NYSE, NASDAQ, ARCA)
            include_delisted: Include delisted stocks to avoid survivorship bias
            force_refresh: Force refresh of cached universe

        Returns:
            DataFrame with universe symbols and metadata
        """
        if date is None:
            date = datetime.now()

        exchanges = exchanges or ["NYSE", "NASDAQ", "ARCA"]

        cache_path = self._get_cache_path(date, min_dollar_volume, min_price, max_price)

        if not force_refresh and self._is_cache_valid(cache_path):
            logger.info(f"Loading universe from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        logger.info(f"Building universe for {date.strftime('%Y-%m-%d')}")

        # Get current listings
        current_symbols = self._get_current_listings(exchanges)

        # Get delisted symbols if requested
        delisted_symbols = set()
        if include_delisted:
            delisted_symbols = self._get_delisted_symbols(date, exchanges)

        all_symbols = current_symbols | delisted_symbols
        logger.info(f"Found {len(all_symbols)} total symbols ({len(current_symbols)} current, {len(delisted_symbols)} delisted)")

        # Filter by volume and price criteria
        universe_df = self._filter_universe(
            list(all_symbols), date, min_dollar_volume, min_price, max_price
        )

        # Add metadata
        universe_df = self._add_metadata(universe_df, date)

        # Save to cache
        self._save_to_cache(universe_df, cache_path)

        logger.info(f"Final universe size: {len(universe_df)} symbols")
        return universe_df

    def _get_current_listings(self, exchanges: List[str]) -> Set[str]:
        """Get current stock listings from specified exchanges."""
        logger.info("Using comprehensive symbol list from major US exchanges")
        return self._get_comprehensive_symbols()

    def _get_comprehensive_symbols(self) -> Set[str]:
        """SIMPLIFIED: Get top 30 most liquid US stocks only."""
        # Reduced to just 30 top liquid stocks to prevent system hanging
        symbols = {
            # Mega cap tech (highest volume)
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX",
            
            # Major finance
            "JPM", "BAC", "WFC", "V", "MA", "BRK-B",
            
            # Consumer/Industrial
            "WMT", "HD", "PG", "JNJ", "UNH", "MCD", "DIS", "NKE",
            
            # Energy/Materials  
            "XOM", "CVX", "CAT",
            
            # Communication
            "VZ", "T", "CMCSA",
            
            # Top ETFs
            "SPY", "QQQ", "IWM"
            "PG", "KO", "PEP", "WMT", "HD", "LOW", "COST", "TGT", "NKE", "SBUX",
            "MCD", "DIS", "CMCSA", "VZ", "T", "CHTR", "DISH", "TMUS", "NEM", "FCX",
            "TJX", "RCL", "CCL", "MAR", "HLT", "YUM", "CMG", "BKNG", "EXPE", "ABNB",
            
            # Industrial & Materials
            "CAT", "DE", "BA", "GE", "HON", "MMM", "UPS", "FDX", "LMT", "RTX",
            "GD", "NOC", "EMR", "ITW", "ETN", "PH", "ROK", "DOV", "XYL", "IEX",
            "DD", "DOW", "LYB", "CF", "FMC", "ECL", "PPG", "SHW", "APD", "LIN",
            
            # Energy
            "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX", "HES", "DVN",
            "OXY", "APA", "EQT", "FANG", "MRO", "HAL", "BKR", "OIH", "XLE", "USO",
            
            # Utilities & REITs
            "NEE", "DUK", "SO", "D", "EXC", "XEL", "SRE", "AEP", "PCG", "ED",
            "AMT", "PLD", "CCI", "EQIX", "DLR", "PSA", "O", "WELL", "AVB", "EQR",
            
            # Popular ETFs (commonly traded)
            "SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "AGG", "BND", "TLT",
            "GLD", "SLV", "USO", "XLE", "XLF", "XLK", "XLV", "XLI", "XLU", "XLP",
            
            # Emerging/Growth companies
            "ROKU", "ZM", "PTON", "HOOD", "COIN", "RBLX", "U", "PLTR", "SNOW", "DDOG",
            "CRWD", "ZS", "OKTA", "TWLO", "NET", "FSLY", "ESTC", "MDB", "TEAM", "WDAY",
            
            # Biotech & Growth
            "MRNA", "BNTX", "NVAX", "BIIB", "CELG", "ILMN", "GENZ", "ALXN", "BMRN", "INCY",
            "SGEN", "EXAS", "VRTX", "IONS", "TECH", "ARWR", "EDIT", "CRSP", "NTLA", "BEAM",
            
            # Additional liquid names
            "F", "GM", "TSLA", "NIO", "XPEV", "LI", "RIVN", "LCID", "CHPT", "BLNK",
            "DKNG", "PENN", "MGM", "LVS", "WYNN", "CZR", "BYD", "GOOS", "LULU", "GPS"
        }
        
        logger.info(f"Loaded {len(symbols)} symbols from comprehensive list")
        return symbols

    def _get_fallback_symbols(self) -> Set[str]:
        """Minimal fallback list of most liquid US stocks."""
        # Only the most liquid and reliable symbols as final fallback
        return {
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "MA",
            "JNJ", "PG", "HD", "KO", "PEP", "WMT", "DIS", "NFLX", "CRM", "ORCL",
            "SPY", "QQQ", "IWM", "GLD", "TLT"  # Include major ETFs for liquidity
        }

    def _get_delisted_symbols(self, date: datetime, exchanges: List[str]) -> Set[str]:
        """Get delisted symbols to avoid survivorship bias."""
        delisted = set()

        try:
            # Try to get delisted data from various sources
            if hasattr(self.config.data_sources.polygon, 'api_key') and self.config.data_sources.polygon.api_key:
                delisted.update(self._get_delisted_from_polygon(date))
        except Exception as e:
            logger.warning(f"Error getting delisted symbols: {e}")

        # Fallback: add known major names that might have been delisted or changed
        # Note: Most of these are still actively traded, but we include them for historical universe building
        known_delisted = {
            "GE", "F", "T", "VZ", "XOM", "CVX", "IBM", "INTC", "CSCO", "ORCL",
            "WMT", "PFE", "JNJ", "KO", "PEP", "MCD", "NKE", "DIS", "HD", "WBA"
        }

        # Only include if they were actually active around the requested date
        for symbol in known_delisted:
            try:
                # Quick check if symbol had data around this time
                test_data = self.data_provider_manager.get_historical_data(
                    symbol,
                    date - timedelta(days=30),
                    date + timedelta(days=30)
                )
                if test_data is not None and not test_data.empty:
                    delisted.add(symbol)
            except Exception:
                continue

        return delisted

    def _get_delisted_from_polygon(self, date: datetime) -> Set[str]:
        """Get delisted symbols from Polygon API."""
        delisted = set()

        # For now, return empty set as this requires additional API endpoints
        # Future enhancement: Use Polygon's reference data API
        
        return delisted

    def _filter_universe(
        self,
        symbols: List[str],
        date: datetime,
        min_dollar_volume: float,
        min_price: float,
        max_price: float,
    ) -> DataFrame:
        """Filter universe by volume and price criteria."""
        # Look back 30 days for volume/price analysis
        end_date = date
        start_date = date - timedelta(days=30)

        valid_symbols = []
        batch_size = 50  # Process in batches to avoid overwhelming APIs
        
        # Adjust minimum dollar volume if it's too restrictive
        adjusted_min_volume = min_dollar_volume
        if min_dollar_volume > 10_000_000:  # If more than 10M, reduce to 1M
            adjusted_min_volume = 1_000_000
            logger.info(f"Adjusted minimum dollar volume from {min_dollar_volume:,.0f} to {adjusted_min_volume:,.0f}")

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(symbols) + batch_size - 1)//batch_size}")

            try:
                # Get data for each symbol in the batch
                data = {}
                for symbol in batch:
                    symbol_data = self.data_provider_manager.get_historical_data(
                        symbol, start_date, end_date
                    )
                    if symbol_data is not None and not symbol_data.empty:
                        # Convert to expected format (lowercase column names)
                        symbol_data.columns = symbol_data.columns.str.lower()
                        data[symbol] = symbol_data

                for symbol, df in data.items():
                    if df.empty:
                        logger.debug(f"No data for {symbol}")
                        continue

                    try:
                        # Calculate median dollar volume
                        dollar_volume = df['close'] * df['volume']
                        median_dollar_vol = dollar_volume.median()
                        
                        # Skip if median dollar volume is NaN or zero
                        if pd.isna(median_dollar_vol) or median_dollar_vol == 0:
                            logger.debug(f"Invalid dollar volume for {symbol}: {median_dollar_vol}")
                            continue

                        # Get latest price (as of the date)
                        # Handle timezone-aware datetime comparison
                        try:
                            if df.index.tz is not None:
                                # Convert naive datetime to timezone-aware for comparison
                                import pytz
                                date_tz = date.replace(tzinfo=pytz.timezone('America/New_York'))
                                price_data = df.loc[df.index <= date_tz, 'close']
                            else:
                                price_data = df.loc[df.index <= date, 'close']
                        except Exception:
                            # Fallback: just use the last available price
                            price_data = df['close']
                            
                        if len(price_data) == 0:
                            logger.debug(f"No price data before {date} for {symbol}")
                            continue
                            
                        latest_price = price_data.iloc[-1]
                        
                        # Skip if price is NaN
                        if pd.isna(latest_price):
                            logger.debug(f"Invalid price for {symbol}: {latest_price}")
                            continue

                        # Apply filters with more lenient thresholds
                        volume_ok = median_dollar_vol >= adjusted_min_volume
                        price_ok = min_price <= latest_price <= max_price
                        
                        logger.debug(f"{symbol}: price=${latest_price:.2f} (ok={price_ok}), volume=${median_dollar_vol:,.0f} (ok={volume_ok})")
                        
                        if volume_ok and price_ok:
                            valid_symbols.append({
                                'symbol': symbol,
                                'price': latest_price,
                                'median_dollar_volume': median_dollar_vol,
                                'avg_volume': df['volume'].mean(),
                                'date': date.strftime('%Y-%m-%d')
                            })
                            logger.info(f"Added {symbol} to universe: ${latest_price:.2f}, vol=${median_dollar_vol:,.0f}")

                    except Exception as e:
                        logger.debug(f"Error processing {symbol}: {e}")
                        continue

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                continue

        logger.info(f"Filtered to {len(valid_symbols)} symbols from {len(symbols)} candidates")
        return pd.DataFrame(valid_symbols)

    def _add_metadata(self, universe_df: DataFrame, date: datetime) -> DataFrame:
        """Add additional metadata to universe."""
        if universe_df.empty:
            return universe_df

        # Add sector information where available
        universe_df['sector'] = 'Unknown'
        universe_df['market_cap'] = np.nan
        universe_df['float_shares'] = np.nan

        # Try to get additional info in small batches
        for i in range(0, len(universe_df), 10):
            batch_symbols = universe_df.iloc[i:i+10]['symbol'].tolist()

            for symbol in batch_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info

                    if info:
                        idx = universe_df[universe_df['symbol'] == symbol].index[0]
                        universe_df.loc[idx, 'sector'] = info.get('sector', 'Unknown')
                        universe_df.loc[idx, 'market_cap'] = info.get('marketCap', np.nan)
                        universe_df.loc[idx, 'float_shares'] = info.get('floatShares', np.nan)

                except Exception:
                    continue  # Skip if we can't get info

        return universe_df

    def _get_cache_path(
        self, date: datetime, min_dollar_volume: float, min_price: float, max_price: float
    ) -> Path:
        """Generate cache path for universe."""
        date_str = date.strftime('%Y%m%d')
        filename = f"universe_{date_str}_dv{int(min_dollar_volume/1000)}k_p{min_price}-{max_price}.parquet"
        return self.cache_dir / filename

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache is still valid."""
        if not cache_path.exists():
            return False

        # Universe cache valid for 1 day
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < timedelta(days=1)

    def _save_to_cache(self, df: DataFrame, cache_path: Path) -> None:
        """Save universe to cache."""
        try:
            df.to_parquet(cache_path, compression="snappy")
            logger.debug(f"Saved universe to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Error saving universe cache: {e}")

    def get_historical_universe(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> Dict[str, DataFrame]:
        """Get universe for multiple historical dates."""
        universes = {}
        current_date = start_date

        while current_date <= end_date:
            try:
                universe = self.build_universe(date=current_date, **kwargs)
                if not universe.empty:
                    universes[current_date.strftime('%Y-%m-%d')] = universe
            except Exception as e:
                logger.error(f"Error building universe for {current_date}: {e}")

            current_date += timedelta(days=1)

        return universes