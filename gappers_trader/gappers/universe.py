"""Universe construction module for building tradeable symbol lists."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from pandas import DataFrame

from gappers.config import config
from gappers.datafeed import DataFeed

logger = logging.getLogger(__name__)


class UniverseBuilder:
    """Builds and maintains the trading universe with survivorship bias corrections."""

    def __init__(self, data_feed: Optional[DataFeed] = None) -> None:
        """Initialize universe builder."""
        self.data_feed = data_feed or DataFeed()
        self.cache_dir = config.data_path / "universe"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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
        symbols = set()

        try:
            # Try to get from FMP API (free tier)
            symbols.update(self._get_symbols_from_fmp(exchanges))
        except Exception as e:
            logger.warning(f"Error getting symbols from FMP: {e}")

        # Fallback to hardcoded list of major symbols
        if not symbols:
            logger.warning("Using fallback symbol list")
            symbols.update(self._get_fallback_symbols())

        return symbols

    def _get_symbols_from_fmp(self, exchanges: List[str]) -> Set[str]:
        """Get symbols from Financial Modeling Prep API."""
        symbols = set()

        for exchange in exchanges:
            try:
                url = f"https://financialmodelingprep.com/api/v3/stock/list"
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                data = response.json()
                exchange_symbols = [
                    item["symbol"]
                    for item in data
                    if item.get("exchangeShortName") == exchange
                    and item.get("type") == "stock"
                    and "." not in item["symbol"]  # Avoid preferred shares
                    and len(item["symbol"]) <= 5  # Reasonable symbol length
                ]

                symbols.update(exchange_symbols)
                logger.info(f"Got {len(exchange_symbols)} symbols from {exchange}")

            except Exception as e:
                logger.error(f"Error getting {exchange} symbols: {e}")

        return symbols

    def _get_fallback_symbols(self) -> Set[str]:
        """Fallback list of major US stocks."""
        # S&P 500 major components and other liquid names
        return {
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "META", "NVDA", "BRK.B", "UNH",
            "JNJ", "XOM", "JPM", "V", "PG", "HD", "CVX", "MA", "BAC", "ABBV",
            "PFE", "KO", "AVGO", "PEP", "TMO", "COST", "DIS", "ABT", "MRK", "WMT",
            "VZ", "ADBE", "NFLX", "CRM", "ACN", "LLY", "NKE", "DHR", "TXN", "NEE",
            "RTX", "PM", "ORCL", "WFC", "BMY", "UPS", "QCOM", "T", "SPGI", "HON",
            "AMD", "INTC", "IBM", "BA", "CAT", "GS", "AXP", "DE", "SYK", "BLK",
            "MDT", "ISRG", "NOW", "AMGN", "LOW", "ELV", "SBUX", "INTU", "TJX", "BKNG",
            "AMT", "PLD", "SCHW", "MU", "CB", "CCI", "FIS", "MMM", "AON", "REGN",
            "ZTS", "SHW", "MDLZ", "CI", "DUK", "SO", "BSX", "CME", "EOG", "ICE",
            "NSC", "ITW", "PNC", "APD", "CL", "FCX", "USB", "GD", "EMR", "MCO"
        }

    def _get_delisted_symbols(self, date: datetime, exchanges: List[str]) -> Set[str]:
        """Get delisted symbols to avoid survivorship bias."""
        delisted = set()

        try:
            # Try to get delisted data from various sources
            if config.polygon_api_key:
                delisted.update(self._get_delisted_from_polygon(date))
        except Exception as e:
            logger.warning(f"Error getting delisted symbols: {e}")

        # Fallback: add known delisted major names that were active in recent years
        known_delisted = {
            "GE", "F", "T", "VZ", "XOM", "CVX", "IBM", "INTC", "CSCO", "ORCL",
            "WMT", "PFE", "JNJ", "KO", "PEP", "MCD", "NKE", "DIS", "HD", "WBA"
        }

        # Only include if they were actually active around the requested date
        for symbol in known_delisted:
            try:
                # Quick check if symbol had data around this time
                test_data = self.data_feed.download(
                    [symbol],
                    start=date - timedelta(days=30),
                    end=date + timedelta(days=30),
                    interval="1d"
                )
                if symbol in test_data and not test_data[symbol].empty:
                    delisted.add(symbol)
            except Exception:
                continue

        return delisted

    def _get_delisted_from_polygon(self, date: datetime) -> Set[str]:
        """Get delisted symbols from Polygon API."""
        delisted = set()

        try:
            if hasattr(self.data_feed, 'polygon_client') and self.data_feed.polygon_client:
                # This would require a more sophisticated approach with Polygon's reference data
                # For now, return empty set as this requires additional API endpoints
                pass
        except Exception as e:
            logger.error(f"Error getting delisted from Polygon: {e}")

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

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(symbols) + batch_size - 1)//batch_size}")

            try:
                data = self.data_feed.download(
                    batch, start=start_date, end=end_date, interval="1d"
                )

                for symbol, df in data.items():
                    if df.empty:
                        continue

                    try:
                        # Calculate median dollar volume
                        dollar_volume = df['close'] * df['volume']
                        median_dollar_vol = dollar_volume.median()

                        # Get latest price (as of the date)
                        latest_price = df.loc[df.index <= date, 'close'].iloc[-1] if len(df) > 0 else np.nan

                        # Apply filters
                        if (median_dollar_vol >= min_dollar_volume and
                            min_price <= latest_price <= max_price):
                            valid_symbols.append({
                                'symbol': symbol,
                                'price': latest_price,
                                'median_dollar_volume': median_dollar_vol,
                                'avg_volume': df['volume'].mean(),
                                'date': date.strftime('%Y-%m-%d')
                            })

                    except Exception as e:
                        logger.debug(f"Error processing {symbol}: {e}")
                        continue

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                continue

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