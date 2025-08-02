"""Data feed module for fetching and caching OHLCV data."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf
from pandas import DataFrame, DatetimeIndex

from gappers.config import config

logger = logging.getLogger(__name__)


class DataFeed:
    """Handles data fetching and caching from multiple sources."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """Initialize data feed with cache directory."""
        self.cache_dir = cache_dir or config.data_path / "ohlcv"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_premium_clients()

    def _init_premium_clients(self) -> None:
        """Initialize premium data feed clients if credentials available."""
        self.iex_client = None
        self.polygon_client = None

        if config.iex_cloud_api_key:
            try:
                from iexfinance.stocks import get_historical_data

                self.iex_client = get_historical_data
                logger.info("IEX Cloud client initialized")
            except ImportError:
                logger.warning("IEX Cloud API key provided but iexfinance not installed")

        if config.polygon_api_key:
            try:
                from polygon import RESTClient

                self.polygon_client = RESTClient(config.polygon_api_key)
                logger.info("Polygon client initialized")
            except ImportError:
                logger.warning("Polygon API key provided but polygon-api-client not installed")

    def download(
        self,
        symbols: List[str],
        start: Union[str, datetime],
        end: Union[str, datetime],
        interval: str = "1d",
        source: str = "auto",
        force_refresh: bool = False,
    ) -> Dict[str, DataFrame]:
        """
        Download OHLCV data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start: Start date (inclusive)
            end: End date (inclusive)
            interval: Data interval ('1d', '1h', '5m', '1m')
            source: Data source ('yfinance', 'iex', 'polygon', 'auto')
            force_refresh: Force re-download even if cached

        Returns:
            Dictionary mapping symbols to DataFrames with OHLCV data
        """
        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)

        results = {}
        for symbol in symbols:
            try:
                df = self._get_symbol_data(
                    symbol, start, end, interval, source, force_refresh
                )
                if not df.empty:
                    results[symbol] = df
                else:
                    logger.warning(f"No data retrieved for {symbol}")
            except Exception as e:
                logger.error(f"Error downloading data for {symbol}: {e}")

        return results

    def _get_symbol_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str,
        source: str,
        force_refresh: bool,
    ) -> DataFrame:
        """Get data for a single symbol with caching."""
        cache_path = self._get_cache_path(symbol, start, end, interval)

        if not force_refresh and cache_path.exists():
            if self._is_cache_valid(cache_path):
                logger.debug(f"Loading {symbol} from cache: {cache_path}")
                return self._load_from_cache(cache_path)

        # Determine source
        if source == "auto":
            if interval == "1m" and self.polygon_client:
                source = "polygon"
            elif interval in ["1d", "1h"] and self.iex_client:
                source = "iex"
            else:
                source = "yfinance"

        # Fetch data
        logger.info(f"Downloading {symbol} from {source}: {start} to {end}, interval={interval}")

        if source == "polygon" and self.polygon_client:
            df = self._fetch_from_polygon(symbol, start, end, interval)
        elif source == "iex" and self.iex_client:
            df = self._fetch_from_iex(symbol, start, end, interval)
        else:
            df = self._fetch_from_yfinance(symbol, start, end, interval)

        if not df.empty:
            self._save_to_cache(df, cache_path)

        return df

    def _fetch_from_yfinance(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> DataFrame:
        """Fetch data from yfinance."""
        try:
            ticker = yf.Ticker(symbol)

            # Add buffer day for end date to ensure we get the full range
            end_buffer = end + timedelta(days=1)

            df = ticker.history(
                start=start,
                end=end_buffer,
                interval=interval,
                actions=True,
                auto_adjust=False,
                prepost=False,
            )

            if df.empty:
                return df

            # Clean column names
            df.columns = [col.lower().replace(" ", "_") for col in df.columns]

            # Ensure we have required columns
            required_cols = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns for {symbol}")
                return pd.DataFrame()

            # Add symbol column
            df["symbol"] = symbol

            # Filter to requested date range (yfinance sometimes returns extra days)
            # Handle timezone comparison issue by converting to timezone-aware if needed
            if df.index.tz is not None:
                # DataFrame index is timezone-aware, convert start/end to match
                start_ts = pd.Timestamp(start)
                end_ts = pd.Timestamp(end)
                if start_ts.tzinfo is None:
                    start_ts = start_ts.tz_localize(df.index.tz)
                else:
                    start_ts = start_ts.tz_convert(df.index.tz)
                if end_ts.tzinfo is None:
                    end_ts = end_ts.tz_localize(df.index.tz)
                else:
                    end_ts = end_ts.tz_convert(df.index.tz)
                df = df.loc[start_ts:end_ts]
            else:
                # DataFrame index is timezone-naive, use original datetime objects
                df = df.loc[start:end]

            return df

        except Exception as e:
            logger.error(f"yfinance error for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_from_polygon(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> DataFrame:
        """Fetch data from Polygon."""
        if not self.polygon_client:
            return pd.DataFrame()

        try:
            # Map interval to Polygon format
            interval_map = {
                "1m": {"multiplier": 1, "timespan": "minute"},
                "5m": {"multiplier": 5, "timespan": "minute"},
                "1h": {"multiplier": 1, "timespan": "hour"},
                "1d": {"multiplier": 1, "timespan": "day"},
            }

            if interval not in interval_map:
                logger.warning(f"Unsupported interval {interval} for Polygon")
                return pd.DataFrame()

            params = interval_map[interval]

            aggs = []
            for agg in self.polygon_client.list_aggs(
                ticker=symbol,
                multiplier=params["multiplier"],
                timespan=params["timespan"],
                from_=start.strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d"),
                limit=50000,
            ):
                aggs.append(agg)

            if not aggs:
                return pd.DataFrame()

            df = pd.DataFrame(
                [
                    {
                        "timestamp": pd.to_datetime(a.timestamp, unit="ms"),
                        "open": a.open,
                        "high": a.high,
                        "low": a.low,
                        "close": a.close,
                        "volume": a.volume,
                        "symbol": symbol,
                    }
                    for a in aggs
                ]
            )

            df.set_index("timestamp", inplace=True)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            return df

        except Exception as e:
            logger.error(f"Polygon error for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_from_iex(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> DataFrame:
        """Fetch data from IEX Cloud."""
        if not self.iex_client:
            return pd.DataFrame()

        try:
            # IEX primarily supports daily data
            if interval != "1d":
                logger.warning(f"IEX Cloud best suited for daily data, not {interval}")
                return pd.DataFrame()

            df = self.iex_client(
                symbol,
                start=start,
                end=end,
                output_format="pandas",
                token=config.iex_cloud_api_key,
            )

            if df.empty:
                return df

            # Standardize columns
            df.rename(
                columns={
                    "fOpen": "open",
                    "fHigh": "high",
                    "fLow": "low",
                    "fClose": "close",
                    "fVolume": "volume",
                },
                inplace=True,
            )

            df["symbol"] = symbol

            return df

        except Exception as e:
            logger.error(f"IEX Cloud error for {symbol}: {e}")
            return pd.DataFrame()

    def _get_cache_path(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> Path:
        """Generate cache file path."""
        date_str = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
        filename = f"{symbol}_{interval}_{date_str}.parquet"
        year = start.year
        return self.cache_dir / f"date={year}" / filename

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is still valid."""
        if not cache_path.exists():
            return False

        # Check age
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        max_age = timedelta(hours=config.cache_expiry_hours)

        return cache_age < max_age

    def _load_from_cache(self, cache_path: Path) -> DataFrame:
        """Load data from parquet cache."""
        try:
            df = pd.read_parquet(cache_path)
            if isinstance(df.index, DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            logger.error(f"Error loading cache {cache_path}: {e}")
            return pd.DataFrame()

    def _save_to_cache(self, df: DataFrame, cache_path: Path) -> None:
        """Save data to parquet cache."""
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, compression="snappy")
            logger.debug(f"Saved to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Error saving cache {cache_path}: {e}")

    def get_splits_dividends(
        self, symbol: str, start: datetime, end: datetime
    ) -> Dict[str, DataFrame]:
        """Get splits and dividends data."""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get splits
            splits = ticker.splits
            if not splits.empty:
                # Handle timezone comparison issue
                start_tz, end_tz = self._match_timezone(splits.index, start, end)
                splits = splits.loc[start_tz:end_tz]
                splits = pd.DataFrame({"split_ratio": splits})
            
            # Get dividends
            dividends = ticker.dividends
            if not dividends.empty:
                # Handle timezone comparison issue
                start_tz, end_tz = self._match_timezone(dividends.index, start, end)
                dividends = dividends.loc[start_tz:end_tz]
                dividends = pd.DataFrame({"dividend": dividends})
            
            return {"splits": splits, "dividends": dividends}
            
        except Exception as e:
            logger.error(f"Error getting splits/dividends for {symbol}: {e}")
            return {"splits": pd.DataFrame(), "dividends": pd.DataFrame()}

    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """Clear cache files older than specified days."""
        count = 0
        cutoff = datetime.now() - timedelta(days=older_than_days or 30)

        for cache_file in self.cache_dir.rglob("*.parquet"):
            if datetime.fromtimestamp(cache_file.stat().st_mtime) < cutoff:
                try:
                    cache_file.unlink()
                    count += 1
                except Exception as e:
                    logger.error(f"Error deleting {cache_file}: {e}")

        logger.info(f"Cleared {count} cache files")
        return count

    def _match_timezone(self, index: DatetimeIndex, start: datetime, end: datetime) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Match timezone between index and start/end datetimes to avoid comparison errors."""
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        
        if index.tz is not None:
            # Index is timezone-aware, convert start/end to match
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize(index.tz)
            else:
                start_ts = start_ts.tz_convert(index.tz)
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize(index.tz)
            else:
                end_ts = end_ts.tz_convert(index.tz)
        else:
            # Index is timezone-naive, ensure start/end are also naive
            if start_ts.tzinfo is not None:
                start_ts = start_ts.tz_localize(None)
            if end_ts.tzinfo is not None:
                end_ts = end_ts.tz_localize(None)
        
        return start_ts, end_ts