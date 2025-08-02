"""Production-grade data collection system with progress bars and robust error handling."""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import pandas as pd
import numpy as np
import yfinance as yf
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, 
    TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.panel import Panel
from rich.table import Table
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Config
from .universe import UniverseBuilder

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)
console = Console()


class DataCollector:
    """Production-grade data collector with robust error handling and progress tracking."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_dir = Path(config.data_collection.data_dir)
        self.cache_dir = Path(config.data_collection.cache_dir)
        self.universe_dir = Path(config.data_collection.universe_dir)
        self.ohlcv_dir = Path(config.data_collection.ohlcv_dir)
        
        # Create directories
        for dir_path in [self.data_dir, self.cache_dir, self.universe_dir, self.ohlcv_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
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
        
        # Initialize universe builder
        self.universe_builder = UniverseBuilder(config)
        
        # Track API usage
        self.api_calls_made = 0
        self.last_api_reset = time.time()
        
        logger.info("DataCollector initialized successfully")

    def collect_universe_data(self, date: datetime) -> bool:
        """Collect universe data for a specific date with progress tracking."""
        console.print(f"\n[bold blue]🌍 Collecting Universe Data for {date.strftime('%Y-%m-%d')}[/bold blue]")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                
                # Build universe
                universe_task = progress.add_task("Building trading universe...", total=100)
                universe_df = self.universe_builder.build_universe(date)
                progress.update(universe_task, completed=100)
                
                if universe_df.empty:
                    console.print("[red]❌ No symbols found for universe[/red]")
                    return False
                
                # Save universe
                save_task = progress.add_task("Saving universe data...", total=100)
                universe_file = self.universe_dir / f"universe_{date.strftime('%Y%m%d')}.parquet"
                universe_df.to_parquet(universe_file)
                progress.update(save_task, completed=100)
                
                console.print(f"[green]✅ Universe saved: {len(universe_df)} symbols[/green]")
                return True
                
        except Exception as e:
            console.print(f"[red]❌ Universe collection failed: {e}[/red]")
            logger.error(f"Universe collection failed: {e}", exc_info=True)
            return False

    def collect_price_data(self, symbols: List[str], start_date: datetime, 
                          end_date: datetime, timeframe: str = "1d") -> Dict[str, pd.DataFrame]:
        """Collect price data for multiple symbols with robust error handling."""
        
        console.print(f"\n[bold blue]📈 Collecting Price Data ({timeframe})[/bold blue]")
        console.print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        console.print(f"Symbols: {len(symbols)}")
        
        collected_data = {}
        failed_symbols = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            main_task = progress.add_task("Downloading price data...", total=len(symbols))
            
            # Use ThreadPoolExecutor for concurrent downloads
            max_workers = min(10, len(symbols))  # Limit concurrent requests
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_symbol = {
                    executor.submit(self._download_symbol_data, symbol, start_date, end_date, timeframe): symbol
                    for symbol in symbols
                }
                
                # Process completed downloads
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    
                    try:
                        data = future.result(timeout=30)  # 30 second timeout per symbol
                        if data is not None and not data.empty:
                            collected_data[symbol] = data
                            progress.update(main_task, advance=1, 
                                          description=f"Downloaded {symbol} ({len(collected_data)}/{len(symbols)})")
                        else:
                            failed_symbols.append(symbol)
                            progress.update(main_task, advance=1)
                            
                    except Exception as e:
                        failed_symbols.append(symbol)
                        logger.warning(f"Failed to download {symbol}: {e}")
                        progress.update(main_task, advance=1)
                        
                    # Rate limiting
                    self._rate_limit()
        
        # Summary
        success_rate = len(collected_data) / len(symbols) * 100 if symbols else 0
        
        summary_table = Table(title="Data Collection Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Symbols", str(len(symbols)))
        summary_table.add_row("Successfully Downloaded", str(len(collected_data)))
        summary_table.add_row("Failed Downloads", str(len(failed_symbols)))
        summary_table.add_row("Success Rate", f"{success_rate:.1f}%")
        
        console.print("\n")
        console.print(summary_table)
        
        if failed_symbols and len(failed_symbols) <= 10:
            console.print(f"\n[yellow]⚠️  Failed symbols: {', '.join(failed_symbols)}[/yellow]")
        elif len(failed_symbols) > 10:
            console.print(f"\n[yellow]⚠️  {len(failed_symbols)} symbols failed (too many to list)[/yellow]")
        
        return collected_data

    def _download_symbol_data(self, symbol: str, start_date: datetime, 
                             end_date: datetime, timeframe: str) -> Optional[pd.DataFrame]:
        """Download data for a single symbol with error handling."""
        try:
            # Check cache first
            cache_file = self._get_cache_file(symbol, start_date, end_date, timeframe)
            if cache_file.exists():
                try:
                    cached_data = pd.read_parquet(cache_file)
                    if self._is_cache_valid(cached_data, start_date, end_date):
                        return cached_data
                except Exception:
                    # If cache is corrupted, continue to download
                    pass
            
            # Download from data source
            ticker = yf.Ticker(symbol)
            
            # Add buffer days to ensure we get all data
            buffer_start = start_date - timedelta(days=5)
            buffer_end = end_date + timedelta(days=1)
            
            data = ticker.history(
                start=buffer_start,
                end=buffer_end,
                interval=timeframe,
                auto_adjust=True,
                prepost=True,
                actions=False
            )
            
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Clean and validate data
            data = self._clean_price_data(data, symbol)
            
            if data.empty:
                logger.warning(f"No valid data after cleaning for {symbol}")
                return None
            
            # Cache the data
            self._cache_data(data, cache_file)
            
            return data
            
        except Exception as e:
            logger.warning(f"Error downloading {symbol}: {e}")
            return None

    def _clean_price_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and validate price data."""
        if data.empty:
            return data
        
        original_len = len(data)
        
        # Remove timezone info to avoid issues
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        # Remove rows with missing critical data
        data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Remove rows with zero or negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            data = data[data[col] > 0]
        
        # Remove rows where High < Low (data errors)
        data = data[data['High'] >= data['Low']]
        
        # Remove rows where Close is outside High/Low range
        data = data[(data['Close'] >= data['Low']) & (data['Close'] <= data['High'])]
        data = data[(data['Open'] >= data['Low']) & (data['Open'] <= data['High'])]
        
        # Remove extreme outliers (price changes > 50% in one day)
        if len(data) > 1:
            pct_change = data['Close'].pct_change().abs()
            data = data[pct_change <= 0.5]
        
        # Log data quality
        removed_rows = original_len - len(data)
        if removed_rows > 0:
            logger.info(f"Cleaned {symbol}: removed {removed_rows}/{original_len} invalid rows")
        
        return data

    def _get_cache_file(self, symbol: str, start_date: datetime, 
                       end_date: datetime, timeframe: str) -> Path:
        """Get cache file path for symbol data."""
        date_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        filename = f"{symbol}_{timeframe}_{date_str}.parquet"
        return self.cache_dir / filename

    def _is_cache_valid(self, cached_data: pd.DataFrame, start_date: datetime, 
                       end_date: datetime) -> bool:
        """Check if cached data is valid and covers the required date range."""
        if cached_data.empty:
            return False
        
        # Check date coverage
        data_start = cached_data.index.min()
        data_end = cached_data.index.max()
        
        # Allow for some flexibility in date ranges (weekends, holidays)
        buffer = timedelta(days=3)
        
        return (data_start <= start_date + buffer and 
                data_end >= end_date - buffer)

    def _cache_data(self, data: pd.DataFrame, cache_file: Path) -> None:
        """Cache data to parquet file."""
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(cache_file)
        except Exception as e:
            logger.warning(f"Failed to cache data to {cache_file}: {e}")

    def _rate_limit(self) -> None:
        """Implement rate limiting for API calls."""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.last_api_reset > 60:
            self.api_calls_made = 0
            self.last_api_reset = current_time
        
        self.api_calls_made += 1
        
        # If approaching rate limit, sleep
        max_calls = self.config.security.max_api_requests_per_minute
        if self.api_calls_made >= max_calls - self.config.security.rate_limit_buffer:
            sleep_time = 60 - (current_time - self.last_api_reset)
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.api_calls_made = 0
                self.last_api_reset = time.time()

    def save_data_partitioned(self, data_dict: Dict[str, pd.DataFrame], 
                             collection_date: datetime) -> bool:
        """Save collected data in partitioned parquet format."""
        console.print(f"\n[bold blue]💾 Saving Partitioned Data[/bold blue]")
        
        if not data_dict:
            console.print("[red]❌ No data to save[/red]")
            return False
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                
                save_task = progress.add_task("Saving symbol data...", total=len(data_dict))
                
                year = collection_date.year
                year_dir = self.ohlcv_dir / f"date={year}"
                year_dir.mkdir(parents=True, exist_ok=True)
                
                saved_count = 0
                for symbol, data in data_dict.items():
                    try:
                        if data.empty:
                            continue
                        
                        # Create filename with date range
                        start_date = data.index.min().strftime('%Y%m%d')
                        end_date = data.index.max().strftime('%Y%m%d')
                        filename = f"{symbol}_1d_{start_date}_{end_date}.parquet"
                        
                        file_path = year_dir / filename
                        data.to_parquet(file_path)
                        saved_count += 1
                        
                        progress.update(save_task, advance=1, 
                                      description=f"Saved {symbol} ({saved_count}/{len(data_dict)})")
                        
                    except Exception as e:
                        logger.error(f"Failed to save {symbol}: {e}")
                        progress.update(save_task, advance=1)
                
                console.print(f"[green]✅ Saved {saved_count}/{len(data_dict)} symbol files[/green]")
                return saved_count > 0
                
        except Exception as e:
            console.print(f"[red]❌ Save operation failed: {e}[/red]")
            logger.error(f"Save operation failed: {e}", exc_info=True)
            return False

    def collect_full_dataset(self, start_date: datetime, end_date: datetime) -> bool:
        """Collect complete dataset for date range."""
        console.print(Panel.fit(
            f"[bold white]🚀 Gap Trading Data Collection[/bold white]\n"
            f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n"
            f"Data Source: {self.config.data_sources.primary}\n"
            f"Universe Size: {self.config.data_collection.universe_size:,} symbols",
            border_style="blue"
        ))
        
        try:
            # Step 1: Build universe for the end date (most recent)
            if not self.collect_universe_data(end_date):
                return False
            
            # Step 2: Load universe symbols
            universe_file = self.universe_dir / f"universe_{end_date.strftime('%Y%m%d')}.parquet"
            universe_df = pd.read_parquet(universe_file)
            symbols = universe_df['symbol'].tolist()
            
            console.print(f"[green]📊 Universe loaded: {len(symbols)} symbols[/green]")
            
            # Step 3: Collect price data
            price_data = self.collect_price_data(symbols, start_date, end_date)
            
            if not price_data:
                console.print("[red]❌ No price data collected[/red]")
                return False
            
            # Step 4: Save partitioned data
            if not self.save_data_partitioned(price_data, end_date):
                return False
            
            # Success summary
            console.print(Panel.fit(
                f"[bold green]✅ Data Collection Complete![/bold green]\n"
                f"• Universe: {len(symbols):,} symbols\n"
                f"• Price Data: {len(price_data):,} symbols\n"
                f"• Success Rate: {len(price_data)/len(symbols)*100:.1f}%\n"
                f"• Storage: Partitioned Parquet format",
                border_style="green"
            ))
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Data collection failed: {e}[/red]")
            logger.error(f"Data collection failed: {e}", exc_info=True)
            return False

    def get_collection_status(self) -> Dict:
        """Get status of current data collection."""
        status = {
            'universe_files': len(list(self.universe_dir.glob('*.parquet'))),
            'price_data_files': len(list(self.ohlcv_dir.rglob('*.parquet'))),
            'cache_files': len(list(self.cache_dir.glob('*.parquet'))),
            'total_size_mb': sum(f.stat().st_size for f in self.data_dir.rglob('*') if f.is_file()) / 1024 / 1024
        }
        
        # Get date range of available data
        universe_files = list(self.universe_dir.glob('universe_*.parquet'))
        if universe_files:
            dates = []
            for f in universe_files:
                try:
                    date_str = f.stem.split('_')[1]
                    dates.append(datetime.strptime(date_str, '%Y%m%d'))
                except:
                    continue
            
            if dates:
                status['earliest_date'] = min(dates).strftime('%Y-%m-%d')
                status['latest_date'] = max(dates).strftime('%Y-%m-%d')
        
        return status