"""Production-grade data management with Parquet storage and efficient querying."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .config_new import Config

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)
console = Console()


class DataManager:
    """Production-grade data manager for efficient storage and retrieval."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_dir = Path(config.data_collection.data_dir)
        self.universe_dir = Path(config.data_collection.universe_dir)
        self.ohlcv_dir = Path(config.data_collection.ohlcv_dir)
        
        # Ensure directories exist
        for dir_path in [self.data_dir, self.universe_dir, self.ohlcv_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("DataManager initialized successfully")

    def get_universe(self, date: datetime) -> pd.DataFrame:
        """Get trading universe for a specific date."""
        universe_file = self.universe_dir / f"universe_{date.strftime('%Y%m%d')}_dv1000k_p5.0-1000.0.parquet"
        
        if not universe_file.exists():
            # Try to find the closest date
            universe_file = self._find_closest_universe_file(date)
            if not universe_file:
                logger.warning(f"No universe file found for {date}")
                return pd.DataFrame()
        
        try:
            universe_df = pd.read_parquet(universe_file)
            logger.info(f"Loaded universe: {len(universe_df)} symbols for {date.strftime('%Y-%m-%d')} (from {universe_file.name})")
            return universe_df
        except Exception as e:
            logger.error(f"Failed to load universe from {universe_file}: {e}")
            return pd.DataFrame()

    def _find_closest_universe_file(self, target_date: datetime) -> Optional[Path]:
        """Find the closest universe file to the target date."""
        universe_files = list(self.universe_dir.glob('universe_*.parquet'))
        
        if not universe_files:
            return None
        
        # Extract dates and find closest
        file_dates = []
        for file_path in universe_files:
            try:
                date_str = file_path.stem.split('_')[1]
                file_date = datetime.strptime(date_str, '%Y%m%d')
                file_dates.append((abs((file_date - target_date).days), file_path))
            except:
                continue
        
        if not file_dates:
            return None
        
        # Return closest file (minimum days difference)
        file_dates.sort(key=lambda x: x[0])
        closest_file = file_dates[0][1]
        
        logger.info(f"Using closest universe file: {closest_file.name} for {target_date.strftime('%Y-%m-%d')}")
        return closest_file

    def get_price_data(self, symbols: List[str], start_date: datetime, 
                      end_date: datetime, columns: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Get price data for symbols in date range with optimized loading."""
        console.print(f"[blue]📊 Loading price data for {len(symbols)} symbols[/blue]")
        
        if columns is None:
            columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        data_dict = {}
        not_found = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            load_task = progress.add_task("Loading price data...", total=len(symbols))
            
            for symbol in symbols:
                try:
                    symbol_data = self._load_symbol_data(symbol, start_date, end_date, columns)
                    if symbol_data is not None and not symbol_data.empty:
                        data_dict[symbol] = symbol_data
                    else:
                        not_found.append(symbol)
                        
                    progress.update(load_task, advance=1)
                    
                except Exception as e:
                    logger.warning(f"Failed to load {symbol}: {e}")
                    not_found.append(symbol)
                    progress.update(load_task, advance=1)
        
        # Log summary
        success_rate = len(data_dict) / len(symbols) * 100 if symbols else 0
        console.print(f"[green]✅ Loaded {len(data_dict)}/{len(symbols)} symbols ({success_rate:.1f}%)[/green]")
        
        if not_found and len(not_found) <= 10:
            console.print(f"[yellow]⚠️  Not found: {', '.join(not_found)}[/yellow]")
        elif len(not_found) > 10:
            console.print(f"[yellow]⚠️  {len(not_found)} symbols not found[/yellow]")
        
        return data_dict

    def _load_symbol_data(self, symbol: str, start_date: datetime, 
                         end_date: datetime, columns: List[str]) -> Optional[pd.DataFrame]:
        """Load data for a single symbol from partitioned storage."""
        symbol_data_frames = []
        
        # Look for files in multiple locations due to inconsistent partitioning
        search_locations = [
            # Current partitioned structure - check all possible year directories
            *[self.ohlcv_dir / f"date={year}" for year in range(start_date.year, end_date.year + 1)],
            # Fallback to cache directory
            self.data_dir / "cache",
            # Direct ohlcv directory (non-partitioned)
            self.ohlcv_dir
        ]
        
        for search_dir in search_locations:
            if not search_dir.exists():
                continue
            
            # Find files for this symbol
            symbol_files = list(search_dir.glob(f"{symbol}_*.parquet"))
            
            for file_path in symbol_files:
                try:
                    # Check if file overlaps with our date range
                    if self._file_overlaps_date_range(file_path, start_date, end_date):
                        df = pd.read_parquet(file_path, columns=columns)
                        if not df.empty:
                            # Ensure index is datetime
                            if not isinstance(df.index, pd.DatetimeIndex):
                                df.index = pd.to_datetime(df.index)
                            symbol_data_frames.append(df)
                except Exception as e:
                    logger.debug(f"Failed to read {file_path}: {e}")
                    continue
        
        if not symbol_data_frames:
            logger.debug(f"No data found for {symbol} in date range {start_date} to {end_date}")
            return None
        
        # Combine all data frames
        combined_df = pd.concat(symbol_data_frames, axis=0)
        
        # Remove duplicates and sort by date
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        combined_df = combined_df.sort_index()
        
        # Filter to exact date range - ensure timezone consistency
        if combined_df.index.tz is not None:
            combined_df.index = combined_df.index.tz_convert(None)
        
        if hasattr(start_date, 'tz') and start_date.tz is not None:
            start_date = start_date.replace(tzinfo=None)
        if hasattr(end_date, 'tz') and end_date.tz is not None:
            end_date = end_date.replace(tzinfo=None)
        
        mask = (combined_df.index >= start_date) & (combined_df.index <= end_date)
        filtered_df = combined_df[mask]
        
        return filtered_df if not filtered_df.empty else None

    def _file_overlaps_date_range(self, file_path: Path, start_date: datetime, 
                                 end_date: datetime) -> bool:
        """Check if a file's date range overlaps with the requested range."""
        try:
            # Extract date range from filename
            # Format: SYMBOL_1d_YYYYMMDD_YYYYMMDD.parquet
            filename = file_path.stem
            parts = filename.split('_')
            
            if len(parts) < 4:
                return True  # If can't parse, assume overlap
            
            file_start_str = parts[2]
            file_end_str = parts[3]
            
            file_start = datetime.strptime(file_start_str, '%Y%m%d')
            file_end = datetime.strptime(file_end_str, '%Y%m%d')
            
            # Check for overlap
            return not (file_end < start_date or file_start > end_date)
            
        except Exception:
            # If can't parse filename, assume overlap to be safe
            return True

    def get_gaps_data(self, date: datetime, min_gap_pct: float = 0.02) -> pd.DataFrame:
        """Get gap data for a specific date with efficient calculation."""
        console.print(f"[blue]📈 Calculating gaps for {date.strftime('%Y-%m-%d')}[/blue]")
        
        # Get universe for this date
        universe_df = self.get_universe(date)
        if universe_df.empty:
            console.print("[red]❌ No universe data available[/red]")
            return pd.DataFrame()
        
        symbols = universe_df['symbol'].tolist()
        
        # Get price data for the date and previous day
        start_date = date - timedelta(days=5)  # Buffer for weekends/holidays
        end_date = date + timedelta(days=1)
        
        price_data = self.get_price_data(symbols, start_date, end_date, ['Open', 'Close'])
        
        if not price_data:
            console.print("[red]❌ No price data available[/red]")
            return pd.DataFrame()
        
        gaps_list = []
        
        for symbol, data in price_data.items():
            try:
                # Find the exact date
                if date not in data.index:
                    continue
                
                today_data = data.loc[date]
                
                # Find previous trading day
                previous_dates = data.index[data.index < date]
                if len(previous_dates) == 0:
                    continue
                
                yesterday_data = data.loc[previous_dates[-1]]
                
                # Calculate gap
                if yesterday_data['Close'] > 0:
                    gap_pct = (today_data['Open'] - yesterday_data['Close']) / yesterday_data['Close']
                    
                    if abs(gap_pct) >= min_gap_pct:
                        gaps_list.append({
                            'symbol': symbol,
                            'date': date,
                            'previous_close': yesterday_data['Close'],
                            'current_open': today_data['Open'],
                            'gap_pct': gap_pct,
                            'gap_abs': abs(gap_pct),
                            'gap_direction': 'up' if gap_pct > 0 else 'down'
                        })
                        
            except Exception as e:
                logger.warning(f"Failed to calculate gap for {symbol}: {e}")
                continue
        
        if not gaps_list:
            console.print("[yellow]⚠️  No significant gaps found[/yellow]")
            return pd.DataFrame()
        
        gaps_df = pd.DataFrame(gaps_list)
        
        # Add universe metadata
        gaps_df = gaps_df.merge(
            universe_df[['symbol', 'sector', 'market_cap', 'avg_volume']],
            on='symbol',
            how='left'
        )
        
        # Sort by absolute gap size
        gaps_df = gaps_df.sort_values('gap_abs', ascending=False).reset_index(drop=True)
        gaps_df['rank'] = range(1, len(gaps_df) + 1)
        
        console.print(f"[green]✅ Found {len(gaps_df)} gaps ≥ {min_gap_pct:.1%}[/green]")
        
        return gaps_df

    def get_available_dates(self) -> List[datetime]:
        """Get list of available trading dates."""
        universe_files = list(self.universe_dir.glob('universe_*.parquet'))
        
        dates = []
        for file_path in universe_files:
            try:
                date_str = file_path.stem.split('_')[1]
                date = datetime.strptime(date_str, '%Y%m%d')
                dates.append(date)
            except Exception:
                continue
        
        return sorted(dates)

    def get_storage_stats(self) -> Dict:
        """Get storage statistics."""
        def get_dir_size(path: Path) -> int:
            """Get total size of directory in bytes."""
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        
        stats = {
            'universe_files': len(list(self.universe_dir.glob('*.parquet'))),
            'price_data_files': len(list(self.ohlcv_dir.rglob('*.parquet'))),
            'total_size_mb': get_dir_size(self.data_dir) / 1024 / 1024,
            'universe_size_mb': get_dir_size(self.universe_dir) / 1024 / 1024,
            'price_data_size_mb': get_dir_size(self.ohlcv_dir) / 1024 / 1024
        }
        
        # Get date range
        available_dates = self.get_available_dates()
        if available_dates:
            stats['earliest_date'] = available_dates[0].strftime('%Y-%m-%d')
            stats['latest_date'] = available_dates[-1].strftime('%Y-%m-%d')
            stats['total_days'] = len(available_dates)
        
        return stats

    def cleanup_old_cache(self, days_old: int = 30) -> int:
        """Clean up old cache files."""
        cache_dir = Path(self.config.data_collection.cache_dir)
        
        if not cache_dir.exists():
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        removed_count = 0
        
        for cache_file in cache_dir.glob('*.parquet'):
            try:
                if datetime.fromtimestamp(cache_file.stat().st_mtime) < cutoff_date:
                    cache_file.unlink()
                    removed_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        logger.info(f"Removed {removed_count} old cache files")
        return removed_count

    def validate_data_integrity(self) -> Dict:
        """Validate data integrity and consistency."""
        console.print("[blue]🔍 Validating data integrity...[/blue]")
        
        validation_results = {
            'total_files_checked': 0,
            'corrupted_files': [],
            'missing_data_ranges': [],
            'inconsistent_files': [],
            'validation_passed': True
        }
        
        try:
            # Check all parquet files
            all_files = list(self.data_dir.rglob('*.parquet'))
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                
                check_task = progress.add_task("Checking files...", total=len(all_files))
                
                for file_path in all_files:
                    try:
                        # Try to read the file
                        df = pd.read_parquet(file_path)
                        
                        # Basic validation
                        if df.empty:
                            validation_results['inconsistent_files'].append(str(file_path))
                        
                        # Check for reasonable data ranges in price files
                        if 'ohlcv' in str(file_path):
                            self._validate_price_data(df, file_path, validation_results)
                        
                        validation_results['total_files_checked'] += 1
                        
                    except Exception as e:
                        validation_results['corrupted_files'].append(str(file_path))
                        logger.warning(f"Corrupted file {file_path}: {e}")
                    
                    progress.update(check_task, advance=1)
            
            # Set overall validation status
            validation_results['validation_passed'] = (
                len(validation_results['corrupted_files']) == 0 and
                len(validation_results['inconsistent_files']) == 0
            )
            
            if validation_results['validation_passed']:
                console.print("[green]✅ Data validation passed[/green]")
            else:
                console.print("[red]❌ Data validation found issues[/red]")
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            validation_results['validation_passed'] = False
        
        return validation_results

    def _validate_price_data(self, df: pd.DataFrame, file_path: Path, 
                           validation_results: Dict) -> None:
        """Validate price data for consistency."""
        if df.empty:
            return
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_results['inconsistent_files'].append(f"{file_path} - Missing columns: {missing_columns}")
            return
        
        # Check for negative prices
        for col in required_columns:
            if (df[col] <= 0).any():
                validation_results['inconsistent_files'].append(f"{file_path} - Negative/zero prices in {col}")
        
        # Check High >= Low
        if (df['High'] < df['Low']).any():
            validation_results['inconsistent_files'].append(f"{file_path} - High < Low violation")
        
        # Check Open/Close within High/Low range
        if ((df['Open'] > df['High']) | (df['Open'] < df['Low'])).any():
            validation_results['inconsistent_files'].append(f"{file_path} - Open outside High/Low range")
        
        if ((df['Close'] > df['High']) | (df['Close'] < df['Low'])).any():
            validation_results['inconsistent_files'].append(f"{file_path} - Close outside High/Low range")