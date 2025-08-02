"""Production-grade gap calculation and ranking engine."""

import logging
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple
import warnings

import pandas as pd
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from .config import Config
from .data_manager import DataManager

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)
console = Console()


class GapEngine:
    """Production-grade gap calculation and ranking engine."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_manager = DataManager(config)
        
        # Gap calculation parameters
        self.min_gap_pct = config.strategy.min_gap_pct
        self.max_gap_pct = config.strategy.max_gap_pct
        self.top_k = config.strategy.top_k
        
        logger.info(f"GapEngine initialized with gap range: {self.min_gap_pct:.1%} to {self.max_gap_pct:.1%}")

    def calculate_daily_gaps(self, date: datetime, 
                           symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Calculate gaps for all symbols on a specific date."""
        console.print(f"[blue]📊 Calculating gaps for {date.strftime('%Y-%m-%d')}[/blue]")
        
        try:
            # Get universe if symbols not provided
            if symbols is None:
                universe_df = self.data_manager.get_universe(date)
                if universe_df.empty:
                    console.print("[red]❌ No universe data available[/red]")
                    return pd.DataFrame()
                symbols = universe_df['symbol'].tolist()
            
            # Get price data for calculation period
            gaps_df = self._calculate_gaps_batch(symbols, date)
            
            if gaps_df.empty:
                console.print("[yellow]⚠️  No gaps found[/yellow]")
                return pd.DataFrame()
            
            # Filter by gap criteria
            gaps_df = self._filter_gaps(gaps_df)
            
            # Rank gaps
            gaps_df = self._rank_gaps(gaps_df)
            
            # Add technical indicators
            gaps_df = self._add_technical_indicators(gaps_df, date)
            
            console.print(f"[green]✅ Found {len(gaps_df)} qualifying gaps[/green]")
            
            return gaps_df
            
        except Exception as e:
            console.print(f"[red]❌ Gap calculation failed: {e}[/red]")
            logger.error(f"Gap calculation failed: {e}", exc_info=True)
            return pd.DataFrame()

    def _calculate_gaps_batch(self, symbols: List[str], date: datetime) -> pd.DataFrame:
        """Calculate gaps for a batch of symbols."""
        
        # We need current day and previous trading day data
        start_date = date - timedelta(days=10)  # Buffer for weekends/holidays
        end_date = date + timedelta(days=1)
        
        # Get price data
        price_data = self.data_manager.get_price_data(
            symbols, start_date, end_date, ['Open', 'Close', 'High', 'Low', 'Volume']
        )
        
        if not price_data:
            return pd.DataFrame()
        
        gaps_list = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            calc_task = progress.add_task("Calculating gaps...", total=len(price_data))
            
            for symbol, data in price_data.items():
                try:
                    gap_info = self._calculate_symbol_gap(symbol, data, date)
                    if gap_info:
                        gaps_list.append(gap_info)
                    
                    progress.update(calc_task, advance=1)
                    
                except Exception as e:
                    logger.debug(f"Failed to calculate gap for {symbol}: {e}")
                    progress.update(calc_task, advance=1)
        
        return pd.DataFrame(gaps_list) if gaps_list else pd.DataFrame()

    def _calculate_symbol_gap(self, symbol: str, data: pd.DataFrame, 
                            target_date: datetime) -> Optional[Dict]:
        """Calculate gap for a single symbol."""
        if data.empty:
            return None
        
        try:
            # Find target date in data
            target_dates = data.index[data.index.date == target_date.date()]
            if len(target_dates) == 0:
                return None
            
            target_idx = target_dates[0]
            today_data = data.loc[target_idx]
            
            # Find previous trading day
            previous_dates = data.index[data.index < target_idx]
            if len(previous_dates) == 0:
                return None
            
            prev_idx = previous_dates[-1]
            yesterday_data = data.loc[prev_idx]
            
            # Calculate gap
            prev_close = yesterday_data['Close']
            current_open = today_data['Open']
            
            if prev_close <= 0 or current_open <= 0:
                return None
            
            gap_pct = (current_open - prev_close) / prev_close
            gap_absolute = current_open - prev_close
            
            # Calculate additional metrics
            current_high = today_data['High']
            current_low = today_data['Low']
            current_close = today_data['Close']
            current_volume = today_data['Volume']
            
            # Intraday performance
            intraday_pct = (current_close - current_open) / current_open if current_open > 0 else 0
            
            # Price action after gap
            high_from_open = (current_high - current_open) / current_open if current_open > 0 else 0
            low_from_open = (current_low - current_open) / current_open if current_open > 0 else 0
            
            return {
                'symbol': symbol,
                'date': target_date,
                'previous_close': prev_close,
                'current_open': current_open,
                'current_high': current_high,
                'current_low': current_low,
                'current_close': current_close,
                'current_volume': current_volume,
                'gap_pct': gap_pct,
                'gap_absolute': gap_absolute,
                'gap_direction': 'up' if gap_pct > 0 else 'down',
                'intraday_pct': intraday_pct,
                'high_from_open_pct': high_from_open,
                'low_from_open_pct': low_from_open,
                'previous_date': prev_idx
            }
            
        except Exception as e:
            logger.debug(f"Error calculating gap for {symbol}: {e}")
            return None

    def _filter_gaps(self, gaps_df: pd.DataFrame) -> pd.DataFrame:
        """Filter gaps by configured criteria."""
        if gaps_df.empty:
            return gaps_df
        
        original_count = len(gaps_df)
        
        # Filter by gap size
        abs_gap = gaps_df['gap_pct'].abs()
        size_filter = (abs_gap >= self.min_gap_pct) & (abs_gap <= self.max_gap_pct)
        gaps_df = gaps_df[size_filter]
        
        # Filter out penny stocks (additional safety)
        price_filter = (gaps_df['previous_close'] >= self.config.data_collection.min_price) & \
                      (gaps_df['current_open'] >= self.config.data_collection.min_price)
        gaps_df = gaps_df[price_filter]
        
        # Filter out extreme volume anomalies
        if 'current_volume' in gaps_df.columns:
            volume_median = gaps_df['current_volume'].median()
            volume_filter = gaps_df['current_volume'] >= volume_median * 0.1  # At least 10% of median volume
            gaps_df = gaps_df[volume_filter]
        
        filtered_count = len(gaps_df)
        
        if filtered_count < original_count:
            console.print(f"[blue]🔧 Filtered gaps: {original_count} → {filtered_count}[/blue]")
        
        return gaps_df

    def _rank_gaps(self, gaps_df: pd.DataFrame) -> pd.DataFrame:
        """Rank gaps by size and quality."""
        if gaps_df.empty:
            return gaps_df
        
        # Primary ranking: absolute gap size
        gaps_df['gap_abs'] = gaps_df['gap_pct'].abs()
        gaps_df = gaps_df.sort_values('gap_abs', ascending=False)
        
        # Add rankings
        gaps_df['gap_rank'] = range(1, len(gaps_df) + 1)
        
        # Quality score (combination of gap size and volume)
        if 'current_volume' in gaps_df.columns:
            # Normalize volume for scoring
            volume_percentile = gaps_df['current_volume'].rank(pct=True) * 100
            gap_percentile = gaps_df['gap_abs'].rank(pct=True) * 100
            
            # Composite quality score (70% gap size, 30% volume)
            gaps_df['quality_score'] = (gap_percentile * 0.7) + (volume_percentile * 0.3)
            gaps_df['quality_rank'] = gaps_df['quality_score'].rank(ascending=False, method='first')
        else:
            gaps_df['quality_score'] = gaps_df['gap_abs']
            gaps_df['quality_rank'] = gaps_df['gap_rank']
        
        # Reset index
        gaps_df = gaps_df.reset_index(drop=True)
        
        return gaps_df

    def _add_technical_indicators(self, gaps_df: pd.DataFrame, date: datetime) -> pd.DataFrame:
        """Add technical indicators to enhance gap analysis."""
        if gaps_df.empty:
            return gaps_df
        
        # Get extended historical data for technical analysis
        symbols = gaps_df['symbol'].tolist()
        lookback_date = date - timedelta(days=30)
        
        try:
            historical_data = self.data_manager.get_price_data(
                symbols, lookback_date, date, ['Open', 'High', 'Low', 'Close', 'Volume']
            )
            
            for idx, row in gaps_df.iterrows():
                symbol = row['symbol']
                
                if symbol not in historical_data:
                    continue
                
                hist_data = historical_data[symbol]
                if hist_data.empty or len(hist_data) < 10:
                    continue
                
                try:
                    # Calculate technical indicators
                    indicators = self._calculate_technical_indicators(hist_data)
                    
                    # Add to dataframe
                    for key, value in indicators.items():
                        gaps_df.at[idx, key] = value
                        
                except Exception as e:
                    logger.debug(f"Failed to calculate indicators for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Failed to add technical indicators: {e}")
        
        return gaps_df

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators for a symbol."""
        indicators = {}
        
        try:
            # Recent price data
            recent_closes = data['Close'].tail(20)
            recent_volumes = data['Volume'].tail(20)
            
            if len(recent_closes) < 5:
                return indicators
            
            # Moving averages
            if len(recent_closes) >= 5:
                indicators['sma_5'] = recent_closes.tail(5).mean()
            if len(recent_closes) >= 10:
                indicators['sma_10'] = recent_closes.tail(10).mean()
            if len(recent_closes) >= 20:
                indicators['sma_20'] = recent_closes.tail(20).mean()
            
            # Volatility (20-day)
            if len(recent_closes) > 1:
                returns = recent_closes.pct_change().dropna()
                if len(returns) > 0:
                    indicators['volatility_20d'] = returns.std() * np.sqrt(252)
            
            # Volume indicators
            if len(recent_volumes) >= 5:
                indicators['avg_volume_5d'] = recent_volumes.tail(5).mean()
            if len(recent_volumes) >= 20:
                indicators['avg_volume_20d'] = recent_volumes.tail(20).mean()
                # Volume ratio (current vs average)
                current_volume = recent_volumes.iloc[-1]
                avg_volume = indicators['avg_volume_20d']
                if avg_volume > 0:
                    indicators['volume_ratio'] = current_volume / avg_volume
            
            # Price position relative to recent range
            if len(data) >= 20:
                recent_highs = data['High'].tail(20)
                recent_lows = data['Low'].tail(20)
                current_price = recent_closes.iloc[-1]
                
                high_20d = recent_highs.max()
                low_20d = recent_lows.min()
                
                if high_20d > low_20d:
                    indicators['price_position'] = (current_price - low_20d) / (high_20d - low_20d)
                
        except Exception as e:
            logger.debug(f"Error calculating technical indicators: {e}")
        
        return indicators

    def get_top_gaps(self, date: datetime, direction: str = 'both', 
                    limit: Optional[int] = None) -> pd.DataFrame:
        """Get top gaps for a specific date and direction."""
        gaps_df = self.calculate_daily_gaps(date)
        
        if gaps_df.empty:
            return gaps_df
        
        # Filter by direction
        if direction.lower() == 'up':
            gaps_df = gaps_df[gaps_df['gap_pct'] > 0]
        elif direction.lower() == 'down':
            gaps_df = gaps_df[gaps_df['gap_pct'] < 0]
        # 'both' includes all gaps
        
        # Apply limit
        if limit is not None:
            gaps_df = gaps_df.head(limit)
        else:
            gaps_df = gaps_df.head(self.top_k)
        
        return gaps_df

    def analyze_gap_patterns(self, start_date: datetime, end_date: datetime) -> Dict:
        """Analyze gap patterns over a date range."""
        console.print(f"[blue]📈 Analyzing gap patterns from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}[/blue]")
        
        all_gaps = []
        
        current_date = start_date
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:  # Monday = 0, Sunday = 6
                try:
                    daily_gaps = self.calculate_daily_gaps(current_date)
                    if not daily_gaps.empty:
                        all_gaps.append(daily_gaps)
                except Exception as e:
                    logger.warning(f"Failed to calculate gaps for {current_date}: {e}")
            
            current_date += timedelta(days=1)
        
        if not all_gaps:
            return {'error': 'No gap data found for the specified period'}
        
        # Combine all gaps
        combined_gaps = pd.concat(all_gaps, ignore_index=True)
        
        # Calculate statistics
        analysis = self._calculate_gap_statistics(combined_gaps)
        
        console.print(f"[green]✅ Analyzed {len(combined_gaps)} gaps over {(end_date - start_date).days} days[/green]")
        
        return analysis

    def _calculate_gap_statistics(self, gaps_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive gap statistics."""
        if gaps_df.empty:
            return {}
        
        stats = {
            'total_gaps': len(gaps_df),
            'up_gaps': len(gaps_df[gaps_df['gap_pct'] > 0]),
            'down_gaps': len(gaps_df[gaps_df['gap_pct'] < 0]),
        }
        
        # Gap size statistics
        gap_sizes = gaps_df['gap_pct'].abs()
        stats.update({
            'avg_gap_size': gap_sizes.mean(),
            'median_gap_size': gap_sizes.median(),
            'max_gap_size': gap_sizes.max(),
            'min_gap_size': gap_sizes.min(),
            'gap_size_std': gap_sizes.std()
        })
        
        # Direction bias
        stats['up_gap_pct'] = stats['up_gaps'] / stats['total_gaps'] if stats['total_gaps'] > 0 else 0
        
        # Intraday follow-through analysis
        if 'intraday_pct' in gaps_df.columns:
            up_gaps_df = gaps_df[gaps_df['gap_pct'] > 0]
            down_gaps_df = gaps_df[gaps_df['gap_pct'] < 0]
            
            if not up_gaps_df.empty:
                stats['up_gap_follow_through'] = (up_gaps_df['intraday_pct'] > 0).mean()
                stats['avg_up_gap_intraday'] = up_gaps_df['intraday_pct'].mean()
            
            if not down_gaps_df.empty:
                stats['down_gap_follow_through'] = (down_gaps_df['intraday_pct'] < 0).mean()
                stats['avg_down_gap_intraday'] = down_gaps_df['intraday_pct'].mean()
        
        # Volume analysis
        if 'volume_ratio' in gaps_df.columns:
            volume_ratios = gaps_df['volume_ratio'].dropna()
            if not volume_ratios.empty:
                stats['avg_volume_ratio'] = volume_ratios.mean()
                stats['high_volume_gaps'] = (volume_ratios > 2.0).sum()  # Above 2x normal volume
        
        # Size distribution
        stats['gap_size_buckets'] = {
            '2-5%': len(gaps_df[(gap_sizes >= 0.02) & (gap_sizes < 0.05)]),
            '5-10%': len(gaps_df[(gap_sizes >= 0.05) & (gap_sizes < 0.10)]),
            '10-15%': len(gaps_df[(gap_sizes >= 0.10) & (gap_sizes < 0.15)]),
            '15%+': len(gaps_df[gap_sizes >= 0.15])
        }
        
        return stats

    def print_gap_summary(self, gaps_df: pd.DataFrame) -> None:
        """Print a formatted summary of gaps."""
        if gaps_df.empty:
            console.print("[yellow]No gaps to display[/yellow]")
            return
        
        # Create summary table
        table = Table(title=f"Top {len(gaps_df)} Gaps Summary")
        
        table.add_column("Rank", style="cyan")
        table.add_column("Symbol", style="magenta")
        table.add_column("Gap %", style="green")
        table.add_column("Direction", style="blue")
        table.add_column("Prev Close", style="yellow")
        table.add_column("Open", style="yellow")
        table.add_column("Intraday %", style="red")
        
        for idx, row in gaps_df.head(20).iterrows():  # Show top 20
            gap_pct = row['gap_pct'] * 100
            intraday_pct = row.get('intraday_pct', 0) * 100
            
            table.add_row(
                str(row.get('gap_rank', idx + 1)),
                row['symbol'],
                f"{gap_pct:+.1f}%",
                "🔺" if row['gap_pct'] > 0 else "🔻",
                f"${row['previous_close']:.2f}",
                f"${row['current_open']:.2f}",
                f"{intraday_pct:+.1f}%" if pd.notna(intraday_pct) else "N/A"
            )
        
        console.print(table)