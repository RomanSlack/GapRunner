"""Signal generation module for calculating overnight gaps and rankings."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from gappers.config import config
from gappers.datafeed import DataFeed
from gappers.universe import UniverseBuilder

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generates gap signals and rankings for the trading universe."""

    def __init__(
        self,
        data_feed: Optional[DataFeed] = None,
        universe_builder: Optional[UniverseBuilder] = None,
    ) -> None:
        """Initialize signal generator."""
        self.data_feed = data_feed or DataFeed()
        self.universe_builder = universe_builder or UniverseBuilder(self.data_feed)

    def calculate_gaps(
        self,
        date: datetime,
        universe_symbols: Optional[List[str]] = None,
        min_gap_pct: float = 0.02,
        max_gap_pct: float = 0.50,
        include_negative_gaps: bool = False,
    ) -> DataFrame:
        """
        Calculate overnight gaps for universe on a given date.

        Args:
            date: Trading date to calculate gaps for
            universe_symbols: Specific symbols to analyze (if None, uses full universe)
            min_gap_pct: Minimum gap percentage to include
            max_gap_pct: Maximum gap percentage to include (filters out halts/events)
            include_negative_gaps: Whether to include negative gaps

        Returns:
            DataFrame with gap calculations and metadata
        """
        logger.info(f"Calculating gaps for {date.strftime('%Y-%m-%d')}")

        # Get universe for this date
        if universe_symbols is None:
            universe_df = self.universe_builder.build_universe(date=date)
            universe_symbols = universe_df['symbol'].tolist()

        if not universe_symbols:
            logger.warning("No symbols in universe")
            return pd.DataFrame()

        # Get price data for gap calculation (need previous day's close and current day's open)
        start_date = date - timedelta(days=10)  # Buffer for weekends/holidays
        end_date = date

        gaps = []
        batch_size = 50

        for i in range(0, len(universe_symbols), batch_size):
            batch_symbols = universe_symbols[i:i + batch_size]
            logger.debug(f"Processing gap batch {i//batch_size + 1}/{(len(universe_symbols) + batch_size - 1)//batch_size}")

            try:
                price_data = self.data_feed.download(
                    batch_symbols,
                    start=start_date,
                    end=end_date,
                    interval="1d"
                )

                for symbol, df in price_data.items():
                    gap_data = self._calculate_symbol_gap(symbol, df, date)
                    if gap_data:
                        gaps.append(gap_data)

            except Exception as e:
                logger.error(f"Error processing gap batch: {e}")
                continue

        if not gaps:
            logger.warning("No gaps calculated")
            return pd.DataFrame()

        gaps_df = pd.DataFrame(gaps)

        # Apply filters
        gaps_df = self._filter_gaps(
            gaps_df, min_gap_pct, max_gap_pct, include_negative_gaps
        )

        # Add technical indicators
        gaps_df = self._add_technical_indicators(gaps_df, date)

        # Sort by gap percentage descending
        gaps_df = gaps_df.sort_values('gap_pct', ascending=False).reset_index(drop=True)

        logger.info(f"Calculated {len(gaps_df)} valid gaps")
        return gaps_df

    def _calculate_symbol_gap(
        self, symbol: str, price_df: DataFrame, date: datetime
    ) -> Optional[Dict]:
        """Calculate gap for a single symbol."""
        if price_df.empty:
            return None

        try:
            # Find the target date and previous trading day
            target_date = pd.Timestamp(date).tz_localize(None)
            
            # Get data up to and including target date
            available_dates = price_df.index.tz_localize(None) if price_df.index.tz is not None else price_df.index
            target_data = price_df[available_dates <= target_date]
            
            if len(target_data) < 2:
                return None

            # Get current day and previous day
            current_day = target_data.iloc[-1]
            previous_day = target_data.iloc[-2]

            current_open = current_day['open']
            previous_close = previous_day['close']

            # Skip if prices are invalid
            if pd.isna(current_open) or pd.isna(previous_close) or previous_close <= 0:
                return None

            # Calculate gap
            gap_pct = (current_open / previous_close) - 1.0
            gap_dollars = current_open - previous_close

            # Additional metrics
            previous_volume = previous_day.get('volume', 0)
            current_volume = current_day.get('volume', 0)
            
            # Price levels
            current_high = current_day.get('high', current_open)
            current_low = current_day.get('low', current_open)
            current_close = current_day.get('close', current_open)

            return {
                'symbol': symbol,
                'date': date.strftime('%Y-%m-%d'),
                'gap_pct': gap_pct,
                'gap_dollars': gap_dollars,
                'previous_close': previous_close,
                'current_open': current_open,
                'current_high': current_high,
                'current_low': current_low,
                'current_close': current_close,
                'previous_volume': previous_volume,
                'current_volume': current_volume,
                'volume_ratio': current_volume / previous_volume if previous_volume > 0 else np.nan,
            }

        except Exception as e:
            logger.debug(f"Error calculating gap for {symbol}: {e}")
            return None

    def _filter_gaps(
        self,
        gaps_df: DataFrame,
        min_gap_pct: float,
        max_gap_pct: float,
        include_negative_gaps: bool,
    ) -> DataFrame:
        """Filter gaps based on criteria."""
        if gaps_df.empty:
            return gaps_df

        # Remove extreme gaps (likely halts or corporate actions)
        gaps_df = gaps_df[abs(gaps_df['gap_pct']) <= max_gap_pct]

        # Filter by minimum gap
        if include_negative_gaps:
            gaps_df = gaps_df[abs(gaps_df['gap_pct']) >= min_gap_pct]
        else:
            gaps_df = gaps_df[gaps_df['gap_pct'] >= min_gap_pct]

        # Remove invalid data
        gaps_df = gaps_df.dropna(subset=['gap_pct', 'previous_close', 'current_open'])

        return gaps_df

    def _add_technical_indicators(self, gaps_df: DataFrame, date: datetime) -> DataFrame:
        """Add technical indicators to gap signals."""
        if gaps_df.empty:
            return gaps_df

        # Initialize new columns
        gaps_df['atr_14'] = np.nan
        gaps_df['rsi_14'] = np.nan
        gaps_df['volume_ma_20'] = np.nan
        gaps_df['price_ma_20'] = np.nan
        gaps_df['sector'] = 'Unknown'

        # Calculate indicators for each symbol
        lookback_days = 30
        start_date = date - timedelta(days=lookback_days)

        for idx, row in gaps_df.iterrows():
            symbol = row['symbol']
            
            try:
                # Get historical data for indicators
                hist_data = self.data_feed.download(
                    [symbol],
                    start=start_date,
                    end=date,
                    interval="1d"
                )

                if symbol not in hist_data or hist_data[symbol].empty:
                    continue

                df = hist_data[symbol]

                # ATR (Average True Range)
                atr = self._calculate_atr(df, period=14)
                if len(atr) > 0:
                    gaps_df.loc[idx, 'atr_14'] = atr.iloc[-1]

                # RSI
                rsi = self._calculate_rsi(df['close'], period=14)
                if len(rsi) > 0:
                    gaps_df.loc[idx, 'rsi_14'] = rsi.iloc[-1]

                # Volume MA
                if len(df) >= 20:
                    gaps_df.loc[idx, 'volume_ma_20'] = df['volume'].rolling(20).mean().iloc[-1]

                # Price MA
                if len(df) >= 20:
                    gaps_df.loc[idx, 'price_ma_20'] = df['close'].rolling(20).mean().iloc[-1]

                # Try to get sector info (lightweight)
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    gaps_df.loc[idx, 'sector'] = info.get('sector', 'Unknown')
                except Exception:
                    pass

            except Exception as e:
                logger.debug(f"Error adding indicators for {symbol}: {e}")
                continue

        return gaps_df

    def _calculate_atr(self, df: DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        if len(df) < period:
            return pd.Series(dtype=float)

        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)

        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - prev_close),
            'lc': abs(low - prev_close)
        }).max(axis=1)

        return tr.rolling(period).mean()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return pd.Series(dtype=float)

        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        avg_gains = gains.rolling(period).mean()
        avg_losses = losses.rolling(period).mean()

        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def rank_gaps(
        self,
        gaps_df: DataFrame,
        top_k: int = 10,
        ranking_method: str = "gap_pct",
        sector_diversification: bool = True,
        max_per_sector: int = 3,
    ) -> DataFrame:
        """
        Rank and select top gap opportunities.

        Args:
            gaps_df: DataFrame with gap calculations
            top_k: Number of top opportunities to select
            ranking_method: Method for ranking ('gap_pct', 'gap_score', 'volume_weighted')
            sector_diversification: Apply sector diversification
            max_per_sector: Maximum positions per sector

        Returns:
            DataFrame with top ranked opportunities
        """
        if gaps_df.empty:
            return gaps_df

        # Calculate ranking score
        gaps_df = self._calculate_ranking_score(gaps_df, ranking_method)

        # Apply sector diversification if requested
        if sector_diversification:
            ranked_df = self._apply_sector_diversification(gaps_df, max_per_sector)
        else:
            ranked_df = gaps_df.sort_values('ranking_score', ascending=False)

        # Select top k
        top_gaps = ranked_df.head(top_k).copy()

        # Add ranking information
        top_gaps['rank'] = range(1, len(top_gaps) + 1)
        top_gaps['selection_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        logger.info(f"Selected top {len(top_gaps)} gaps using {ranking_method} method")
        return top_gaps

    def _calculate_ranking_score(self, gaps_df: DataFrame, method: str) -> DataFrame:
        """Calculate ranking score based on method."""
        df = gaps_df.copy()

        if method == "gap_pct":
            df['ranking_score'] = df['gap_pct']

        elif method == "gap_score":
            # Composite score considering gap size, volume, and volatility
            gap_score = df['gap_pct'].rank(pct=True)
            volume_score = df['volume_ratio'].fillna(1).rank(pct=True)
            
            # Penalize very high RSI (overbought)
            rsi_penalty = np.where(df['rsi_14'] > 70, 0.5, 1.0)
            rsi_penalty = np.where(df['rsi_14'] < 30, 1.2, rsi_penalty)  # Bonus for oversold
            
            df['ranking_score'] = (gap_score * 0.6 + volume_score * 0.4) * rsi_penalty

        elif method == "volume_weighted":
            # Weight gap by relative volume
            volume_weight = np.log1p(df['volume_ratio'].fillna(1))
            df['ranking_score'] = df['gap_pct'] * volume_weight

        else:
            raise ValueError(f"Unknown ranking method: {method}")

        return df

    def _apply_sector_diversification(
        self, gaps_df: DataFrame, max_per_sector: int
    ) -> DataFrame:
        """Apply sector diversification to gap selection."""
        diversified = []
        sector_counts = {}

        # Sort by ranking score first
        sorted_gaps = gaps_df.sort_values('ranking_score', ascending=False)

        for _, row in sorted_gaps.iterrows():
            sector = row.get('sector', 'Unknown')
            
            if sector_counts.get(sector, 0) < max_per_sector:
                diversified.append(row)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        return pd.DataFrame(diversified)

    def get_historical_gaps(
        self,
        start_date: datetime,
        end_date: datetime,
        top_k: int = 10,
        **kwargs
    ) -> Dict[str, DataFrame]:
        """Get historical gap rankings for backtesting."""
        historical_gaps = {}
        current_date = start_date

        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue

            try:
                gaps_df = self.calculate_gaps(current_date, **kwargs)
                if not gaps_df.empty:
                    ranked_gaps = self.rank_gaps(gaps_df, top_k=top_k)
                    historical_gaps[current_date.strftime('%Y-%m-%d')] = ranked_gaps

                logger.debug(f"Processed gaps for {current_date.strftime('%Y-%m-%d')}")

            except Exception as e:
                logger.error(f"Error processing gaps for {current_date}: {e}")

            current_date += timedelta(days=1)

        logger.info(f"Generated historical gaps for {len(historical_gaps)} trading days")
        return historical_gaps

    def validate_gap_calculation(
        self, symbol: str, date: datetime, expected_gap: Optional[float] = None
    ) -> Dict:
        """Validate gap calculation for a specific symbol and date."""
        gaps_df = self.calculate_gaps(date, universe_symbols=[symbol])
        
        if gaps_df.empty:
            return {"status": "error", "message": "No gap calculated"}

        gap_data = gaps_df.iloc[0].to_dict()
        
        result = {
            "status": "success",
            "symbol": symbol,
            "date": date.strftime('%Y-%m-%d'),
            "calculated_gap": gap_data['gap_pct'],
            "gap_details": gap_data
        }

        if expected_gap is not None:
            diff = abs(gap_data['gap_pct'] - expected_gap)
            result["expected_gap"] = expected_gap
            result["difference"] = diff
            result["within_tolerance"] = diff < 0.001  # 0.1% tolerance

        return result