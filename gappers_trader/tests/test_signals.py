"""Tests for the signals module."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from gappers.signals import SignalGenerator


class TestSignalGenerator:
    """Test cases for SignalGenerator class."""
    
    def test_init(self, mock_data_feed, mock_universe_builder):
        """Test SignalGenerator initialization."""
        signal_gen = SignalGenerator(mock_data_feed, mock_universe_builder)
        assert signal_gen.data_feed is not None
        assert signal_gen.universe_builder is not None
    
    def test_calculate_symbol_gap_basic(self, mock_data_feed, mock_universe_builder):
        """Test basic gap calculation for a single symbol."""
        signal_gen = SignalGenerator(mock_data_feed, mock_universe_builder)
        
        # Create test data with a clear gap
        test_date = datetime(2023, 6, 15)
        dates = [test_date - timedelta(days=1), test_date]
        
        price_df = pd.DataFrame({
            'open': [105.0, 110.0],  # 4.76% gap up
            'high': [106.0, 112.0],
            'low': [99.0, 109.0],
            'close': [100.0, 111.0],
            'volume': [1000000, 1500000]
        }, index=dates)
        
        result = signal_gen._calculate_symbol_gap('AAPL', price_df, test_date)
        
        assert result is not None
        assert result['symbol'] == 'AAPL'
        assert abs(result['gap_pct'] - 0.10) < 0.01  # ~10% gap
        assert result['previous_close'] == 100.0
        assert result['current_open'] == 110.0
        assert result['volume_ratio'] == 1.5
    
    def test_calculate_symbol_gap_no_gap(self, mock_data_feed, mock_universe_builder):
        """Test gap calculation when there's no significant gap."""
        signal_gen = SignalGenerator(mock_data_feed, mock_universe_builder)
        
        test_date = datetime(2023, 6, 15)
        dates = [test_date - timedelta(days=1), test_date]
        
        # No gap scenario
        price_df = pd.DataFrame({
            'open': [100.0, 100.5],  # Small 0.5% gap
            'high': [101.0, 102.0],
            'low': [99.0, 99.5],
            'close': [100.0, 101.0],
            'volume': [1000000, 1000000]
        }, index=dates)
        
        result = signal_gen._calculate_symbol_gap('AAPL', price_df, test_date)
        
        assert result is not None
        assert abs(result['gap_pct'] - 0.005) < 0.001  # ~0.5% gap
    
    def test_calculate_symbol_gap_insufficient_data(self, mock_data_feed, mock_universe_builder):
        """Test gap calculation with insufficient data."""
        signal_gen = SignalGenerator(mock_data_feed, mock_universe_builder)
        
        test_date = datetime(2023, 6, 15)
        
        # Only one day of data
        price_df = pd.DataFrame({
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.0],
            'volume': [1000000]
        }, index=[test_date])
        
        result = signal_gen._calculate_symbol_gap('AAPL', price_df, test_date)
        
        assert result is None
    
    def test_calculate_symbol_gap_invalid_prices(self, mock_data_feed, mock_universe_builder):
        """Test gap calculation with invalid prices."""
        signal_gen = SignalGenerator(mock_data_feed, mock_universe_builder)
        
        test_date = datetime(2023, 6, 15)
        dates = [test_date - timedelta(days=1), test_date]
        
        # Invalid data (NaN values)
        price_df = pd.DataFrame({
            'open': [np.nan, 110.0],
            'high': [106.0, 112.0],
            'low': [99.0, 109.0],
            'close': [100.0, 111.0],
            'volume': [1000000, 1500000]
        }, index=dates)
        
        result = signal_gen._calculate_symbol_gap('AAPL', price_df, test_date)
        
        assert result is None
    
    def test_filter_gaps(self, mock_data_feed, mock_universe_builder):
        """Test gap filtering functionality."""
        signal_gen = SignalGenerator(mock_data_feed, mock_universe_builder)
        
        # Create sample gaps data
        gaps_df = pd.DataFrame({
            'symbol': ['A', 'B', 'C', 'D', 'E'],
            'gap_pct': [0.05, -0.03, 0.01, 0.40, 0.08],  # Various gap sizes
            'previous_close': [100, 100, 100, 100, 100],
            'current_open': [105, 97, 101, 140, 108]
        })
        
        # Filter: min 2%, max 30%, positive only
        filtered = signal_gen._filter_gaps(
            gaps_df, 
            min_gap_pct=0.02, 
            max_gap_pct=0.30, 
            include_negative_gaps=False
        )
        
        # Should include A (5%) and E (8%), exclude others
        assert len(filtered) == 2
        assert set(filtered['symbol']) == {'A', 'E'}
    
    def test_filter_gaps_include_negative(self, mock_data_feed, mock_universe_builder):
        """Test gap filtering with negative gaps included."""
        signal_gen = SignalGenerator(mock_data_feed, mock_universe_builder)
        
        gaps_df = pd.DataFrame({
            'symbol': ['A', 'B', 'C'],
            'gap_pct': [0.05, -0.03, 0.01],
            'previous_close': [100, 100, 100],
            'current_open': [105, 97, 101]
        })
        
        # Include negative gaps with 2% minimum absolute
        filtered = signal_gen._filter_gaps(
            gaps_df, 
            min_gap_pct=0.02, 
            max_gap_pct=0.30, 
            include_negative_gaps=True
        )
        
        # Should include A (5%) and B (-3%), exclude C (1%)
        assert len(filtered) == 2
        assert set(filtered['symbol']) == {'A', 'B'}
    
    def test_calculate_atr(self, mock_data_feed, mock_universe_builder):
        """Test Average True Range calculation."""
        signal_gen = SignalGenerator(mock_data_feed, mock_universe_builder)
        
        # Create test data
        df = pd.DataFrame({
            'high': [105, 108, 107, 110, 109],
            'low': [95, 97, 98, 105, 106],
            'close': [100, 105, 103, 108, 107]
        })
        
        atr = signal_gen._calculate_atr(df, period=3)
        
        assert len(atr) == len(df)
        assert not atr.isna().all()
        assert atr.iloc[-1] > 0  # ATR should be positive
    
    def test_calculate_rsi(self, mock_data_feed, mock_universe_builder):
        """Test RSI calculation."""
        signal_gen = SignalGenerator(mock_data_feed, mock_universe_builder)
        
        # Create trending price data
        prices = pd.Series([100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111, 113, 112, 114, 116])
        
        rsi = signal_gen._calculate_rsi(prices, period=14)
        
        assert len(rsi) == len(prices)
        assert not rsi.isna().all()
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert all(0 <= val <= 100 for val in valid_rsi)
    
    def test_calculate_ranking_score_gap_pct(self, mock_data_feed, mock_universe_builder):
        """Test ranking score calculation using gap percentage method."""
        signal_gen = SignalGenerator(mock_data_feed, mock_universe_builder)
        
        gaps_df = pd.DataFrame({
            'gap_pct': [0.05, 0.08, 0.03, 0.10],
            'volume_ratio': [1.5, 2.0, 1.2, 1.8],
            'rsi_14': [60, 70, 50, 80]
        })
        
        result = signal_gen._calculate_ranking_score(gaps_df, 'gap_pct')
        
        assert 'ranking_score' in result.columns
        assert result['ranking_score'].equals(result['gap_pct'])
    
    def test_calculate_ranking_score_composite(self, mock_data_feed, mock_universe_builder):
        """Test composite ranking score calculation."""
        signal_gen = SignalGenerator(mock_data_feed, mock_universe_builder)
        
        gaps_df = pd.DataFrame({
            'gap_pct': [0.05, 0.08, 0.03, 0.10],
            'volume_ratio': [1.5, 2.0, 1.2, 1.8],
            'rsi_14': [60, 75, 50, 25]  # High RSI should be penalized
        })
        
        result = signal_gen._calculate_ranking_score(gaps_df, 'gap_score')
        
        assert 'ranking_score' in result.columns
        # All scores should be positive
        assert all(result['ranking_score'] > 0)
        # Score should be influenced by RSI penalty
        assert result.loc[1, 'ranking_score'] < result.loc[0, 'ranking_score']  # High RSI penalty
    
    def test_apply_sector_diversification(self, mock_data_feed, mock_universe_builder):
        """Test sector diversification logic."""
        signal_gen = SignalGenerator(mock_data_feed, mock_universe_builder)
        
        gaps_df = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'BAC', 'GS'],
            'ranking_score': [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            'sector': ['Tech', 'Tech', 'Tech', 'Finance', 'Finance', 'Finance']
        })
        
        # Allow max 2 per sector
        diversified = signal_gen._apply_sector_diversification(gaps_df, max_per_sector=2)
        
        # Should have 4 total (2 Tech + 2 Finance)
        assert len(diversified) == 4
        
        # Check sector distribution
        tech_count = sum(diversified['sector'] == 'Tech')
        finance_count = sum(diversified['sector'] == 'Finance')
        assert tech_count <= 2
        assert finance_count <= 2
    
    def test_rank_gaps_basic(self, mock_data_feed, mock_universe_builder, sample_gap_data):
        """Test basic gap ranking functionality."""
        signal_gen = SignalGenerator(mock_data_feed, mock_universe_builder)
        
        result = signal_gen.rank_gaps(sample_gap_data, top_k=5)
        
        assert len(result) <= 5
        assert 'rank' in result.columns
        assert 'selection_date' in result.columns
        # Should be sorted by ranking score descending
        assert result['ranking_score'].is_monotonic_decreasing
    
    def test_rank_gaps_with_diversification(self, mock_data_feed, mock_universe_builder):
        """Test gap ranking with sector diversification."""
        signal_gen = SignalGenerator(mock_data_feed, mock_universe_builder)
        
        gaps_df = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'BAC'],
            'gap_pct': [0.08, 0.07, 0.06, 0.05, 0.04],
            'volume_ratio': [1.5, 1.4, 1.3, 1.2, 1.1],
            'rsi_14': [60, 65, 70, 55, 50],
            'sector': ['Tech', 'Tech', 'Tech', 'Finance', 'Finance']
        })
        
        result = signal_gen.rank_gaps(
            gaps_df, 
            top_k=10, 
            sector_diversification=True, 
            max_per_sector=2
        )
        
        # Should respect sector limits
        tech_count = sum(result['sector'] == 'Tech')
        finance_count = sum(result['sector'] == 'Finance')
        assert tech_count <= 2
        assert finance_count <= 2
    
    @patch.object(SignalGenerator, 'calculate_gaps')
    def test_get_historical_gaps(self, mock_calculate_gaps, mock_data_feed, mock_universe_builder, sample_gap_data):
        """Test historical gap generation."""
        mock_calculate_gaps.return_value = sample_gap_data
        
        signal_gen = SignalGenerator(mock_data_feed, mock_universe_builder)
        
        start_date = datetime(2023, 6, 15)
        end_date = datetime(2023, 6, 17)
        
        historical = signal_gen.get_historical_gaps(start_date, end_date, top_k=5)
        
        # Should have entries for weekdays only
        expected_dates = ['2023-06-15', '2023-06-16']  # Skip weekend
        assert len(historical) == len(expected_dates)
        
        for date_str in expected_dates:
            assert date_str in historical
            assert len(historical[date_str]) <= 5
    
    def test_validate_gap_calculation(self, mock_data_feed, mock_universe_builder):
        """Test gap calculation validation."""
        with patch.object(SignalGenerator, 'calculate_gaps') as mock_calc:
            # Mock successful gap calculation 
            mock_gaps = pd.DataFrame({
                'symbol': ['AAPL'],
                'gap_pct': [0.05],
                'previous_close': [100.0],
                'current_open': [105.0]
            })
            mock_calc.return_value = mock_gaps
            
            signal_gen = SignalGenerator(mock_data_feed, mock_universe_builder)
            
            result = signal_gen.validate_gap_calculation(
                'AAPL', 
                datetime(2023, 6, 15),
                expected_gap=0.05
            )
            
            assert result['status'] == 'success'
            assert result['symbol'] == 'AAPL'
            assert result['calculated_gap'] == 0.05
            assert result['within_tolerance'] is True
    
    def test_validate_gap_calculation_no_data(self, mock_data_feed, mock_universe_builder):
        """Test gap validation with no data."""
        with patch.object(SignalGenerator, 'calculate_gaps', return_value=pd.DataFrame()):
            signal_gen = SignalGenerator(mock_data_feed, mock_universe_builder)
            
            result = signal_gen.validate_gap_calculation('AAPL', datetime(2023, 6, 15))
            
            assert result['status'] == 'error'
            assert 'No gap calculated' in result['message']
    
    def test_calculate_gaps_integration(self, mock_data_feed, mock_universe_builder):
        """Test full gap calculation workflow."""
        signal_gen = SignalGenerator(mock_data_feed, mock_universe_builder)
        
        test_date = datetime(2023, 6, 15)
        
        # Mock the universe builder to return test symbols
        with patch.object(mock_universe_builder, 'build_universe') as mock_universe:
            mock_universe.return_value = pd.DataFrame({
                'symbol': ['AAPL', 'MSFT'],
                'price': [150, 250],
                'median_dollar_volume': [1e9, 8e8]
            })
            
            result = signal_gen.calculate_gaps(test_date, universe_symbols=['AAPL'])
            
            # Should process the specified symbols
            assert isinstance(result, pd.DataFrame)
            # Result could be empty if mock data doesn't have gaps
    
    def test_error_handling_in_gap_calculation(self, mock_data_feed, mock_universe_builder):
        """Test error handling during gap calculation."""
        # Mock data feed to raise an exception
        mock_data_feed.download = MagicMock(side_effect=Exception("Network error"))
        
        signal_gen = SignalGenerator(mock_data_feed, mock_universe_builder)
        
        # Should not raise exception, should return empty DataFrame
        result = signal_gen.calculate_gaps(datetime(2023, 6, 15), universe_symbols=['AAPL'])
        
        assert isinstance(result, pd.DataFrame)
        # Could be empty due to error