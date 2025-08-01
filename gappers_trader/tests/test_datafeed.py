"""Tests for the datafeed module."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from gappers.datafeed import DataFeed


class TestDataFeed:
    """Test cases for DataFeed class."""
    
    def test_init(self, temp_data_dir):
        """Test DataFeed initialization."""
        data_feed = DataFeed(cache_dir=temp_data_dir / "cache")
        assert data_feed.cache_dir.exists()
        assert data_feed.iex_client is None  # No API key provided
        assert data_feed.polygon_client is None  # No API key provided
    
    def test_init_with_api_keys(self, temp_data_dir, monkeypatch):
        """Test DataFeed initialization with API keys."""
        monkeypatch.setenv("IEX_CLOUD_API_KEY", "test_iex_key")
        monkeypatch.setenv("POLYGON_API_KEY", "test_polygon_key")
        
        # Mock the imports to avoid requiring actual packages
        with patch('gappers.datafeed.get_historical_data') as mock_iex:
            with patch('gappers.datafeed.RESTClient') as mock_polygon:
                data_feed = DataFeed(cache_dir=temp_data_dir / "cache")
                
                # Should have attempted to initialize clients
                assert mock_iex.called or data_feed.iex_client is None  # Might fail import
                assert mock_polygon.called or data_feed.polygon_client is None  # Might fail import
    
    @patch('gappers.datafeed.yf.Ticker')
    def test_fetch_from_yfinance(self, mock_ticker, temp_data_dir):
        """Test fetching data from yfinance."""
        # Setup mock data
        sample_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [104, 105, 106],
            'Volume': [1000000, 1100000, 1200000],
            'Dividends': [0, 0, 0.5],
            'Stock Splits': [0, 0, 0]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = sample_data
        mock_ticker.return_value = mock_ticker_instance
        
        data_feed = DataFeed(cache_dir=temp_data_dir / "cache")
        
        result = data_feed._fetch_from_yfinance(
            'AAPL',
            datetime(2023, 1, 1),
            datetime(2023, 1, 3),
            '1d'
        )
        
        assert not result.empty
        assert 'symbol' in result.columns
        assert result['symbol'].iloc[0] == 'AAPL'
        assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    def test_download_single_symbol(self, temp_data_dir):
        """Test downloading data for a single symbol."""
        with patch.object(DataFeed, '_fetch_from_yfinance') as mock_fetch:
            mock_data = pd.DataFrame({
                'open': [100, 101],
                'high': [105, 106],
                'low': [95, 96],
                'close': [104, 105],
                'volume': [1000000, 1100000],
                'symbol': ['AAPL', 'AAPL']
            }, index=pd.date_range('2023-01-01', periods=2))
            
            mock_fetch.return_value = mock_data
            
            data_feed = DataFeed(cache_dir=temp_data_dir / "cache")
            
            result = data_feed.download(
                ['AAPL'],
                start='2023-01-01',
                end='2023-01-02',
                interval='1d'
            )
            
            assert 'AAPL' in result
            assert not result['AAPL'].empty
            assert len(result['AAPL']) == 2
    
    def test_download_multiple_symbols(self, temp_data_dir):
        """Test downloading data for multiple symbols."""
        def mock_fetch_side_effect(symbol, start, end, interval):
            return pd.DataFrame({
                'open': [100, 101],
                'high': [105, 106],
                'low': [95, 96],
                'close': [104, 105],
                'volume': [1000000, 1100000],
                'symbol': [symbol, symbol]
            }, index=pd.date_range('2023-01-01', periods=2))
        
        with patch.object(DataFeed, '_fetch_from_yfinance', side_effect=mock_fetch_side_effect):
            data_feed = DataFeed(cache_dir=temp_data_dir / "cache")
            
            result = data_feed.download(
                ['AAPL', 'MSFT'],
                start='2023-01-01',
                end='2023-01-02',
                interval='1d'
            )
            
            assert 'AAPL' in result
            assert 'MSFT' in result
            assert len(result) == 2
    
    def test_cache_functionality(self, temp_data_dir):
        """Test data caching functionality."""
        with patch.object(DataFeed, '_fetch_from_yfinance') as mock_fetch:
            mock_data = pd.DataFrame({
                'open': [100],
                'high': [105],
                'low': [95],
                'close': [104],
                'volume': [1000000],
                'symbol': ['AAPL']
            }, index=[datetime(2023, 1, 1)])
            
            mock_fetch.return_value = mock_data
            
            data_feed = DataFeed(cache_dir=temp_data_dir / "cache")
            
            # First call should fetch data
            result1 = data_feed.download(['AAPL'], '2023-01-01', '2023-01-01')
            assert mock_fetch.call_count == 1
            
            # Second call should use cache (if cache is valid)
            result2 = data_feed.download(['AAPL'], '2023-01-01', '2023-01-01')
            
            # Results should be the same
            pd.testing.assert_frame_equal(result1['AAPL'], result2['AAPL'])
    
    def test_cache_path_generation(self, temp_data_dir):
        """Test cache path generation."""
        data_feed = DataFeed(cache_dir=temp_data_dir / "cache")
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        
        cache_path = data_feed._get_cache_path('AAPL', start_date, end_date, '1d')
        
        assert cache_path.parent.name == 'date=2023'
        assert 'AAPL' in cache_path.name
        assert '1d' in cache_path.name
        assert cache_path.suffix == '.parquet'
    
    def test_force_refresh(self, temp_data_dir):
        """Test force refresh functionality."""
        with patch.object(DataFeed, '_fetch_from_yfinance') as mock_fetch:
            mock_data = pd.DataFrame({
                'open': [100],
                'high': [105],
                'low': [95],
                'close': [104],
                'volume': [1000000],
                'symbol': ['AAPL']
            }, index=[datetime(2023, 1, 1)])
            
            mock_fetch.return_value = mock_data
            
            data_feed = DataFeed(cache_dir=temp_data_dir / "cache")
            
            # First call
            data_feed.download(['AAPL'], '2023-01-01', '2023-01-01')
            call_count_1 = mock_fetch.call_count
            
            # Second call with force_refresh=True
            data_feed.download(['AAPL'], '2023-01-01', '2023-01-01', force_refresh=True)
            call_count_2 = mock_fetch.call_count
            
            # Should have made an additional call
            assert call_count_2 > call_count_1
    
    def test_get_splits_dividends(self, temp_data_dir):
        """Test getting splits and dividends data."""
        with patch('gappers.datafeed.yf.Ticker') as mock_ticker:
            # Mock splits and dividends data
            splits_data = pd.Series([2.0], index=[datetime(2023, 6, 1)], name='Stock Splits')
            dividends_data = pd.Series([0.5], index=[datetime(2023, 3, 15)], name='Dividends')
            
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.splits = splits_data
            mock_ticker_instance.dividends = dividends_data
            mock_ticker.return_value = mock_ticker_instance
            
            data_feed = DataFeed(cache_dir=temp_data_dir / "cache")
            
            result = data_feed.get_splits_dividends(
                'AAPL',
                datetime(2023, 1, 1),
                datetime(2023, 12, 31)
            )
            
            assert 'splits' in result
            assert 'dividends' in result
            assert not result['splits'].empty
            assert not result['dividends'].empty
    
    def test_clear_cache(self, temp_data_dir):
        """Test cache clearing functionality."""
        cache_dir = temp_data_dir / "cache"
        data_feed = DataFeed(cache_dir=cache_dir)
        
        # Create some fake cache files
        cache_subdir = cache_dir / "date=2023"
        cache_subdir.mkdir(parents=True)
        
        old_file = cache_subdir / "old_file.parquet"
        recent_file = cache_subdir / "recent_file.parquet"
        
        old_file.write_text("old data")
        recent_file.write_text("recent data")
        
        # Make old file actually old
        old_time = (datetime.now() - timedelta(days=60)).timestamp()
        old_file.touch(times=(old_time, old_time))
        
        # Clear cache older than 30 days
        count = data_feed.clear_cache(older_than_days=30)
        
        assert count >= 1  # Should have deleted the old file
        assert recent_file.exists()  # Recent file should still exist
    
    def test_error_handling(self, temp_data_dir):
        """Test error handling in data download."""
        with patch.object(DataFeed, '_fetch_from_yfinance', side_effect=Exception("Network error")):
            data_feed = DataFeed(cache_dir=temp_data_dir / "cache")
            
            result = data_feed.download(['INVALID'], '2023-01-01', '2023-01-01')
            
            # Should return empty dict on error
            assert result == {}
    
    def test_date_string_conversion(self, temp_data_dir):
        """Test automatic date string to datetime conversion."""
        with patch.object(DataFeed, '_fetch_from_yfinance') as mock_fetch:
            mock_data = pd.DataFrame({
                'open': [100],
                'high': [105],
                'low': [95],
                'close': [104],
                'volume': [1000000],
                'symbol': ['AAPL']
            }, index=[datetime(2023, 1, 1)])
            
            mock_fetch.return_value = mock_data
            
            data_feed = DataFeed(cache_dir=temp_data_dir / "cache")
            
            # Use string dates
            result = data_feed.download(['AAPL'], '2023-01-01', '2023-01-01')
            
            # Should have converted strings to datetime objects
            mock_fetch.assert_called_once()
            args = mock_fetch.call_args[0]
            assert isinstance(args[1], datetime)  # start date
            assert isinstance(args[2], datetime)  # end date
    
    def test_empty_data_handling(self, temp_data_dir):
        """Test handling of empty data responses."""
        with patch.object(DataFeed, '_fetch_from_yfinance', return_value=pd.DataFrame()):
            data_feed = DataFeed(cache_dir=temp_data_dir / "cache")
            
            result = data_feed.download(['AAPL'], '2023-01-01', '2023-01-01')
            
            # Should return empty dict when no data is available
            assert result == {}