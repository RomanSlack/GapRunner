"""Tests for the universe module."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from gappers.universe import UniverseBuilder


class TestUniverseBuilder:
    """Test cases for UniverseBuilder class."""
    
    def test_init(self, mock_data_feed, temp_data_dir):
        """Test UniverseBuilder initialization."""
        universe_builder = UniverseBuilder(mock_data_feed)
        assert universe_builder.data_feed is not None
        assert universe_builder.cache_dir.exists()
    
    def test_get_fallback_symbols(self, mock_data_feed):
        """Test fallback symbol list."""
        universe_builder = UniverseBuilder(mock_data_feed)
        
        fallback_symbols = universe_builder._get_fallback_symbols()
        
        assert isinstance(fallback_symbols, set)
        assert len(fallback_symbols) > 50  # Should have major stocks
        assert 'AAPL' in fallback_symbols
        assert 'MSFT' in fallback_symbols
        assert 'GOOGL' in fallback_symbols
    
    @patch('requests.get')
    def test_get_symbols_from_fmp_success(self, mock_get, mock_data_feed):
        """Test successful symbol retrieval from FMP API."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [
            {'symbol': 'AAPL', 'exchangeShortName': 'NYSE', 'type': 'stock'},
            {'symbol': 'MSFT', 'exchangeShortName': 'NASDAQ', 'type': 'stock'},
            {'symbol': 'GOOGL', 'exchangeShortName': 'NASDAQ', 'type': 'stock'},
            {'symbol': 'BOND.PR', 'exchangeShortName': 'NYSE', 'type': 'stock'},  # Should be filtered
            {'symbol': 'TOOLONG', 'exchangeShortName': 'NYSE', 'type': 'stock'},  # Should be kept
        ]
        mock_get.return_value = mock_response
        
        universe_builder = UniverseBuilder(mock_data_feed)
        
        symbols = universe_builder._get_symbols_from_fmp(['NYSE', 'NASDAQ'])
        
        assert isinstance(symbols, set)
        assert 'AAPL' in symbols
        assert 'MSFT' in symbols
        assert 'GOOGL' in symbols
        assert 'BOND.PR' not in symbols  # Should be filtered out (has dot)
        assert 'TOOLONG' in symbols  # 7 chars is still acceptable
    
    @patch('requests.get')
    def test_get_symbols_from_fmp_api_error(self, mock_get, mock_data_feed):
        """Test FMP API error handling."""
        # Mock API error
        mock_get.side_effect = requests.RequestException("API Error")
        
        universe_builder = UniverseBuilder(mock_data_feed)
        
        symbols = universe_builder._get_symbols_from_fmp(['NYSE'])
        
        assert symbols == set()  # Should return empty set on error
    
    @patch('requests.get')
    def test_get_symbols_from_fmp_invalid_response(self, mock_get, mock_data_feed):
        """Test FMP API with invalid response."""
        # Mock invalid JSON response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = "invalid"  # Not a list
        mock_get.return_value = mock_response
        
        universe_builder = UniverseBuilder(mock_data_feed)
        
        symbols = universe_builder._get_symbols_from_fmp(['NYSE'])
        
        assert symbols == set()
    
    def test_get_current_listings_fallback(self, mock_data_feed):
        """Test current listings with fallback to hardcoded symbols."""
        with patch.object(UniverseBuilder, '_get_symbols_from_fmp', return_value=set()):
            universe_builder = UniverseBuilder(mock_data_feed)
            
            symbols = universe_builder._get_current_listings(['NYSE', 'NASDAQ'])
            
            # Should use fallback symbols
            assert len(symbols) > 50
            assert 'AAPL' in symbols
    
    def test_get_delisted_symbols_no_api(self, mock_data_feed):
        """Test delisted symbol retrieval without API."""
        universe_builder = UniverseBuilder(mock_data_feed)
        
        test_date = datetime(2023, 6, 15)
        delisted = universe_builder._get_delisted_symbols(test_date, ['NYSE'])
        
        # Should return some known delisted symbols (from fallback)
        assert isinstance(delisted, set)
    
    def test_filter_universe_basic(self, mock_data_feed, sample_ohlcv_data):
        """Test basic universe filtering."""
        universe_builder = UniverseBuilder(mock_data_feed)
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        test_date = datetime(2023, 6, 15)
        
        # Should use mock data feed
        result = universe_builder._filter_universe(
            symbols, 
            test_date, 
            min_dollar_volume=1_000_000, 
            min_price=50.0, 
            max_price=3000.0
        )
        
        assert isinstance(result, pd.DataFrame)
        # Results depend on mock data characteristics
    
    def test_filter_universe_empty_data(self, mock_data_feed):
        """Test universe filtering with no data."""
        # Mock data feed to return empty data
        mock_data_feed.download = MagicMock(return_value={})
        
        universe_builder = UniverseBuilder(mock_data_feed)
        
        result = universe_builder._filter_universe(
            ['INVALID'], 
            datetime(2023, 6, 15), 
            1_000_000, 
            50.0, 
            3000.0
        )
        
        assert result.empty
    
    def test_filter_universe_price_filters(self, mock_data_feed):
        """Test universe filtering with price constraints."""
        # Mock data with specific price ranges
        def mock_download(symbols, start, end, interval):
            result = {}
            for symbol in symbols:
                if symbol == 'CHEAP':
                    # Below minimum price
                    df = pd.DataFrame({
                        'close': [4.0, 4.5, 4.2],
                        'volume': [1000000] * 3
                    }, index=pd.date_range(start, periods=3))
                    result[symbol] = df
                elif symbol == 'EXPENSIVE':
                    # Above maximum price  
                    df = pd.DataFrame({
                        'close': [1100.0, 1150.0, 1200.0],
                        'volume': [1000000] * 3
                    }, index=pd.date_range(start, periods=3))
                    result[symbol] = df
                elif symbol == 'VALID':
                    # Within price range
                    df = pd.DataFrame({
                        'close': [100.0, 105.0, 102.0],
                        'volume': [2000000] * 3
                    }, index=pd.date_range(start, periods=3))
                    result[symbol] = df
            return result
        
        mock_data_feed.download = mock_download
        
        universe_builder = UniverseBuilder(mock_data_feed)
        
        result = universe_builder._filter_universe(
            ['CHEAP', 'EXPENSIVE', 'VALID'], 
            datetime(2023, 6, 15), 
            min_dollar_volume=1_000_000,
            min_price=50.0, 
            max_price=1000.0
        )
        
        # Only VALID should pass filters
        assert len(result) == 1
        assert result.iloc[0]['symbol'] == 'VALID'
    
    def test_add_metadata_basic(self, mock_data_feed):
        """Test adding metadata to universe."""
        universe_builder = UniverseBuilder(mock_data_feed)
        
        universe_df = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'price': [150.0, 250.0],
            'median_dollar_volume': [1e9, 8e8]
        })
        
        with patch('yfinance.Ticker') as mock_ticker:
            # Mock yfinance ticker info
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.info = {
                'sector': 'Technology',
                'marketCap': 3e12,
                'floatShares': 16e9
            }
            mock_ticker.return_value = mock_ticker_instance
            
            result = universe_builder._add_metadata(universe_df, datetime(2023, 6, 15))
            
            assert 'sector' in result.columns
            assert 'market_cap' in result.columns
            assert 'float_shares' in result.columns
    
    def test_add_metadata_api_error(self, mock_data_feed):
        """Test metadata addition with API errors."""
        universe_builder = UniverseBuilder(mock_data_feed)
        
        universe_df = pd.DataFrame({
            'symbol': ['AAPL'],
            'price': [150.0],
            'median_dollar_volume': [1e9]
        })
        
        with patch('yfinance.Ticker', side_effect=Exception("API Error")):
            result = universe_builder._add_metadata(universe_df, datetime(2023, 6, 15))
            
            # Should still have metadata columns with default values
            assert 'sector' in result.columns
            assert result.iloc[0]['sector'] == 'Unknown'
    
    def test_cache_path_generation(self, mock_data_feed):
        """Test cache path generation."""
        universe_builder = UniverseBuilder(mock_data_feed)
        
        test_date = datetime(2023, 6, 15)
        cache_path = universe_builder._get_cache_path(
            test_date, 1_000_000, 5.0, 1000.0
        )
        
        assert 'universe_20230615' in cache_path.name
        assert 'dv1000k' in cache_path.name  # Dollar volume
        assert 'p5.0-1000.0' in cache_path.name  # Price range
        assert cache_path.suffix == '.parquet'
    
    def test_cache_validity_check(self, mock_data_feed, temp_data_dir):
        """Test cache validity checking."""
        universe_builder = UniverseBuilder(mock_data_feed)
        
        # Create a cache file
        cache_path = temp_data_dir / "test_cache.parquet"
        cache_path.write_text("fake cache data")
        
        # Fresh file should be valid
        assert universe_builder._is_cache_valid(cache_path) is True
        
        # Make file old
        old_time = (datetime.now() - timedelta(days=2)).timestamp()
        cache_path.touch(times=(old_time, old_time))
        
        # Old file should be invalid
        assert universe_builder._is_cache_valid(cache_path) is False
    
    def test_save_to_cache(self, mock_data_feed, temp_data_dir):
        """Test saving universe to cache."""
        universe_builder = UniverseBuilder(mock_data_feed)
        
        universe_df = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'price': [150.0, 250.0]
        })
        
        cache_path = temp_data_dir / "test_cache.parquet"
        
        universe_builder._save_to_cache(universe_df, cache_path)
        
        assert cache_path.exists()
        
        # Verify contents
        loaded_df = pd.read_parquet(cache_path)
        pd.testing.assert_frame_equal(universe_df, loaded_df)
    
    def test_build_universe_basic(self, mock_data_feed):
        """Test basic universe building."""
        with patch.object(UniverseBuilder, '_get_current_listings') as mock_current:
            with patch.object(UniverseBuilder, '_get_delisted_symbols') as mock_delisted:
                with patch.object(UniverseBuilder, '_filter_universe') as mock_filter:
                    with patch.object(UniverseBuilder, '_add_metadata') as mock_metadata:
                        # Setup mocks
                        mock_current.return_value = {'AAPL', 'MSFT'}
                        mock_delisted.return_value = {'GE'}
                        
                        filtered_df = pd.DataFrame({
                            'symbol': ['AAPL', 'MSFT'],
                            'price': [150.0, 250.0]
                        })
                        mock_filter.return_value = filtered_df
                        mock_metadata.return_value = filtered_df
                        
                        universe_builder = UniverseBuilder(mock_data_feed)
                        
                        result = universe_builder.build_universe(
                            date=datetime(2023, 6, 15),
                            force_refresh=True
                        )
                        
                        assert not result.empty
                        assert len(result) == 2
                        assert set(result['symbol']) == {'AAPL', 'MSFT'}
    
    def test_build_universe_with_cache(self, mock_data_feed, temp_data_dir):
        """Test universe building with cache."""
        universe_builder = UniverseBuilder(mock_data_feed)
        universe_builder.cache_dir = temp_data_dir
        
        test_date = datetime(2023, 6, 15)
        
        # Create cached data
        cached_df = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT'],
            'price': [150.0, 250.0]
        })
        
        cache_path = universe_builder._get_cache_path(test_date, 1_000_000, 5.0, 1000.0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached_df.to_parquet(cache_path)
        
        # Should load from cache
        result = universe_builder.build_universe(date=test_date)
        
        pd.testing.assert_frame_equal(result, cached_df)
    
    def test_build_universe_no_delisted(self, mock_data_feed):
        """Test universe building without delisted symbols."""
        with patch.object(UniverseBuilder, '_get_current_listings') as mock_current:
            with patch.object(UniverseBuilder, '_filter_universe') as mock_filter:
                with patch.object(UniverseBuilder, '_add_metadata') as mock_metadata:
                    mock_current.return_value = {'AAPL', 'MSFT'}
                    
                    filtered_df = pd.DataFrame({
                        'symbol': ['AAPL'],
                        'price': [150.0]
                    })
                    mock_filter.return_value = filtered_df
                    mock_metadata.return_value = filtered_df
                    
                    universe_builder = UniverseBuilder(mock_data_feed)
                    
                    result = universe_builder.build_universe(
                        date=datetime(2023, 6, 15),
                        include_delisted=False,
                        force_refresh=True
                    )
                    
                    assert not result.empty
                    assert len(result) == 1
    
    def test_get_historical_universe(self, mock_data_feed):
        """Test historical universe generation."""
        with patch.object(UniverseBuilder, 'build_universe') as mock_build:
            mock_universe = pd.DataFrame({
                'symbol': ['AAPL', 'MSFT'],
                'price': [150.0, 250.0]
            })
            mock_build.return_value = mock_universe
            
            universe_builder = UniverseBuilder(mock_data_feed)
            
            start_date = datetime(2023, 6, 15)
            end_date = datetime(2023, 6, 17)
            
            historical = universe_builder.get_historical_universe(start_date, end_date)
            
            # Should have 3 days of data
            assert len(historical) == 3
            assert '2023-06-15' in historical
            assert '2023-06-16' in historical
            assert '2023-06-17' in historical
    
    def test_get_historical_universe_with_errors(self, mock_data_feed):
        """Test historical universe with some errors."""
        def mock_build_with_error(date=None, **kwargs):
            if date and date.day == 16:  # Error on the 16th
                raise Exception("API Error")
            return pd.DataFrame({
                'symbol': ['AAPL'],
                'price': [150.0]
            })
        
        with patch.object(UniverseBuilder, 'build_universe', side_effect=mock_build_with_error):
            universe_builder = UniverseBuilder(mock_data_feed)
            
            start_date = datetime(2023, 6, 15)
            end_date = datetime(2023, 6, 17)
            
            historical = universe_builder.get_historical_universe(start_date, end_date)
            
            # Should have 2 days (15th and 17th), skip 16th due to error
            assert len(historical) == 2
            assert '2023-06-15' in historical
            assert '2023-06-16' not in historical
            assert '2023-06-17' in historical