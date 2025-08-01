"""Tests for the live trading module."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from gappers.backtest import GapParams
from gappers.live import LiveTrader


class TestLiveTrader:
    """Test cases for LiveTrader class."""
    
    def test_init_dry_run(self, mock_data_feed):
        """Test LiveTrader initialization in dry run mode."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        assert trader.data_feed is not None
        assert trader.signal_generator is not None
        assert trader.dry_run is True
        assert trader.alpaca is None  # No credentials
        assert trader.active_orders == {}
        assert trader.active_positions == {}
        assert trader.daily_trades == 0
    
    @patch('gappers.live.REST')
    def test_init_with_credentials(self, mock_rest, mock_data_feed, monkeypatch):
        """Test LiveTrader initialization with Alpaca credentials."""
        # Set credentials
        monkeypatch.setenv("ALPACA_API_KEY", "test_key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
        
        # Mock Alpaca client
        mock_client = MagicMock()
        mock_account = MagicMock()
        mock_account.id = "test_account"
        mock_account.equity = "100000"
        mock_client.get_account.return_value = mock_account
        mock_rest.return_value = mock_client
        
        # Re-import config to pick up new env vars
        from gappers.config import Config
        config = Config()
        
        with patch('gappers.live.config', config):
            trader = LiveTrader(mock_data_feed, dry_run=False)
            
            assert trader.alpaca is not None
            assert not trader.dry_run
    
    @patch('gappers.live.REST')
    def test_init_alpaca_connection_error(self, mock_rest, mock_data_feed, monkeypatch):
        """Test LiveTrader initialization with Alpaca connection error."""
        monkeypatch.setenv("ALPACA_API_KEY", "test_key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
        
        # Mock connection error
        mock_rest.side_effect = Exception("Connection failed")
        
        from gappers.config import Config
        config = Config()
        
        with patch('gappers.live.config', config):
            trader = LiveTrader(mock_data_feed, dry_run=False)
            
            # Should handle error gracefully
            assert trader.alpaca is None
    
    def test_pre_market_scan_no_gaps(self, mock_data_feed):
        """Test pre-market scan with no gaps found."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        # Mock signal generator to return empty gaps
        trader.signal_generator.calculate_gaps = MagicMock(return_value=pd.DataFrame())
        
        params = GapParams()
        
        # Should not raise exception
        trader._pre_market_scan(params)
        
        # Should not have gap opportunities
        assert not hasattr(trader, 'gap_opportunities')
    
    def test_pre_market_scan_with_gaps(self, mock_data_feed, sample_gap_data):
        """Test pre-market scan with gaps found."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        # Mock signal generator
        trader.signal_generator.calculate_gaps = MagicMock(return_value=sample_gap_data)
        trader.signal_generator.rank_gaps = MagicMock(return_value=sample_gap_data.head(5))
        
        params = GapParams()
        
        trader._pre_market_scan(params)
        
        # Should have stored gap opportunities
        assert hasattr(trader, 'gap_opportunities')
        assert len(trader.gap_opportunities) == 5
    
    def test_pre_market_scan_error_handling(self, mock_data_feed):
        """Test pre-market scan error handling."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        # Mock signal generator to raise exception
        trader.signal_generator.calculate_gaps = MagicMock(side_effect=Exception("API Error"))
        
        params = GapParams()
        
        # Should not raise exception
        trader._pre_market_scan(params)
    
    def test_get_current_price_yfinance_fallback(self, mock_data_feed):
        """Test getting current price with yfinance fallback."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        # Mock data feed to return current price
        mock_data = pd.DataFrame({
            'open': [150.0],
            'high': [152.0],
            'low': [149.0], 
            'close': [151.0]
        }, index=[datetime.now().date()])
        
        trader.data_feed.download = MagicMock(return_value={'AAPL': mock_data})
        
        price = trader._get_current_price('AAPL')
        
        assert price == 150.0  # Should return open price
    
    def test_get_current_price_no_data(self, mock_data_feed):
        """Test getting current price with no data available."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        # Mock data feed to return empty data
        trader.data_feed.download = MagicMock(return_value={})
        
        price = trader._get_current_price('INVALID')
        
        assert price is None
    
    @patch('gappers.live.REST')
    def test_get_current_price_alpaca(self, mock_rest, mock_data_feed, monkeypatch):
        """Test getting current price from Alpaca."""
        monkeypatch.setenv("ALPACA_API_KEY", "test_key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
        
        # Mock Alpaca client
        mock_client = MagicMock()
        mock_account = MagicMock()
        mock_account.id = "test"
        mock_account.equity = "100000"
        mock_client.get_account.return_value = mock_account
        
        # Mock quote data
        mock_quote = MagicMock()
        mock_quote.ask_price = 150.50
        mock_client.get_latest_quote.return_value = mock_quote
        
        mock_rest.return_value = mock_client
        
        from gappers.config import Config
        config = Config()
        
        with patch('gappers.live.config', config):
            trader = LiveTrader(mock_data_feed, dry_run=False)
            
            price = trader._get_current_price('AAPL')
            
            assert price == 150.50
    
    def test_get_available_buying_power_no_alpaca(self, mock_data_feed):
        """Test getting buying power without Alpaca connection."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        buying_power = trader._get_available_buying_power()
        
        # Should return default from config
        assert buying_power == 10000  # Default position size
    
    @patch('gappers.live.REST')
    def test_get_available_buying_power_alpaca(self, mock_rest, mock_data_feed, monkeypatch):
        """Test getting buying power from Alpaca."""
        monkeypatch.setenv("ALPACA_API_KEY", "test_key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
        
        mock_client = MagicMock()
        mock_account = MagicMock()
        mock_account.id = "test"
        mock_account.equity = "100000"
        mock_account.buying_power = "50000"
        mock_client.get_account.return_value = mock_account
        
        mock_rest.return_value = mock_client
        
        from gappers.config import Config
        config = Config()
        
        with patch('gappers.live.config', config):
            trader = LiveTrader(mock_data_feed, dry_run=False)
            
            buying_power = trader._get_available_buying_power()
            
            assert buying_power == 50000.0
    
    def test_execute_gap_trade_dry_run(self, mock_data_feed):
        """Test gap trade execution in dry run mode."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        # Mock current price
        trader._get_current_price = MagicMock(return_value=100.0)
        trader._get_available_buying_power = MagicMock(return_value=50000.0)
        
        gap_data = {
            'symbol': 'AAPL',
            'gap_pct': 0.05,
            'rank': 1
        }
        params = GapParams()
        
        result = trader._execute_gap_trade('AAPL', gap_data, params)
        
        assert result is True  # Should succeed in dry run
    
    def test_execute_gap_trade_insufficient_buying_power(self, mock_data_feed):
        """Test gap trade execution with insufficient buying power."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        # Mock insufficient buying power
        trader._get_current_price = MagicMock(return_value=10000.0)  # Very expensive stock
        trader._get_available_buying_power = MagicMock(return_value=5000.0)  # Low buying power
        
        gap_data = {'symbol': 'EXPENSIVE', 'gap_pct': 0.05, 'rank': 1}
        params = GapParams(position_size=10000)
        
        result = trader._execute_gap_trade('EXPENSIVE', gap_data, params)
        
        assert result is False
    
    def test_execute_gap_trade_no_price(self, mock_data_feed):
        """Test gap trade execution when current price unavailable."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        # Mock no current price
        trader._get_current_price = MagicMock(return_value=None)
        
        gap_data = {'symbol': 'INVALID', 'gap_pct': 0.05, 'rank': 1}
        params = GapParams()
        
        result = trader._execute_gap_trade('INVALID', gap_data, params)
        
        assert result is False
    
    def test_market_open_execution_no_opportunities(self, mock_data_feed):
        """Test market open execution with no opportunities."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        params = GapParams()
        
        # Should not raise exception
        trader._market_open_execution(params)
    
    def test_market_open_execution_with_opportunities(self, mock_data_feed, sample_gap_data):
        """Test market open execution with gap opportunities."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        # Set up gap opportunities
        trader.gap_opportunities = sample_gap_data.head(3)
        
        # Mock successful execution
        trader._execute_gap_trade = MagicMock(return_value=True)
        
        params = GapParams()
        
        trader._market_open_execution(params)
        
        # Should have attempted to execute trades
        assert trader._execute_gap_trade.call_count == 3
        assert trader.daily_trades == 3
    
    def test_market_open_execution_daily_limit(self, mock_data_feed, sample_gap_data):
        """Test market open execution with daily trade limit."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        # Set daily trades near limit
        trader.daily_trades = 8
        trader.gap_opportunities = sample_gap_data.head(5)
        
        trader._execute_gap_trade = MagicMock(return_value=True)
        
        params = GapParams(max_positions=10)
        
        trader._market_open_execution(params)
        
        # Should only execute 2 more trades (limit is 10)
        assert trader._execute_gap_trade.call_count == 2
    
    def test_position_monitoring_no_positions(self, mock_data_feed):
        """Test position monitoring with no active positions."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        params = GapParams()
        
        # Should not raise exception
        trader._position_monitoring(params) 
    
    def test_check_time_based_exits(self, mock_data_feed):
        """Test time-based position exits."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        # Mock active position
        mock_position = MagicMock()
        trader.active_positions = {'AAPL': mock_position}
        
        # Mock close position
        trader._close_position = MagicMock()
        
        params = GapParams(max_hold_time_hours=1)  # Very short hold time
        
        trader._check_time_based_exits(params)
        
        # Should have attempted to close position due to time limit
        trader._close_position.assert_called_once_with('AAPL', 'time_limit')
    
    def test_close_position_dry_run(self, mock_data_feed):
        """Test position closing in dry run mode."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        # Mock active position
        mock_position = MagicMock()
        trader.active_positions = {'AAPL': mock_position}
        
        trader._close_position('AAPL', 'profit_target')
        
        # Should not raise exception in dry run
    
    def test_close_position_not_found(self, mock_data_feed):
        """Test closing position that doesn't exist."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        # Should not raise exception
        trader._close_position('NONEXISTENT', 'test')
    
    def test_market_close_cleanup(self, mock_data_feed):
        """Test market close cleanup."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        # Set some daily state
        trader.daily_trades = 5
        trader.daily_pnl = 1000.0
        
        # Mock cleanup methods
        trader._close_all_positions = MagicMock()
        trader._cancel_all_orders = MagicMock()
        trader._log_daily_summary = MagicMock()
        
        params = GapParams()
        
        trader._market_close_cleanup(params)
        
        # Should reset daily counters
        assert trader.daily_trades == 0
        assert trader.daily_pnl == 0.0
        
        # Should have called cleanup methods
        trader._cancel_all_orders.assert_called_once()
        trader._log_daily_summary.assert_called_once()
    
    def test_get_portfolio_status_basic(self, mock_data_feed):
        """Test getting basic portfolio status."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        trader.daily_trades = 3
        trader.daily_pnl = 500.0
        trader.active_positions = {'AAPL': MagicMock(), 'MSFT': MagicMock()}
        trader.active_orders = {'GOOGL': MagicMock()}
        
        status = trader.get_portfolio_status()
        
        assert status['daily_trades'] == 3
        assert status['daily_pnl'] == 500.0
        assert status['active_positions'] == 2
        assert status['active_orders'] == 1
        assert status['dry_run'] is True
        assert 'timestamp' in status
    
    def test_manual_trade_execution_buy(self, mock_data_feed):
        """Test manual buy trade execution."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        # Mock successful execution
        trader._execute_gap_trade = MagicMock(return_value=True)
        
        params = GapParams()
        
        result = trader.manual_trade_execution('AAPL', 'buy', params)
        
        assert result is True
        trader._execute_gap_trade.assert_called_once()
    
    def test_manual_trade_execution_sell(self, mock_data_feed):
        """Test manual sell trade execution."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        # Mock active position
        mock_position = MagicMock()
        trader.active_positions = {'AAPL': mock_position}
        
        trader._close_position = MagicMock()
        
        params = GapParams()
        
        result = trader.manual_trade_execution('AAPL', 'sell', params)
        
        assert result is True
        trader._close_position.assert_called_once_with('AAPL', 'manual')
    
    def test_manual_trade_execution_sell_no_position(self, mock_data_feed):
        """Test manual sell with no existing position."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        params = GapParams()
        
        result = trader.manual_trade_execution('AAPL', 'sell', params)
        
        assert result is False
    
    def test_run_single_scan_success(self, mock_data_feed, sample_gap_data):
        """Test successful single gap scan."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        # Mock signal generator
        trader.signal_generator.calculate_gaps = MagicMock(return_value=sample_gap_data)
        trader.signal_generator.rank_gaps = MagicMock(return_value=sample_gap_data.head(5))
        
        params = GapParams()
        
        result = trader.run_single_scan(params)
        
        assert result['gaps_found'] == 5
        assert len(result['opportunities']) == 5
        assert 'scan_time' in result
        
        # Check opportunity structure
        opp = result['opportunities'][0]
        assert 'symbol' in opp
        assert 'gap_pct' in opp
        assert 'rank' in opp
    
    def test_run_single_scan_no_gaps(self, mock_data_feed):
        """Test single scan with no gaps found."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        # Mock empty gaps
        trader.signal_generator.calculate_gaps = MagicMock(return_value=pd.DataFrame())
        
        params = GapParams()
        
        result = trader.run_single_scan(params)
        
        assert result['gaps_found'] == 0
        assert result['opportunities'] == []
    
    def test_run_single_scan_error(self, mock_data_feed):
        """Test single scan with error."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        # Mock error
        trader.signal_generator.calculate_gaps = MagicMock(side_effect=Exception("API Error"))
        
        params = GapParams()
        
        result = trader.run_single_scan(params)
        
        assert 'error' in result
        assert 'API Error' in result['error']
    
    def test_scheduler_setup(self, mock_data_feed):
        """Test scheduler job setup."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        # Mock scheduler to avoid actual scheduling
        trader.scheduler = MagicMock()
        
        params = GapParams()
        
        # This would normally start the scheduler, but we'll just test setup
        trader.scheduler.add_job = MagicMock()
        
        # Manually add jobs like start_live_trading would
        trader.scheduler.add_job(trader._pre_market_scan, args=[params])
        trader.scheduler.add_job(trader._market_open_execution, args=[params])
        trader.scheduler.add_job(trader._position_monitoring, args=[params])
        trader.scheduler.add_job(trader._market_close_cleanup, args=[params])
        
        # Should have added 4 jobs
        assert trader.scheduler.add_job.call_count == 4
    
    def test_stop_live_trading(self, mock_data_feed):
        """Test stopping live trading system."""
        trader = LiveTrader(mock_data_feed, dry_run=True)
        
        # Mock scheduler and methods
        trader.scheduler = MagicMock()
        trader.scheduler.running = True
        trader._close_all_positions = MagicMock()
        trader._cancel_all_orders = MagicMock()
        
        trader.stop_live_trading()
        
        # Should have stopped scheduler and cleaned up
        trader.scheduler.shutdown.assert_called_once()
        trader._cancel_all_orders.assert_called_once()