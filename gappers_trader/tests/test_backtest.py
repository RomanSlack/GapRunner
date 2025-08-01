"""Tests for the backtest module."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from gappers.backtest import Backtester, GapParams, TradeResult


class TestGapParams:
    """Test cases for GapParams dataclass."""
    
    def test_default_params(self):
        """Test default parameter values."""
        params = GapParams()
        
        assert params.profit_target == 0.05
        assert params.stop_loss == 0.02
        assert params.max_hold_time_hours == 6
        assert params.top_k == 10
        assert params.position_size == 10000
        assert params.commission_per_share == 0.005
        assert params.slippage_bps == 10
    
    def test_custom_params(self):
        """Test custom parameter values."""
        params = GapParams(
            profit_target=0.08,
            stop_loss=0.03,
            top_k=15,
            position_size=20000
        )
        
        assert params.profit_target == 0.08
        assert params.stop_loss == 0.03
        assert params.top_k == 15
        assert params.position_size == 20000
        # Other params should remain default
        assert params.max_hold_time_hours == 6


class TestTradeResult:
    """Test cases for TradeResult dataclass."""
    
    def test_trade_result_creation(self):
        """Test creating a TradeResult instance."""
        entry_date = datetime(2023, 6, 15, 9, 30)
        exit_date = datetime(2023, 6, 15, 14, 30)
        
        trade = TradeResult(
            symbol='AAPL',
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=100.0,
            exit_price=105.0,
            position_size=100,
            pnl_gross=500.0,
            pnl_net=499.0,
            return_pct=0.05,
            hold_time_hours=5.0,
            exit_reason='profit_target',
            gap_pct=0.03,
            rank=1
        )
        
        assert trade.symbol == 'AAPL'
        assert trade.return_pct == 0.05
        assert trade.exit_reason == 'profit_target'


class TestBacktester:
    """Test cases for Backtester class."""
    
    def test_init(self, mock_data_feed):
        """Test Backtester initialization."""
        backtester = Backtester(mock_data_feed)
        assert backtester.data_feed is not None
        assert backtester.signal_generator is not None
    
    def test_find_exit_point_profit_target(self, mock_data_feed):
        """Test finding exit point when profit target is hit."""
        backtester = Backtester(mock_data_feed)
        
        # Create intraday data that hits profit target
        times = pd.date_range('2023-06-15 09:30', periods=60, freq='1min')
        intraday_df = pd.DataFrame({
            'open': np.linspace(100, 108, 60),
            'high': np.linspace(101, 109, 60),  # High prices increase
            'low': np.linspace(99, 107, 60),
            'close': np.linspace(100.5, 108.5, 60),
        }, index=times)
        
        profit_target = 105.0  # Should be hit around minute 30
        stop_loss = 95.0
        params = GapParams()
        
        result = backtester._find_exit_point(intraday_df, profit_target, stop_loss, params)
        
        assert result is not None
        exit_price, exit_time, exit_reason = result
        assert exit_reason == 'profit_target'
        assert exit_price == profit_target
    
    def test_find_exit_point_stop_loss(self, mock_data_feed):
        """Test finding exit point when stop loss is hit."""
        backtester = Backtester(mock_data_feed)
        
        # Create intraday data that hits stop loss
        times = pd.date_range('2023-06-15 09:30', periods=60, freq='1min')
        intraday_df = pd.DataFrame({
            'open': np.linspace(100, 92, 60),
            'high': np.linspace(101, 93, 60),
            'low': np.linspace(99, 91, 60),  # Low prices decrease
            'close': np.linspace(100.5, 92.5, 60),
        }, index=times)
        
        profit_target = 110.0
        stop_loss = 95.0  # Should be hit around minute 30
        params = GapParams()
        
        result = backtester._find_exit_point(intraday_df, profit_target, stop_loss, params)
        
        assert result is not None
        exit_price, exit_time, exit_reason = result
        assert exit_reason == 'stop_loss'
        assert exit_price == stop_loss
    
    def test_find_exit_point_time_limit(self, mock_data_feed):
        """Test finding exit point when time limit is reached."""
        backtester = Backtester(mock_data_feed)
        
        # Create intraday data that doesn't hit targets
        times = pd.date_range('2023-06-15 09:30', periods=8*60, freq='1min')  # 8 hours
        intraday_df = pd.DataFrame({
            'open': [100] * len(times),
            'high': [102] * len(times),
            'low': [98] * len(times),
            'close': [101] * len(times),
        }, index=times)
        
        profit_target = 110.0  # Won't be hit
        stop_loss = 90.0  # Won't be hit
        params = GapParams(max_hold_time_hours=4)  # 4 hour limit
        
        result = backtester._find_exit_point(intraday_df, profit_target, stop_loss, params)
        
        assert result is not None
        exit_price, exit_time, exit_reason = result
        assert exit_reason == 'time_limit'
        assert exit_price == 101  # Close price
    
    def test_find_exit_point_eod(self, mock_data_feed):
        """Test finding exit point at end of day."""
        backtester = Backtester(mock_data_feed)
        
        # Create short intraday data (market closes early)
        times = pd.date_range('2023-06-15 09:30', periods=30, freq='1min')
        intraday_df = pd.DataFrame({
            'open': [100] * 30,
            'high': [102] * 30,
            'low': [98] * 30,
            'close': [101] * 30,
        }, index=times)
        
        profit_target = 110.0
        stop_loss = 90.0
        params = GapParams(max_hold_time_hours=8)  # Long time limit
        
        result = backtester._find_exit_point(intraday_df, profit_target, stop_loss, params)
        
        assert result is not None
        exit_price, exit_time, exit_reason = result
        assert exit_reason == 'eod'
        assert exit_price == 101
    
    def test_simulate_single_trade_success(self, mock_data_feed, sample_intraday_data):
        """Test successful single trade simulation."""
        backtester = Backtester(mock_data_feed)
        
        gap_row = {
            'symbol': 'AAPL',
            'gap_pct': 0.05,
            'rank': 1
        }
        
        params = GapParams(profit_target=0.03, stop_loss=0.02)
        trade_date = datetime(2023, 6, 15)
        
        # Use sample intraday data
        intraday_df = sample_intraday_data['AAPL']
        
        result = backtester._simulate_single_trade(
            'AAPL', gap_row, intraday_df, params, trade_date
        )
        
        assert result is not None
        assert isinstance(result, TradeResult)
        assert result.symbol == 'AAPL'
        assert result.gap_pct == 0.05
        assert result.position_size > 0
    
    def test_simulate_single_trade_empty_data(self, mock_data_feed):
        """Test single trade simulation with empty data."""
        backtester = Backtester(mock_data_feed)
        
        gap_row = {'symbol': 'AAPL', 'gap_pct': 0.05, 'rank': 1}
        params = GapParams()
        trade_date = datetime(2023, 6, 15)
        
        # Empty intraday data
        intraday_df = pd.DataFrame()
        
        result = backtester._simulate_single_trade(
            'AAPL', gap_row, intraday_df, params, trade_date
        )
        
        assert result is None
    
    def test_calculate_summary_metrics(self, mock_data_feed, sample_trades):
        """Test calculation of summary metrics."""
        backtester = Backtester(mock_data_feed)
        
        # Create sample portfolio values
        dates = pd.date_range('2023-01-01', periods=len(sample_trades), freq='D')
        portfolio_values = pd.DataFrame({
            'value': np.cumsum([100000] + [trade.pnl_net for trade in sample_trades])
        }, index=dates)
        
        metrics = backtester._calculate_summary_metrics(sample_trades, portfolio_values)
        
        assert 'total_trades' in metrics
        assert 'win_rate' in metrics
        assert 'avg_return_pct' in metrics
        assert 'total_return_pct' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown_pct' in metrics
        
        assert metrics['total_trades'] == len(sample_trades)
        assert 0 <= metrics['win_rate'] <= 1
    
    def test_calculate_summary_metrics_empty(self, mock_data_feed):
        """Test summary metrics with empty data."""
        backtester = Backtester(mock_data_feed)
        
        metrics = backtester._calculate_summary_metrics([], pd.DataFrame())
        
        assert metrics == {}
    
    @patch.object(Backtester, '_run_vectorized_simulation')
    @patch.object(Backtester, '_get_benchmark_data')
    def test_run_backtest(self, mock_benchmark, mock_simulation, mock_data_feed, sample_trades):
        """Test full backtest run."""
        # Mock the simulation to return sample trades
        mock_portfolio = pd.DataFrame({
            'value': [100000, 105000, 103000],
            'daily_pnl': [0, 5000, -2000],
            'trades_count': [0, 2, 1]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        mock_simulation.return_value = (sample_trades[:3], mock_portfolio)
        mock_benchmark.return_value = pd.DataFrame({'close': [400, 402, 401]})
        
        # Mock signal generator
        with patch.object(Backtester, '__init__', lambda x, y, z=None: None):
            backtester = Backtester(None, None)
            backtester.data_feed = mock_data_feed
            backtester.signal_generator = MagicMock()
            backtester.signal_generator.get_historical_gaps.return_value = {
                '2023-01-01': pd.DataFrame({'symbol': ['AAPL'], 'gap_pct': [0.05]})
            }
            
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 3)
            params = GapParams()
            
            results = backtester.run_backtest(start_date, end_date, params)
            
            assert 'trades' in results
            assert 'portfolio_values' in results
            assert 'benchmark' in results
            assert 'params' in results
            assert results['total_trades'] == 3
    
    def test_validate_backtest_success(self, mock_data_feed, sample_trades):
        """Test successful backtest validation."""
        backtester = Backtester(mock_data_feed)
        
        # Create good portfolio values
        portfolio_values = pd.DataFrame({
            'value': np.linspace(100000, 120000, len(sample_trades))  # 20% gain
        }, index=pd.date_range('2023-01-01', periods=len(sample_trades)))
        
        results = {
            'trades': sample_trades,
            'portfolio_values': portfolio_values
        }
        
        validation = backtester.validate_backtest(results, min_trades=10)
        
        assert validation['passed'] is True
        assert len(validation['issues']) == 0
        assert 'summary' in validation
    
    def test_validate_backtest_insufficient_trades(self, mock_data_feed, sample_trades):
        """Test backtest validation with insufficient trades."""
        backtester = Backtester(mock_data_feed)
        
        portfolio_values = pd.DataFrame({
            'value': [100000, 105000]
        }, index=pd.date_range('2023-01-01', periods=2))
        
        results = {
            'trades': sample_trades[:5],  # Only 5 trades
            'portfolio_values': portfolio_values
        }
        
        validation = backtester.validate_backtest(results, min_trades=10)
        
        assert validation['passed'] is False
        assert any('Insufficient trades' in issue for issue in validation['issues'])
    
    def test_validate_backtest_excessive_drawdown(self, mock_data_feed, sample_trades):
        """Test backtest validation with excessive drawdown."""
        backtester = Backtester(mock_data_feed)
        
        # Create portfolio with large drawdown
        values = [100000]
        for i in range(len(sample_trades)):
            if i == 10:  # Large loss at trade 10
                values.append(values[-1] * 0.7)  # 30% drawdown
            else:
                values.append(values[-1] * 1.01)  # Small gains
        
        portfolio_values = pd.DataFrame({
            'value': values[1:]  # Remove initial value
        }, index=pd.date_range('2023-01-01', periods=len(sample_trades)))
        
        results = {
            'trades': sample_trades,
            'portfolio_values': portfolio_values
        }
        
        validation = backtester.validate_backtest(results, max_drawdown=-0.20)
        
        assert validation['passed'] is False
        assert any('Excessive drawdown' in issue for issue in validation['issues'])
    
    def test_get_benchmark_data(self, mock_data_feed):
        """Test benchmark data retrieval."""
        backtester = Backtester(mock_data_feed)
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        
        benchmark = backtester._get_benchmark_data('SPY', start_date, end_date)
        
        # Should return data from mock_data_feed
        assert isinstance(benchmark, pd.DataFrame)
        if not benchmark.empty:
            assert 'returns' in benchmark.columns
            assert 'cumulative_returns' in benchmark.columns
    
    def test_get_benchmark_data_missing_symbol(self, mock_data_feed):
        """Test benchmark data with missing symbol."""
        # Modify mock to return empty for missing symbol
        original_download = mock_data_feed.download
        
        def mock_download_missing(symbols, **kwargs):
            if 'MISSING' in symbols:
                return {}
            return original_download(symbols, **kwargs)
        
        mock_data_feed.download = mock_download_missing
        
        backtester = Backtester(mock_data_feed)
        
        benchmark = backtester._get_benchmark_data('MISSING', datetime(2023, 1, 1), datetime(2023, 1, 31))
        
        assert benchmark.empty
    
    def test_run_parameter_sweep_basic(self, mock_data_feed):
        """Test basic parameter sweep functionality."""
        with patch.object(Backtester, 'run_backtest') as mock_backtest:
            # Mock backtest results
            mock_trades = [MagicMock(return_pct=0.05, pnl_net=500) for _ in range(10)]
            mock_portfolio = pd.DataFrame({
                'value': np.linspace(100000, 110000, 10)
            }, index=pd.date_range('2023-01-01', periods=10))
            
            mock_backtest.return_value = {
                'trades': mock_trades,
                'portfolio_values': mock_portfolio
            }
            
            backtester = Backtester(mock_data_feed)
            
            param_grid = {
                'profit_target': [0.03, 0.05],
                'stop_loss': [0.01, 0.02]
            }
            
            results_df = backtester.run_parameter_sweep(
                datetime(2023, 1, 1),
                datetime(2023, 1, 31),
                param_grid
            )
            
            assert not results_df.empty
            assert len(results_df) == 4  # 2 x 2 combinations
            assert 'param_profit_target' in results_df.columns
            assert 'param_stop_loss' in results_df.columns
            assert 'sharpe_ratio' in results_df.columns
    
    def test_run_parameter_sweep_error_handling(self, mock_data_feed):
        """Test parameter sweep with errors."""
        with patch.object(Backtester, 'run_backtest', side_effect=Exception("Test error")):
            backtester = Backtester(mock_data_feed)
            
            param_grid = {
                'profit_target': [0.05],
                'stop_loss': [0.02]
            }
            
            results_df = backtester.run_parameter_sweep(
                datetime(2023, 1, 1),
                datetime(2023, 1, 31),
                param_grid
            )
            
            # Should return empty DataFrame when all combinations fail
            assert results_df.empty