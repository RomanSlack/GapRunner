"""Tests for the CLI module."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from click.testing import CliRunner

from gappers.cli import cli


class TestCLI:
    """Test cases for CLI commands."""
    
    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert 'Gap Trading System' in result.output
        assert 'backtest' in result.output
        assert 'sweep' in result.output
        assert 'live' in result.output
    
    def test_cli_version(self):
        """Test CLI version option."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
    
    def test_cli_verbose(self):
        """Test CLI verbose option."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--verbose', '--help'])
        
        assert result.exit_code == 0
    
    @patch('gappers.cli.Backtester')
    @patch('gappers.cli.DataFeed')
    @patch('gappers.cli.SignalGenerator')
    @patch('gappers.cli.UniverseBuilder')
    @patch('gappers.cli.PerformanceAnalyzer')
    def test_backtest_command_basic(self, mock_analyzer, mock_universe, mock_signal, mock_datafeed, mock_backtester):
        """Test basic backtest command."""
        # Setup mocks
        mock_backtester_instance = MagicMock()
        mock_analyzer_instance = MagicMock()
        
        mock_backtester.return_value = mock_backtester_instance
        mock_analyzer.return_value = mock_analyzer_instance
        
        # Mock backtest results
        mock_trades = [MagicMock(pnl_net=100, return_pct=0.05)]
        mock_portfolio = pd.DataFrame({'value': [100000, 105000]})
        mock_results = {
            'trades': mock_trades,
            'portfolio_values': mock_portfolio,
            'benchmark': pd.DataFrame()
        }
        mock_backtester_instance.run_backtest.return_value = mock_results
        
        # Mock analysis
        mock_analysis = {
            'trade_analysis': {
                'total_trades': 1,
                'win_rate': 1.0,
                'profit_factor': 2.0
            },
            'performance_metrics': {
                'total_return_pct': 5.0,
                'sharpe_ratio': 1.5
            }
        }
        mock_analyzer_instance.analyze_backtest_results.return_value = mock_analysis
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'backtest',
            '--start-date', '2023-01-01',
            '--end-date', '2023-01-31',
            '--profit-target', '0.05',
            '--stop-loss', '0.02'
        ])
        
        assert result.exit_code == 0
        assert 'Running backtest' in result.output
        assert 'Total Trades' in result.output
    
    @patch('gappers.cli.Backtester')
    @patch('gappers.cli.DataFeed')
    def test_backtest_command_with_output(self, mock_datafeed, mock_backtester, tmp_path):
        """Test backtest command with output file."""
        # Setup mocks
        mock_backtester_instance = MagicMock()
        mock_backtester.return_value = mock_backtester_instance
        
        mock_trades = [MagicMock(
            symbol='AAPL',
            entry_date=datetime(2023, 1, 1),
            exit_date=datetime(2023, 1, 1),
            entry_price=100.0,
            exit_price=105.0,
            position_size=100,
            pnl_gross=500.0,
            pnl_net=499.0,
            return_pct=0.05,
            hold_time_hours=2.0,
            exit_reason='profit_target',
            gap_pct=0.03,
            rank=1
        )]
        
        mock_results = {
            'trades': mock_trades,
            'portfolio_values': pd.DataFrame({'value': [100000, 105000]}),
            'benchmark': pd.DataFrame()
        }
        mock_backtester_instance.run_backtest.return_value = mock_results
        
        # Mock analyzer
        with patch('gappers.cli.PerformanceAnalyzer') as mock_analyzer:
            mock_analyzer_instance = MagicMock()
            mock_analyzer.return_value = mock_analyzer_instance
            mock_analyzer_instance.analyze_backtest_results.return_value = {
                'trade_analysis': {'total_trades': 1},
                'performance_metrics': {'sharpe_ratio': 1.5}
            }
            
            output_file = tmp_path / "results.csv"
            
            runner = CliRunner()
            result = runner.invoke(cli, [
                'backtest',
                '--start-date', '2023-01-01',
                '--end-date', '2023-01-31',
                '--output', str(output_file)
            ])
            
            assert result.exit_code == 0
            assert output_file.exists()
    
    @patch('gappers.cli.Backtester')
    @patch('gappers.cli.DataFeed')
    def test_backtest_command_error(self, mock_datafeed, mock_backtester):
        """Test backtest command with error."""
        # Setup mock to raise exception
        mock_backtester_instance = MagicMock()
        mock_backtester.return_value = mock_backtester_instance
        mock_backtester_instance.run_backtest.side_effect = Exception("Test error")
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'backtest',
            '--start-date', '2023-01-01',
            '--end-date', '2023-01-31'
        ])
        
        assert result.exit_code == 1
        assert 'Error running backtest' in result.output
    
    @patch('gappers.cli.Backtester')
    @patch('gappers.cli.DataFeed')
    def test_sweep_command_basic(self, mock_datafeed, mock_backtester, tmp_path):
        """Test parameter sweep command."""
        # Setup mocks
        mock_backtester_instance = MagicMock()
        mock_backtester.return_value = mock_backtester_instance
        
        # Mock sweep results
        mock_results_df = pd.DataFrame({
            'param_profit_target': [0.03, 0.05, 0.07],
            'param_stop_loss': [0.01, 0.02, 0.03],
            'total_trades': [100, 150, 120],
            'sharpe_ratio': [1.2, 1.8, 1.4],
            'total_return_pct': [15.0, 25.0, 18.0]
        })
        mock_backtester_instance.run_parameter_sweep.return_value = mock_results_df
        
        output_file = tmp_path / "sweep_results.csv"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'sweep',
            '--profit-target', '0.03', '--profit-target', '0.05',
            '--stop-loss', '0.01', '--stop-loss', '0.02',
            '--output', str(output_file)
        ])
        
        assert result.exit_code == 0
        assert 'Running parameter sweep' in result.output
        assert 'Top 5 Parameter Combinations' in result.output
        assert output_file.exists()
    
    @patch('gappers.cli.Backtester')
    @patch('gappers.cli.DataFeed')
    def test_sweep_command_empty_results(self, mock_datafeed, mock_backtester, tmp_path):
        """Test parameter sweep with empty results."""
        mock_backtester_instance = MagicMock()
        mock_backtester.return_value = mock_backtester_instance
        mock_backtester_instance.run_parameter_sweep.return_value = pd.DataFrame()
        
        output_file = tmp_path / "empty_results.csv"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'sweep',
            '--output', str(output_file)
        ])
        
        assert result.exit_code == 1
        assert 'No results from parameter sweep' in result.output
    
    @patch('gappers.cli.LiveTrader')
    def test_live_command_scan_only(self, mock_live_trader):
        """Test live command with scan-only mode."""
        # Setup mock trader
        mock_trader_instance = MagicMock()
        mock_live_trader.return_value = mock_trader_instance
        
        # Mock scan results
        mock_scan_results = {
            'gaps_found': 3,
            'opportunities': [
                {'rank': 1, 'symbol': 'AAPL', 'gap_pct': 0.05, 'previous_close': 150.0, 'current_open': 157.5},
                {'rank': 2, 'symbol': 'MSFT', 'gap_pct': 0.04, 'previous_close': 250.0, 'current_open': 260.0},
                {'rank': 3, 'symbol': 'GOOGL', 'gap_pct': 0.03, 'previous_close': 2500.0, 'current_open': 2575.0}
            ]
        }
        mock_trader_instance.run_single_scan.return_value = mock_scan_results
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'live',
            '--scan-only',
            '--dry-run'
        ])
        
        assert result.exit_code == 0
        assert 'Running gap scan' in result.output
        assert 'Found 3 gap opportunities' in result.output
        assert 'AAPL' in result.output
    
    @patch('gappers.cli.LiveTrader')
    def test_live_command_scan_error(self, mock_live_trader):
        """Test live command scan with error."""
        mock_trader_instance = MagicMock()
        mock_live_trader.return_value = mock_trader_instance
        
        # Mock scan error
        mock_trader_instance.run_single_scan.return_value = {
            'error': 'API connection failed'
        }
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'live',
            '--scan-only'
        ])
        
        assert result.exit_code == 1
        assert 'Scan error' in result.output
    
    @patch('gappers.cli.LiveTrader')
    @patch('gappers.cli.config')
    def test_live_command_no_credentials(self, mock_config, mock_live_trader):
        """Test live command without Alpaca credentials."""
        mock_config.has_alpaca_credentials = False
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'live'  # No --dry-run flag
        ])
        
        assert result.exit_code == 1
        assert 'Alpaca credentials required' in result.output
    
    @patch('gappers.cli.LiveTrader')
    @patch('gappers.cli.config')
    def test_live_command_with_credentials(self, mock_config, mock_live_trader):
        """Test live command with credentials."""
        mock_config.has_alpaca_credentials = True
        
        mock_trader_instance = MagicMock()
        mock_live_trader.return_value = mock_trader_instance
        
        # Mock KeyboardInterrupt to stop live trading
        mock_trader_instance.start_live_trading.side_effect = KeyboardInterrupt()
        
        runner = CliRunner()
        result = runner.invoke(cli, ['live'])
        
        assert result.exit_code == 0
        assert 'Live trading stopped by user' in result.output
    
    @patch('gappers.cli.SignalGenerator')
    @patch('gappers.cli.DataFeed')
    @patch('gappers.cli.UniverseBuilder')
    def test_scan_command_basic(self, mock_universe, mock_datafeed, mock_signal):
        """Test scan command."""
        # Setup mocks
        mock_signal_instance = MagicMock()
        mock_signal.return_value = mock_signal_instance
        
        # Mock gap data
        mock_gaps = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'gap_pct': [0.05, 0.04, 0.03],
            'previous_close': [150.0, 250.0, 2500.0],
            'current_open': [157.5, 260.0, 2575.0],
            'rank': [1, 2, 3],
            'sector': ['Technology', 'Technology', 'Technology']
        })
        
        mock_signal_instance.calculate_gaps.return_value = mock_gaps
        mock_signal_instance.rank_gaps.return_value = mock_gaps
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'scan',
            '--date', '2023-06-15',
            '--min-gap', '0.02',
            '--top-k', '5'
        ])
        
        assert result.exit_code == 0
        assert 'Scanning for gaps' in result.output
        assert 'Found 3 gap opportunities' in result.output
        assert 'AAPL' in result.output
    
    @patch('gappers.cli.SignalGenerator')
    @patch('gappers.cli.DataFeed')
    def test_scan_command_no_gaps(self, mock_datafeed, mock_signal):
        """Test scan command with no gaps found."""
        mock_signal_instance = MagicMock()
        mock_signal.return_value = mock_signal_instance
        
        # Return empty DataFrame
        mock_signal_instance.calculate_gaps.return_value = pd.DataFrame()
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'scan',
            '--date', '2023-06-15'
        ])
        
        assert result.exit_code == 0
        assert 'No gaps found matching criteria' in result.output
    
    @patch('gappers.cli.SignalGenerator')
    @patch('gappers.cli.DataFeed')
    def test_scan_command_with_output(self, mock_datafeed, mock_signal, tmp_path):
        """Test scan command with output file."""
        mock_signal_instance = MagicMock()
        mock_signal.return_value = mock_signal_instance
        
        mock_gaps = pd.DataFrame({
            'symbol': ['AAPL'],
            'gap_pct': [0.05],
            'previous_close': [150.0],
            'current_open': [157.5],
            'rank': [1]
        })
        
        mock_signal_instance.calculate_gaps.return_value = mock_gaps
        mock_signal_instance.rank_gaps.return_value = mock_gaps
        
        output_file = tmp_path / "scan_results.csv"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'scan',
            '--output', str(output_file)
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
    
    @patch('gappers.cli.DataFeed')
    def test_download_command_basic(self, mock_datafeed):
        """Test download command."""
        mock_datafeed_instance = MagicMock()
        mock_datafeed.return_value = mock_datafeed_instance
        
        # Mock downloaded data
        mock_data = {
            'AAPL': pd.DataFrame({'close': [150, 155, 152]}),
            'MSFT': pd.DataFrame({'close': [250, 260, 255]})
        }
        mock_datafeed_instance.download.return_value = mock_data
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'download',
            '--symbols', 'AAPL',
            '--symbols', 'MSFT',
            '--start-date', '2023-01-01',
            '--end-date', '2023-01-31'
        ])
        
        assert result.exit_code == 0
        assert 'Downloading 2 symbols' in result.output
        assert 'Downloaded data for 2 symbols' in result.output
        assert 'AAPL: 3 bars' in result.output
    
    @patch('gappers.cli.DataFeed')
    def test_download_command_with_output_dir(self, mock_datafeed, tmp_path):
        """Test download command with output directory."""
        mock_datafeed_instance = MagicMock()
        mock_datafeed.return_value = mock_datafeed_instance
        
        mock_data = {
            'AAPL': pd.DataFrame({'close': [150, 155, 152]})
        }
        mock_datafeed_instance.download.return_value = mock_data
        
        output_dir = tmp_path / "data_output"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'download',
            '--symbols', 'AAPL',
            '--output-dir', str(output_dir)
        ])
        
        assert result.exit_code == 0
        assert output_dir.exists()
        
        # Check that CSV file was created
        csv_files = list(output_dir.glob("*.csv"))
        assert len(csv_files) == 1
        assert 'AAPL' in csv_files[0].name
    
    @patch('gappers.cli.DataFeed')
    def test_status_command_clear_cache(self, mock_datafeed):
        """Test status command with cache clearing."""
        mock_datafeed_instance = MagicMock()
        mock_datafeed.return_value = mock_datafeed_instance
        mock_datafeed_instance.clear_cache.return_value = 15
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'status',
            '--clear-cache'
        ])
        
        assert result.exit_code == 0
        assert 'Cleared 15 cache files' in result.output
    
    @patch('gappers.cli.config')
    def test_status_command_check_config(self, mock_config):
        """Test status command with config check."""
        mock_config.data_path = "/test/data"
        mock_config.has_alpaca_credentials = True
        mock_config.has_premium_feeds = False
        mock_config.log_level = "INFO"
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'status',
            '--check-config'
        ])
        
        assert result.exit_code == 0
        assert 'Configuration Status' in result.output
        assert '/test/data' in result.output
        assert '✅' in result.output  # Alpaca credentials
    
    @patch('gappers.cli.DataFeed')
    @patch('gappers.cli.LiveTrader')
    @patch('gappers.cli.config')
    def test_status_command_test_connection(self, mock_config, mock_live_trader, mock_datafeed):
        """Test status command with connection testing."""
        # Mock successful yfinance connection
        mock_datafeed_instance = MagicMock()
        mock_datafeed.return_value = mock_datafeed_instance
        mock_datafeed_instance.download.return_value = {
            'AAPL': pd.DataFrame({'close': [150]})
        }
        
        # Mock Alpaca connection
        mock_config.has_alpaca_credentials = True
        mock_trader_instance = MagicMock()
        mock_live_trader.return_value = mock_trader_instance
        mock_trader_instance.get_portfolio_status.return_value = {'status': 'ok'}
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'status',
            '--test-connection'
        ])
        
        assert result.exit_code == 0
        assert 'Testing connections' in result.output
        assert 'yfinance: ✅' in result.output
    
    def test_status_command_no_options(self):
        """Test status command with no options."""
        runner = CliRunner()
        result = runner.invoke(cli, ['status'])
        
        assert result.exit_code == 0
        # Should complete without error even with no options
    
    @patch('gappers.cli.SignalGenerator')
    def test_scan_command_error(self, mock_signal):
        """Test scan command with error."""
        mock_signal_instance = MagicMock()
        mock_signal.return_value = mock_signal_instance
        mock_signal_instance.calculate_gaps.side_effect = Exception("Data error")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['scan'])
        
        assert result.exit_code == 1
        assert 'Error scanning for gaps' in result.output
    
    @patch('gappers.cli.DataFeed')  
    def test_download_command_error(self, mock_datafeed):
        """Test download command with error."""
        mock_datafeed_instance = MagicMock()
        mock_datafeed.return_value = mock_datafeed_instance
        mock_datafeed_instance.download.side_effect = Exception("Network error")
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'download',
            '--symbols', 'AAPL'
        ])
        
        assert result.exit_code == 1
        assert 'Error downloading data' in result.output