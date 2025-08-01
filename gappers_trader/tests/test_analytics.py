"""Tests for the analytics module."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from gappers.analytics import PerformanceAnalyzer


class TestPerformanceAnalyzer:
    """Test cases for PerformanceAnalyzer class."""
    
    def test_init(self):
        """Test PerformanceAnalyzer initialization."""
        analyzer = PerformanceAnalyzer()
        assert analyzer.figures_created == []
    
    def test_analyze_trades_basic(self, sample_trades):
        """Test basic trade analysis."""
        analyzer = PerformanceAnalyzer()
        
        analysis = analyzer._analyze_trades(sample_trades)
        
        assert 'total_trades' in analysis
        assert 'winners' in analysis
        assert 'losers' in analysis
        assert 'win_rate' in analysis
        assert 'avg_return_pct' in analysis
        assert 'profit_factor' in analysis
        
        assert analysis['total_trades'] == len(sample_trades)
        assert 0 <= analysis['win_rate'] <= 1
        assert analysis['winners'] + analysis['losers'] == len(sample_trades)
    
    def test_analyze_trades_empty(self):
        """Test trade analysis with empty trade list."""
        analyzer = PerformanceAnalyzer()
        
        analysis = analyzer._analyze_trades([])
        
        assert analysis == {}
    
    def test_analyze_exit_reasons(self, sample_trades):
        """Test exit reason analysis."""
        analyzer = PerformanceAnalyzer()
        
        exit_reasons = analyzer._analyze_exit_reasons(sample_trades)
        
        assert isinstance(exit_reasons, dict)
        assert all(0 <= ratio <= 1 for ratio in exit_reasons.values())
        assert abs(sum(exit_reasons.values()) - 1.0) < 1e-6  # Should sum to 1
    
    def test_calculate_profit_factor(self):
        """Test profit factor calculation."""
        analyzer = PerformanceAnalyzer()
        
        # Test with profits and losses
        pnl_list = [100, -50, 200, -75, 150]
        profit_factor = analyzer._calculate_profit_factor(pnl_list)
        
        expected = (100 + 200 + 150) / (50 + 75)  # 450 / 125 = 3.6
        assert abs(profit_factor - expected) < 1e-6
    
    def test_calculate_profit_factor_no_losses(self):
        """Test profit factor with no losses."""
        analyzer = PerformanceAnalyzer()
        
        pnl_list = [100, 200, 150]
        profit_factor = analyzer._calculate_profit_factor(pnl_list)
        
        assert profit_factor == float('inf')
    
    def test_calculate_profit_factor_no_profits(self):
        """Test profit factor with no profits."""
        analyzer = PerformanceAnalyzer()
        
        pnl_list = [-100, -200, -150]
        profit_factor = analyzer._calculate_profit_factor(pnl_list)
        
        assert profit_factor == 0
    
    def test_calculate_performance_metrics(self, sample_trades):
        """Test portfolio performance metrics calculation."""
        analyzer = PerformanceAnalyzer()
        
        # Create portfolio values
        portfolio_values = pd.DataFrame({
            'value': np.cumsum([100000] + [trade.pnl_net for trade in sample_trades])
        }, index=pd.date_range('2023-01-01', periods=len(sample_trades) + 1))
        
        metrics = analyzer._calculate_performance_metrics(sample_trades, portfolio_values)
        
        assert 'total_return_pct' in metrics
        assert 'cagr_pct' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'volatility_pct' in metrics
        assert 'max_drawdown_pct' in metrics
        
        # Validate ranges
        assert metrics['sharpe_ratio'] >= -10  # Reasonable range
        assert metrics['sharpe_ratio'] <= 10
        assert metrics['max_drawdown_pct'] <= 0  # Drawdown should be negative
    
    def test_calculate_performance_metrics_empty(self):
        """Test performance metrics with empty data."""
        analyzer = PerformanceAnalyzer()
        
        metrics = analyzer._calculate_performance_metrics([], pd.DataFrame())
        
        assert metrics == {}
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        analyzer = PerformanceAnalyzer()
        
        # Create sample daily returns
        daily_returns = pd.Series([0.01, -0.005, 0.02, 0.005, -0.01, 0.015])
        
        sharpe = analyzer._calculate_sharpe_ratio(daily_returns, risk_free_rate=0.02)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_calculate_sharpe_ratio_insufficient_data(self):
        """Test Sharpe ratio with insufficient data."""
        analyzer = PerformanceAnalyzer()
        
        daily_returns = pd.Series([0.01])  # Only one return
        
        sharpe = analyzer._calculate_sharpe_ratio(daily_returns)
        
        assert sharpe == 0
    
    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        analyzer = PerformanceAnalyzer()
        
        daily_returns = pd.Series([0.01, -0.005, 0.02, 0.005, -0.01, 0.015])
        
        sortino = analyzer._calculate_sortino_ratio(daily_returns, risk_free_rate=0.02)
        
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)
    
    def test_calculate_sortino_ratio_no_downside(self):
        """Test Sortino ratio with no downside deviation."""
        analyzer = PerformanceAnalyzer()
        
        # All positive returns
        daily_returns = pd.Series([0.01, 0.015, 0.02, 0.005, 0.03, 0.025])
        
        sortino = analyzer._calculate_sortino_ratio(daily_returns, risk_free_rate=0.0)
        
        assert sortino == float('inf')
    
    def test_calculate_drawdown_metrics(self):
        """Test drawdown metrics calculation."""
        analyzer = PerformanceAnalyzer()
        
        # Create portfolio values with a clear drawdown
        values = [100000, 105000, 103000, 98000, 99000, 104000, 106000]
        portfolio_values = pd.Series(values)
        
        metrics = analyzer._calculate_drawdown_metrics(portfolio_values)
        
        assert 'max_drawdown_pct' in metrics
        assert 'avg_drawdown_duration_days' in metrics
        assert 'max_drawdown_duration_days' in metrics
        assert 'num_drawdown_periods' in metrics
        
        assert metrics['max_drawdown_pct'] < 0  # Should be negative
        assert metrics['num_drawdown_periods'] >= 0
    
    def test_calculate_risk_metrics(self, sample_trades):
        """Test risk metrics calculation."""
        analyzer = PerformanceAnalyzer()
        
        portfolio_values = pd.DataFrame({
            'value': np.cumsum([100000] + [trade.pnl_net for trade in sample_trades])
        }, index=pd.date_range('2023-01-01', periods=len(sample_trades) + 1))
        
        risk_metrics = analyzer._calculate_risk_metrics(sample_trades, portfolio_values)
        
        assert 'var_95_pct' in risk_metrics
        assert 'var_99_pct' in risk_metrics
        assert 'cvar_95_pct' in risk_metrics
        assert 'cvar_99_pct' in risk_metrics
        assert 'beta' in risk_metrics
        assert 'trade_return_volatility_pct' in risk_metrics
        
        # VaR should be more extreme than CVaR
        assert risk_metrics['var_99_pct'] <= risk_metrics['var_95_pct']
    
    def test_analyze_by_sector(self, sample_trades):
        """Test sector analysis."""
        analyzer = PerformanceAnalyzer()
        
        # Add sector attribute to sample trades
        sectors = ['Technology', 'Finance', 'Healthcare'] * (len(sample_trades) // 3 + 1)
        for i, trade in enumerate(sample_trades):
            trade.sector = sectors[i] if i < len(sectors) else 'Technology'
        
        sector_analysis = analyzer._analyze_by_sector(sample_trades)
        
        assert isinstance(sector_analysis, dict)
        assert len(sector_analysis) > 0
        
        for sector, data in sector_analysis.items():
            assert 'trade_count' in data
            assert 'win_rate' in data
            assert 'avg_return_pct' in data
            assert 'total_pnl' in data
            assert 'volatility_pct' in data
    
    def test_analyze_temporal_patterns(self, sample_trades):
        """Test temporal pattern analysis."""
        analyzer = PerformanceAnalyzer()
        
        temporal_analysis = analyzer._analyze_temporal_patterns(sample_trades)
        
        assert 'monthly' in temporal_analysis
        assert 'day_of_week' in temporal_analysis
        assert 'hourly' in temporal_analysis
        
        # Check structure of monthly data
        if temporal_analysis['monthly']:
            month_data = list(temporal_analysis['monthly'].values())[0]
            assert 'count' in month_data
            assert 'avg_return_pct' in month_data
            assert 'win_rate' in month_data
            assert 'std_pct' in month_data
    
    def test_compare_to_benchmark(self):
        """Test benchmark comparison."""
        analyzer = PerformanceAnalyzer()
        
        dates = pd.date_range('2023-01-01', periods=10)
        
        portfolio_values = pd.DataFrame({
            'value': np.linspace(100000, 110000, 10)  # 10% gain
        }, index=dates)
        
        benchmark = pd.DataFrame({
            'close': np.linspace(400, 404, 10)  # 1% gain
        }, index=dates)
        
        comparison = analyzer._compare_to_benchmark(portfolio_values, benchmark)
        
        assert 'strategy_total_return_pct' in comparison
        assert 'benchmark_total_return_pct' in comparison
        assert 'excess_return_pct' in comparison
        assert 'correlation' in comparison
        assert 'beta' in comparison
        assert 'tracking_error_pct' in comparison
        
        # Strategy should have higher return
        assert comparison['strategy_total_return_pct'] > comparison['benchmark_total_return_pct']
        assert comparison['excess_return_pct'] > 0
    
    def test_compare_to_benchmark_empty(self):
        """Test benchmark comparison with empty data."""
        analyzer = PerformanceAnalyzer()
        
        comparison = analyzer._compare_to_benchmark(pd.DataFrame(), pd.DataFrame())
        
        assert comparison == {}
    
    def test_compare_to_benchmark_no_common_dates(self):
        """Test benchmark comparison with no overlapping dates."""
        analyzer = PerformanceAnalyzer()
        
        portfolio_values = pd.DataFrame({
            'value': [100000, 110000]
        }, index=pd.date_range('2023-01-01', periods=2))
        
        benchmark = pd.DataFrame({
            'close': [400, 404]
        }, index=pd.date_range('2023-02-01', periods=2))  # Different dates
        
        comparison = analyzer._compare_to_benchmark(portfolio_values, benchmark)
        
        assert comparison == {}
    
    def test_analyze_backtest_results_comprehensive(self, sample_trades):
        """Test comprehensive backtest results analysis."""
        analyzer = PerformanceAnalyzer()
        
        # Create portfolio values
        portfolio_values = pd.DataFrame({
            'value': np.cumsum([100000] + [trade.pnl_net for trade in sample_trades])
        }, index=pd.date_range('2023-01-01', periods=len(sample_trades) + 1))
        
        # Create benchmark
        benchmark = pd.DataFrame({
            'close': np.linspace(400, 404, len(sample_trades) + 1)
        }, index=portfolio_values.index)
        
        backtest_results = {
            'trades': sample_trades,
            'portfolio_values': portfolio_values,
            'benchmark': benchmark
        }
        
        analysis = analyzer.analyze_backtest_results(backtest_results)
        
        assert 'trade_analysis' in analysis
        assert 'performance_metrics' in analysis
        assert 'risk_metrics' in analysis
        assert 'sector_analysis' in analysis
        assert 'temporal_analysis' in analysis
        assert 'benchmark_comparison' in analysis
    
    def test_analyze_backtest_results_no_trades(self):
        """Test analysis with no trades."""
        analyzer = PerformanceAnalyzer()
        
        backtest_results = {
            'trades': [],
            'portfolio_values': pd.DataFrame(),
            'benchmark': pd.DataFrame()
        }
        
        analysis = analyzer.analyze_backtest_results(backtest_results)
        
        assert analysis == {}
    
    def test_create_trade_analysis_plots(self, sample_trades):
        """Test trade analysis plot creation."""
        analyzer = PerformanceAnalyzer()
        
        figures = analyzer.create_trade_analysis_plots(sample_trades)
        
        assert len(figures) > 0
        assert all(hasattr(fig, 'savefig') for fig in figures)  # matplotlib figures
        
        # Should have created various plot types
        assert len(figures) >= 4  # At least 4 different plots
    
    def test_create_trade_analysis_plots_empty(self):
        """Test plot creation with empty trades."""
        analyzer = PerformanceAnalyzer()
        
        figures = analyzer.create_trade_analysis_plots([])
        
        assert figures == []
    
    def test_create_trade_analysis_plots_with_save(self, sample_trades, temp_data_dir):
        """Test plot creation with file saving."""
        analyzer = PerformanceAnalyzer()
        
        save_dir = temp_data_dir / "plots"
        
        figures = analyzer.create_trade_analysis_plots(sample_trades, save_dir=str(save_dir))
        
        assert len(figures) > 0
        assert save_dir.exists()
        
        # Check that files were saved
        plot_files = list(save_dir.glob("*.png"))
        assert len(plot_files) > 0
    
    def test_generate_performance_report(self, sample_trades):
        """Test performance report generation."""
        analyzer = PerformanceAnalyzer()
        
        # Create minimal analysis data
        analysis = {
            'trade_analysis': {
                'total_trades': len(sample_trades),
                'win_rate': 0.6,
                'avg_return_pct': 2.5,
                'profit_factor': 1.8,
                'total_pnl': 5000
            },
            'performance_metrics': {
                'total_return_pct': 15.0,
                'cagr_pct': 12.0,
                'sharpe_ratio': 1.5,
                'max_drawdown_pct': -8.5
            }
        }
        
        report = analyzer.generate_performance_report(analysis)
        
        assert isinstance(report, str)
        assert 'PERFORMANCE REPORT' in report
        assert 'Total Trades' in report
        assert 'Win Rate' in report
        assert 'Sharpe Ratio' in report
        assert str(len(sample_trades)) in report
    
    def test_generate_performance_report_empty(self):
        """Test report generation with empty analysis."""
        analyzer = PerformanceAnalyzer()
        
        report = analyzer.generate_performance_report({})
        
        assert "No analysis data available" in report
    
    def test_generate_performance_report_save(self, temp_data_dir):
        """Test report generation with file saving."""
        analyzer = PerformanceAnalyzer()
        
        analysis = {
            'trade_analysis': {
                'total_trades': 100,
                'win_rate': 0.6,
                'profit_factor': 1.8
            }
        }
        
        report_path = temp_data_dir / "report.txt"
        
        report = analyzer.generate_performance_report(analysis, save_path=str(report_path))
        
        assert report_path.exists()
        
        # Verify file contents
        saved_content = report_path.read_text()
        assert saved_content == report
    
    def test_cleanup_figures(self):
        """Test figure cleanup functionality."""
        analyzer = PerformanceAnalyzer()
        
        # Add some mock figures
        mock_fig1 = MagicMock()
        mock_fig2 = MagicMock()
        analyzer.figures_created = [mock_fig1, mock_fig2]
        
        with patch('matplotlib.pyplot.close') as mock_close:
            analyzer.cleanup_figures()
            
            assert mock_close.call_count == 2
            assert analyzer.figures_created == []
    
    @patch('plotly.graph_objects.Figure')
    def test_create_performance_dashboard(self, mock_figure):
        """Test interactive dashboard creation."""
        analyzer = PerformanceAnalyzer()
        
        analysis = {
            'trade_analysis': {'total_trades': 100},
            'performance_metrics': {'sharpe_ratio': 1.5}
        }
        
        # Mock plotly figure
        mock_fig_instance = MagicMock()
        mock_figure.return_value = mock_fig_instance
        
        with patch('plotly.subplots.make_subplots') as mock_subplots:
            mock_subplots.return_value = mock_fig_instance
            
            fig = analyzer.create_performance_dashboard(analysis)
            
            assert fig is not None
            # Should have called various plotly methods
            assert mock_fig_instance.add_trace.called or mock_fig_instance.update_layout.called