"""Analytics module for performance analysis and visualization."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from pandas import DataFrame
from plotly.subplots import make_subplots
from scipy import stats

from gappers.backtest import TradeResult

logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PerformanceAnalyzer:
    """Comprehensive performance analysis and visualization."""

    def __init__(self) -> None:
        """Initialize performance analyzer."""
        self.figures_created = []

    def analyze_backtest_results(self, backtest_results: Dict) -> Dict:
        """
        Comprehensive analysis of backtest results.

        Args:
            backtest_results: Results from Backtester.run_backtest()

        Returns:
            Dictionary with detailed analysis
        """
        trades = backtest_results.get('trades', [])
        portfolio_values = backtest_results.get('portfolio_values', pd.DataFrame())
        benchmark = backtest_results.get('benchmark', pd.DataFrame())

        if not trades:
            logger.warning("No trades to analyze")
            return {}

        analysis = {
            'trade_analysis': self._analyze_trades(trades),
            'performance_metrics': self._calculate_performance_metrics(trades, portfolio_values),
            'risk_metrics': self._calculate_risk_metrics(trades, portfolio_values),
            'sector_analysis': self._analyze_by_sector(trades),
            'temporal_analysis': self._analyze_temporal_patterns(trades),
            'benchmark_comparison': self._compare_to_benchmark(portfolio_values, benchmark),
        }

        return analysis

    def _analyze_trades(self, trades: List[TradeResult]) -> Dict:
        """Analyze individual trade characteristics."""
        if not trades:
            return {}

        returns = [trade.return_pct for trade in trades]
        pnl = [trade.pnl_net for trade in trades]
        hold_times = [trade.hold_time_hours for trade in trades]
        gaps = [trade.gap_pct for trade in trades]

        winners = [trade for trade in trades if trade.return_pct > 0]
        losers = [trade for trade in trades if trade.return_pct < 0]

        analysis = {
            'total_trades': len(trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(trades) if trades else 0,
            
            # Return statistics
            'avg_return_pct': np.mean(returns) * 100,
            'median_return_pct': np.median(returns) * 100,
            'std_return_pct': np.std(returns) * 100,
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            
            # Winner/Loser analysis
            'avg_winner_pct': np.mean([w.return_pct for w in winners]) * 100 if winners else 0,
            'avg_loser_pct': np.mean([l.return_pct for l in losers]) * 100 if losers else 0,
            'largest_winner_pct': max(returns) * 100 if returns else 0,
            'largest_loser_pct': min(returns) * 100 if returns else 0,
            
            # Hold time analysis
            'avg_hold_time_hours': np.mean(hold_times),
            'median_hold_time_hours': np.median(hold_times),
            
            # Gap analysis
            'avg_gap_pct': np.mean(gaps) * 100,
            'gap_return_correlation': np.corrcoef(gaps, returns)[0, 1] if len(gaps) > 1 else 0,
            
            # Exit reason breakdown
            'exit_reasons': self._analyze_exit_reasons(trades),
            
            # PnL statistics
            'total_pnl': sum(pnl),
            'avg_pnl_per_trade': np.mean(pnl),
            'profit_factor': self._calculate_profit_factor(pnl),
        }

        return analysis

    def _analyze_exit_reasons(self, trades: List[TradeResult]) -> Dict:
        """Analyze exit reason distribution."""
        exit_counts = {}
        for trade in trades:
            reason = trade.exit_reason
            exit_counts[reason] = exit_counts.get(reason, 0) + 1

        total = len(trades)
        return {reason: count / total for reason, count in exit_counts.items()}

    def _calculate_profit_factor(self, pnl_list: List[float]) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_profit = sum(pnl for pnl in pnl_list if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnl_list if pnl < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    def _calculate_performance_metrics(
        self, trades: List[TradeResult], portfolio_values: DataFrame
    ) -> Dict:
        """Calculate portfolio-level performance metrics."""
        if portfolio_values.empty:
            return {}

        # Calculate returns
        portfolio_values = portfolio_values.copy()
        portfolio_values['returns'] = portfolio_values['value'].pct_change()
        daily_returns = portfolio_values['returns'].dropna()

        if len(portfolio_values) < 2:
            return {}

        # Basic metrics
        initial_value = portfolio_values['value'].iloc[0]
        final_value = portfolio_values['value'].iloc[-1]
        total_return = (final_value / initial_value) - 1

        # Time-based metrics
        days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
        years = days / 365.25

        cagr = (final_value / initial_value) ** (1 / years) - 1 if years > 0 else 0

        # Risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        sortino_ratio = self._calculate_sortino_ratio(daily_returns)
        calmar_ratio = self._calculate_calmar_ratio(daily_returns, cagr)

        # Drawdown analysis
        drawdown_analysis = self._calculate_drawdown_metrics(portfolio_values['value'])

        metrics = {
            'total_return_pct': total_return * 100,
            'cagr_pct': cagr * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'volatility_pct': daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 1 else 0,
            'trading_days': len(portfolio_values),
            'avg_daily_return_pct': daily_returns.mean() * 100 if len(daily_returns) > 0 else 0,
        }

        metrics.update(drawdown_analysis)
        return metrics

    def _calculate_sharpe_ratio(self, daily_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(daily_returns) < 2:
            return 0

        excess_returns = daily_returns - (risk_free_rate / 252)
        return excess_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

    def _calculate_sortino_ratio(self, daily_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (using downside deviation)."""
        if len(daily_returns) < 2:
            return 0

        excess_returns = daily_returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = downside_returns.std()
        return excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0

    def _calculate_calmar_ratio(self, daily_returns: pd.Series, cagr: float) -> float:
        """Calculate Calmar ratio (CAGR / Max Drawdown)."""
        portfolio_values = (1 + daily_returns).cumprod()
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        return cagr / max_drawdown if max_drawdown > 0 else 0

    def _calculate_drawdown_metrics(self, portfolio_values: pd.Series) -> Dict:
        """Calculate comprehensive drawdown metrics."""
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        
        max_drawdown = drawdown.min()
        max_drawdown_pct = max_drawdown * 100
        
        # Drawdown duration analysis
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        # Add final period if still in drawdown
        if current_period > 0:
            drawdown_periods.append(current_period)

        return {
            'max_drawdown_pct': max_drawdown_pct,
            'avg_drawdown_duration_days': np.mean(drawdown_periods) if drawdown_periods else 0,
            'max_drawdown_duration_days': max(drawdown_periods) if drawdown_periods else 0,
            'num_drawdown_periods': len(drawdown_periods),
        }

    def _calculate_risk_metrics(
        self, trades: List[TradeResult], portfolio_values: DataFrame
    ) -> Dict:
        """Calculate risk-related metrics."""
        if not trades or portfolio_values.empty:
            return {}

        returns = [trade.return_pct for trade in trades]
        daily_returns = portfolio_values['value'].pct_change().dropna()

        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5) * 100 if returns else 0
        var_99 = np.percentile(returns, 1) * 100 if returns else 0

        # Conditional Value at Risk (CVaR)
        cvar_95 = np.mean([r for r in returns if r <= np.percentile(returns, 5)]) * 100 if returns else 0
        cvar_99 = np.mean([r for r in returns if r <= np.percentile(returns, 1)]) * 100 if returns else 0

        # Beta calculation (simplified, assuming market return = 0.1% daily)
        market_return = 0.001
        beta = np.cov(daily_returns, [market_return] * len(daily_returns))[0, 1] / np.var([market_return] * len(daily_returns)) if len(daily_returns) > 1 else 1

        return {
            'var_95_pct': var_95,
            'var_99_pct': var_99,
            'cvar_95_pct': cvar_95,
            'cvar_99_pct': cvar_99,
            'beta': beta,
            'trade_return_volatility_pct': np.std(returns) * 100 if returns else 0,
        }

    def _analyze_by_sector(self, trades: List[TradeResult]) -> Dict:
        """Analyze performance by sector."""
        sector_data = {}
        
        for trade in trades:
            # Get sector from trade (would need to be added to TradeResult)
            sector = getattr(trade, 'sector', 'Unknown')
            
            if sector not in sector_data:
                sector_data[sector] = []
            sector_data[sector].append(trade)

        sector_analysis = {}
        for sector, sector_trades in sector_data.items():
            returns = [t.return_pct for t in sector_trades]
            pnl = [t.pnl_net for t in sector_trades]
            
            sector_analysis[sector] = {
                'trade_count': len(sector_trades),
                'win_rate': sum(1 for r in returns if r > 0) / len(returns),
                'avg_return_pct': np.mean(returns) * 100,
                'total_pnl': sum(pnl),
                'volatility_pct': np.std(returns) * 100,
            }

        return sector_analysis

    def _analyze_temporal_patterns(self, trades: List[TradeResult]) -> Dict:
        """Analyze performance patterns over time."""
        if not trades:
            return {}

        # Group by month, day of week, hour
        monthly_performance = {}
        dow_performance = {}
        hourly_performance = {}

        for trade in trades:
            # Monthly analysis
            month = trade.entry_date.month
            if month not in monthly_performance:
                monthly_performance[month] = []
            monthly_performance[month].append(trade.return_pct)

            # Day of week analysis
            dow = trade.entry_date.weekday()  # 0=Monday
            if dow not in dow_performance:
                dow_performance[dow] = []
            dow_performance[dow].append(trade.return_pct)

            # Hourly analysis (entry hour)
            hour = trade.entry_date.hour
            if hour not in hourly_performance:
                hourly_performance[hour] = []
            hourly_performance[hour].append(trade.return_pct)

        # Calculate statistics for each grouping
        def calc_group_stats(group_data):
            return {
                key: {
                    'count': len(values),
                    'avg_return_pct': np.mean(values) * 100,
                    'win_rate': sum(1 for v in values if v > 0) / len(values),
                    'std_pct': np.std(values) * 100,
                }
                for key, values in group_data.items()
            }

        return {
            'monthly': calc_group_stats(monthly_performance),
            'day_of_week': calc_group_stats(dow_performance),
            'hourly': calc_group_stats(hourly_performance),
        }

    def _compare_to_benchmark(
        self, portfolio_values: DataFrame, benchmark: DataFrame
    ) -> Dict:
        """Compare strategy performance to benchmark."""
        if portfolio_values.empty or benchmark.empty:
            return {}

        try:
            # Align dates
            common_dates = portfolio_values.index.intersection(benchmark.index)
            if len(common_dates) < 2:
                return {}

            strategy_values = portfolio_values.loc[common_dates, 'value']
            benchmark_values = benchmark.loc[common_dates, 'close']

            # Normalize to same starting point
            strategy_returns = strategy_values / strategy_values.iloc[0]
            benchmark_returns = benchmark_values / benchmark_values.iloc[0]

            # Calculate metrics
            strategy_total_return = (strategy_returns.iloc[-1] / strategy_returns.iloc[0]) - 1
            benchmark_total_return = (benchmark_returns.iloc[-1] / benchmark_returns.iloc[0]) - 1

            excess_return = strategy_total_return - benchmark_total_return

            # Calculate correlation
            strategy_daily_returns = strategy_returns.pct_change().dropna()
            benchmark_daily_returns = benchmark_returns.pct_change().dropna()
            
            correlation = np.corrcoef(strategy_daily_returns, benchmark_daily_returns)[0, 1] if len(strategy_daily_returns) > 1 else 0

            # Beta
            beta = np.cov(strategy_daily_returns, benchmark_daily_returns)[0, 1] / np.var(benchmark_daily_returns) if np.var(benchmark_daily_returns) > 0 else 1

            return {
                'strategy_total_return_pct': strategy_total_return * 100,
                'benchmark_total_return_pct': benchmark_total_return * 100,
                'excess_return_pct': excess_return * 100,
                'correlation': correlation,
                'beta': beta,
                'tracking_error_pct': np.std(strategy_daily_returns - benchmark_daily_returns) * np.sqrt(252) * 100,
            }

        except Exception as e:
            logger.error(f"Error comparing to benchmark: {e}")
            return {}

    def create_performance_dashboard(self, analysis: Dict, save_path: Optional[str] = None) -> go.Figure:
        """Create interactive performance dashboard using Plotly."""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Equity Curve', 'Drawdown', 'Monthly Returns Heatmap', 
                          'Return Distribution', 'Rolling Sharpe Ratio', 'Trade Statistics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"colspan": 2}, None],
                   [{"secondary_y": False}, {"type": "table"}]],
            vertical_spacing=0.08
        )

        # This is a placeholder - in a real implementation, you would need
        # the actual time series data to create these plots
        
        # Add placeholder content
        fig.add_trace(
            go.Scatter(x=[1, 2, 3], y=[100, 105, 110], name="Portfolio Value"),
            row=1, col=1
        )

        fig.update_layout(
            height=1000,
            title_text="Gap Trading Strategy Performance Dashboard",
            showlegend=True
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def create_trade_analysis_plots(
        self, trades: List[TradeResult], save_dir: Optional[str] = None
    ) -> List[plt.Figure]:
        """Create detailed trade analysis plots using matplotlib."""
        
        if not trades:
            return []

        figures = []

        # 1. Return distribution histogram
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        returns = [trade.return_pct * 100 for trade in trades]
        ax1.hist(returns, bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Return (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Trade Returns')
        ax1.grid(True, alpha=0.3)
        figures.append(fig1)

        # 2. Gap vs Return scatter plot
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        gaps = [trade.gap_pct * 100 for trade in trades]
        ax2.scatter(gaps, returns, alpha=0.6)
        ax2.set_xlabel('Gap Size (%)')
        ax2.set_ylabel('Trade Return (%)')
        ax2.set_title('Gap Size vs Trade Return')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        if len(gaps) > 1:
            z = np.polyfit(gaps, returns, 1)
            p = np.poly1d(z)
            ax2.plot(gaps, p(gaps), "r--", alpha=0.8)
        figures.append(fig2)

        # 3. Hold time analysis
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        hold_times = [trade.hold_time_hours for trade in trades]
        ax3.hist(hold_times, bins=30, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Hold Time (hours)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Hold Times')
        ax3.grid(True, alpha=0.3)
        figures.append(fig3)

        # 4. Exit reason pie chart
        fig4, ax4 = plt.subplots(figsize=(8, 8))
        exit_reasons = [trade.exit_reason for trade in trades]
        exit_counts = pd.Series(exit_reasons).value_counts()
        ax4.pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%')
        ax4.set_title('Exit Reason Distribution')
        figures.append(fig4)

        # 5. Performance over time
        if trades:
            fig5, ax5 = plt.subplots(figsize=(12, 6))
            trade_dates = [trade.entry_date for trade in trades]
            cumulative_returns = np.cumsum([trade.return_pct for trade in trades])
            
            ax5.plot(trade_dates, cumulative_returns)
            ax5.set_xlabel('Date')
            ax5.set_ylabel('Cumulative Return')
            ax5.set_title('Cumulative Returns Over Time')
            ax5.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            figures.append(fig5)

        # Save figures if directory provided
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            for i, fig in enumerate(figures):
                fig.savefig(f"{save_dir}/trade_analysis_{i+1}.png", dpi=300, bbox_inches='tight')

        self.figures_created.extend(figures)
        return figures

    def generate_performance_report(
        self, analysis: Dict, save_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive text-based performance report."""
        
        if not analysis:
            return "No analysis data available."

        report_lines = [
            "=" * 80,
            "GAP TRADING STRATEGY PERFORMANCE REPORT",
            "=" * 80,
            "",
        ]

        # Trade Analysis Section
        if 'trade_analysis' in analysis:
            trade_data = analysis['trade_analysis']
            report_lines.extend([
                "TRADE ANALYSIS",
                "-" * 40,
                f"Total Trades: {trade_data.get('total_trades', 0):,}",
                f"Winners: {trade_data.get('winners', 0):,} ({trade_data.get('win_rate', 0)*100:.1f}%)",
                f"Losers: {trade_data.get('losers', 0):,}",
                "",
                f"Average Return: {trade_data.get('avg_return_pct', 0):.2f}%",
                f"Average Winner: {trade_data.get('avg_winner_pct', 0):.2f}%",
                f"Average Loser: {trade_data.get('avg_loser_pct', 0):.2f}%",
                f"Largest Winner: {trade_data.get('largest_winner_pct', 0):.2f}%",
                f"Largest Loser: {trade_data.get('largest_loser_pct', 0):.2f}%",
                "",
                f"Profit Factor: {trade_data.get('profit_factor', 0):.2f}",
                f"Total P&L: ${trade_data.get('total_pnl', 0):,.2f}",
                "",
                f"Average Hold Time: {trade_data.get('avg_hold_time_hours', 0):.1f} hours",
                f"Average Gap Size: {trade_data.get('avg_gap_pct', 0):.2f}%",
                "",
            ])

        # Performance Metrics Section
        if 'performance_metrics' in analysis:
            perf_data = analysis['performance_metrics']
            report_lines.extend([
                "PERFORMANCE METRICS",
                "-" * 40,
                f"Total Return: {perf_data.get('total_return_pct', 0):.2f}%",
                f"CAGR: {perf_data.get('cagr_pct', 0):.2f}%",
                f"Sharpe Ratio: {perf_data.get('sharpe_ratio', 0):.2f}",
                f"Sortino Ratio: {perf_data.get('sortino_ratio', 0):.2f}",
                f"Calmar Ratio: {perf_data.get('calmar_ratio', 0):.2f}",
                f"Volatility: {perf_data.get('volatility_pct', 0):.2f}%",
                "",
                f"Max Drawdown: {perf_data.get('max_drawdown_pct', 0):.2f}%",
                f"Avg Drawdown Duration: {perf_data.get('avg_drawdown_duration_days', 0):.1f} days",
                "",
            ])

        # Risk Metrics Section
        if 'risk_metrics' in analysis:
            risk_data = analysis['risk_metrics']
            report_lines.extend([
                "RISK METRICS",
                "-" * 40,
                f"VaR (95%): {risk_data.get('var_95_pct', 0):.2f}%",
                f"VaR (99%): {risk_data.get('var_99_pct', 0):.2f}%",
                f"CVaR (95%): {risk_data.get('cvar_95_pct', 0):.2f}%",
                f"CVaR (99%): {risk_data.get('cvar_99_pct', 0):.2f}%",
                f"Beta: {risk_data.get('beta', 0):.2f}",
                "",
            ])

        # Benchmark Comparison
        if 'benchmark_comparison' in analysis and analysis['benchmark_comparison']:
            bench_data = analysis['benchmark_comparison']
            report_lines.extend([
                "BENCHMARK COMPARISON",
                "-" * 40,
                f"Strategy Return: {bench_data.get('strategy_total_return_pct', 0):.2f}%",
                f"Benchmark Return: {bench_data.get('benchmark_total_return_pct', 0):.2f}%",
                f"Excess Return: {bench_data.get('excess_return_pct', 0):.2f}%",
                f"Correlation: {bench_data.get('correlation', 0):.2f}",
                f"Tracking Error: {bench_data.get('tracking_error_pct', 0):.2f}%",
                "",
            ])

        report_lines.extend([
            "=" * 80,
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
        ])

        report = "\n".join(report_lines)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)

        return report

    def cleanup_figures(self) -> None:
        """Clean up created matplotlib figures to free memory."""
        for fig in self.figures_created:
            plt.close(fig)
        self.figures_created.clear()