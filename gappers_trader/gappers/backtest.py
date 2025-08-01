"""Backtesting engine with vectorbt integration for gap trading strategies."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import vectorbt as vbt
from pandas import DataFrame
from scipy import stats

from gappers.config import config
from gappers.datafeed import DataFeed
from gappers.signals import SignalGenerator

logger = logging.getLogger(__name__)

# Configure vectorbt settings
vbt.settings.caching['enabled'] = True
vbt.settings.broadcasting['align_index'] = True


@dataclass
class GapParams:
    """Parameters for gap trading strategy."""
    
    # Entry/exit parameters
    profit_target: float = 0.05  # 5% profit target
    stop_loss: float = 0.02  # 2% stop loss
    max_hold_time_hours: int = 6  # Maximum hold time in hours (until 3:30 PM)
    
    # Selection parameters
    top_k: int = 10  # Number of top gaps to trade
    min_gap_pct: float = 0.02  # Minimum 2% gap
    max_gap_pct: float = 0.30  # Maximum 30% gap (filter outliers)
    
    # Risk management
    position_size: float = 10000  # Dollar amount per position
    max_positions: int = 10  # Maximum concurrent positions
    sector_diversification: bool = True
    max_per_sector: int = 3
    
    # Costs
    commission_per_share: float = 0.005  # $0.005 per share
    slippage_bps: float = 10  # 10 basis points slippage
    
    # Universe filters
    min_dollar_volume: float = 1_000_000  # $1M min daily dollar volume
    min_price: float = 5.0
    max_price: float = 1000.0


@dataclass
class TradeResult:
    """Individual trade result."""
    
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    position_size: int  # shares
    pnl_gross: float
    pnl_net: float  # after costs
    return_pct: float
    hold_time_hours: float
    exit_reason: str  # 'profit_target', 'stop_loss', 'time_limit', 'eod'
    gap_pct: float
    rank: int


class Backtester:
    """Vectorized backtesting engine for gap trading strategies."""
    
    def __init__(
        self,
        data_feed: Optional[DataFeed] = None,
        signal_generator: Optional[SignalGenerator] = None,
    ) -> None:
        """Initialize backtester."""
        self.data_feed = data_feed or DataFeed()
        self.signal_generator = signal_generator or SignalGenerator(self.data_feed)
        
    def run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        params: GapParams,
        benchmark_symbol: str = "SPY",
    ) -> Dict:
        """
        Run comprehensive backtest of gap trading strategy.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            params: Strategy parameters
            benchmark_symbol: Benchmark for comparison
            
        Returns:
            Dictionary containing all backtest results
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Get historical gap signals
        historical_gaps = self.signal_generator.get_historical_gaps(
            start_date=start_date,
            end_date=end_date,
            top_k=params.top_k,
            min_gap_pct=params.min_gap_pct,
            max_gap_pct=params.max_gap_pct,
            include_negative_gaps=False
        )
        
        if not historical_gaps:
            raise ValueError("No historical gap data found")
        
        # Run vectorized simulation
        trades, portfolio_values = self._run_vectorized_simulation(
            historical_gaps, params, start_date, end_date
        )
        
        # Get benchmark data
        benchmark_data = self._get_benchmark_data(benchmark_symbol, start_date, end_date)
        
        # Compile results
        results = {
            'trades': trades,
            'portfolio_values': portfolio_values,
            'benchmark': benchmark_data,
            'params': params,
            'start_date': start_date,
            'end_date': end_date,
            'total_trades': len(trades),
            'signals_generated': sum(len(df) for df in historical_gaps.values()),
        }
        
        logger.info(f"Backtest completed: {len(trades)} trades executed")
        return results
    
    def _run_vectorized_simulation(
        self,
        historical_gaps: Dict[str, DataFrame],
        params: GapParams,
        start_date: datetime,
        end_date: datetime,
    ) -> Tuple[List[TradeResult], DataFrame]:
        """Run vectorized simulation using vectorbt."""
        
        all_trades = []
        portfolio_values = []
        current_cash = 100000  # Start with $100k
        
        # Process each trading day
        for date_str, gaps_df in historical_gaps.items():
            trade_date = datetime.strptime(date_str, '%Y-%m-%d')
            
            if gaps_df.empty:
                continue
                
            # Simulate trades for this day
            day_trades = self._simulate_day_trades(gaps_df, params, trade_date)
            all_trades.extend(day_trades)
            
            # Update portfolio value
            day_pnl = sum(trade.pnl_net for trade in day_trades)
            current_cash += day_pnl
            
            portfolio_values.append({
                'date': trade_date,
                'value': current_cash,
                'daily_pnl': day_pnl,
                'trades_count': len(day_trades)
            })
        
        portfolio_df = pd.DataFrame(portfolio_values)
        if not portfolio_df.empty:
            portfolio_df.set_index('date', inplace=True)
        
        return all_trades, portfolio_df
    
    def _simulate_day_trades(
        self, gaps_df: DataFrame, params: GapParams, trade_date: datetime
    ) -> List[TradeResult]:
        """Simulate trades for a single day."""
        
        trades = []
        
        # Get intraday data for selected gaps
        symbols = gaps_df['symbol'].tolist()[:params.max_positions]
        
        try:
            # Get 1-minute data for the trading day
            intraday_data = self.data_feed.download(
                symbols,
                start=trade_date,
                end=trade_date + timedelta(days=1),
                interval="1m"
            )
            
            for _, gap_row in gaps_df.head(params.max_positions).iterrows():
                symbol = gap_row['symbol']
                
                if symbol not in intraday_data:
                    continue
                    
                trade_result = self._simulate_single_trade(
                    symbol, gap_row, intraday_data[symbol], params, trade_date
                )
                
                if trade_result:
                    trades.append(trade_result)
                    
        except Exception as e:
            logger.error(f"Error simulating trades for {trade_date}: {e}")
        
        return trades
    
    def _simulate_single_trade(
        self,
        symbol: str,
        gap_row: Dict,
        intraday_df: DataFrame,
        params: GapParams,
        trade_date: datetime,
    ) -> Optional[TradeResult]:
        """Simulate a single trade with realistic execution."""
        
        if intraday_df.empty:
            return None
            
        try:
            # Entry at market open (9:30 AM ET)
            market_open = trade_date.replace(hour=9, minute=30, second=0)
            
            # Filter to regular trading hours (9:30 AM - 4:00 PM ET)
            trading_hours = intraday_df.between_time('09:30', '16:00')
            
            if trading_hours.empty:
                return None
            
            # Entry price (first minute of trading with slippage)
            entry_price = trading_hours.iloc[0]['open']
            slippage_amount = entry_price * (params.slippage_bps / 10000)
            entry_price_net = entry_price + slippage_amount  # Buy at slightly higher price
            
            # Calculate position size
            position_value = min(params.position_size, 100000)  # Cap position size
            shares = int(position_value / entry_price_net)
            
            if shares < 1:
                return None
            
            # Define exit levels
            profit_target_price = entry_price_net * (1 + params.profit_target)
            stop_loss_price = entry_price_net * (1 - params.stop_loss)
            
            # Find exit point
            exit_info = self._find_exit_point(
                trading_hours, profit_target_price, stop_loss_price, params
            )
            
            if not exit_info:
                return None
            
            exit_price, exit_time, exit_reason = exit_info
            
            # Apply exit slippage
            if exit_reason == 'profit_target':
                exit_price_net = exit_price - slippage_amount  # Sell at slightly lower
            else:
                exit_price_net = exit_price - slippage_amount  # Conservative exit
            
            # Calculate returns
            gross_pnl = (exit_price_net - entry_price_net) * shares
            commission = params.commission_per_share * shares * 2  # Buy + sell
            net_pnl = gross_pnl - commission
            
            return_pct = (exit_price_net - entry_price_net) / entry_price_net
            
            # Calculate hold time
            hold_time = (exit_time - market_open).total_seconds() / 3600
            
            return TradeResult(
                symbol=symbol,
                entry_date=market_open,
                exit_date=exit_time,
                entry_price=entry_price_net,
                exit_price=exit_price_net,
                position_size=shares,
                pnl_gross=gross_pnl,
                pnl_net=net_pnl,
                return_pct=return_pct,
                hold_time_hours=hold_time,
                exit_reason=exit_reason,
                gap_pct=gap_row.get('gap_pct', 0),
                rank=gap_row.get('rank', 0),
            )
            
        except Exception as e:
            logger.debug(f"Error simulating trade for {symbol}: {e}")
            return None
    
    def _find_exit_point(
        self,
        intraday_df: DataFrame,
        profit_target: float,
        stop_loss: float,
        params: GapParams,
    ) -> Optional[Tuple[float, datetime, str]]:
        """Find the exit point for a trade."""
        
        max_hold_time = timedelta(hours=params.max_hold_time_hours)
        entry_time = intraday_df.index[0]
        
        for timestamp, row in intraday_df.iterrows():
            current_time = pd.Timestamp(timestamp)
            
            # Check profit target (using high of the bar)
            if row['high'] >= profit_target:
                return profit_target, current_time, 'profit_target'
            
            # Check stop loss (using low of the bar)
            if row['low'] <= stop_loss:
                return stop_loss, current_time, 'stop_loss'
            
            # Check time limit
            if (current_time - entry_time) >= max_hold_time:
                return row['close'], current_time, 'time_limit'
        
        # Exit at end of day
        last_row = intraday_df.iloc[-1]
        last_time = pd.Timestamp(intraday_df.index[-1])
        return last_row['close'], last_time, 'eod'
    
    def _get_benchmark_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> DataFrame:
        """Get benchmark data for comparison."""
        try:
            benchmark_data = self.data_feed.download(
                [symbol], start=start_date, end=end_date, interval="1d"
            )
            
            if symbol in benchmark_data:
                df = benchmark_data[symbol].copy()
                df['returns'] = df['close'].pct_change()
                df['cumulative_returns'] = (1 + df['returns']).cumprod()
                return df
            
        except Exception as e:
            logger.error(f"Error getting benchmark data: {e}")
        
        return pd.DataFrame()
    
    def run_parameter_sweep(
        self,
        start_date: datetime,
        end_date: datetime,
        param_grid: Dict[str, List],
        base_params: Optional[GapParams] = None,
    ) -> DataFrame:
        """
        Run parameter sweep for strategy optimization.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            param_grid: Dictionary of parameters to sweep
            base_params: Base parameters to modify
            
        Returns:
            DataFrame with results for each parameter combination
        """
        if base_params is None:
            base_params = GapParams()
            
        results = []
        
        # Generate all parameter combinations
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        combinations = list(product(*param_values))
        total_combinations = len(combinations)
        
        logger.info(f"Running parameter sweep: {total_combinations} combinations")
        
        for i, combo in enumerate(combinations):
            # Create parameters for this combination
            params = GapParams(**base_params.__dict__)
            
            for param_name, param_value in zip(param_names, combo):
                setattr(params, param_name, param_value)
            
            try:
                # Run backtest
                backtest_results = self.run_backtest(start_date, end_date, params)
                
                # Calculate summary metrics
                trades = backtest_results['trades']
                portfolio_values = backtest_results['portfolio_values']
                
                if trades and not portfolio_values.empty:
                    metrics = self._calculate_summary_metrics(trades, portfolio_values)
                    
                    # Add parameter values
                    result_row = {f'param_{name}': value for name, value in zip(param_names, combo)}
                    result_row.update(metrics)
                    results.append(result_row)
                
                logger.info(f"Completed combination {i+1}/{total_combinations}")
                
            except Exception as e:
                logger.error(f"Error in parameter combination {i+1}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def _calculate_summary_metrics(
        self, trades: List[TradeResult], portfolio_values: DataFrame
    ) -> Dict:
        """Calculate summary performance metrics."""
        
        if not trades or portfolio_values.empty:
            return {}
        
        # Trade-level metrics
        returns = [trade.return_pct for trade in trades]
        pnl = [trade.pnl_net for trade in trades]
        
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        avg_return = np.mean(returns)
        avg_winner = np.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0
        avg_loser = np.mean([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 0
        
        # Portfolio-level metrics
        daily_returns = portfolio_values['value'].pct_change().dropna()
        
        if len(daily_returns) > 0:
            total_return = (portfolio_values['value'].iloc[-1] / portfolio_values['value'].iloc[0]) - 1
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
            
            # Maximum drawdown
            running_max = portfolio_values['value'].expanding().max()
            drawdown = (portfolio_values['value'] - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # CAGR
            years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25
            cagr = (portfolio_values['value'].iloc[-1] / portfolio_values['value'].iloc[0]) ** (1/years) - 1 if years > 0 else 0
        else:
            total_return = sharpe_ratio = max_drawdown = cagr = 0
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_return_pct': avg_return * 100,
            'avg_winner_pct': avg_winner * 100,
            'avg_loser_pct': avg_loser * 100,
            'total_pnl': sum(pnl),
            'total_return_pct': total_return * 100,
            'cagr_pct': cagr * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'profit_factor': abs(sum(p for p in pnl if p > 0) / sum(p for p in pnl if p < 0)) if any(p < 0 for p in pnl) else float('inf'),
        }
    
    def validate_backtest(
        self,
        results: Dict,
        min_trades: int = 100,
        min_sharpe: float = 0.5,
        max_drawdown: float = -0.20,
    ) -> Dict:
        """Validate backtest results against criteria."""
        
        trades = results.get('trades', [])
        portfolio_values = results.get('portfolio_values', pd.DataFrame())
        
        validation = {
            'passed': True,
            'issues': [],
            'warnings': [],
            'summary': {}
        }
        
        # Check minimum trade count
        if len(trades) < min_trades:
            validation['issues'].append(f"Insufficient trades: {len(trades)} < {min_trades}")
            validation['passed'] = False
        
        # Calculate metrics for validation
        if trades and not portfolio_values.empty:
            metrics = self._calculate_summary_metrics(trades, portfolio_values)
            validation['summary'] = metrics
            
            # Check Sharpe ratio
            if metrics.get('sharpe_ratio', 0) < min_sharpe:
                validation['warnings'].append(f"Low Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            
            # Check maximum drawdown
            if metrics.get('max_drawdown_pct', 0) < max_drawdown * 100:
                validation['issues'].append(f"Excessive drawdown: {metrics.get('max_drawdown_pct', 0):.1f}%")
                validation['passed'] = False
        
        return validation