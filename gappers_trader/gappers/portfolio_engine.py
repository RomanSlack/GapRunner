"""Production-grade portfolio simulation engine implementing the momentum-gap strategy."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple
import warnings

import pandas as pd
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .config_new import Config
from .gap_engine import GapEngine
from .data_manager import DataManager

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)
console = Console()


@dataclass
class Trade:
    """Individual trade record."""
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    shares: float = 0
    gap_pct: float = 0
    rank: int = 0
    sector: str = "Unknown"
    pnl_gross: float = 0
    pnl_net: float = 0
    return_pct: float = 0
    hold_time_hours: float = 0
    max_price: float = 0
    min_price: float = 0
    trailing_stop_price: float = 0


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time."""
    date: datetime
    cash: float
    positions_value: float
    total_value: float
    daily_pnl: float
    open_positions: int
    trades_today: int


class PortfolioEngine:
    """Production-grade portfolio simulation engine."""
    
    def __init__(self, config: Config):
        self.config = config
        self.gap_engine = GapEngine(config)
        self.data_manager = DataManager(config)
        
        # Strategy parameters
        self.top_k = config.strategy.top_k
        self.min_gap_pct = config.strategy.min_gap_pct
        self.max_gap_pct = config.strategy.max_gap_pct
        self.profit_target_pct = config.strategy.profit_target_pct
        self.trailing_stop_pct = config.strategy.trailing_stop_pct
        self.hard_stop_pct = config.strategy.hard_stop_pct
        self.time_stop_hour = config.strategy.time_stop_hour
        self.position_size_usd = config.strategy.position_size_usd
        self.max_positions = config.strategy.max_positions
        
        # Cost parameters
        self.commission_per_share = config.costs.commission_per_share
        self.slippage_bps = config.costs.slippage_bps
        
        # Portfolio state
        self.initial_capital = config.backtest.initial_capital
        self.cash = self.initial_capital
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.portfolio_history: List[PortfolioSnapshot] = []
        
        logger.info(f"PortfolioEngine initialized with ${self.initial_capital:,} capital")

    def run_backtest(self, start_date: datetime, end_date: datetime) -> Dict:
        """Run complete backtest simulation."""
        console.print(f"[blue]💼 Running Portfolio Backtest[/blue]")
        console.print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        console.print(f"Strategy: Top {self.top_k} gaps, ${self.position_size_usd:,} per position")
        
        # Ensure dates are timezone-naive for consistent handling
        if hasattr(start_date, 'tz') and start_date.tz is not None:
            start_date = start_date.tz_localize(None)
        if hasattr(end_date, 'tz') and end_date.tz is not None:
            end_date = end_date.tz_localize(None)
        
        # Reset portfolio state
        self.cash = self.initial_capital
        self.open_trades = []
        self.closed_trades = []
        self.portfolio_history = []
        
        # Generate trading dates
        trading_dates = pd.bdate_range(start=start_date, end=end_date)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            backtest_task = progress.add_task("Running backtest...", total=len(trading_dates))
            
            for current_date in trading_dates:
                try:
                    self._process_trading_day(current_date.to_pydatetime())
                    progress.update(backtest_task, advance=1)
                    
                except Exception as e:
                    logger.warning(f"Error processing {current_date}: {e}")
                    progress.update(backtest_task, advance=1)
        
        # Final portfolio snapshot
        self._record_portfolio_snapshot(trading_dates[-1].to_pydatetime(), 0)
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics()
        
        console.print(f"[green]✅ Backtest completed: {len(self.closed_trades)} trades, "
                     f"{results['final_value']:,.0f} final value[/green]")
        
        return results

    def _process_trading_day(self, date: datetime) -> None:
        """Process a single trading day."""
        daily_trades = 0
        
        # Step 1: Exit existing positions (market open)
        self._process_exits(date)
        
        # Step 2: Find and rank gaps
        gaps_df = self.gap_engine.calculate_daily_gaps(date)
        
        if gaps_df.empty:
            self._record_portfolio_snapshot(date, daily_trades)
            return
        
        # Step 3: Filter for valid entry candidates
        entry_candidates = self._filter_entry_candidates(gaps_df)
        
        # Step 4: Enter new positions
        daily_trades = self._process_entries(entry_candidates, date)
        
        # Step 5: Update portfolio tracking
        self._record_portfolio_snapshot(date, daily_trades)

    def _process_exits(self, date: datetime) -> None:
        """Process exits for all open positions."""
        if not self.open_trades:
            return
        
        # Get intraday data for all open positions
        symbols = [trade.symbol for trade in self.open_trades]
        price_data = self.data_manager.get_price_data(
            symbols, date, date + timedelta(days=1), ['Open', 'High', 'Low', 'Close']
        )
        
        trades_to_close = []
        
        for trade in self.open_trades:
            if trade.symbol not in price_data:
                continue
            
            symbol_data = price_data[trade.symbol]
            if symbol_data.empty or date not in symbol_data.index:
                continue
            
            today_data = symbol_data.loc[date]
            
            # Update trade tracking
            trade.max_price = max(trade.max_price, today_data['High'])
            trade.min_price = min(trade.min_price, today_data['Low'])
            
            # Update trailing stop
            if trade.max_price > trade.entry_price:
                new_trailing_stop = trade.max_price * (1 - self.trailing_stop_pct)
                trade.trailing_stop_price = max(trade.trailing_stop_price, new_trailing_stop)
            
            # Check exit conditions
            exit_price, exit_reason = self._check_exit_conditions(trade, today_data, date)
            
            if exit_price:
                trade.exit_date = date
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                trades_to_close.append(trade)
        
        # Close flagged trades
        for trade in trades_to_close:
            self._close_trade(trade)

    def _check_exit_conditions(self, trade: Trade, today_data: pd.Series, date: datetime) -> Tuple[Optional[float], Optional[str]]:
        """Check if trade should be exited and return exit price and reason."""
        
        # Time stop (15:55 ET)
        current_time = date.replace(hour=self.time_stop_hour, minute=55)
        if date >= current_time:
            return today_data['Close'], 'time_stop'
        
        # Hard stop loss
        hard_stop_price = trade.entry_price * (1 - self.hard_stop_pct)
        if today_data['Low'] <= hard_stop_price:
            return hard_stop_price, 'hard_stop'
        
        # Trailing stop
        if trade.trailing_stop_price > 0 and today_data['Low'] <= trade.trailing_stop_price:
            return trade.trailing_stop_price, 'trailing_stop'
        
        # Profit target
        profit_target_price = trade.entry_price * (1 + self.profit_target_pct)
        if today_data['High'] >= profit_target_price:
            return profit_target_price, 'profit_target'
        
        return None, None

    def _filter_entry_candidates(self, gaps_df: pd.DataFrame) -> pd.DataFrame:
        """Filter gaps for valid entry candidates."""
        if gaps_df.empty:
            return gaps_df
        
        # Only up gaps
        up_gaps = gaps_df[gaps_df['gap_pct'] > 0].copy()
        
        # Apply gap size filters
        size_filter = (up_gaps['gap_pct'] >= self.min_gap_pct) & (up_gaps['gap_pct'] <= self.max_gap_pct)
        filtered_gaps = up_gaps[size_filter]
        
        # Take top K
        top_gaps = filtered_gaps.head(self.top_k)
        
        # Check sector diversification if enabled
        if self.config.strategy.sector_diversification:
            top_gaps = self._apply_sector_diversification(top_gaps)
        
        return top_gaps

    def _apply_sector_diversification(self, gaps_df: pd.DataFrame) -> pd.DataFrame:
        """Apply sector diversification limits."""
        if gaps_df.empty or 'sector' not in gaps_df.columns:
            return gaps_df
        
        diversified_gaps = []
        sector_counts = {}
        max_per_sector = self.config.strategy.max_per_sector
        
        for _, row in gaps_df.iterrows():
            sector = row.get('sector', 'Unknown')
            current_count = sector_counts.get(sector, 0)
            
            if current_count < max_per_sector:
                diversified_gaps.append(row)
                sector_counts[sector] = current_count + 1
        
        return pd.DataFrame(diversified_gaps) if diversified_gaps else pd.DataFrame()

    def _process_entries(self, entry_candidates: pd.DataFrame, date: datetime) -> int:
        """Process new position entries."""
        if entry_candidates.empty:
            return 0
        
        trades_entered = 0
        available_slots = self.max_positions - len(self.open_trades)
        
        for _, candidate in entry_candidates.head(available_slots).iterrows():
            # Check if we have enough cash
            total_cost = self.position_size_usd * (1 + self.slippage_bps / 10000)
            if self.cash < total_cost:
                break
            
            # Create new trade
            trade = self._create_trade(candidate, date)
            if trade:
                self.open_trades.append(trade)
                self.cash -= total_cost
                trades_entered += 1
        
        return trades_entered

    def _create_trade(self, candidate: pd.Series, date: datetime) -> Optional[Trade]:
        """Create a new trade from gap candidate."""
        try:
            entry_price = candidate['current_open']
            
            # Apply slippage
            slippage_factor = 1 + (self.slippage_bps / 10000)
            adjusted_entry_price = entry_price * slippage_factor
            
            # Calculate shares
            shares = self.position_size_usd / adjusted_entry_price
            
            # Calculate commission
            commission = shares * self.commission_per_share
            
            trade = Trade(
                symbol=candidate['symbol'],
                entry_date=date,
                entry_price=adjusted_entry_price,
                shares=shares,
                gap_pct=candidate['gap_pct'],
                rank=candidate.get('gap_rank', 0),
                sector=candidate.get('sector', 'Unknown'),
                max_price=adjusted_entry_price,
                min_price=adjusted_entry_price,
                trailing_stop_price=0
            )
            
            return trade
            
        except Exception as e:
            logger.warning(f"Failed to create trade for {candidate.get('symbol', 'Unknown')}: {e}")
            return None

    def _close_trade(self, trade: Trade) -> None:
        """Close a trade and calculate P&L."""
        if not trade.exit_price or not trade.exit_date:
            return
        
        # Calculate gross P&L
        trade.pnl_gross = trade.shares * (trade.exit_price - trade.entry_price)
        
        # Calculate costs
        entry_commission = trade.shares * self.commission_per_share
        exit_commission = trade.shares * self.commission_per_share
        total_commission = entry_commission + exit_commission
        
        # Net P&L
        trade.pnl_net = trade.pnl_gross - total_commission
        
        # Return percentage
        trade.return_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
        
        # Hold time
        if trade.entry_date and trade.exit_date:
            hold_time_delta = trade.exit_date - trade.entry_date
            trade.hold_time_hours = hold_time_delta.total_seconds() / 3600
        
        # Add cash back to portfolio
        proceeds = trade.shares * trade.exit_price - exit_commission
        self.cash += proceeds
        
        # Move to closed trades
        self.closed_trades.append(trade)
        self.open_trades.remove(trade)
        
        logger.debug(f"Closed {trade.symbol}: {trade.return_pct:.2%} return, ${trade.pnl_net:.2f} P&L")

    def _record_portfolio_snapshot(self, date: datetime, daily_trades: int) -> None:
        """Record daily portfolio snapshot."""
        # Calculate open positions value
        positions_value = 0
        if self.open_trades:
            symbols = [trade.symbol for trade in self.open_trades]
            try:
                price_data = self.data_manager.get_price_data(symbols, date, date + timedelta(days=1), ['Close'])
                
                for trade in self.open_trades:
                    if trade.symbol in price_data and not price_data[trade.symbol].empty:
                        symbol_data = price_data[trade.symbol]
                        if date in symbol_data.index:
                            current_price = symbol_data.loc[date, 'Close']
                            positions_value += trade.shares * current_price
            except Exception as e:
                logger.debug(f"Error calculating positions value: {e}")
        
        total_value = self.cash + positions_value
        
        # Calculate daily P&L
        daily_pnl = 0
        if self.portfolio_history:
            daily_pnl = total_value - self.portfolio_history[-1].total_value
        
        snapshot = PortfolioSnapshot(
            date=date,
            cash=self.cash,
            positions_value=positions_value,
            total_value=total_value,
            daily_pnl=daily_pnl,
            open_positions=len(self.open_trades),
            trades_today=daily_trades
        )
        
        self.portfolio_history.append(snapshot)

    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.portfolio_history or not self.closed_trades:
            return {
                'final_value': self.initial_capital,
                'total_return': 0,
                'total_return_pct': 0,
                'num_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'portfolio_df': pd.DataFrame(),
                'trades_df': pd.DataFrame()
            }
        
        # Portfolio performance
        final_value = self.portfolio_history[-1].total_value
        total_return = final_value - self.initial_capital
        total_return_pct = total_return / self.initial_capital
        
        # Trade statistics
        winning_trades = [t for t in self.closed_trades if t.pnl_net > 0]
        win_rate = len(winning_trades) / len(self.closed_trades) if self.closed_trades else 0
        avg_return = sum(t.return_pct for t in self.closed_trades) / len(self.closed_trades) if self.closed_trades else 0
        
        # Portfolio DataFrame
        portfolio_data = []
        for snapshot in self.portfolio_history:
            portfolio_data.append({
                'date': snapshot.date,
                'total_value': snapshot.total_value,
                'cash': snapshot.cash,
                'positions_value': snapshot.positions_value,
                'daily_pnl': snapshot.daily_pnl,
                'open_positions': snapshot.open_positions,
                'trades_today': snapshot.trades_today
            })
        
        portfolio_df = pd.DataFrame(portfolio_data)
        if not portfolio_df.empty:
            portfolio_df.set_index('date', inplace=True)
        
        # Trades DataFrame
        trades_data = []
        for trade in self.closed_trades:
            trades_data.append({
                'symbol': trade.symbol,
                'entry_date': trade.entry_date,
                'exit_date': trade.exit_date,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'gap_pct': trade.gap_pct,
                'return_pct': trade.return_pct,
                'pnl_net': trade.pnl_net,
                'hold_time_hours': trade.hold_time_hours,
                'exit_reason': trade.exit_reason,
                'rank': trade.rank,
                'sector': trade.sector
            })
        
        trades_df = pd.DataFrame(trades_data)
        
        # Risk metrics
        if len(portfolio_df) > 1:
            returns = portfolio_df['total_value'].pct_change().dropna()
            
            # Sharpe ratio (assuming 252 trading days)
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            else:
                sharpe_ratio = 0
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Exit reason analysis
        exit_reasons = {}
        for trade in self.closed_trades:
            reason = trade.exit_reason or 'unknown'
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        return {
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'num_trades': len(self.closed_trades),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'avg_winner': np.mean([t.return_pct for t in winning_trades]) if winning_trades else 0,
            'avg_loser': np.mean([t.return_pct for t in self.closed_trades if t.pnl_net <= 0]) if self.closed_trades else 0,
            'largest_winner': max([t.return_pct for t in self.closed_trades]) if self.closed_trades else 0,
            'largest_loser': min([t.return_pct for t in self.closed_trades]) if self.closed_trades else 0,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'exit_reasons': exit_reasons,
            'portfolio_df': portfolio_df,
            'trades_df': trades_df
        }