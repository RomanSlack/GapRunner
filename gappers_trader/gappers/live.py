"""Live trading module with Alpaca paper trading integration."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
from alpaca_trade_api import REST, TimeFrame
from alpaca_trade_api.entity import Order, Position
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from gappers.backtest import GapParams
from gappers.config import config
from gappers.datafeed import DataFeed
from gappers.signals import SignalGenerator
from gappers.universe import UniverseBuilder

logger = logging.getLogger(__name__)


class LiveTrader:
    """Live trading system with Alpaca paper trading integration."""

    def __init__(
        self,
        data_feed: Optional[DataFeed] = None,
        signal_generator: Optional[SignalGenerator] = None,
        dry_run: bool = True,
    ) -> None:
        """
        Initialize live trader.

        Args:
            data_feed: Data feed instance
            signal_generator: Signal generator instance
            dry_run: If True, only log trades without executing
        """
        self.data_feed = data_feed or DataFeed()
        self.signal_generator = signal_generator or SignalGenerator(
            self.data_feed, UniverseBuilder(self.data_feed)
        )
        self.dry_run = dry_run

        # Initialize Alpaca client
        self.alpaca = self._initialize_alpaca_client()

        # Active orders and positions
        self.active_orders: Dict[str, Order] = {}
        self.active_positions: Dict[str, Position] = {}

        # Trading state
        self.is_trading_active = False
        self.daily_trades = 0
        self.daily_pnl = 0.0

        # Scheduler for market events
        self.scheduler = BlockingScheduler()

    def _initialize_alpaca_client(self) -> Optional[REST]:
        """Initialize Alpaca REST client."""
        if not config.has_alpaca_credentials:
            logger.warning("Alpaca credentials not configured - running in simulation mode")
            return None

        try:
            client = REST(
                key_id=config.alpaca_api_key,
                secret_key=config.alpaca_secret_key,
                base_url=config.alpaca_base_url,
                api_version='v2'
            )

            # Test connection
            account = client.get_account()
            logger.info(f"Connected to Alpaca: Account {account.id} (${account.equity})")

            return client

        except Exception as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
            return None

    def start_live_trading(self, params: GapParams) -> None:
        """
        Start live trading with scheduled market events.

        Args:
            params: Trading parameters
        """
        logger.info("Starting live trading system")

        # Schedule market events (Eastern Time)
        self.scheduler.add_job(
            func=self._pre_market_scan,
            trigger=CronTrigger(hour=9, minute=29, timezone='US/Eastern'),
            args=[params],
            id='pre_market_scan',
            name='Pre-market gap scan'
        )

        self.scheduler.add_job(
            func=self._market_open_execution,
            trigger=CronTrigger(hour=9, minute=30, timezone='US/Eastern'),
            args=[params],
            id='market_open',
            name='Market open execution'
        )

        self.scheduler.add_job(
            func=self._position_monitoring,
            trigger=CronTrigger(minute='*/1', timezone='US/Eastern'),
            args=[params],
            id='position_monitoring',
            name='Position monitoring'
        )

        self.scheduler.add_job(
            func=self._market_close_cleanup,
            trigger=CronTrigger(hour=16, minute=0, timezone='US/Eastern'),
            args=[params],
            id='market_close',
            name='Market close cleanup'
        )

        # Start scheduler
        try:
            self.scheduler.start()
        except KeyboardInterrupt:
            logger.info("Live trading stopped by user")
            self.stop_live_trading()

    def stop_live_trading(self) -> None:
        """Stop live trading and cleanup."""
        logger.info("Stopping live trading system")

        # Stop scheduler
        if self.scheduler.running:
            self.scheduler.shutdown()

        # Close all positions if requested
        if not self.dry_run:
            self._close_all_positions()

        # Cancel all orders
        self._cancel_all_orders()

        logger.info("Live trading system stopped")

    def _pre_market_scan(self, params: GapParams) -> None:
        """Scan for gaps before market open."""
        logger.info("Running pre-market gap scan")

        try:
            # Get today's gaps
            today = datetime.now()
            gaps_df = self.signal_generator.calculate_gaps(
                today,
                min_gap_pct=params.min_gap_pct,
                max_gap_pct=params.max_gap_pct
            )

            if gaps_df.empty:
                logger.info("No significant gaps found today")
                return

            # Rank gaps
            top_gaps = self.signal_generator.rank_gaps(
                gaps_df,
                top_k=params.max_positions,
                sector_diversification=params.sector_diversification,
                max_per_sector=params.max_per_sector
            )

            logger.info(f"Found {len(top_gaps)} gap opportunities:")
            for _, gap in top_gaps.iterrows():
                logger.info(f"  {gap['symbol']}: {gap['gap_pct']:.2%} gap")

            # Store for market open execution
            self.gap_opportunities = top_gaps

        except Exception as e:
            logger.error(f"Error in pre-market scan: {e}")

    def _market_open_execution(self, params: GapParams) -> None:
        """Execute trades at market open."""
        if not hasattr(self, 'gap_opportunities'):
            logger.warning("No gap opportunities available for execution")
            return

        logger.info("Executing market open trades")

        try:
            for _, gap in self.gap_opportunities.iterrows():
                symbol = gap['symbol']

                # Check if already holding position
                if symbol in self.active_positions:
                    logger.info(f"Already holding position in {symbol}, skipping")
                    continue

                # Check daily trade limit
                if self.daily_trades >= params.max_positions:
                    logger.info("Daily trade limit reached")
                    break

                # Execute trade
                success = self._execute_gap_trade(symbol, gap, params)
                if success:
                    self.daily_trades += 1

        except Exception as e:
            logger.error(f"Error in market open execution: {e}")

    def _execute_gap_trade(self, symbol: str, gap_data: Dict, params: GapParams) -> bool:
        """
        Execute a gap trade with bracket orders.

        Args:
            symbol: Stock symbol
            gap_data: Gap information
            params: Trading parameters

        Returns:
            True if trade executed successfully
        """
        try:
            # Get current price
            current_price = self._get_current_price(symbol)
            if not current_price:
                logger.error(f"Could not get current price for {symbol}")
                return False

            # Calculate position size
            position_value = min(params.position_size, self._get_available_buying_power())
            shares = int(position_value / current_price)

            if shares < 1:
                logger.warning(f"Insufficient buying power for {symbol}")
                return False

            # Calculate target and stop prices
            target_price = current_price * (1 + params.profit_target)
            stop_price = current_price * (1 - params.stop_loss)

            logger.info(f"Executing gap trade for {symbol}:")
            logger.info(f"  Shares: {shares}")
            logger.info(f"  Entry: ${current_price:.2f}")
            logger.info(f"  Target: ${target_price:.2f}")
            logger.info(f"  Stop: ${stop_price:.2f}")

            if self.dry_run:
                logger.info(f"[DRY RUN] Would execute bracket order for {symbol}")
                return True

            # Execute bracket order
            if self.alpaca:
                order = self._submit_bracket_order(
                    symbol, shares, target_price, stop_price
                )

                if order:
                    self.active_orders[symbol] = order
                    logger.info(f"Bracket order submitted for {symbol}: {order.id}")
                    return True

        except Exception as e:
            logger.error(f"Error executing gap trade for {symbol}: {e}")

        return False

    def _submit_bracket_order(
        self, symbol: str, shares: int, target_price: float, stop_price: float
    ) -> Optional[Order]:
        """Submit bracket order to Alpaca."""
        try:
            # Submit parent market order
            parent_order = self.alpaca.submit_order(
                symbol=symbol,
                qty=shares,
                side='buy',
                type='market',
                time_in_force='day',
                order_class='bracket',
                take_profit={'limit_price': f"{target_price:.2f}"},
                stop_loss={'stop_price': f"{stop_price:.2f}"}
            )

            return parent_order

        except Exception as e:
            logger.error(f"Error submitting bracket order for {symbol}: {e}")
            return None

    def _position_monitoring(self, params: GapParams) -> None:
        """Monitor active positions and manage risk."""
        if not self.active_orders and not self.active_positions:
            return

        try:
            # Update position status
            self._update_positions()

            # Check for time-based exits
            self._check_time_based_exits(params)

            # Update daily P&L
            self._update_daily_pnl()

        except Exception as e:
            logger.error(f"Error in position monitoring: {e}")

    def _update_positions(self) -> None:
        """Update active positions from Alpaca."""
        if not self.alpaca:
            return

        try:
            positions = self.alpaca.list_positions()
            self.active_positions = {pos.symbol: pos for pos in positions if float(pos.qty) != 0}

            # Update orders
            orders = self.alpaca.list_orders(status='open')
            self.active_orders = {order.symbol: order for order in orders}

        except Exception as e:
            logger.error(f"Error updating positions: {e}")

    def _check_time_based_exits(self, params: GapParams) -> None:
        """Check for time-based position exits."""
        current_time = datetime.now(timezone.utc)
        market_open = current_time.replace(hour=13, minute=30, second=0, microsecond=0)  # 9:30 AM ET in UTC
        max_hold_time = timedelta(hours=params.max_hold_time_hours)

        for symbol, position in self.active_positions.items():
            # Check if position has been held too long
            if (current_time - market_open) >= max_hold_time:
                logger.info(f"Time-based exit for {symbol}")
                self._close_position(symbol, "time_limit")

    def _close_position(self, symbol: str, reason: str) -> None:
        """Close a position."""
        if symbol not in self.active_positions:
            return

        try:
            if self.dry_run:
                logger.info(f"[DRY RUN] Would close position in {symbol} (reason: {reason})")
                return

            if self.alpaca:
                # Cancel any open orders for this symbol
                self._cancel_orders_for_symbol(symbol)

                # Submit market sell order
                position = self.active_positions[symbol]
                self.alpaca.submit_order(
                    symbol=symbol,
                    qty=abs(int(float(position.qty))),
                    side='sell',
                    type='market',
                    time_in_force='day'
                )

                logger.info(f"Market sell order submitted for {symbol}")

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")

    def _cancel_orders_for_symbol(self, symbol: str) -> None:
        """Cancel all orders for a specific symbol."""
        if not self.alpaca:
            return

        try:
            orders = self.alpaca.list_orders(status='open', symbols=[symbol])
            for order in orders:
                self.alpaca.cancel_order(order.id)
                logger.info(f"Cancelled order {order.id} for {symbol}")

        except Exception as e:
            logger.error(f"Error cancelling orders for {symbol}: {e}")

    def _market_close_cleanup(self, params: GapParams) -> None:
        """Clean up at market close."""
        logger.info("Market close cleanup")

        try:
            # Close all remaining positions
            if not self.dry_run:
                self._close_all_positions()

            # Cancel all orders
            self._cancel_all_orders()

            # Log daily summary
            self._log_daily_summary()

            # Reset daily counters
            self.daily_trades = 0
            self.daily_pnl = 0.0

        except Exception as e:
            logger.error(f"Error during market close cleanup: {e}")

    def _close_all_positions(self) -> None:
        """Close all open positions."""
        if not self.alpaca:
            return

        try:
            self.alpaca.close_all_positions()
            logger.info("All positions closed")

        except Exception as e:
            logger.error(f"Error closing all positions: {e}")

    def _cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        if not self.alpaca:
            return

        try:
            self.alpaca.cancel_all_orders()
            logger.info("All orders cancelled")

        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            if self.alpaca:
                # Get latest quote
                quote = self.alpaca.get_latest_quote(symbol)
                return float(quote.ask_price)
            else:
                # Fallback to yfinance
                data = self.data_feed.download([symbol], 
                                             start=datetime.now().date(),
                                             end=datetime.now().date(),
                                             interval="1d")
                if symbol in data and not data[symbol].empty:
                    return float(data[symbol]['open'].iloc[-1])

        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")

        return None

    def _get_available_buying_power(self) -> float:
        """Get available buying power."""
        if not self.alpaca:
            return config.default_position_size

        try:
            account = self.alpaca.get_account()
            return float(account.buying_power)

        except Exception as e:
            logger.error(f"Error getting buying power: {e}")
            return config.default_position_size

    def _update_daily_pnl(self) -> None:
        """Update daily P&L."""
        if not self.alpaca:
            return

        try:
            account = self.alpaca.get_account()
            today_pnl = float(account.todays_change)
            self.daily_pnl = today_pnl

        except Exception as e:
            logger.error(f"Error updating daily P&L: {e}")

    def _log_daily_summary(self) -> None:
        """Log daily trading summary."""
        logger.info("=" * 50)
        logger.info("DAILY TRADING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Trades executed: {self.daily_trades}")
        logger.info(f"Daily P&L: ${self.daily_pnl:.2f}")
        logger.info(f"Active positions: {len(self.active_positions)}")
        logger.info(f"Active orders: {len(self.active_orders)}")
        logger.info("=" * 50)

    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'is_trading_active': self.is_trading_active,
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'active_positions': len(self.active_positions),
            'active_orders': len(self.active_orders),
            'dry_run': self.dry_run
        }

        if self.alpaca:
            try:
                account = self.alpaca.get_account()
                status.update({
                    'account_equity': float(account.equity),
                    'buying_power': float(account.buying_power),
                    'day_trade_count': int(account.daytrade_count)
                })
            except Exception as e:
                logger.error(f"Error getting account status: {e}")

        return status

    def manual_trade_execution(
        self, symbol: str, action: str, params: GapParams
    ) -> bool:
        """
        Manually execute a trade.

        Args:
            symbol: Stock symbol
            action: 'buy' or 'sell'
            params: Trading parameters

        Returns:
            True if successful
        """
        logger.info(f"Manual trade execution: {action} {symbol}")

        try:
            if action.lower() == 'buy':
                # Create fake gap data for manual execution
                gap_data = {
                    'symbol': symbol,
                    'gap_pct': 0.03,  # Assume 3% gap
                    'rank': 1
                }
                return self._execute_gap_trade(symbol, gap_data, params)

            elif action.lower() == 'sell':
                if symbol in self.active_positions:
                    self._close_position(symbol, "manual")
                    return True
                else:
                    logger.warning(f"No position found for {symbol}")
                    return False

        except Exception as e:
            logger.error(f"Error in manual trade execution: {e}")
            return False

        return False

    def run_single_scan(self, params: GapParams) -> Dict:
        """
        Run a single gap scan (useful for testing).

        Args:
            params: Trading parameters

        Returns:
            Dictionary with scan results
        """
        logger.info("Running single gap scan")

        try:
            # Get today's gaps
            today = datetime.now()
            gaps_df = self.signal_generator.calculate_gaps(
                today,
                min_gap_pct=params.min_gap_pct,
                max_gap_pct=params.max_gap_pct
            )

            if gaps_df.empty:
                return {'gaps_found': 0, 'opportunities': []}

            # Rank gaps
            top_gaps = self.signal_generator.rank_gaps(
                gaps_df,
                top_k=params.max_positions,
                sector_diversification=params.sector_diversification,
                max_per_sector=params.max_per_sector
            )

            opportunities = []
            for _, gap in top_gaps.iterrows():
                opportunities.append({
                    'symbol': gap['symbol'],
                    'gap_pct': gap['gap_pct'],
                    'previous_close': gap['previous_close'],
                    'current_open': gap['current_open'],
                    'rank': gap['rank']
                })

            return {
                'gaps_found': len(top_gaps),
                'opportunities': opportunities,
                'scan_time': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in single scan: {e}")
            return {'error': str(e)}


def main():
    """Main function for running live trader from command line."""
    import argparse

    parser = argparse.ArgumentParser(description='Gap Trading Live System')
    parser.add_argument('--dry-run', action='store_true', help='Run in dry-run mode')
    parser.add_argument('--scan-only', action='store_true', help='Run single scan only')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create trader
    trader = LiveTrader(dry_run=args.dry_run)

    # Default parameters
    params = GapParams()

    if args.scan_only:
        # Run single scan
        results = trader.run_single_scan(params)
        print("Scan Results:")
        print(f"Gaps found: {results.get('gaps_found', 0)}")
        for opp in results.get('opportunities', []):
            print(f"  {opp['symbol']}: {opp['gap_pct']:.2%} gap")
    else:
        # Start live trading
        trader.start_live_trading(params)


if __name__ == "__main__":
    main()