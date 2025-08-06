#!/usr/bin/env python3
"""CLI Paper Trading script with configuration support."""

import logging
import os
import sys
import time
from datetime import datetime, timedelta, time as time_obj
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import signal
import schedule

import click
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

# Try to import alpaca API with fallback
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    tradeapi = None

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Try to import project components with fallback
try:
    from gappers.config_new import Config, get_config
    from gappers.data_collector import DataCollector
    from gappers.data_manager import DataManager
    from gappers.gap_engine import GapEngine
    from gappers.universe import UniverseBuilder
    from gappers.simulation_manager import SimulationManager
    PROJECT_COMPONENTS_AVAILABLE = True
except ImportError as e:
    PROJECT_COMPONENTS_AVAILABLE = False
    logger.error(f"Failed to import project components: {e}")
    console.print(f"[red]✗[/red] Project components not available: {e}")

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paper_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
console = Console()

# Global variables for graceful shutdown
running = True
current_positions = {}


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global running
    console.print("\n[yellow]Received shutdown signal. Stopping trading...[/yellow]")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class PaperTradingBot:
    """Paper trading bot using Alpaca API and saved configurations."""
    
    def __init__(self, config_name: str, dry_run: bool = False):
        self.config_name = config_name
        self.dry_run = dry_run
        
        # Check dependencies
        if not ALPACA_AVAILABLE and not dry_run:
            raise ImportError("Alpaca Trade API not available. Install with: pip install alpaca-trade-api")
        
        if not PROJECT_COMPONENTS_AVAILABLE:
            raise ImportError("Project components not available. Check project structure and dependencies")
        
        # Initialize system components
        self.config = get_config()
        self.data_collector = DataCollector(self.config)
        self.data_manager = DataManager(self.config)
        self.gap_engine = GapEngine(self.config)
        self.universe_builder = UniverseBuilder(self.config)
        self.sim_manager = SimulationManager()
        
        # Load saved configuration
        self.trading_config = self._load_configuration()
        
        # Initialize Alpaca API (if not in dry run mode)
        if not dry_run and ALPACA_AVAILABLE:
            self.api = self._initialize_alpaca()
        else:
            self.api = None
        
        # Trading state
        self.positions = {}
        self.pending_orders = {}
        
    def _load_configuration(self) -> Dict[str, Any]:
        """Load saved configuration from SimulationManager."""
        try:
            simulations = self.sim_manager.list_simulations()
            matching_sims = [s for s in simulations if s.get('name') == self.config_name]
            
            if not matching_sims:
                available = [s.get('name', 'Unknown') for s in simulations]
                raise ValueError(f"Configuration '{self.config_name}' not found. Available: {available}")
            
            # Load the full simulation data
            filename = matching_sims[0].get('filename')
            if not filename:
                raise ValueError(f"No filename found for configuration '{self.config_name}'")
            
            save = self.sim_manager.load_simulation(filename)
            if not save:
                raise ValueError(f"Failed to load configuration '{self.config_name}'")
            
            config = save.config
            
            console.print(f"[green]✓[/green] Loaded configuration: {self.config_name}")
            console.print(f"[blue]Strategy:[/blue] {config.get('strategy', {}).get('name', 'Unknown')}")
            
            return config
            
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load configuration: {e}")
            raise
    
    def _initialize_alpaca(self) -> Optional[tradeapi.REST]:
        """Initialize Alpaca API connection."""
        try:
            if not ALPACA_AVAILABLE:
                raise ImportError("Alpaca Trade API not available")
                
            api_key = os.getenv('ALPACA_API_KEY')
            api_secret = os.getenv('ALPACA_API_SECRET')
            base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
            
            if not api_key or not api_secret:
                raise ValueError("Alpaca API credentials not found in environment variables")
            
            api = tradeapi.REST(api_key, api_secret, base_url)
            
            # Test connection
            account = api.get_account()
            console.print(f"[green]✓[/green] Connected to Alpaca (Paper Trading)")
            console.print(f"[blue]Account:[/blue] ${float(account.buying_power):,.2f} buying power")
            
            return api
            
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to initialize Alpaca API: {e}")
            if self.dry_run:
                console.print("[yellow]Continuing in dry-run mode[/yellow]")
                return None
            raise
    
    def _is_trading_hours(self) -> bool:
        """Check if current time is within trading hours."""
        now = datetime.now()
        trading_start = self.trading_config.get('automation', {}).get('trading_hours_start', '09:30')
        trading_end = self.trading_config.get('automation', {}).get('trading_hours_end', '16:00')
        
        start_time = datetime.strptime(trading_start, '%H:%M').time()
        end_time = datetime.strptime(trading_end, '%H:%M').time()
        current_time = now.time()
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        return start_time <= current_time <= end_time
    
    def scan_gaps(self, date: datetime = None) -> pd.DataFrame:
        """Scan for gap opportunities using the project's gap engine."""
        try:
            if date is None:
                date = datetime.now()
            
            console.print(f"[blue]Scanning gaps for {date.strftime('%Y-%m-%d')}...[/blue]")
            
            # Get universe
            universe_df = self.universe_builder.build_universe()
            symbols = universe_df['symbol'].tolist() if not universe_df.empty else []
            
            if not symbols:
                console.print("[yellow]No symbols found in universe[/yellow]")
                return pd.DataFrame()
            
            console.print(f"[blue]Analyzing {len(symbols)} symbols...[/blue]")
            
            # Calculate gaps (date first, then symbols)
            gaps_df = self.gap_engine.calculate_daily_gaps(date, symbols)
            
            if gaps_df.empty:
                console.print("[yellow]No gaps found[/yellow]")
                return pd.DataFrame()
            
            # Filter based on strategy criteria
            strategy_config = self.trading_config.get('strategy', {})
            min_gap = strategy_config.get('min_gap_percent', 0.02)
            max_gap = strategy_config.get('max_gap_percent', 0.20)
            
            filtered_gaps = gaps_df[
                (abs(gaps_df['gap_percent']) >= min_gap) &
                (abs(gaps_df['gap_percent']) <= max_gap)
            ]
            
            console.print(f"[green]Found {len(filtered_gaps)} qualifying gaps[/green]")
            
            return filtered_gaps.head(10)  # Limit to top 10 opportunities
            
        except Exception as e:
            console.print(f"[red]✗[/red] Gap scanning failed: {e}")
            logger.error(f"Gap scanning error: {e}")
            return pd.DataFrame()
    
    def calculate_position_size(self, symbol: str, price: float) -> int:
        """Calculate position size based on risk management rules."""
        try:
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            
            risk_config = self.trading_config.get('risk_management', {})
            max_position_size = risk_config.get('max_position_size', 0.05)  # 5% max per position
            
            max_dollar_amount = buying_power * max_position_size
            shares = int(max_dollar_amount / price)
            
            # Minimum position size
            min_shares = max(1, int(1000 / price))  # At least $1000 or 1 share
            
            return max(min_shares, shares)
            
        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            return 0
    
    def place_order(self, symbol: str, side: str, qty: int, order_type: str = 'market') -> Optional[str]:
        """Place an order through Alpaca API."""
        try:
            if self.dry_run:
                console.print(f"[yellow]DRY RUN:[/yellow] Would place {side} order for {qty} shares of {symbol}")
                return f"dry_run_order_{symbol}_{side}_{qty}"
            
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force='day'
            )
            
            console.print(f"[green]✓[/green] Placed {side} order: {qty} shares of {symbol} (Order ID: {order.id})")
            return order.id
            
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to place {side} order for {symbol}: {e}")
            logger.error(f"Order placement error: {e}")
            return None
    
    def execute_strategy(self, gaps_df: pd.DataFrame) -> None:
        """Execute trading strategy based on gap opportunities."""
        if gaps_df.empty:
            return
        
        strategy_config = self.trading_config.get('strategy', {})
        entry_timing = strategy_config.get('entry_timing', 'gap_open')
        
        for _, gap_data in gaps_df.iterrows():
            symbol = gap_data['symbol']
            gap_percent = gap_data['gap_percent']
            current_price = gap_data.get('current_price', gap_data.get('open_price'))
            
            if symbol in self.positions:
                continue  # Already have a position
            
            # Determine trade direction based on gap and strategy
            if entry_timing == 'gap_open' and gap_percent > 0:
                # Buy gap-up stocks (momentum strategy)
                side = 'buy'
            elif entry_timing == 'gap_fill' and gap_percent > 0:
                # Sell gap-up stocks expecting reversion
                side = 'sell'
            elif entry_timing == 'gap_open' and gap_percent < 0:
                # Buy gap-down stocks (contrarian strategy)
                side = 'buy'
            else:
                continue
            
            # Calculate position size
            qty = self.calculate_position_size(symbol, current_price)
            if qty <= 0:
                continue
            
            # Place order
            order_id = self.place_order(symbol, side, qty)
            if order_id:
                self.pending_orders[order_id] = {
                    'symbol': symbol,
                    'side': side,
                    'qty': qty,
                    'timestamp': datetime.now()
                }
    
    def monitor_positions(self) -> None:
        """Monitor existing positions and execute exit strategy."""
        if not self.api:
            if self.dry_run:
                console.print("[yellow]Dry run mode: Skipping position monitoring[/yellow]")
            return
            
        try:
            positions = self.api.list_positions()
            current_positions = {pos.symbol: pos for pos in positions}
            
            if not current_positions:
                return
            
            strategy_config = self.trading_config.get('strategy', {})
            profit_target = strategy_config.get('profit_target', 0.05)
            stop_loss = strategy_config.get('stop_loss', 0.03)
            
            for symbol, position in current_positions.items():
                market_value = float(position.market_value)
                unrealized_pl = float(position.unrealized_pl)
                unrealized_plpc = float(position.unrealized_plpc)
                
                # Check exit conditions
                should_exit = False
                exit_reason = ""
                
                if unrealized_plpc >= profit_target:
                    should_exit = True
                    exit_reason = f"profit target ({unrealized_plpc:.2%})"
                elif unrealized_plpc <= -stop_loss:
                    should_exit = True
                    exit_reason = f"stop loss ({unrealized_plpc:.2%})"
                
                if should_exit:
                    side = 'sell' if position.side == 'long' else 'buy'
                    qty = abs(int(position.qty))
                    
                    order_id = self.place_order(symbol, side, qty)
                    if order_id:
                        console.print(f"[yellow]Exit:[/yellow] {symbol} due to {exit_reason}")
            
        except Exception as e:
            logger.error(f"Position monitoring error: {e}")
    
    def display_status(self) -> None:
        """Display current trading status."""
        if not self.api:
            if self.dry_run:
                console.print("[yellow]Dry run mode: No account status available[/yellow]")
                return
            else:
                console.print("[red]API not available[/red]")
                return
                
        try:
            # Get account info
            account = self.api.get_account()
            positions = self.api.list_positions()
            
            # Create status table
            table = Table(title="Paper Trading Status")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("Account Value", f"${float(account.portfolio_value):,.2f}")
            table.add_row("Buying Power", f"${float(account.buying_power):,.2f}")
            table.add_row("Day P&L", f"${float(account.todays_pl):,.2f}")
            table.add_row("Total P&L", f"${float(account.total_pl):,.2f}")
            table.add_row("Open Positions", str(len(positions)))
            
            console.print(table)
            
            # Show positions if any
            if positions:
                pos_table = Table(title="Current Positions")
                pos_table.add_column("Symbol")
                pos_table.add_column("Side")
                pos_table.add_column("Qty")
                pos_table.add_column("Market Value")
                pos_table.add_column("Unrealized P&L")
                
                for pos in positions:
                    pos_table.add_row(
                        pos.symbol,
                        pos.side,
                        pos.qty,
                        f"${float(pos.market_value):,.2f}",
                        f"${float(pos.unrealized_pl):,.2f}"
                    )
                
                console.print(pos_table)
                
        except Exception as e:
            logger.error(f"Status display error: {e}")
    
    def run_trading_session(self) -> None:
        """Run a single trading session."""
        console.print(f"[green]Starting trading session at {datetime.now()}[/green]")
        
        # Check if it's trading hours
        if not self._is_trading_hours():
            console.print("[yellow]Outside trading hours, skipping session[/yellow]")
            return
        
        # Scan for gaps
        gaps_df = self.scan_gaps()
        
        # Execute strategy
        if not gaps_df.empty:
            self.execute_strategy(gaps_df)
        
        # Monitor existing positions
        self.monitor_positions()
        
        # Display status
        self.display_status()
    
    def run_continuous(self, scan_interval: int = 300) -> None:
        """Run continuous trading with scheduled scans."""
        console.print(f"[green]Starting continuous paper trading (scan every {scan_interval}s)[/green]")
        console.print("[yellow]Press Ctrl+C to stop[/yellow]")
        
        # Schedule trading sessions
        schedule.every(scan_interval).seconds.do(self.run_trading_session)
        
        # Initial run
        self.run_trading_session()
        
        # Main loop
        while running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                break
        
        console.print("[green]Trading stopped[/green]")


@click.group()
@click.version_option()
def cli():
    """CLI Paper Trading System - Trade with saved configurations."""
    pass


@cli.command()
@click.argument('config_name')
@click.option('--dry-run', is_flag=True, help='Run in dry-run mode (no actual orders)')
@click.option('--scan-interval', default=300, help='Gap scan interval in seconds')
def run(config_name: str, dry_run: bool, scan_interval: int):
    """Run continuous paper trading with saved configuration."""
    try:
        bot = PaperTradingBot(config_name, dry_run=dry_run)
        bot.run_continuous(scan_interval)
    except Exception as e:
        console.print(f"[red]✗[/red] Trading failed: {e}")
        logger.error(f"Trading error: {e}")
        sys.exit(1)


@cli.command()
@click.argument('config_name')
@click.option('--dry-run', is_flag=True, help='Run in dry-run mode (no actual orders)')
def scan(config_name: str, dry_run: bool):
    """Run a single gap scan and trading session."""
    try:
        bot = PaperTradingBot(config_name, dry_run=dry_run)
        bot.run_trading_session()
    except Exception as e:
        console.print(f"[red]✗[/red] Scan failed: {e}")
        logger.error(f"Scan error: {e}")
        sys.exit(1)


@cli.command()
def list_configs():
    """List available saved configurations."""
    try:
        sim_manager = SimulationManager()
        simulations = sim_manager.list_simulations()
        
        if not simulations:
            console.print("[yellow]No saved configurations found[/yellow]")
            return
        
        table = Table(title="Available Configurations")
        table.add_column("Name", style="cyan")
        table.add_column("File", style="blue")
        table.add_column("Created", style="magenta")
        table.add_column("Size", style="white")
        
        for sim in simulations:
            name = sim.get('name', 'Unknown')
            filename = sim.get('filename', 'Unknown')
            created = sim.get('created', 'Unknown')
            size = sim.get('size', 'Unknown')
            
            table.add_row(name, filename, created, size)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to list configurations: {e}")
        sys.exit(1)


@cli.command()
@click.argument('config_name')
def status(config_name: str):
    """Show current account and position status."""
    try:
        bot = PaperTradingBot(config_name, dry_run=False)
        bot.display_status()
    except Exception as e:
        console.print(f"[red]✗[/red] Status failed: {e}")
        sys.exit(1)


@cli.command()
def test():
    """Test CLI functionality and dependencies."""
    console.print("[blue]Testing CLI Paper Trading Dependencies...[/blue]")
    
    # Test Alpaca API
    if ALPACA_AVAILABLE:
        console.print("[green]✓[/green] Alpaca Trade API available")
    else:
        console.print("[yellow]⚠[/yellow] Alpaca Trade API not available (install with: pip install alpaca-trade-api)")
    
    # Test project components
    if PROJECT_COMPONENTS_AVAILABLE:
        console.print("[green]✓[/green] Project components available")
    else:
        console.print("[red]✗[/red] Project components not available")
        return
    
    # Test configuration loading
    try:
        sim_manager = SimulationManager()
        simulations = sim_manager.list_simulations()
        console.print(f"[green]✓[/green] Found {len(simulations)} saved configurations")
        
        if simulations:
            console.print("Available configurations:")
            for sim in simulations[:3]:  # Show first 3
                name = sim.get('name', 'Unknown')
                console.print(f"  - {name}")
        else:
            console.print("[yellow]⚠[/yellow] No saved configurations found")
    except Exception as e:
        console.print(f"[red]✗[/red] Configuration loading failed: {e}")
    
    # Test environment variables
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')
    
    if api_key and api_secret:
        console.print("[green]✓[/green] Alpaca API credentials found in .env")
    else:
        console.print("[yellow]⚠[/yellow] Alpaca API credentials not found in .env file")
    
    console.print("\n[blue]CLI is ready to use![/blue]")
    console.print("Run 'python cli_paper_trading.py list-configs' to see available configurations")
    console.print("Run 'python cli_paper_trading.py scan <config_name> --dry-run' to test gap scanning")


if __name__ == '__main__':
    cli()