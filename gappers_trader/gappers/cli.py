"""Command-line interface for the gap trading system."""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import click
import pandas as pd

from gappers import (
    Backtester,
    DataFeed,
    GapParams,
    LiveTrader,
    PerformanceAnalyzer,
    SignalGenerator,
    UniverseBuilder,
)
from gappers.config import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose: bool) -> None:
    """Gap Trading System - Production-grade overnight gap trading with live paper-trading support."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option('--start-date', '-s', type=click.DateTime(['%Y-%m-%d']), 
              default=str((datetime.now() - timedelta(days=365)).date()),
              help='Backtest start date (YYYY-MM-DD)')
@click.option('--end-date', '-e', type=click.DateTime(['%Y-%m-%d']),
              default=str((datetime.now() - timedelta(days=1)).date()),
              help='Backtest end date (YYYY-MM-DD)')
@click.option('--profit-target', '-p', type=float, default=0.05,
              help='Profit target percentage (default: 0.05)')
@click.option('--stop-loss', '-l', type=float, default=0.02,
              help='Stop loss percentage (default: 0.02)')
@click.option('--top-k', '-k', type=int, default=10,
              help='Number of top gaps to trade (default: 10)')
@click.option('--min-gap', type=float, default=0.02,
              help='Minimum gap percentage (default: 0.02)')
@click.option('--max-gap', type=float, default=0.30,
              help='Maximum gap percentage (default: 0.30)')
@click.option('--position-size', type=float, default=10000,
              help='Position size in dollars (default: 10000)')
@click.option('--max-positions', type=int, default=10,
              help='Maximum concurrent positions (default: 10)')
@click.option('--commission', type=float, default=0.005,
              help='Commission per share (default: 0.005)')
@click.option('--slippage-bps', type=float, default=10,
              help='Slippage in basis points (default: 10)')
@click.option('--output', '-o', type=click.Path(), 
              help='Output file for results (CSV format)')
@click.option('--report', '-r', type=click.Path(),
              help='Generate detailed report to file')
def backtest(
    start_date: datetime,
    end_date: datetime,
    profit_target: float,
    stop_loss: float,
    top_k: int,
    min_gap: float,
    max_gap: float,
    position_size: float,
    max_positions: int,
    commission: float,
    slippage_bps: float,
    output: Optional[str],
    report: Optional[str]
) -> None:
    """Run backtest with specified parameters."""
    
    click.echo(f"🎯 Running backtest from {start_date.date()} to {end_date.date()}")
    
    # Create parameters
    params = GapParams(
        profit_target=profit_target,
        stop_loss=stop_loss,
        top_k=top_k,
        min_gap_pct=min_gap,
        max_gap_pct=max_gap,
        position_size=position_size,
        max_positions=max_positions,
        commission_per_share=commission,
        slippage_bps=slippage_bps,
    )
    
    try:
        # Initialize components
        data_feed = DataFeed()
        signal_generator = SignalGenerator(data_feed, UniverseBuilder(data_feed))
        backtester = Backtester(data_feed, signal_generator)
        analyzer = PerformanceAnalyzer()
        
        # Run backtest
        with click.progressbar(length=1, label='Running backtest') as bar:
            results = backtester.run_backtest(start_date, end_date, params)
            bar.update(1)
        
        # Analyze results
        analysis = analyzer.analyze_backtest_results(results)
        
        # Display summary
        display_backtest_summary(analysis)
        
        # Save results
        if output:
            save_backtest_results(results, output)
            click.echo(f"📁 Results saved to {output}")
        
        # Generate report
        if report:
            report_text = analyzer.generate_performance_report(analysis)
            Path(report).write_text(report_text)
            click.echo(f"📄 Report saved to {report}")
            
    except Exception as e:
        click.echo(f"❌ Error running backtest: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--profit-target', '-p', type=float, multiple=True,
              default=[0.03, 0.05, 0.07], help='Profit target values to test')
@click.option('--stop-loss', '-l', type=float, multiple=True,
              default=[0.01, 0.02, 0.03], help='Stop loss values to test')
@click.option('--top-k', '-k', type=int, multiple=True,
              default=[5, 10, 15], help='Top K values to test')
@click.option('--start-date', '-s', type=click.DateTime(['%Y-%m-%d']),
              default=str((datetime.now() - timedelta(days=365)).date()),
              help='Backtest start date')
@click.option('--end-date', '-e', type=click.DateTime(['%Y-%m-%d']),
              default=str((datetime.now() - timedelta(days=1)).date()),
              help='Backtest end date')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output CSV file for sweep results')
@click.option('--max-combinations', type=int, default=100,
              help='Maximum parameter combinations to test')
def sweep(
    profit_target: List[float],
    stop_loss: List[float],
    top_k: List[int],
    start_date: datetime,
    end_date: datetime,
    output: str,
    max_combinations: int
) -> None:
    """Run parameter sweep optimization."""
    
    param_grid = {
        'profit_target': list(profit_target),
        'stop_loss': list(stop_loss),
        'top_k': list(top_k),
    }
    
    # Calculate total combinations
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    
    if total_combinations > max_combinations:
        click.echo(f"⚠️  Too many combinations ({total_combinations}), limiting to {max_combinations}")
        # Could implement sampling logic here
    
    click.echo(f"🔄 Running parameter sweep: {min(total_combinations, max_combinations)} combinations")
    
    try:
        # Initialize components
        data_feed = DataFeed()
        signal_generator = SignalGenerator(data_feed, UniverseBuilder(data_feed))
        backtester = Backtester(data_feed, signal_generator)
        
        # Run sweep
        results_df = backtester.run_parameter_sweep(
            start_date, end_date, param_grid
        )
        
        if results_df.empty:
            click.echo("❌ No results from parameter sweep", err=True)
            sys.exit(1)
        
        # Sort by Sharpe ratio (or other metric)
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)
        
        # Save results
        results_df.to_csv(output, index=False)
        click.echo(f"📁 Sweep results saved to {output}")
        
        # Display top results
        click.echo("\n🏆 Top 5 Parameter Combinations:")
        click.echo(results_df.head().to_string(index=False))
        
    except Exception as e:
        click.echo(f"❌ Error running parameter sweep: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--dry-run', is_flag=True, default=True,
              help='Run in dry-run mode (default: True)')
@click.option('--scan-only', is_flag=True,
              help='Run single scan only, don\'t start live trading')
@click.option('--profit-target', '-p', type=float, default=0.05,
              help='Profit target percentage')
@click.option('--stop-loss', '-l', type=float, default=0.02,
              help='Stop loss percentage')
@click.option('--top-k', '-k', type=int, default=10,
              help='Number of top gaps to trade')
def live(
    dry_run: bool,
    scan_only: bool,
    profit_target: float,
    stop_loss: float,
    top_k: int
) -> None:
    """Start live trading system."""
    
    if not config.has_alpaca_credentials and not dry_run:
        click.echo("❌ Alpaca credentials required for live trading", err=True)
        click.echo("💡 Use --dry-run flag for simulation mode")
        sys.exit(1)
    
    mode = "DRY RUN" if dry_run else "LIVE"
    click.echo(f"🚀 Starting live trading system [{mode}]")
    
    # Create parameters
    params = GapParams(
        profit_target=profit_target,
        stop_loss=stop_loss,
        top_k=top_k,
    )
    
    try:
        # Initialize trader
        trader = LiveTrader(dry_run=dry_run)
        
        if scan_only:
            # Run single scan
            click.echo("📊 Running gap scan...")
            results = trader.run_single_scan(params)
            
            if 'error' in results:
                click.echo(f"❌ Scan error: {results['error']}", err=True)
                sys.exit(1)
            
            click.echo(f"✅ Found {results['gaps_found']} gap opportunities:")
            for opp in results.get('opportunities', []):
                gap_pct = opp['gap_pct'] * 100
                click.echo(f"  {opp['rank']:2d}. {opp['symbol']:<6} {gap_pct:+6.2f}% "
                          f"(${opp['previous_close']:.2f} → ${opp['current_open']:.2f})")
        
        else:
            # Start live trading
            click.echo("📈 Starting scheduled trading...")
            click.echo("⏰ Market events will be triggered automatically")
            click.echo("🛑 Press Ctrl+C to stop")
            
            trader.start_live_trading(params)
            
    except KeyboardInterrupt:
        click.echo("\n🛑 Live trading stopped by user")
    except Exception as e:
        click.echo(f"❌ Error in live trading: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--date', '-d', type=click.DateTime(['%Y-%m-%d']),
              default=str(datetime.now().date()),
              help='Date to scan for gaps')
@click.option('--min-gap', type=float, default=0.02,
              help='Minimum gap percentage')
@click.option('--max-gap', type=float, default=0.50,
              help='Maximum gap percentage')
@click.option('--top-k', '-k', type=int, default=20,
              help='Number of top gaps to show')
@click.option('--output', '-o', type=click.Path(),
              help='Save results to CSV file')
def scan(
    date: datetime,
    min_gap: float,
    max_gap: float,
    top_k: int,
    output: Optional[str]
) -> None:
    """Scan for gap opportunities on a specific date."""
    
    click.echo(f"📊 Scanning for gaps on {date.date()}")
    
    try:
        # Initialize components
        data_feed = DataFeed()
        universe_builder = UniverseBuilder(data_feed)
        signal_generator = SignalGenerator(data_feed, universe_builder)
        
        # Calculate gaps
        with click.progressbar(length=1, label='Calculating gaps') as bar:
            gaps_df = signal_generator.calculate_gaps(
                date,
                min_gap_pct=min_gap,
                max_gap_pct=max_gap
            )
            bar.update(1)
        
        if gaps_df.empty:
            click.echo("❌ No gaps found matching criteria")
            return
        
        # Rank gaps
        top_gaps = signal_generator.rank_gaps(gaps_df, top_k=top_k)
        
        # Display results
        click.echo(f"\n✅ Found {len(top_gaps)} gap opportunities:")
        click.echo("-" * 80)
        click.echo(f"{'Rank':<4} {'Symbol':<8} {'Gap %':<8} {'Prev Close':<12} {'Open':<12} {'Sector':<15}")
        click.echo("-" * 80)
        
        for _, gap in top_gaps.iterrows():
            click.echo(f"{gap['rank']:<4} {gap['symbol']:<8} "
                      f"{gap['gap_pct']*100:+6.2f}% "
                      f"${gap['previous_close']:<10.2f} "
                      f"${gap['current_open']:<10.2f} "
                      f"{gap.get('sector', 'Unknown'):<15}")
        
        # Save to file if requested
        if output:
            top_gaps.to_csv(output, index=False)
            click.echo(f"\n📁 Results saved to {output}")
            
    except Exception as e:
        click.echo(f"❌ Error scanning for gaps: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--symbols', '-s', multiple=True, required=True,
              help='Symbols to download (can specify multiple)')
@click.option('--start-date', type=click.DateTime(['%Y-%m-%d']),
              default=str((datetime.now() - timedelta(days=30)).date()),
              help='Start date for data download')
@click.option('--end-date', type=click.DateTime(['%Y-%m-%d']),
              default=str(datetime.now().date()),
              help='End date for data download')
@click.option('--interval', type=click.Choice(['1d', '1h', '5m', '1m']),
              default='1d', help='Data interval')
@click.option('--source', type=click.Choice(['auto', 'yfinance', 'iex', 'polygon']),
              default='auto', help='Data source')
@click.option('--output-dir', '-o', type=click.Path(),
              help='Output directory for data files')
def download(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    interval: str,
    source: str,
    output_dir: Optional[str]
) -> None:
    """Download historical data for specified symbols."""
    
    click.echo(f"📥 Downloading {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
    
    try:
        # Initialize data feed
        data_feed = DataFeed()
        
        # Download data
        with click.progressbar(symbols, label='Downloading') as symbol_list:
            all_data = {}
            for symbol in symbol_list:
                data = data_feed.download(
                    [symbol],
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    source=source
                )
                all_data.update(data)
        
        # Display summary
        click.echo(f"\n✅ Downloaded data for {len(all_data)} symbols:")
        for symbol, df in all_data.items():
            click.echo(f"  {symbol}: {len(df)} bars")
        
        # Save to files if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for symbol, df in all_data.items():
                filename = f"{symbol}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                filepath = output_path / filename
                df.to_csv(filepath)
                
            click.echo(f"📁 Data saved to {output_dir}")
            
    except Exception as e:
        click.echo(f"❌ Error downloading data: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--clear-cache', is_flag=True, help='Clear data cache')
@click.option('--check-config', is_flag=True, help='Check configuration')
@click.option('--test-connection', is_flag=True, help='Test API connections')
def status(clear_cache: bool, check_config: bool, test_connection: bool) -> None:
    """Check system status and perform maintenance tasks."""
    
    if clear_cache:
        try:
            data_feed = DataFeed()
            count = data_feed.clear_cache()
            click.echo(f"🧹 Cleared {count} cache files")
        except Exception as e:
            click.echo(f"❌ Error clearing cache: {e}", err=True)
    
    if check_config:
        click.echo("⚙️  Configuration Status:")
        click.echo(f"  Data path: {config.data_path}")
        click.echo(f"  Alpaca credentials: {'✅' if config.has_alpaca_credentials else '❌'}")
        click.echo(f"  Premium feeds: {'✅' if config.has_premium_feeds else '❌'}")
        click.echo(f"  Log level: {config.log_level}")
    
    if test_connection:
        click.echo("🔌 Testing connections...")
        
        # Test yfinance
        try:
            data_feed = DataFeed()
            test_data = data_feed.download(['AAPL'], 
                                         start=datetime.now() - timedelta(days=5),
                                         end=datetime.now(),
                                         interval='1d')
            if 'AAPL' in test_data and not test_data['AAPL'].empty:
                click.echo("  yfinance: ✅")
            else:
                click.echo("  yfinance: ❌")
        except Exception:
            click.echo("  yfinance: ❌")
        
        # Test Alpaca if configured
        if config.has_alpaca_credentials:
            try:
                trader = LiveTrader(dry_run=True)
                status = trader.get_portfolio_status()
                if status:
                    click.echo("  Alpaca: ✅")
                else:
                    click.echo("  Alpaca: ❌")
            except Exception:
                click.echo("  Alpaca: ❌")
        else:
            click.echo("  Alpaca: ⚠️  (not configured)")


def display_backtest_summary(analysis: Dict) -> None:
    """Display backtest summary to console."""
    trade_data = analysis.get('trade_analysis', {})
    perf_data = analysis.get('performance_metrics', {})
    
    click.echo("\n" + "="*60)
    click.echo("📊 BACKTEST SUMMARY")
    click.echo("="*60)
    
    # Key metrics
    click.echo(f"Total Trades:     {trade_data.get('total_trades', 0):,}")
    click.echo(f"Win Rate:         {trade_data.get('win_rate', 0)*100:.1f}%")
    click.echo(f"Total Return:     {perf_data.get('total_return_pct', 0):+.2f}%")
    click.echo(f"CAGR:             {perf_data.get('cagr_pct', 0):+.2f}%")
    click.echo(f"Sharpe Ratio:     {perf_data.get('sharpe_ratio', 0):.2f}")
    click.echo(f"Max Drawdown:     {perf_data.get('max_drawdown_pct', 0):.2f}%")
    click.echo(f"Profit Factor:    {trade_data.get('profit_factor', 0):.2f}")
    
    # Trade statistics
    click.echo(f"\nAverage Return:   {trade_data.get('avg_return_pct', 0):+.2f}%")
    click.echo(f"Average Winner:   {trade_data.get('avg_winner_pct', 0):+.2f}%")
    click.echo(f"Average Loser:    {trade_data.get('avg_loser_pct', 0):+.2f}%")
    click.echo(f"Total P&L:        ${trade_data.get('total_pnl', 0):,.2f}")
    
    click.echo("="*60)


def save_backtest_results(results: Dict, output_path: str) -> None:
    """Save backtest results to CSV file."""
    trades = results.get('trades', [])
    
    if not trades:
        return
    
    # Convert trades to DataFrame
    trade_data = []
    for trade in trades:
        trade_data.append({
            'symbol': trade.symbol,
            'entry_date': trade.entry_date.isoformat(),
            'exit_date': trade.exit_date.isoformat(),
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'position_size': trade.position_size,
            'pnl_gross': trade.pnl_gross,
            'pnl_net': trade.pnl_net,
            'return_pct': trade.return_pct,
            'hold_time_hours': trade.hold_time_hours,
            'exit_reason': trade.exit_reason,
            'gap_pct': trade.gap_pct,
            'rank': trade.rank,
        })
    
    df = pd.DataFrame(trade_data)
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    cli()