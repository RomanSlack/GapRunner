#!/usr/bin/env python3
"""CLI interface for gap analysis."""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gappers.config_new import Config
from gappers.gap_engine import GapEngine
from gappers.data_manager import DataManager

console = Console()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Gap Trading System - Gap Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze gaps for yesterday
  python cli_gaps.py --yesterday
  
  # Analyze gaps for specific date
  python cli_gaps.py --date 2024-01-15
  
  # Analyze only up gaps, top 5
  python cli_gaps.py --date 2024-01-15 --direction up --limit 5
  
  # Export results to CSV
  python cli_gaps.py --date 2024-01-15 --export gaps_20240115.csv
        """
    )
    
    # Date options
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        "--date",
        type=str,
        help="Specific date to analyze (YYYY-MM-DD format)"
    )
    date_group.add_argument(
        "--yesterday",
        action="store_true",
        help="Analyze gaps for yesterday"
    )
    
    # Analysis options
    parser.add_argument(
        "--direction",
        choices=["up", "down", "both"],
        default="both",
        help="Gap direction to analyze (default: both)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of gaps to show (default: 20)"
    )
    
    # Output options
    parser.add_argument(
        "--export",
        type=str,
        help="Export results to CSV file"
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary statistics only"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Show banner
    console.print(Panel.fit(
        "[bold blue]Gap Trading System - Gap Analysis[/bold blue]\n"
        "[dim]Real-time gap detection and ranking[/dim]",
        border_style="blue"
    ))
    
    try:
        # Load configuration
        config = Config.load(args.config)
        console.print(f"[green]✓[/green] Configuration loaded")
        
        # Initialize components
        data_manager = DataManager(config)
        gap_engine = GapEngine(config)
        console.print(f"[green]✓[/green] Gap engine initialized")
        
        # Determine analysis date
        if args.yesterday:
            analysis_date = datetime.now() - timedelta(days=1)
        else:
            analysis_date = datetime.strptime(args.date, '%Y-%m-%d')
        
        console.print(f"[blue]📅 Analysis date: {analysis_date.strftime('%Y-%m-%d')}[/blue]")
        
        # Analyze gaps
        console.print("\n[bold]Analyzing gaps...[/bold]")
        gaps_df = gap_engine.get_top_gaps(
            analysis_date,
            direction=args.direction,
            limit=args.limit
        )
        
        if gaps_df.empty:
            console.print("[yellow]⚠️  No gaps found for the specified criteria[/yellow]")
            sys.exit(0)
        
        # Display results
        if args.summary:
            display_summary_only(gaps_df, args.direction)
        else:
            display_full_results(gaps_df, analysis_date, args.direction)
        
        # Export if requested
        if args.export:
            try:
                gaps_df.to_csv(args.export, index=False)
                console.print(f"[green]✓[/green] Results exported to {args.export}")
            except Exception as e:
                console.print(f"[red]❌ Export failed: {e}[/red]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Analysis interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


def display_summary_only(gaps_df, direction):
    """Display summary statistics only."""
    console.print("\n[bold]📊 Gap Summary[/bold]")
    
    # Create summary table
    table = Table(title="Gap Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    total_gaps = len(gaps_df)
    up_gaps = len(gaps_df[gaps_df['gap_pct'] > 0])
    down_gaps = len(gaps_df[gaps_df['gap_pct'] < 0])
    
    table.add_row("Total Gaps", str(total_gaps))
    table.add_row("Up Gaps", str(up_gaps))
    table.add_row("Down Gaps", str(down_gaps))
    
    if total_gaps > 0:
        avg_gap = gaps_df['gap_pct'].abs().mean()
        max_gap = gaps_df['gap_pct'].abs().max()
        min_gap = gaps_df['gap_pct'].abs().min()
        
        table.add_row("Average Gap", f"{avg_gap:.2%}")
        table.add_row("Largest Gap", f"{max_gap:.2%}")
        table.add_row("Smallest Gap", f"{min_gap:.2%}")
        
        if up_gaps > 0:
            up_follow_through = 0
            if 'intraday_pct' in gaps_df.columns:
                up_gaps_df = gaps_df[gaps_df['gap_pct'] > 0]
                up_follow_through = (up_gaps_df['intraday_pct'] > 0).mean()
            table.add_row("Up Gap Follow-Through", f"{up_follow_through:.1%}")
    
    console.print(table)


def display_full_results(gaps_df, analysis_date, direction):
    """Display full gap analysis results."""
    # Summary metrics
    display_summary_only(gaps_df, direction)
    
    # Detailed gap table
    console.print(f"\n[bold]🏆 Top Gaps - {analysis_date.strftime('%Y-%m-%d')}[/bold]")
    
    # Create detailed table
    table = Table(title=f"Gap Details ({direction.title()} Direction)")
    
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Symbol", style="magenta", width=8)
    table.add_column("Gap %", style="green", width=8)
    table.add_column("Direction", style="blue", width=10)
    table.add_column("Prev Close", style="yellow", width=10)
    table.add_column("Open", style="yellow", width=10)
    table.add_column("Intraday %", style="red", width=10)
    table.add_column("Volume", style="dim", width=12)
    
    for idx, row in gaps_df.head(20).iterrows():  # Show top 20
        gap_pct = row['gap_pct'] * 100
        intraday_pct = row.get('intraday_pct', 0) * 100 if 'intraday_pct' in gaps_df.columns else 0
        
        # Format volume
        volume = row.get('current_volume', 0)
        if volume >= 1_000_000:
            volume_str = f"{volume/1_000_000:.1f}M"
        elif volume >= 1_000:
            volume_str = f"{volume/1_000:.0f}K"
        else:
            volume_str = f"{volume:.0f}"
        
        table.add_row(
            str(row.get('gap_rank', idx + 1)),
            row['symbol'],
            f"{gap_pct:+.1f}%",
            "🔺 Up" if row['gap_pct'] > 0 else "🔻 Down",
            f"${row['previous_close']:.2f}",
            f"${row['current_open']:.2f}",
            f"{intraday_pct:+.1f}%" if intraday_pct != 0 else "N/A",
            volume_str
        )
    
    console.print(table)
    
    # Additional insights
    if len(gaps_df) > 20:
        console.print(f"[dim]... and {len(gaps_df) - 20} more gaps[/dim]")
    
    # Show top performers if intraday data available
    if 'intraday_pct' in gaps_df.columns:
        best_performers = gaps_df.nlargest(5, 'intraday_pct')
        if not best_performers.empty:
            console.print("\n[bold]🚀 Best Intraday Performers[/bold]")
            
            perf_table = Table()
            perf_table.add_column("Symbol", style="magenta")
            perf_table.add_column("Gap %", style="green")
            perf_table.add_column("Intraday %", style="red")
            perf_table.add_column("Total %", style="blue")
            
            for _, row in best_performers.iterrows():
                gap_pct = row['gap_pct'] * 100
                intraday_pct = row['intraday_pct'] * 100
                total_pct = gap_pct + intraday_pct
                
                perf_table.add_row(
                    row['symbol'],
                    f"{gap_pct:+.1f}%",
                    f"{intraday_pct:+.1f}%",
                    f"{total_pct:+.1f}%"
                )
            
            console.print(perf_table)


if __name__ == "__main__":
    main()