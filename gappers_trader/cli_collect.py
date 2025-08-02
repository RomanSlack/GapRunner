#!/usr/bin/env python3
"""CLI interface for gap trading data collection."""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.panel import Panel

from gappers.config_new import Config
from gappers.data_collector import DataCollector

console = Console()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Gap Trading System - Data Collection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect last 30 days of data
  python cli_collect.py --days 30
  
  # Collect specific date range
  python cli_collect.py --start 2024-01-01 --end 2024-01-31
  
  # Collect with custom configuration
  python cli_collect.py --config custom_config.yaml --days 7
        """
    )
    
    # Date range options
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        "--days",
        type=int,
        help="Number of days to collect (from today backwards)"
    )
    date_group.add_argument(
        "--start-end",
        nargs=2,
        metavar=("START", "END"),
        help="Start and end dates (YYYY-MM-DD format)"
    )
    
    # Configuration options
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (default: auto-detect)"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate data integrity after collection"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-collection of existing data"
    )
    
    args = parser.parse_args()
    
    # Show banner
    console.print(Panel.fit(
        "[bold blue]Gap Trading System - Data Collection[/bold blue]\n"
        "[dim]Production-grade data collection with progress tracking[/dim]",
        border_style="blue"
    ))
    
    try:
        # Load configuration
        config = Config.load(args.config)
        console.print(f"[green]✓[/green] Configuration loaded")
        
        # Validate configuration
        validation_issues = config.validate()
        if validation_issues:
            console.print("[red]Configuration validation failed:[/red]")
            for issue in validation_issues:
                console.print(f"  • {issue}")
            sys.exit(1)
        
        # Setup logging
        config.setup_logging()
        
        # Initialize data collector
        data_collector = DataCollector(config)
        console.print(f"[green]✓[/green] Data collector initialized")
        
        # Determine date range
        if args.days:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=args.days)
        else:
            start_date = datetime.strptime(args.start_end[0], '%Y-%m-%d')
            end_date = datetime.strptime(args.start_end[1], '%Y-%m-%d')
        
        console.print(f"[blue]📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}[/blue]")
        
        # Collect data
        console.print("\n[bold]Starting data collection...[/bold]")
        success = data_collector.collect_full_dataset(start_date, end_date)
        
        if success:
            console.print(Panel.fit(
                "[bold green]✅ Data collection completed successfully![/bold green]",
                border_style="green"
            ))
            
            # Validate data if requested
            if args.validate:
                console.print("\n[bold]Validating data integrity...[/bold]")
                
                from gappers.data_manager import DataManager
                data_manager = DataManager(config)
                
                validation_results = data_manager.validate_data_integrity()
                
                if validation_results['validation_passed']:
                    console.print("[green]✅ Data validation passed![/green]")
                else:
                    console.print("[red]❌ Data validation found issues:[/red]")
                    
                    if validation_results['corrupted_files']:
                        console.print("Corrupted files:")
                        for file in validation_results['corrupted_files'][:10]:  # Show first 10
                            console.print(f"  • {file}")
                    
                    if validation_results['inconsistent_files']:
                        console.print("Inconsistent files:")
                        for file in validation_results['inconsistent_files'][:10]:  # Show first 10
                            console.print(f"  • {file}")
                
                console.print(f"Total files checked: {validation_results['total_files_checked']}")
            
        else:
            console.print(Panel.fit(
                "[bold red]❌ Data collection failed![/bold red]\n"
                "Check the logs for detailed error information.",
                border_style="red"
            ))
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Data collection interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()