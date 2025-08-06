"""Production-grade Streamlit dashboard for gap trading system."""

import logging
import sys
from datetime import datetime, timedelta, date as date_type
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gappers.config_new import Config, get_config
from gappers.data_collector import DataCollector
from gappers.data_manager import DataManager
from gappers.gap_engine import GapEngine
from gappers.universe import UniverseBuilder
from gappers.simulation_manager import SimulationManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

# Page configuration
st.set_page_config(
    page_title="Gap Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin-bottom: 1rem;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system_components():
    """Load and cache system components."""
    try:
        config = get_config()
        data_collector = DataCollector(config)
        data_manager = DataManager(config)
        gap_engine = GapEngine(config)
        universe_builder = UniverseBuilder(config)
        
        return config, data_collector, data_manager, gap_engine, universe_builder
    except Exception as e:
        st.error(f"Failed to initialize system components: {e}")
        st.stop()


def main():
    """Main dashboard application."""
    st.title("🎯 Production Gap Trading System")
    st.markdown("**Real-time gap detection, backtesting, and portfolio management**")
    
    # Load system components
    config, data_collector, data_manager, gap_engine, universe_builder = load_system_components()
    
    # Sidebar navigation
    with st.sidebar:
        st.header("📊 Navigation")
        
        page = st.selectbox(
            "Select Page",
            [
                "🏠 Dashboard",
                "📊 Data Collection", 
                "🔍 Gap Analysis",
                "💼 Portfolio Simulation",
                "🔖 Paper Trading",
                "💾 Simulation Manager",
                "⚙️ Configuration",
                "📈 System Status"
            ]
        )
    
    # Route to appropriate page
    if page == "🏠 Dashboard":
        show_dashboard_page(config, data_manager, gap_engine)
    elif page == "📊 Data Collection":
        show_data_collection_page(config, data_collector, data_manager)
    elif page == "🔍 Gap Analysis":
        show_gap_analysis_page(config, gap_engine, data_manager)
    elif page == "💼 Portfolio Simulation":
        show_portfolio_simulation_page(config, gap_engine, data_manager)
    elif page == "🔖 Paper Trading":
        show_paper_trading_page()
    elif page == "💾 Simulation Manager":
        show_simulation_manager_page()
    elif page == "⚙️ Configuration":
        show_configuration_page(config)
    elif page == "📈 System Status":
        show_system_status_page(config, data_manager)


def show_dashboard_page(config: Config, data_manager: DataManager, gap_engine: GapEngine):
    """Show main dashboard page."""
    st.header("📊 System Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        storage_stats = data_manager.get_storage_stats()
        st.metric(
            "Data Files",
            f"{storage_stats.get('price_data_files', 0):,}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Storage Size",
            f"{storage_stats.get('total_size_mb', 0):.1f} MB",
            delta=None
        )
    
    with col3:
        available_dates = data_manager.get_available_dates()
        st.metric(
            "Date Range",
            f"{len(available_dates)} days",
            delta=None
        )
    
    with col4:
        st.metric(
            "Universe Size",
            f"{config.data_collection.universe_size:,}",
            delta=None
        )
    
    st.markdown("---")
    
    # Quick gap analysis
    st.subheader("🔍 Recent Gap Activity")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        analysis_date = st.date_input(
            "Analysis Date",
            value=datetime.now().date() - timedelta(days=1),
            max_value=datetime.now().date()
        )
    
    if st.button("🚀 Analyze Gaps", type="primary"):
        with st.spinner("Analyzing gaps..."):
            try:
                analysis_datetime = datetime.combine(analysis_date, datetime.min.time())
                gaps_df = gap_engine.calculate_daily_gaps(analysis_datetime)
                
                if not gaps_df.empty:
                    # Summary stats
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        total_gaps = len(gaps_df)
                        st.metric("Total Gaps", total_gaps)
                    
                    with col2:
                        up_gaps = len(gaps_df[gaps_df['gap_pct'] > 0])
                        st.metric("Up Gaps", up_gaps)
                    
                    with col3:
                        down_gaps = len(gaps_df[gaps_df['gap_pct'] < 0])
                        st.metric("Down Gaps", down_gaps)
                    
                    # Top gaps table
                    st.subheader("🏆 Top Gaps")
                    display_gaps_table(gaps_df.head(10))
                    
                else:
                    st.warning("No gaps found for the selected date.")
                    
            except Exception as e:
                st.error(f"Gap analysis failed: {e}")


def show_data_collection_page(config: Config, data_collector: DataCollector, data_manager: DataManager):
    """Show data collection page."""
    st.header("📊 Data Collection & Management")
    
    # Data collection status
    storage_stats = data_manager.get_storage_stats()
    
    # Status metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Universe Files", storage_stats.get('universe_files', 0))
    
    with col2:
        st.metric("Price Data Files", storage_stats.get('price_data_files', 0))
    
    with col3:
        st.metric("Total Size", f"{storage_stats.get('total_size_mb', 0):.1f} MB")
    
    with col4:
        if 'earliest_date' in storage_stats and 'latest_date' in storage_stats:
            date_range = f"{storage_stats['earliest_date']} to {storage_stats['latest_date']}"
        else:
            date_range = "No data"
        st.text("Date Range")
        st.text(date_range)
    
    st.markdown("---")
    
    # Data collection controls
    st.subheader("🔄 Data Collection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=30)
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now().date()
        )
    
    if st.button("🚀 Collect Data", type="primary"):
        if start_date >= end_date:
            st.error("Start date must be before end date")
        else:
            with st.spinner("Collecting data... This may take several minutes."):
                try:
                    start_datetime = datetime.combine(start_date, datetime.min.time())
                    end_datetime = datetime.combine(end_date, datetime.min.time())
                    
                    # Create a placeholder for real-time updates
                    status_placeholder = st.empty()
                    
                    # Run data collection
                    success = data_collector.collect_full_dataset(start_datetime, end_datetime)
                    
                    if success:
                        st.success("✅ Data collection completed successfully!")
                        st.rerun()  # Refresh the page to show updated stats
                    else:
                        st.error("❌ Data collection failed. Check logs for details.")
                        
                except Exception as e:
                    st.error(f"Data collection error: {e}")
    
    # Data validation
    st.markdown("---")
    st.subheader("🔍 Data Validation")
    
    if st.button("🔍 Validate Data Integrity"):
        with st.spinner("Validating data integrity..."):
            try:
                validation_results = data_manager.validate_data_integrity()
                
                if validation_results['validation_passed']:
                    st.success("✅ Data validation passed!")
                else:
                    st.error("❌ Data validation found issues:")
                    
                    if validation_results['corrupted_files']:
                        st.write("**Corrupted files:**")
                        for file in validation_results['corrupted_files']:
                            st.write(f"- {file}")
                    
                    if validation_results['inconsistent_files']:
                        st.write("**Inconsistent files:**")
                        for file in validation_results['inconsistent_files']:
                            st.write(f"- {file}")
                
                st.write(f"**Total files checked:** {validation_results['total_files_checked']}")
                
            except Exception as e:
                st.error(f"Validation error: {e}")


def show_gap_analysis_page(config: Config, gap_engine: GapEngine, data_manager: DataManager):
    """Show gap analysis page."""
    st.header("🔍 Gap Analysis & Detection")
    
    # Analysis controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analysis_date = st.date_input(
            "Analysis Date",
            value=datetime.now().date() - timedelta(days=1),
            max_value=datetime.now().date()
        )
    
    with col2:
        gap_direction = st.selectbox(
            "Gap Direction",
            ["Both", "Up Only", "Down Only"]
        )
    
    with col3:
        max_results = st.number_input(
            "Max Results",
            min_value=5,
            max_value=100,
            value=20
        )
    
    if st.button("🔍 Analyze Gaps", type="primary"):
        with st.spinner("Analyzing gaps..."):
            try:
                analysis_datetime = datetime.combine(analysis_date, datetime.min.time())
                
                # Get gaps based on direction
                direction_map = {"Both": "both", "Up Only": "up", "Down Only": "down"}
                gaps_df = gap_engine.get_top_gaps(
                    analysis_datetime, 
                    direction=direction_map[gap_direction],
                    limit=max_results
                )
                
                if not gaps_df.empty:
                    # Display summary metrics
                    display_gap_summary_metrics(gaps_df)
                    
                    # Gap distribution chart
                    st.subheader("📊 Gap Distribution")
                    display_gap_distribution_chart(gaps_df)
                    
                    # Detailed gaps table
                    st.subheader("📋 Gap Details")
                    display_gaps_table(gaps_df)
                    
                    # Download option
                    csv = gaps_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Gap Data",
                        data=csv,
                        file_name=f"gaps_{analysis_date.strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.warning("No gaps found for the selected criteria.")
                    
            except Exception as e:
                st.error(f"Gap analysis failed: {e}")
    
    # Historical gap patterns
    st.markdown("---")
    st.subheader("🕒 Historical Gap Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pattern_start = st.date_input(
            "Pattern Analysis Start",
            value=datetime.now().date() - timedelta(days=30)
        )
    
    with col2:
        pattern_end = st.date_input(
            "Pattern Analysis End",
            value=datetime.now().date()
        )
    
    if st.button("📈 Analyze Patterns"):
        if pattern_start >= pattern_end:
            st.error("Start date must be before end date")
        else:
            with st.spinner("Analyzing historical patterns..."):
                try:
                    start_datetime = datetime.combine(pattern_start, datetime.min.time())
                    end_datetime = datetime.combine(pattern_end, datetime.min.time())
                    
                    patterns = gap_engine.analyze_gap_patterns(start_datetime, end_datetime)
                    
                    if 'error' not in patterns:
                        display_gap_patterns(patterns)
                    else:
                        st.error(patterns['error'])
                        
                except Exception as e:
                    st.error(f"Pattern analysis failed: {e}")


def show_portfolio_simulation_page(config: Config, gap_engine: GapEngine, data_manager: DataManager):
    """Show portfolio simulation page."""
    st.header("💼 Portfolio Simulation & Backtesting")
    st.markdown("**Full momentum-gap strategy implementation with risk management**")
    
    # Initialize simulation manager
    @st.cache_resource
    def get_simulation_manager():
        return SimulationManager()
    
    sim_manager = get_simulation_manager()
    
    # Import portfolio engine
    from gappers.portfolio_engine import PortfolioEngine
    
    # Simulation parameters
    with st.expander("🔧 Strategy Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Entry Rules")
            top_k = st.number_input(
                "Top K Gaps", 
                value=config.strategy.top_k, 
                min_value=1, 
                max_value=50,
                help="Number of top-ranked gap stocks to trade each day. Strategy selects the K largest overnight gaps that meet criteria."
            )
            min_gap = st.number_input(
                "Min Gap (%)", 
                value=config.strategy.min_gap_pct * 100, 
                min_value=0.1, 
                max_value=10.0,
                help="Minimum overnight gap percentage required to enter a trade. Filters out small gaps that may lack momentum."
            ) / 100
            max_gap = st.number_input(
                "Max Gap (%)", 
                value=config.strategy.max_gap_pct * 100, 
                min_value=5.0, 
                max_value=50.0,
                help="Maximum overnight gap percentage allowed. Filters out extreme gaps that may reverse or be halted."
            ) / 100
            position_size = st.number_input(
                "Position Size ($)", 
                value=config.strategy.position_size_usd, 
                min_value=100, 
                max_value=50000,
                help="Dollar amount invested in each position. Total risk = Position Size × Max Positions."
            )
        
        with col2:
            st.subheader("Exit Rules")
            profit_target = st.number_input(
                "Profit Target (%)", 
                value=config.strategy.profit_target_pct * 100, 
                min_value=1.0, 
                max_value=50.0,
                help="Target profit percentage to automatically close position. Trade exits when stock rises this % above entry price."
            ) / 100
            hard_stop = st.number_input(
                "Hard Stop Loss (%)", 
                value=config.strategy.hard_stop_pct * 100, 
                min_value=0.5, 
                max_value=20.0,
                help="Maximum loss percentage before closing position. Trade exits immediately if stock falls this % below entry price."
            ) / 100
            trailing_stop = st.number_input(
                "Trailing Stop (%)", 
                value=config.strategy.trailing_stop_pct * 100, 
                min_value=0.5, 
                max_value=10.0,
                help="Trailing stop percentage from session high. Once profitable, trade exits if stock falls this % from its intraday peak."
            ) / 100
            time_stop = st.number_input(
                "Time Stop (Hour)", 
                value=config.strategy.time_stop_hour, 
                min_value=13, 
                max_value=16,
                help="Hour (24h format) to force-close all positions. Default 15 = 3:00 PM ET to avoid overnight risk."
            )
        
        with col3:
            st.subheader("Risk Management")
            max_positions = st.number_input(
                "Max Positions", 
                value=config.strategy.max_positions, 
                min_value=1, 
                max_value=20,
                help="Maximum number of simultaneous open positions. Limits total portfolio exposure and concentration risk."
            )
            initial_capital = st.number_input(
                "Initial Capital ($)", 
                value=config.backtest.initial_capital, 
                min_value=10000, 
                max_value=1000000,
                help="Starting portfolio value for backtesting. Higher capital allows for more positions and better diversification."
            )
            sector_div = st.checkbox(
                "Sector Diversification", 
                value=config.strategy.sector_diversification,
                help="Limit positions per sector to reduce concentration risk. Prevents over-exposure to single industry moves."
            )
            commission = st.number_input(
                "Commission/Share ($)", 
                value=config.costs.commission_per_share, 
                min_value=0.0, 
                max_value=0.01, 
                format="%.4f",
                help="Brokerage commission charged per share traded. Typical range: $0.001-$0.005 per share for retail brokers."
            )
    
    # Backtest period
    st.subheader("📅 Backtest Period")
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.strptime(config.backtest.start_date, '%Y-%m-%d').date(),
            min_value=datetime(2020, 1, 1).date(),
            max_value=datetime.now().date()
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.strptime(config.backtest.end_date, '%Y-%m-%d').date(),
            min_value=start_date,
            max_value=datetime.now().date()
        )
    
    # Save/Load controls
    st.markdown("---")
    st.subheader("💾 Save/Load Simulations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**💾 Save Current Setup**")
        save_name = st.text_input("Save Name", placeholder="My Simulation")
        save_description = st.text_area("Description (Optional)", placeholder="Description of this simulation setup...")
        save_tags = st.text_input("Tags (comma-separated)", placeholder="backtesting, strategy1")
        
        if st.button("💾 Save Configuration", help="Save current parameters for later use"):
            if save_name.strip():
                # Create config dict with current parameters
                current_config = {
                    'strategy': {
                        'top_k': top_k,
                        'min_gap_pct': min_gap,
                        'max_gap_pct': max_gap,
                        'profit_target_pct': profit_target,
                        'hard_stop_pct': hard_stop,
                        'trailing_stop_pct': trailing_stop,
                        'time_stop_hour': time_stop,
                        'position_size_usd': position_size,
                        'max_positions': max_positions,
                        'sector_diversification': sector_div
                    },
                    'costs': {
                        'commission_per_share': commission
                    },
                    'backtest': {
                        'initial_capital': initial_capital,
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat()
                    }
                }
                
                # Parse tags
                tags_list = [tag.strip() for tag in save_tags.split(',') if tag.strip()] if save_tags else []
                
                # Save configuration (empty results for now)
                success = sim_manager.save_simulation(
                    save_name.strip(),
                    current_config,
                    {},  # Empty results - just saving config
                    save_description,
                    tags_list
                )
                
                if success:
                    st.success(f"✅ Configuration '{save_name}' saved successfully!")
                else:
                    st.error("❌ Failed to save configuration")
            else:
                st.error("Please enter a save name")
    
    with col2:
        st.markdown("**📂 Load Saved Simulation**")
        
        # List available simulations
        simulations = sim_manager.list_simulations()
        
        if simulations:
            # Create display options
            sim_options = {}
            for sim in simulations:
                timestamp = datetime.fromisoformat(sim['timestamp']).strftime('%Y-%m-%d %H:%M')
                display_name = f"{sim['name']} ({timestamp})"
                sim_options[display_name] = sim['filename']
            
            selected_sim = st.selectbox(
                "Select Simulation",
                options=list(sim_options.keys()),
                help="Choose a saved simulation to load"
            )
            
            if selected_sim and st.button("📂 Load Simulation"):
                filename = sim_options[selected_sim]
                loaded_save = sim_manager.load_simulation(filename)
                
                if loaded_save:
                    # Update session state with loaded config
                    st.session_state.loaded_config = loaded_save.config
                    st.session_state.loaded_results = loaded_save.results
                    st.success(f"✅ Loaded simulation: {loaded_save.name}")
                    st.rerun()
                else:
                    st.error("❌ Failed to load simulation")
        else:
            st.info("No saved simulations found")
    
    with col3:
        st.markdown("**🗂️ Manage Simulations**")
        
        if simulations:
            # Show storage stats
            stats = sim_manager.get_storage_stats()
            st.metric("Total Saves", stats['total_simulations'])
            st.metric("Storage Used", f"{stats['total_size_mb']:.1f} MB")
            
            # Delete simulation
            sim_to_delete = st.selectbox(
                "Delete Simulation",
                options=[''] + [sim['name'] for sim in simulations],
                help="Select a simulation to delete"
            )
            
            if sim_to_delete and st.button("🗑️ Delete", help="Permanently delete selected simulation"):
                # Find the filename for the selected simulation
                filename_to_delete = None
                for sim in simulations:
                    if sim['name'] == sim_to_delete:
                        filename_to_delete = sim['filename']
                        break
                
                if filename_to_delete and sim_manager.delete_simulation(filename_to_delete):
                    st.success(f"✅ Deleted simulation: {sim_to_delete}")
                    st.rerun()
                else:
                    st.error("❌ Failed to delete simulation")
    
    # Load saved configuration if available
    if hasattr(st.session_state, 'loaded_config') and st.session_state.loaded_config:
        loaded_config = st.session_state.loaded_config
        
        st.info("🔄 Configuration loaded from saved simulation. Parameters updated above.")
        
        # Update parameters from loaded config
        if 'strategy' in loaded_config:
            strategy = loaded_config['strategy']
            top_k = strategy.get('top_k', top_k)
            min_gap = strategy.get('min_gap_pct', min_gap)
            max_gap = strategy.get('max_gap_pct', max_gap)
            profit_target = strategy.get('profit_target_pct', profit_target)
            hard_stop = strategy.get('hard_stop_pct', hard_stop)
            trailing_stop = strategy.get('trailing_stop_pct', trailing_stop)
            time_stop = strategy.get('time_stop_hour', time_stop)
            position_size = strategy.get('position_size_usd', position_size)
            max_positions = strategy.get('max_positions', max_positions)
            sector_div = strategy.get('sector_diversification', sector_div)
        
        if 'costs' in loaded_config:
            commission = loaded_config['costs'].get('commission_per_share', commission)
        
        if 'backtest' in loaded_config:
            backtest_config = loaded_config['backtest']
            initial_capital = backtest_config.get('initial_capital', initial_capital)
            if 'start_date' in backtest_config:
                start_date = datetime.fromisoformat(backtest_config['start_date']).date()
            if 'end_date' in backtest_config:
                end_date = datetime.fromisoformat(backtest_config['end_date']).date()
        
        # Clear the loaded config to prevent reloading
        del st.session_state.loaded_config
    
    st.markdown("---")
    
    # Run backtest button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        run_backtest = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
    
    with col2:
        save_after_run = st.checkbox("💾 Save Results", help="Automatically save results after successful backtest")
    
    if run_backtest:
        if start_date >= end_date:
            st.error("Start date must be before end date")
        else:
            # Update config with user parameters
            config.strategy.top_k = top_k
            config.strategy.min_gap_pct = min_gap
            config.strategy.max_gap_pct = max_gap
            config.strategy.profit_target_pct = profit_target
            config.strategy.hard_stop_pct = hard_stop
            config.strategy.trailing_stop_pct = trailing_stop
            config.strategy.time_stop_hour = time_stop
            config.strategy.position_size_usd = position_size
            config.strategy.max_positions = max_positions
            config.strategy.sector_diversification = sector_div
            config.costs.commission_per_share = commission
            config.backtest.initial_capital = initial_capital
            
            with st.spinner("Running backtest... This may take a few minutes."):
                try:
                    # Initialize portfolio engine
                    portfolio_engine = PortfolioEngine(config)
                    
                    # Run backtest
                    results = portfolio_engine.run_backtest(
                        datetime.combine(start_date, datetime.min.time()),
                        datetime.combine(end_date, datetime.min.time())
                    )
                    
                    # Store results in session state
                    st.session_state.backtest_results = results
                    
                    # Auto-save results if requested
                    if save_after_run:
                        auto_save_name = f"Backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        current_config = {
                            'strategy': {
                                'top_k': top_k,
                                'min_gap_pct': min_gap,
                                'max_gap_pct': max_gap,
                                'profit_target_pct': profit_target,
                                'hard_stop_pct': hard_stop,
                                'trailing_stop_pct': trailing_stop,
                                'time_stop_hour': time_stop,
                                'position_size_usd': position_size,
                                'max_positions': max_positions,
                                'sector_diversification': sector_div
                            },
                            'costs': {
                                'commission_per_share': commission
                            },
                            'backtest': {
                                'initial_capital': initial_capital,
                                'start_date': start_date.isoformat(),
                                'end_date': end_date.isoformat()
                            }
                        }
                        
                        save_success = sim_manager.save_simulation(
                            auto_save_name,
                            current_config,
                            results,
                            f"Automatic save of backtest results from {start_date} to {end_date}",
                            ['auto-save', 'backtest']
                        )
                        
                        if save_success:
                            st.success(f"✅ Backtest completed and saved as '{auto_save_name}'!")
                        else:
                            st.success("✅ Backtest completed successfully!")
                            st.warning("⚠️ Auto-save failed, but results are available below")
                    else:
                        st.success("✅ Backtest completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Backtest failed: {e}")
                    logger.error(f"Backtest error: {e}", exc_info=True)
    
    # Display results if available
    if hasattr(st.session_state, 'backtest_results') and st.session_state.backtest_results:
        display_backtest_results(st.session_state.backtest_results)


def show_paper_trading_page():
    """Show paper trading page with strategy-based trading using saved configurations."""
    st.header("🔖 Paper Trading")
    st.markdown("**Execute gap trading strategies in real-time with paper money**")
    
    try:
        import alpaca_trade_api as tradeapi
        import os
        from dotenv import load_dotenv
        from datetime import datetime, timedelta
        
        # Load environment variables
        load_dotenv()
    except ImportError as e:
        missing_lib = "alpaca-trade-api" if "alpaca" in str(e) else "python-dotenv"
        st.error(f"❌ {missing_lib} library not installed. Please run: pip install {missing_lib}")
        return
    
    # Get credentials from environment variables
    api_key = os.getenv('ALPACA_API_KEY', '')
    api_secret = os.getenv('ALPACA_API_SECRET', '')
    base_url = "https://paper-api.alpaca.markets"
    
    if not api_key or not api_secret:
        st.error("⚠️ Alpaca API credentials not found in .env file")
        st.info("💡 Please add ALPACA_API_KEY and ALPACA_API_SECRET to your .env file")
        return
    
    # Initialize components
    try:
        api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        
        # Initialize simulation manager to load saved configs
        from gappers.simulation_manager import SimulationManager
        sim_manager = SimulationManager()
        
        # Load system components for gap detection
        config, data_collector, data_manager, gap_engine, universe_builder = load_system_components()
        
    except Exception as e:
        st.error(f"❌ Failed to initialize components: {e}")
        return
    
    # Paper Trading Configuration Section
    st.subheader("⚙️ Paper Trading Setup")
    
    tabs = st.tabs(["📋 Configuration", "🔍 Gap Scanner", "🚀 Strategy Execution"])
    
    with tabs[0]:  # Configuration Only
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**📂 Load Saved Configuration**")
            
            # Load saved configurations
            simulations = sim_manager.list_simulations()
            
            if simulations:
                # Create display options for saved configs
                sim_options = {"Select Configuration": None}
                for sim in simulations:
                    timestamp = datetime.fromisoformat(sim['timestamp']).strftime('%Y-%m-%d %H:%M')
                    display_name = f"{sim['name']} ({timestamp})"
                    sim_options[display_name] = sim['filename']
                
                selected_config = st.selectbox(
                    "Choose Configuration",
                    options=list(sim_options.keys()),
                    help="Select a saved configuration to load"
                )
                
                if st.button("🔄 Refresh Configs"):
                    st.rerun()
                
                # Load selected configuration
                if selected_config and selected_config != "Select Configuration":
                    filename = sim_options[selected_config]
                    loaded_save = sim_manager.load_simulation(filename)
                    
                    if loaded_save and loaded_save.config:
                        st.session_state.paper_trading_config = loaded_save.config
                        st.success(f"✅ Loaded: {loaded_save.name}")
                    else:
                        st.error("❌ Failed to load configuration")
            else:
                st.info("💡 No saved configurations found.")
        
        with col2:
            st.markdown("**⚙️ Current Configuration Status**")
            
            current_config = st.session_state.get('paper_trading_config')
            if current_config:
                strategy_config = current_config.get('strategy', {})
                timing_config = current_config.get('paper_trading', {})
                costs_config = current_config.get('costs', {})
                backtest_config = current_config.get('backtest', {})
                
                st.success("✅ Configuration Loaded")
                
                # Create expandable detailed view
                with st.expander("📋 Configuration Details", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Strategy**")
                        st.metric("Gap Range", f"{strategy_config.get('min_gap_pct', 0)*100:.1f}% - {strategy_config.get('max_gap_pct', 0)*100:.1f}%")
                        st.metric("Top K Gaps", strategy_config.get('top_k', 0))
                        st.metric("Position Size", f"${strategy_config.get('position_size_usd', 0):,}")
                        st.metric("Max Positions", strategy_config.get('max_positions', 0))
                        
                        st.markdown("**Exit Rules**")
                        st.metric("Profit Target", f"{strategy_config.get('profit_target_pct', 0)*100:.1f}%")
                        st.metric("Hard Stop", f"{strategy_config.get('hard_stop_pct', 0)*100:.1f}%")
                        st.metric("Trailing Stop", f"{strategy_config.get('trailing_stop_pct', 0)*100:.1f}%")
                        st.metric("Time Stop", f"{strategy_config.get('time_stop_hour', 15)}:00")
                    
                    with col2:
                        st.markdown("**Trading Costs**")
                        st.metric("Commission/Share", f"${costs_config.get('commission_per_share', 0):.4f}")
                        st.metric("Slippage", f"{costs_config.get('slippage_bps', 0)} bps")
                        st.metric("Borrowing Rate", f"{costs_config.get('borrowing_rate', 0)*100:.2f}%")
                        st.metric("Spread Cost", f"{costs_config.get('spread_cost_bps', 0)} bps")
                        
                        st.markdown("**Timing**")
                        st.metric("Trading Hours", f"{timing_config.get('trading_start', 'Not set')} - {timing_config.get('trading_end', 'Not set')}")
                        st.metric("Gap Scan Time", timing_config.get('gap_scan_time', 'Not set'))
                        st.metric("Auto Scan", "✅" if timing_config.get('auto_scan_enabled', False) else "❌")
                        st.metric("Auto Trade", "✅" if timing_config.get('auto_trade_enabled', False) else "❌")
                
                # Summary metrics always visible
                st.metric("Gap Range", f"{strategy_config.get('min_gap_pct', 0)*100:.1f}% - {strategy_config.get('max_gap_pct', 0)*100:.1f}%")
                st.metric("Position Size", f"${strategy_config.get('position_size_usd', 0):,}")
                st.metric("Trading Hours", f"{timing_config.get('trading_start', 'Not set')} - {timing_config.get('trading_end', 'Not set')}")
            else:
                st.warning("⚠️ No Configuration Loaded")
                st.info("Load a saved configuration or create one below")
        
        # Create new configuration section
        st.markdown("---")
        st.markdown("**🔧 Create New Configuration**")
        
        with st.expander("Create New Paper Trading Configuration", expanded=True):
            # Load existing config if available
            existing_config = st.session_state.get('paper_trading_config', {})
            strategy_config = existing_config.get('strategy', {})
            timing_config = existing_config.get('paper_trading', {})
            costs_config = existing_config.get('costs', {})
            backtest_config = existing_config.get('backtest', {})
            
            # Strategy Parameters - COMPLETE from Portfolio Simulation
            st.subheader("🎯 Strategy Parameters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Entry Rules**")
                top_k = st.number_input(
                    "Top K Gaps", 
                    value=strategy_config.get('top_k', 5), 
                    min_value=1, max_value=50,
                    help="Number of top-ranked gap stocks to trade each day"
                )
                min_gap = st.number_input(
                    "Min Gap (%)", 
                    value=strategy_config.get('min_gap_pct', 0.02) * 100, 
                    min_value=0.1, max_value=10.0,
                    help="Minimum overnight gap percentage required to enter a trade"
                ) / 100
                max_gap = st.number_input(
                    "Max Gap (%)", 
                    value=strategy_config.get('max_gap_pct', 0.15) * 100, 
                    min_value=5.0, max_value=50.0,
                    help="Maximum overnight gap percentage allowed"
                ) / 100
                position_size = st.number_input(
                    "Position Size ($)", 
                    value=strategy_config.get('position_size_usd', 1000), 
                    min_value=100, max_value=50000,
                    help="Dollar amount invested in each position"
                )
            
            with col2:
                st.markdown("**Exit Rules**")
                profit_target = st.number_input(
                    "Profit Target (%)", 
                    value=strategy_config.get('profit_target_pct', 0.08) * 100, 
                    min_value=1.0, max_value=50.0,
                    help="Target profit percentage to automatically close position"
                ) / 100
                hard_stop = st.number_input(
                    "Hard Stop Loss (%)", 
                    value=strategy_config.get('hard_stop_pct', 0.05) * 100, 
                    min_value=0.5, max_value=20.0,
                    help="Maximum loss percentage before closing position"
                ) / 100
                trailing_stop = st.number_input(
                    "Trailing Stop (%)", 
                    value=strategy_config.get('trailing_stop_pct', 0.03) * 100, 
                    min_value=0.5, max_value=10.0,
                    help="Trailing stop percentage from session high"
                ) / 100
                time_stop = st.number_input(
                    "Time Stop (Hour)", 
                    value=strategy_config.get('time_stop_hour', 15), 
                    min_value=13, max_value=16,
                    help="Hour (24h format) to force-close all positions"
                )
            
            with col3:
                st.markdown("**Risk Management**")
                max_positions = st.number_input(
                    "Max Positions", 
                    value=strategy_config.get('max_positions', 5), 
                    min_value=1, max_value=20,
                    help="Maximum number of simultaneous open positions"
                )
                initial_capital = st.number_input(
                    "Initial Capital ($)", 
                    value=backtest_config.get('initial_capital', 100000), 
                    min_value=10000, max_value=1000000,
                    help="Starting portfolio value for tracking purposes"
                )
                sector_div = st.checkbox(
                    "Sector Diversification", 
                    value=strategy_config.get('sector_diversification', False),
                    help="Limit positions per sector to reduce concentration risk"
                )
                commission = st.number_input(
                    "Commission/Share ($)", 
                    value=costs_config.get('commission_per_share', 0.001), 
                    min_value=0.0, max_value=0.01, 
                    format="%.4f",
                    help="Brokerage commission charged per share traded"
                )
            
            # Costs Configuration - COMPLETE from Portfolio Simulation
            st.subheader("💰 Trading Costs")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                slippage_bps = st.number_input(
                    "Slippage (bps)", 
                    value=costs_config.get('slippage_bps', 5), 
                    min_value=0, max_value=50,
                    help="Expected slippage in basis points (1 bp = 0.01%)"
                )
            
            with col2:
                borrowing_rate = st.number_input(
                    "Borrowing Rate (%)", 
                    value=costs_config.get('borrowing_rate', 0.02) * 100, 
                    min_value=0.0, max_value=10.0,
                    help="Annual borrowing rate for short positions"
                ) / 100
            
            with col3:
                spread_cost = st.number_input(
                    "Spread Cost (bps)", 
                    value=costs_config.get('spread_cost_bps', 10), 
                    min_value=0, max_value=100,
                    help="Expected bid-ask spread cost in basis points"
                )
            
            # Paper Trading Specific Configuration
            st.subheader("⏰ Paper Trading Timing")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Trading Hours**")
                trading_start = st.time_input(
                    "Trading Start Time",
                    value=datetime.strptime(timing_config.get('trading_start', '09:35'), '%H:%M').time(),
                    help="Earliest time to place trades"
                )
                
                trading_end = st.time_input(
                    "Trading End Time", 
                    value=datetime.strptime(timing_config.get('trading_end', '15:30'), '%H:%M').time(),
                    help="Latest time to place new trades"
                )
            
            with col2:
                st.markdown("**Automation Settings**")
                gap_scan_time = st.time_input(
                    "Daily Gap Scan Time",
                    value=datetime.strptime(timing_config.get('gap_scan_time', '09:35'), '%H:%M').time(),
                    help="Time to automatically scan for gap opportunities"
                )
                
                auto_scan_enabled = st.checkbox(
                    "Enable Automatic Gap Scanning",
                    value=timing_config.get('auto_scan_enabled', False),
                    help="Automatically scan for gaps at specified time"
                )
                
                auto_trade_enabled = st.checkbox(
                    "Enable Automatic Trading",
                    value=timing_config.get('auto_trade_enabled', False),
                    help="Automatically execute trades on qualifying gaps"
                )
            
            # Save Configuration
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                config_name = st.text_input("Configuration Name", placeholder="My Paper Trading Config")
            
            with col2:
                config_description = st.text_area("Description (Optional)", placeholder="Strategy description...")
            
            with col3:
                config_tags = st.text_input("Tags (comma-separated)", placeholder="paper-trading, gaps")
                
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("💾 Save Configuration", type="primary"):
                    if config_name.strip():
                        # Create COMPLETE configuration dictionary
                        paper_config = {
                            'strategy': {
                                'top_k': top_k,
                                'min_gap_pct': min_gap,
                                'max_gap_pct': max_gap,
                                'profit_target_pct': profit_target,
                                'hard_stop_pct': hard_stop,
                                'trailing_stop_pct': trailing_stop,
                                'time_stop_hour': time_stop,
                                'position_size_usd': position_size,
                                'max_positions': max_positions,
                                'sector_diversification': sector_div
                            },
                            'costs': {
                                'commission_per_share': commission,
                                'slippage_bps': slippage_bps,
                                'borrowing_rate': borrowing_rate,
                                'spread_cost_bps': spread_cost
                            },
                            'backtest': {
                                'initial_capital': initial_capital
                            },
                            'paper_trading': {
                                'trading_start': trading_start.strftime('%H:%M'),
                                'trading_end': trading_end.strftime('%H:%M'),
                                'gap_scan_time': gap_scan_time.strftime('%H:%M'),
                                'auto_scan_enabled': auto_scan_enabled,
                                'auto_trade_enabled': auto_trade_enabled
                            },
                            'config_type': 'paper_trading'
                        }
                        
                        # Parse tags
                        tags_list = [tag.strip() for tag in config_tags.split(',') if tag.strip()] if config_tags else ['paper-trading']
                        
                        # Save configuration
                        success = sim_manager.save_simulation(
                            config_name.strip(),
                            paper_config,
                            {},  # No results for config-only save
                            config_description.strip() if config_description.strip() else f"Paper trading configuration created on {datetime.now().strftime('%Y-%m-%d')}",
                            tags_list
                        )
                        
                        if success:
                            st.session_state.paper_trading_config = paper_config
                            st.success(f"✅ Configuration '{config_name}' saved successfully!")
                        else:
                            st.error("❌ Failed to save configuration")
                    else:
                        st.error("Please enter a configuration name")
    
    with tabs[1]:  # Gap Scanner Only
        st.markdown("**🔍 Gap Opportunity Scanner**")
        
        # Check if config is loaded
        strategy_config = st.session_state.get('paper_trading_config', {}).get('strategy', {})
        if not strategy_config:
            st.warning("⚠️ Load a configuration first to enable gap scanning")
            return
        
        # Gap scanning controls
        col1, col2 = st.columns([3, 1])
        
        with col1:
            scan_date = st.date_input(
                "Scan Date",
                value=datetime.now().date(),
                help="Date to scan for gap opportunities"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            scan_gaps = st.button("🔍 Scan for Gaps", type="primary")
        
        # Handle gap scanning
        current_gaps = None
        
        if scan_gaps:
            with st.spinner("Scanning for gap opportunities using project's gap engine..."):
                try:
                    scan_datetime = datetime.combine(scan_date, datetime.min.time())
                    gaps_df = gap_engine.calculate_daily_gaps(scan_datetime)
                    
                    if not gaps_df.empty:
                        # Filter gaps based on strategy criteria
                        min_gap = strategy_config.get('min_gap_pct', 0.02)
                        max_gap = strategy_config.get('max_gap_pct', 0.15)
                        top_k = strategy_config.get('top_k', 5)
                        
                        # Filter by gap size
                        filtered_gaps = gaps_df[
                            (abs(gaps_df['gap_pct']) >= min_gap) & 
                            (abs(gaps_df['gap_pct']) <= max_gap)
                        ]
                        
                        # Sort by absolute gap size (largest gaps first)
                        filtered_gaps = filtered_gaps.reindex(filtered_gaps['gap_pct'].abs().sort_values(ascending=False).index)
                        
                        # Take top K
                        current_gaps = filtered_gaps.head(top_k)
                        st.session_state.current_gaps = current_gaps  # Store for execution tab
                        
                        if not current_gaps.empty:
                            st.success(f"✅ Found {len(current_gaps)} qualifying gap opportunities")
                            display_gaps_table(current_gaps)
                            
                            # Show timing info
                            timing_config = st.session_state.get('paper_trading_config', {}).get('paper_trading', {})
                            if timing_config:
                                current_time = datetime.now().time()
                                trading_start = datetime.strptime(timing_config.get('trading_start', '09:35'), '%H:%M').time()
                                trading_end = datetime.strptime(timing_config.get('trading_end', '15:30'), '%H:%M').time()
                                
                                if trading_start <= current_time <= trading_end:
                                    st.success(f"🟢 Within trading hours ({timing_config.get('trading_start')} - {timing_config.get('trading_end')})")
                                else:
                                    st.warning(f"🟡 Outside trading hours ({timing_config.get('trading_start')} - {timing_config.get('trading_end')})")
                        else:
                            st.warning("No gaps found matching your strategy criteria")
                            st.info(f"📊 Strategy filters: {min_gap*100:.1f}% - {max_gap*100:.1f}% gap range, top {top_k} positions")
                    else:
                        st.warning(f"No gaps found for {scan_date.strftime('%Y-%m-%d')}")
                        st.info("💡 Try a different date or check if market data is available")
                        
                except Exception as e:
                    st.error(f"❌ Gap scanning failed: {e}")
                    st.info("💡 Make sure the gap engine has data for the selected date")
    
    with tabs[2]:  # Strategy Execution Only
        st.markdown("**🚀 Strategy Execution & Trading**")
        
        # Check if config and gaps are available
        strategy_config = st.session_state.get('paper_trading_config', {}).get('strategy', {})
        current_gaps = st.session_state.get('current_gaps')
        
        if not strategy_config:
            st.warning("⚠️ Load a configuration first")
            return
        
        if current_gaps is None or current_gaps.empty:
            st.warning("⚠️ Scan for gaps first in the Gap Scanner tab")
            return
        
        # Check trading hours
        timing_config = st.session_state.get('paper_trading_config', {}).get('paper_trading', {})
        current_time = datetime.now().time()
        within_trading_hours = True
        
        if timing_config:
            trading_start = datetime.strptime(timing_config.get('trading_start', '09:35'), '%H:%M').time()
            trading_end = datetime.strptime(timing_config.get('trading_end', '15:30'), '%H:%M').time()
            within_trading_hours = trading_start <= current_time <= trading_end
        
        # Strategy execution controls
        st.info("💡 Ready to execute strategy-based trades")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            auto_trade_symbols = st.multiselect(
                "Select Stocks to Trade",
                options=current_gaps['symbol'].tolist(),
                default=current_gaps['symbol'].tolist()[:min(3, len(current_gaps))],
                help="Choose which gap stocks to trade"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            
            button_disabled = timing_config and not within_trading_hours
            button_help = "Outside configured trading hours" if button_disabled else "Execute strategy trades"
            
            execute_strategy = st.button(
                "🎯 Execute Strategy", 
                type="primary", 
                use_container_width=True,
                disabled=button_disabled,
                help=button_help
            )
        
        # Handle strategy execution
        if execute_strategy and auto_trade_symbols:
            with st.spinner("Executing strategy trades..."):
                position_size = strategy_config.get('position_size_usd', 1000)
                max_positions = strategy_config.get('max_positions', 5)
                
                # Check current position count
                try:
                    current_positions = api.list_positions()
                    position_count = len([p for p in current_positions if float(p.qty) != 0])
                    
                    if position_count >= max_positions:
                        st.warning(f"⚠️ Maximum positions ({max_positions}) already reached. Close some positions first.")
                        return
                        
                except Exception as e:
                    st.warning(f"Could not check current positions: {e}")
                
                successful_orders = 0
                failed_orders = 0
                
                for symbol in auto_trade_symbols[:max_positions - position_count]:
                    try:
                        # Get current price
                        quote = api.get_latest_quote(symbol)
                        if quote and quote._raw:
                            ask_price = float(quote._raw.get('ap', 0))
                            bid_price = float(quote._raw.get('bp', 0))
                            current_price = ask_price if ask_price > 0 else bid_price
                            
                            if current_price > 0:
                                qty = max(1, int(position_size / current_price))
                                
                                order = api.submit_order(
                                    symbol=symbol,
                                    qty=qty,
                                    side='buy',
                                    type='market',
                                    time_in_force='day'
                                )
                                
                                st.success(f"✅ {symbol}: Bought {qty} shares @ ~${current_price:.2f} (Order: {order.id})")
                                successful_orders += 1
                            else:
                                st.error(f"❌ {symbol}: No valid price available")
                                failed_orders += 1
                                
                    except Exception as e:
                        st.error(f"❌ {symbol}: Failed to place order - {e}")
                        failed_orders += 1
                
                # Summary
                if successful_orders > 0:
                    st.success(f"🎯 Strategy executed: {successful_orders} orders placed successfully")
                    if failed_orders > 0:
                        st.warning(f"⚠️ {failed_orders} orders failed")
                    st.info("🔔 Monitor positions in Account Overview below")
                else:
                    st.error("❌ No orders were successfully placed")
    
    # Get final strategy config for other sections
    strategy_config = st.session_state.get('paper_trading_config', {}).get('strategy', {})
    
    
    # Manual Trading Section
    st.markdown("---")
    st.subheader("🔧 Manual Trading")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        manual_ticker = st.text_input("Ticker", placeholder="AAPL").upper()
    
    with col2:
        manual_qty = st.number_input("Quantity", min_value=1, value=100)
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        manual_buy = st.button("🟢 BUY", disabled=not manual_ticker)
    
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        manual_sell = st.button("🔴 SELL", disabled=not manual_ticker)
    
    # Handle manual orders
    if (manual_buy or manual_sell) and manual_ticker:
        side = 'buy' if manual_buy else 'sell'
        with st.spinner(f"Placing {side} order for {manual_qty} shares of {manual_ticker}..."):
            try:
                order = api.submit_order(
                    symbol=manual_ticker,
                    qty=manual_qty,
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                st.success(f"✅ {side.title()} order submitted for {manual_ticker} (Order: {order.id})")
            except Exception as e:
                st.error(f"❌ Failed to place {side} order: {e}")
    
    # Account Overview Section
    st.markdown("---")
    st.subheader("📊 Paper Account Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💼 Current Positions"):
            try:
                positions = api.list_positions()
                if positions:
                    pos_data = []
                    for pos in positions:
                        if float(pos.qty) != 0:
                            pos_data.append({
                                "Symbol": pos.symbol,
                                "Qty": int(float(pos.qty)),
                                "Avg Cost": f"${float(pos.avg_entry_price):.2f}",
                                "Market Value": f"${float(pos.market_value):,.2f}",
                                "P&L": f"${float(pos.unrealized_pl):,.2f}",
                                "P&L %": f"{float(pos.unrealized_plpc)*100:.1f}%"
                            })
                    
                    if pos_data:
                        st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
                    else:
                        st.info("No positions")
                else:
                    st.info("No positions found")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        if st.button("📋 Recent Orders"):
            try:
                orders = api.list_orders(status='all', limit=10)
                if orders:
                    order_data = []
                    for order in orders:
                        order_data.append({
                            "Symbol": order.symbol,
                            "Side": order.side.upper(),
                            "Qty": int(order.qty),
                            "Status": order.status.upper(),
                            "Time": order.submitted_at.strftime("%H:%M") if order.submitted_at else "N/A"
                        })
                    st.dataframe(pd.DataFrame(order_data), use_container_width=True)
                else:
                    st.info("No recent orders")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col3:
        if st.button("💰 Account Info"):
            try:
                account = api.get_account()
                st.metric("Buying Power", f"${float(account.buying_power):,.2f}")
                st.metric("Portfolio Value", f"${float(account.portfolio_value):,.2f}")
                st.metric("Day P&L", f"${float(account.todays_change):,.2f}")
            except Exception as e:
                st.error(f"Error: {e}")


def show_configuration_page(config: Config):
    """Show configuration management page."""
    st.header("⚙️ System Configuration")
    
    tabs = st.tabs(["📊 Data Sources", "🎯 Strategy", "💰 Costs", "⏰ Schedule", "🔒 Security"])
    
    with tabs[0]:  # Data Sources
        st.subheader("Data Source Configuration")
        
        primary_source = st.selectbox(
            "Primary Data Source",
            ["yfinance", "polygon", "tiingo"],
            index=["yfinance", "polygon", "tiingo"].index(config.data_sources.primary)
        )
        
        st.multiselect(
            "Fallback Sources",
            ["yfinance", "polygon", "tiingo"],
            default=config.data_sources.fallback
        )
        
        st.text_input("Polygon API Key", value=config.data_sources.polygon.api_key, type="password")
        st.text_input("Tiingo API Key", value=config.data_sources.tiingo.api_key, type="password")
    
    with tabs[1]:  # Strategy
        st.subheader("Strategy Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("Top K Gaps", value=config.strategy.top_k, min_value=1)
            st.number_input("Min Gap %", value=config.strategy.min_gap_pct * 100, min_value=0.1) / 100
            st.number_input("Max Gap %", value=config.strategy.max_gap_pct * 100, min_value=1.0) / 100
        
        with col2:
            st.number_input("Profit Target %", value=config.strategy.profit_target_pct * 100, min_value=0.1) / 100
            st.number_input("Stop Loss %", value=config.strategy.hard_stop_pct * 100, min_value=0.1) / 100
            st.number_input("Position Size $", value=config.strategy.position_size_usd, min_value=100)
    
    with tabs[2]:  # Costs
        st.subheader("Trading Costs")
        
        st.number_input("Commission per Share", value=config.costs.commission_per_share, min_value=0.0, format="%.4f")
        st.number_input("Slippage (bps)", value=config.costs.slippage_bps, min_value=0)
        st.number_input("Borrowing Rate", value=config.costs.borrowing_rate, min_value=0.0)
    
    with tabs[3]:  # Schedule
        st.subheader("Data Collection Schedule")
        
        st.number_input("Collection Frequency (minutes)", value=config.data_collection.frequency_minutes, min_value=1)
        st.number_input("Universe Size", value=config.data_collection.universe_size, min_value=100)
        st.number_input("Min Dollar Volume", value=config.data_collection.min_dollar_volume, min_value=10000)
    
    with tabs[4]:  # Security
        st.subheader("Security Settings")
        
        st.number_input("Max API Requests/Min", value=config.security.max_api_requests_per_minute, min_value=1)
        st.checkbox("Encrypt Sensitive Data", value=config.security.encrypt_sensitive_data)
        st.checkbox("Audit Trail", value=config.security.audit_trail)


def show_system_status_page(config: Config, data_manager: DataManager):
    """Show system status and health page."""
    st.header("📈 System Status & Health")
    
    # System health indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("System Status", "🟢 Healthy")
    
    with col2:
        storage_stats = data_manager.get_storage_stats()
        total_size_gb = storage_stats.get('total_size_mb', 0) / 1024
        st.metric("Storage Usage", f"{total_size_gb:.2f} GB")
    
    with col3:
        st.metric("Data Freshness", "📅 Current")
    
    # Detailed system information
    with st.expander("🔧 System Configuration"):
        st.json({
            "Data Source": config.data_sources.primary,
            "Universe Size": config.data_collection.universe_size,
            "Min Dollar Volume": f"${config.data_collection.min_dollar_volume:,}",
            "Gap Range": f"{config.strategy.min_gap_pct:.1%} - {config.strategy.max_gap_pct:.1%}",
            "Position Size": f"${config.strategy.position_size_usd:,}",
            "Max Positions": config.strategy.max_positions
        })
    
    # Data storage breakdown
    st.subheader("💾 Data Storage Breakdown")
    
    storage_stats = data_manager.get_storage_stats()
    
    if storage_stats:
        # Create pie chart of storage usage
        storage_data = {
            'Universe Data': storage_stats.get('universe_size_mb', 0),
            'Price Data': storage_stats.get('price_data_size_mb', 0)
        }
        
        fig = px.pie(
            values=list(storage_data.values()),
            names=list(storage_data.keys()),
            title="Storage Usage by Category"
        )
        st.plotly_chart(fig, use_container_width=True)


def display_yahoo_gaps_table(gaps_df: pd.DataFrame):
    """Display formatted gaps table from Yahoo Finance data."""
    if gaps_df.empty:
        st.info("No gaps to display")
        return
    
    # Format the dataframe for display
    display_df = gaps_df.copy()
    
    # Format percentage columns
    display_df['gap_pct_formatted'] = display_df['gap_pct'].apply(lambda x: f"{x:.2%}")
    
    # Format price and volume
    display_df['current_price_formatted'] = display_df['current_price'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
    display_df['current_volume_formatted'] = display_df['current_volume'].apply(lambda x: f"{x:,.0f}" if x > 0 else "N/A")
    
    # Select and rename columns for display
    display_columns = {
        'symbol': 'Symbol',
        'gap_pct_formatted': 'Gap %',
        'gap_direction': 'Direction',
        'current_price_formatted': 'Current Price',
        'current_volume_formatted': 'Volume',
        'source': 'Source'
    }
    
    final_df = display_df[list(display_columns.keys())].rename(columns=display_columns)
    
    st.dataframe(
        final_df,
        use_container_width=True,
        hide_index=True
    )


def display_gaps_table(gaps_df: pd.DataFrame):
    """Display formatted gaps table."""
    if gaps_df.empty:
        st.info("No gaps to display")
        return
    
    # Format the dataframe for display
    display_df = gaps_df.copy()
    
    # Format percentage columns
    for col in ['gap_pct', 'intraday_pct']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
    
    # Format price columns
    for col in ['previous_close', 'current_open', 'current_high', 'current_low', 'current_close']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
    
    # Format volume
    if 'current_volume' in display_df.columns:
        display_df['current_volume'] = display_df['current_volume'].apply(
            lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A"
        )
    
    # Select and rename columns for display
    display_columns = {
        'symbol': 'Symbol',
        'gap_pct': 'Gap %',
        'gap_direction': 'Direction',
        'previous_close': 'Prev Close',
        'current_open': 'Open',
        'intraday_pct': 'Intraday %',
        'current_volume': 'Volume'
    }
    
    available_columns = {k: v for k, v in display_columns.items() if k in display_df.columns}
    
    st.dataframe(
        display_df[list(available_columns.keys())].rename(columns=available_columns),
        use_container_width=True
    )


def display_gap_summary_metrics(gaps_df: pd.DataFrame):
    """Display gap summary metrics."""
    if gaps_df.empty:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_gaps = len(gaps_df)
        st.metric("Total Gaps", total_gaps)
    
    with col2:
        up_gaps = len(gaps_df[gaps_df['gap_pct'] > 0])
        st.metric("Up Gaps", up_gaps, delta=f"{up_gaps/total_gaps:.1%}" if total_gaps > 0 else "0%")
    
    with col3:
        avg_gap = gaps_df['gap_pct'].abs().mean()
        st.metric("Avg Gap Size", f"{avg_gap:.2%}")
    
    with col4:
        max_gap = gaps_df['gap_pct'].abs().max()
        st.metric("Max Gap Size", f"{max_gap:.2%}")


def display_gap_distribution_chart(gaps_df: pd.DataFrame):
    """Display gap distribution chart."""
    if gaps_df.empty:
        return
    
    # Create histogram of gap sizes
    fig = px.histogram(
        gaps_df,
        x='gap_pct',
        nbins=20,
        title="Gap Size Distribution",
        labels={'gap_pct': 'Gap Percentage', 'count': 'Frequency'}
    )
    
    fig.update_layout(
        xaxis_tickformat='.1%',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_gap_patterns(patterns: Dict):
    """Display historical gap patterns analysis."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Gap Statistics")
        
        metrics_df = pd.DataFrame([
            {"Metric": "Total Gaps", "Value": patterns.get('total_gaps', 0)},
            {"Metric": "Up Gaps", "Value": patterns.get('up_gaps', 0)},
            {"Metric": "Down Gaps", "Value": patterns.get('down_gaps', 0)},
            {"Metric": "Avg Gap Size", "Value": f"{patterns.get('avg_gap_size', 0):.2%}"},
            {"Metric": "Up Gap Follow-Through", "Value": f"{patterns.get('up_gap_follow_through', 0):.1%}"},
        ])
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("📈 Gap Size Distribution")
        
        if 'gap_size_buckets' in patterns:
            buckets = patterns['gap_size_buckets']
            
            fig = px.bar(
                x=list(buckets.keys()),
                y=list(buckets.values()),
                title="Gaps by Size Range"
            )
            
            st.plotly_chart(fig, use_container_width=True)


def display_backtest_results(results: Dict):
    """Display comprehensive backtest results."""
    
    st.markdown("---")
    st.header("📊 Backtest Results")
    
    # Key performance metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_return = results.get('total_return', 0)
        total_return_pct = results.get('total_return_pct', 0)
        st.metric(
            "Total Return", 
            f"${total_return:,.0f}",
            delta=f"{total_return_pct:.1%}"
        )
    
    with col2:
        final_value = results.get('final_value', 0)
        st.metric("Final Portfolio Value", f"${final_value:,.0f}")
    
    with col3:
        num_trades = results.get('num_trades', 0)
        win_rate = results.get('win_rate', 0)
        st.metric(
            "Total Trades", 
            f"{num_trades:,}",
            delta=f"{win_rate:.1%} wins"
        )
    
    with col4:
        sharpe_ratio = results.get('sharpe_ratio', 0)
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    with col5:
        max_drawdown_pct = results.get('max_drawdown_pct', 0)
        st.metric("Max Drawdown", f"{max_drawdown_pct:.1f}%")
    
    # Portfolio performance chart
    portfolio_df = results.get('portfolio_df', pd.DataFrame())
    if not portfolio_df.empty:
        st.subheader("📈 Portfolio Performance")
        
        fig = go.Figure()
        
        # Portfolio value line
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['total_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Initial capital line
        initial_capital = results.get('final_value', 100000) - results.get('total_return', 0)
        fig.add_hline(
            y=initial_capital, 
            line_dash="dash", 
            line_color="gray",
            annotation_text=f"Initial Capital (${initial_capital:,.0f})"
        )
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Daily P&L chart
        if 'daily_pnl' in portfolio_df.columns:
            st.subheader("📊 Daily P&L")
            
            fig_pnl = go.Figure()
            
            # Color positive/negative P&L differently
            colors = ['green' if x >= 0 else 'red' for x in portfolio_df['daily_pnl']]
            
            fig_pnl.add_trace(go.Bar(
                x=portfolio_df.index,
                y=portfolio_df['daily_pnl'],
                name='Daily P&L',
                marker_color=colors
            ))
            
            fig_pnl.update_layout(
                title="Daily Profit & Loss",
                xaxis_title="Date",
                yaxis_title="Daily P&L ($)",
                height=400
            )
            
            st.plotly_chart(fig_pnl, use_container_width=True)
    
    # Detailed analysis tabs
    tabs = st.tabs(["📋 Trade Details", "📊 Exit Analysis", "🎯 Performance Stats", "💾 Raw Data"])
    
    with tabs[0]:  # Trade Details
        trades_df = results.get('trades_df', pd.DataFrame())
        if not trades_df.empty:
            st.subheader("Individual Trade Results")
            
            # Format the dataframe for display
            display_trades_df = trades_df.copy()
            
            # Format columns
            display_trades_df['gap_pct'] = display_trades_df['gap_pct'].apply(lambda x: f"{x:.2%}")
            display_trades_df['return_pct'] = display_trades_df['return_pct'].apply(lambda x: f"{x:.2%}")
            display_trades_df['pnl_net'] = display_trades_df['pnl_net'].apply(lambda x: f"${x:.2f}")
            display_trades_df['entry_price'] = display_trades_df['entry_price'].apply(lambda x: f"${x:.2f}")
            display_trades_df['exit_price'] = display_trades_df['exit_price'].apply(lambda x: f"${x:.2f}")
            display_trades_df['hold_time_hours'] = display_trades_df['hold_time_hours'].apply(lambda x: f"{x:.1f}h")
            
            # Rename columns for display
            display_trades_df = display_trades_df.rename(columns={
                'symbol': 'Symbol',
                'entry_date': 'Entry Date',
                'exit_date': 'Exit Date',
                'gap_pct': 'Gap %',
                'return_pct': 'Return %',
                'pnl_net': 'P&L',
                'entry_price': 'Entry $',
                'exit_price': 'Exit $',
                'hold_time_hours': 'Hold Time',
                'exit_reason': 'Exit Reason',
                'sector': 'Sector'
            })
            
            st.dataframe(display_trades_df, use_container_width=True)
            
            # Download button
            csv = trades_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Trade Data",
                data=csv,
                file_name=f"backtest_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No trade data available")
    
    with tabs[1]:  # Exit Analysis
        exit_reasons = results.get('exit_reasons', {})
        if exit_reasons:
            st.subheader("Exit Reason Breakdown")
            
            # Create pie chart
            fig_exits = px.pie(
                values=list(exit_reasons.values()),
                names=list(exit_reasons.keys()),
                title="Distribution of Exit Reasons"
            )
            st.plotly_chart(fig_exits, use_container_width=True)
            
            # Exit reason table
            exit_df = pd.DataFrame([
                {"Exit Reason": reason, "Count": count, "Percentage": f"{count/sum(exit_reasons.values()):.1%}"}
                for reason, count in exit_reasons.items()
            ])
            st.dataframe(exit_df, use_container_width=True)
        else:
            st.warning("No exit reason data available")
    
    with tabs[2]:  # Performance Stats
        st.subheader("Detailed Performance Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Return Analysis**")
            stats_data = [
                {"Metric": "Average Return", "Value": f"{results.get('avg_return', 0):.2%}"},
                {"Metric": "Average Winner", "Value": f"{results.get('avg_winner', 0):.2%}"},
                {"Metric": "Average Loser", "Value": f"{results.get('avg_loser', 0):.2%}"},
                {"Metric": "Largest Winner", "Value": f"{results.get('largest_winner', 0):.2%}"},
                {"Metric": "Largest Loser", "Value": f"{results.get('largest_loser', 0):.2%}"},
            ]
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Risk Metrics**")
            risk_data = [
                {"Metric": "Sharpe Ratio", "Value": f"{results.get('sharpe_ratio', 0):.2f}"},
                {"Metric": "Max Drawdown", "Value": f"{results.get('max_drawdown_pct', 0):.1f}%"},
                {"Metric": "Win Rate", "Value": f"{results.get('win_rate', 0):.1%}"},
                {"Metric": "Total Trades", "Value": f"{results.get('num_trades', 0):,}"},
                {"Metric": "Final Value", "Value": f"${results.get('final_value', 0):,.0f}"},
            ]
            st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)
    
    with tabs[3]:  # Raw Data
        st.subheader("Raw Portfolio Data")
        if not portfolio_df.empty:
            st.dataframe(portfolio_df, use_container_width=True)
            
            # Download portfolio data
            portfolio_csv = portfolio_df.to_csv()
            st.download_button(
                label="📥 Download Portfolio Data",
                data=portfolio_csv,
                file_name=f"backtest_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No portfolio data available")


def show_simulation_manager_page():
    """Show simulation manager page."""
    st.header("💾 Simulation Manager")
    st.markdown("**Manage saved simulation configurations and results**")
    
    # Initialize simulation manager
    @st.cache_resource
    def get_simulation_manager():
        return SimulationManager()
    
    sim_manager = get_simulation_manager()
    
    # Get all saved simulations
    simulations = sim_manager.list_simulations()
    
    # Storage stats
    stats = sim_manager.get_storage_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Simulations", stats['total_simulations'])
    
    with col2:
        st.metric("Storage Used", f"{stats['total_size_mb']:.1f} MB")
    
    with col3:
        st.metric("JSON Files", stats['json_files'])
    
    with col4:
        st.metric("Pickle Files", stats['pickle_files'])
    
    if simulations:
        st.markdown("---")
        st.subheader("📋 Saved Simulations")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Simulation List", "🔍 Details", "🔧 Management"])
        
        with tab1:
            # Create a DataFrame for display
            display_data = []
            for sim in simulations:
                timestamp = datetime.fromisoformat(sim['timestamp'])
                display_data.append({
                    'Name': sim['name'],
                    'Date': timestamp.strftime('%Y-%m-%d'),
                    'Time': timestamp.strftime('%H:%M:%S'),
                    'Size (MB)': f"{sim['size_mb']:.2f}",
                    'Format': sim['format'].upper(),
                    'Tags': ', '.join(sim['tags']) if sim['tags'] else 'None',
                    'Description': sim['description'][:50] + '...' if len(sim['description']) > 50 else sim['description']
                })
            
            sim_df = pd.DataFrame(display_data)
            st.dataframe(sim_df, use_container_width=True)
        
        with tab2:
            # Detailed view of selected simulation
            sim_names = [f"{sim['name']} ({datetime.fromisoformat(sim['timestamp']).strftime('%Y-%m-%d %H:%M')})" for sim in simulations]
            selected_sim_name = st.selectbox("Select Simulation for Details", sim_names)
            
            if selected_sim_name:
                # Find the selected simulation
                selected_index = sim_names.index(selected_sim_name)
                selected_sim = simulations[selected_index]
                
                # Load the full simulation
                loaded_sim = sim_manager.load_simulation(selected_sim['filename'])
                
                if loaded_sim:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Configuration")
                        if loaded_sim.config:
                            st.json(loaded_sim.config)
                        else:
                            st.info("No configuration data available")
                    
                    with col2:
                        st.subheader("📈 Results Summary")
                        if loaded_sim.results:
                            # Show key metrics if available
                            if 'final_value' in loaded_sim.results:
                                st.metric("Final Value", f"${loaded_sim.results['final_value']:,.0f}")
                            if 'total_return_pct' in loaded_sim.results:
                                st.metric("Total Return", f"{loaded_sim.results['total_return_pct']:.1%}")
                            if 'num_trades' in loaded_sim.results:
                                st.metric("Total Trades", loaded_sim.results['num_trades'])
                            if 'win_rate' in loaded_sim.results:
                                st.metric("Win Rate", f"{loaded_sim.results['win_rate']:.1%}")
                            
                            # Show full results
                            with st.expander("🔍 Full Results"):
                                st.json({k: str(v) for k, v in loaded_sim.results.items() if not isinstance(v, pd.DataFrame)})
                        else:
                            st.info("No results data available (configuration only)")
                    
                    # Action buttons
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("🔄 Load Configuration", key=f"load_{selected_sim['filename']}"):
                            st.session_state.loaded_config = loaded_sim.config
                            st.success("✅ Configuration loaded! Go to Portfolio Simulation to apply it.")
                    
                    with col2:
                        export_filename = st.text_input("Export filename", value=f"{loaded_sim.name}_export.json")
                        if st.button("📤 Export", key=f"export_{selected_sim['filename']}"):
                            if export_filename:
                                if sim_manager.export_simulation(selected_sim['filename'], export_filename):
                                    st.success(f"✅ Exported to {export_filename}")
                                else:
                                    st.error("❌ Export failed")
                    
                    with col3:
                        if st.button("🗑️ Delete", key=f"delete_{selected_sim['filename']}", type="secondary"):
                            if sim_manager.delete_simulation(selected_sim['filename']):
                                st.success("✅ Simulation deleted")
                                st.rerun()
                            else:
                                st.error("❌ Delete failed")
                else:
                    st.error("Failed to load simulation details")
        
        with tab3:
            # Management operations
            st.subheader("🔧 Management Operations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🧹 Cleanup Operations**")
                
                keep_count = st.number_input(
                    "Keep most recent simulations",
                    min_value=1,
                    max_value=100,
                    value=20,
                    help="Number of most recent simulations to keep"
                )
                
                if st.button("🧹 Cleanup Old Saves"):
                    deleted_count = sim_manager.cleanup_old_saves(keep_count)
                    if deleted_count > 0:
                        st.success(f"✅ Deleted {deleted_count} old simulation saves")
                        st.rerun()
                    else:
                        st.info("No old saves to clean up")
            
            with col2:
                st.markdown("**📥 Import/Export**")
                
                # Import simulation
                uploaded_file = st.file_uploader(
                    "Import Simulation",
                    type=['json'],
                    help="Upload a previously exported simulation file"
                )
                
                if uploaded_file is not None:
                    try:
                        # Save uploaded file temporarily
                        temp_path = f"temp_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(temp_path, 'wb') as f:
                            f.write(uploaded_file.getvalue())
                        
                        # Import the simulation
                        if sim_manager.import_simulation(temp_path):
                            st.success("✅ Simulation imported successfully")
                            # Clean up temp file
                            Path(temp_path).unlink(missing_ok=True)
                            st.rerun()
                        else:
                            st.error("❌ Import failed")
                            Path(temp_path).unlink(missing_ok=True)
                    except Exception as e:
                        st.error(f"❌ Import error: {e}")
                
                # Export all simulations
                if st.button("📤 Export All Simulations"):
                    st.info("This feature will be implemented in a future update")
    
    else:
        st.info("🎯 No saved simulations found. Run a backtest in the Portfolio Simulation page to create your first save!")
        
        # Show quick start guide
        st.markdown("---")
        st.subheader("🚀 Quick Start Guide")
        
        st.markdown("""
        1. **Go to Portfolio Simulation** page
        2. **Configure your strategy** parameters
        3. **Set backtest period** (start and end dates)
        4. **Enter a save name** and click "💾 Save Configuration"
        5. **Run backtest** with "💾 Save Results" checked
        6. **Return here** to manage your saved simulations
        """)


if __name__ == "__main__":
    main()