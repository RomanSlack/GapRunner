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
    
    # Run backtest button
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
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
                    st.success("✅ Backtest completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Backtest failed: {e}")
                    logger.error(f"Backtest error: {e}", exc_info=True)
    
    # Display results if available
    if hasattr(st.session_state, 'backtest_results') and st.session_state.backtest_results:
        display_backtest_results(st.session_state.backtest_results)


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


if __name__ == "__main__":
    main()