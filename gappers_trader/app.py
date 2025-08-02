"""Streamlit dashboard for the gap trading system."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from gappers.dead_simple import DeadSimpleTrader
from gappers.config import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Gap Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .positive {
        color: #00C851;
    }
    .negative {
        color: #ff4444;
    }
    .stAlert > div {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_simple_trader():
    """Load dead simple trader."""
    return DeadSimpleTrader()


def main():
    """Main dashboard application."""
    st.title("🎯 Gap Trading Strategy Dashboard")
    st.markdown("Production-grade overnight gap trading system with live paper-trading support")
    
    # Initialize simple trader
    try:
        trader = load_simple_trader()
    except Exception as e:
        st.error(f"Error initializing trader: {e}")
        st.stop()
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("⚙️ Strategy Parameters")
        
        # Date range selection
        st.subheader("Backtest Period")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime(2024, 8, 2).date(),
                max_value=datetime.now() - timedelta(days=1)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime(2024, 10, 15).date(),
                max_value=datetime.now() - timedelta(days=1)
            )
        
        # Strategy parameters
        st.subheader("Entry/Exit Rules")
        profit_target = st.slider("Profit Target (%)", 1.0, 20.0, 5.0, 0.5) / 100
        stop_loss = st.slider("Stop Loss (%)", 0.5, 10.0, 2.0, 0.5) / 100
        max_hold_hours = st.slider("Max Hold Time (hours)", 1, 8, 6)
        
        st.subheader("Selection Criteria")
        top_k = st.slider("Top K Gaps", 5, 50, 10)
        min_gap = st.slider("Min Gap (%)", 1.0, 10.0, 2.0, 0.5) / 100
        max_gap = st.slider("Max Gap (%)", 10.0, 50.0, 30.0, 1.0) / 100
        
        st.subheader("Risk Management")
        position_size = st.number_input("Position Size ($)", 1000, 100000, 10000, 1000)
        max_positions = st.slider("Max Positions", 1, 20, 10)
        sector_diversification = st.checkbox("Sector Diversification", True)
        
        if sector_diversification:
            max_per_sector = st.slider("Max per Sector", 1, 5, 3)
        else:
            max_per_sector = max_positions
        
        st.subheader("Costs")
        commission = st.number_input("Commission per Share ($)", 0.0, 0.01, 0.005, 0.001, format="%.4f")
        slippage_bps = st.slider("Slippage (bps)", 0, 50, 10)
        
        # Run backtest button
        if st.button("🚀 Run Backtest", type="primary"):
            st.session_state.run_backtest = True
    
    # Main content area
    if not hasattr(st.session_state, 'run_backtest'):
        show_welcome_screen()
    else:
        # Create simple parameters
        params = {
            'profit_target': profit_target,
            'stop_loss': stop_loss,
            'top_k': top_k,
            'min_gap': min_gap,
            'position_size': position_size
        }
        
        run_simple_backtest(
            trader, datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date, datetime.min.time()), params
        )


def show_welcome_screen():
    """Show welcome screen with system overview."""
    st.markdown("## 👋 Welcome to the Gap Trading Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Features
        - **Real-time gap detection** with survivorship bias correction
        - **Vectorized backtesting** using advanced analytics
        - **Risk management** with position sizing and diversification
        - **Live paper trading** via Alpaca integration
        """)
    
    with col2:
        st.markdown("""
        ### ⚡ Performance
        - Process 20+ years of data in under 60 seconds
        - Advanced performance metrics and risk analysis
        - Interactive visualizations and reports
        - Parameter optimization and sensitivity analysis
        """)
    
    with col3:
        st.markdown("""
        ### 🛡️ Production Ready
        - Comprehensive error handling and logging
        - SEC Rule 201 compliance for short sales
        - Multi-source data feeds (yfinance, IEX, Polygon)
        - Docker containerization and CI/CD pipeline
        """)
    
    st.markdown("---")
    st.info("👈 Configure your strategy parameters in the sidebar and click 'Run Backtest' to get started!")
    
    # Show current market gaps (if available)
    # Commented out to prevent automatic loading on startup
    # show_current_gaps()


def show_current_gaps():
    """Show current market gaps for today."""
    st.markdown("## 📈 Today's Gap Opportunities")
    
    try:
        with st.spinner("Loading current gaps..."):
            data_feed = load_data_feed()
            universe_builder, signal_generator, _, _ = load_components(data_feed)
            
            today = datetime.now()
            gaps_df = signal_generator.calculate_gaps(today, min_gap_pct=0.02)
            
            if not gaps_df.empty:
                top_gaps = signal_generator.rank_gaps(gaps_df, top_k=10)
                
                # Display top gaps
                st.dataframe(
                    top_gaps[['symbol', 'gap_pct', 'previous_close', 'current_open', 'sector', 'rank']]
                    .rename(columns={
                        'symbol': 'Symbol',
                        'gap_pct': 'Gap %',
                        'previous_close': 'Prev Close',
                        'current_open': 'Open',
                        'sector': 'Sector',
                        'rank': 'Rank'
                    })
                    .style.format({
                        'Gap %': '{:.2%}',
                        'Prev Close': '${:.2f}',
                        'Open': '${:.2f}'
                    }),
                    use_container_width=True
                )
            else:
                st.warning("No significant gaps found for today.")
                
    except Exception as e:
        st.warning(f"Could not load current gaps: {e}")


def run_simple_backtest(trader, start_date, end_date, params):
    """Run simple backtest."""
    with st.spinner("Running simple backtest..."):
        try:
            # Run backtest
            results = trader.run_backtest(start_date, end_date, params)
            
            # Display results
            display_simple_results(results)
            
        except Exception as e:
            st.error(f"Error running backtest: {e}")
            logger.error(f"Backtest error: {e}", exc_info=True)


def display_simple_results(results: Dict):
    """Display simple backtest results."""
    st.markdown("## 📊 Backtest Results")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total P&L", f"${results['total_return']:.0f}")
    with col2:
        st.metric("Total Return", f"{results.get('total_return_pct', 0):.1%}")
    with col3:
        st.metric("Final Value", f"${results.get('final_portfolio_value', 100000):.0f}")
    with col4:
        st.metric("Win Rate", f"{results['win_rate']:.1%}")
    with col5:
        st.metric("Total Trades", results['num_trades'])
    
    # Portfolio chart
    if 'portfolio_values' in results and not results['portfolio_values'].empty:
        st.markdown("## 📈 Portfolio Performance")
        
        portfolio_df = results['portfolio_values']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add starting value line
        fig.add_hline(y=100000, line_dash="dash", line_color="gray", 
                     annotation_text="Starting Value ($100k)")
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Trade details
    if results['trades']:
        st.markdown("## 📋 Trade Details")
        trades_df = pd.DataFrame(results['trades'])
        trades_df['pnl'] = trades_df['pnl'].round(2)
        trades_df['gap_pct'] = (trades_df['gap_pct'] * 100).round(2)
        trades_df['pnl_pct'] = (trades_df['pnl_pct'] * 100).round(2)
        
        st.dataframe(trades_df, use_container_width=True)
    else:
        st.warning("No trades found in backtest period.")


def display_backtest_results(results: Dict, analysis: Dict):
    """Display comprehensive backtest results."""
    
    trades = results.get('trades', [])
    portfolio_values = results.get('portfolio_values', pd.DataFrame())
    
    if not trades:
        st.warning("No trades were executed in the backtest period.")
        return
    
    # Key metrics at the top
    display_key_metrics(analysis)
    
    # Main charts
    display_performance_charts(portfolio_values, trades)
    
    # Detailed analysis tabs
    display_detailed_analysis(analysis, trades)
    
    # Trade table
    display_trade_table(trades)


def display_key_metrics(analysis: Dict):
    """Display key performance metrics."""
    st.markdown("## 📊 Key Performance Metrics")
    
    trade_data = analysis.get('trade_analysis', {})
    perf_data = analysis.get('performance_metrics', {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_return = perf_data.get('total_return_pct', 0)
        color = "positive" if total_return > 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Return</h4>
            <h2 class="{color}">{total_return:+.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        sharpe = perf_data.get('sharpe_ratio', 0)
        color = "positive" if sharpe > 1 else "negative" if sharpe < 0 else "neutral"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Sharpe Ratio</h4>
            <h2 class="{color}">{sharpe:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        win_rate = trade_data.get('win_rate', 0) * 100
        color = "positive" if win_rate > 50 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Win Rate</h4>
            <h2 class="{color}">{win_rate:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        max_dd = perf_data.get('max_drawdown_pct', 0)
        color = "positive" if max_dd > -10 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Max Drawdown</h4>
            <h2 class="{color}">{max_dd:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        total_trades = trade_data.get('total_trades', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Trades</h4>
            <h2>{total_trades:,}</h2>
        </div>
        """, unsafe_allow_html=True)


def display_performance_charts(portfolio_values: pd.DataFrame, trades: List):
    """Display main performance charts."""
    st.markdown("## 📈 Performance Analysis")
    
    if portfolio_values.empty:
        st.warning("No portfolio data available for charting.")
        return
    
    # Equity curve
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=portfolio_values.index,
        y=portfolio_values['value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig_equity.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # Drawdown chart
    if len(portfolio_values) > 1:
        running_max = portfolio_values['value'].expanding().max()
        drawdown = (portfolio_values['value'] - running_max) / running_max * 100
        
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=portfolio_values.index,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            fill='tonexty',
            line=dict(color='red', width=1)
        ))
        
        fig_dd.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_dd, use_container_width=True)


def display_detailed_analysis(analysis: Dict, trades: List):
    """Display detailed analysis in tabs."""
    st.markdown("## 🔍 Detailed Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Trade Analysis", "Risk Metrics", "Sector Analysis", "Temporal Patterns"])
    
    with tab1:
        display_trade_analysis(analysis.get('trade_analysis', {}))
    
    with tab2:
        display_risk_analysis(analysis.get('risk_metrics', {}))
    
    with tab3:
        display_sector_analysis(analysis.get('sector_analysis', {}))
    
    with tab4:
        display_temporal_analysis(analysis.get('temporal_analysis', {}))


def display_trade_analysis(trade_data: Dict):
    """Display trade-level analysis."""
    if not trade_data:
        st.warning("No trade analysis data available.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Return Statistics")
        st.write(f"Average Return: {trade_data.get('avg_return_pct', 0):.2f}%")
        st.write(f"Median Return: {trade_data.get('median_return_pct', 0):.2f}%")
        st.write(f"Standard Deviation: {trade_data.get('std_return_pct', 0):.2f}%")
        st.write(f"Skewness: {trade_data.get('skewness', 0):.3f}")
        st.write(f"Kurtosis: {trade_data.get('kurtosis', 0):.3f}")
    
    with col2:
        st.subheader("Winner/Loser Analysis")
        st.write(f"Average Winner: {trade_data.get('avg_winner_pct', 0):.2f}%")
        st.write(f"Average Loser: {trade_data.get('avg_loser_pct', 0):.2f}%")
        st.write(f"Largest Winner: {trade_data.get('largest_winner_pct', 0):.2f}%")
        st.write(f"Largest Loser: {trade_data.get('largest_loser_pct', 0):.2f}%")
        st.write(f"Profit Factor: {trade_data.get('profit_factor', 0):.2f}")
    
    # Exit reasons pie chart
    exit_reasons = trade_data.get('exit_reasons', {})
    if exit_reasons:
        fig_pie = px.pie(
            values=list(exit_reasons.values()),
            names=list(exit_reasons.keys()),
            title="Exit Reason Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)


def display_risk_analysis(risk_data: Dict):
    """Display risk metrics."""
    if not risk_data:
        st.warning("No risk analysis data available.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Value at Risk")
        st.write(f"VaR (95%): {risk_data.get('var_95_pct', 0):.2f}%")
        st.write(f"VaR (99%): {risk_data.get('var_99_pct', 0):.2f}%")
        st.write(f"CVaR (95%): {risk_data.get('cvar_95_pct', 0):.2f}%")
        st.write(f"CVaR (99%): {risk_data.get('cvar_99_pct', 0):.2f}%")
    
    with col2:
        st.subheader("Market Risk")
        st.write(f"Beta: {risk_data.get('beta', 0):.2f}")
        st.write(f"Trade Volatility: {risk_data.get('trade_return_volatility_pct', 0):.2f}%")


def display_sector_analysis(sector_data: Dict):
    """Display sector-wise performance."""
    if not sector_data:
        st.warning("No sector analysis data available.")
        return
    
    # Convert to DataFrame for easier display
    sector_df = pd.DataFrame.from_dict(sector_data, orient='index')
    
    if not sector_df.empty:
        st.dataframe(
            sector_df.style.format({
                'win_rate': '{:.1%}',
                'avg_return_pct': '{:.2f}%',
                'total_pnl': '${:,.2f}',
                'volatility_pct': '{:.2f}%'
            }),
            use_container_width=True
        )
        
        # Sector performance chart
        fig_sector = px.bar(
            x=sector_df.index,
            y=sector_df['avg_return_pct'],
            title="Average Return by Sector"
        )
        st.plotly_chart(fig_sector, use_container_width=True)


def display_temporal_analysis(temporal_data: Dict):
    """Display temporal pattern analysis."""
    if not temporal_data:
        st.warning("No temporal analysis data available.")
        return
    
    # Monthly performance
    monthly_data = temporal_data.get('monthly', {})
    if monthly_data:
        st.subheader("Monthly Performance")
        monthly_df = pd.DataFrame.from_dict(monthly_data, orient='index')
        
        if not monthly_df.empty:
            fig_monthly = px.bar(
                x=monthly_df.index,
                y=monthly_df['avg_return_pct'],
                title="Average Return by Month"
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Day of week performance
    dow_data = temporal_data.get('day_of_week', {})
    if dow_data:
        st.subheader("Day of Week Performance")
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        dow_df = pd.DataFrame.from_dict(dow_data, orient='index')
        
        if not dow_df.empty:
            dow_df.index = [dow_names[i] for i in dow_df.index if i < len(dow_names)]
            
            fig_dow = px.bar(
                x=dow_df.index,
                y=dow_df['avg_return_pct'],
                title="Average Return by Day of Week"
            )
            st.plotly_chart(fig_dow, use_container_width=True)


def display_trade_table(trades: List):
    """Display detailed trade table."""
    st.markdown("## 📋 Trade Details")
    
    if not trades:
        st.warning("No trades to display.")
        return
    
    # Convert trades to DataFrame
    trade_data = []
    for trade in trades:
        trade_data.append({
            'Symbol': trade.symbol,
            'Entry Date': trade.entry_date.strftime('%Y-%m-%d %H:%M'),
            'Exit Date': trade.exit_date.strftime('%Y-%m-%d %H:%M'),
            'Gap %': f"{trade.gap_pct:.2%}",
            'Entry Price': f"${trade.entry_price:.2f}",
            'Exit Price': f"${trade.exit_price:.2f}",
            'Return %': f"{trade.return_pct:.2%}",
            'P&L': f"${trade.pnl_net:.2f}",
            'Hold Time (h)': f"{trade.hold_time_hours:.1f}",
            'Exit Reason': trade.exit_reason,
            'Rank': trade.rank
        })
    
    trade_df = pd.DataFrame(trade_data)
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol_filter = st.multiselect(
            "Filter by Symbol",
            options=sorted(trade_df['Symbol'].unique()),
            default=[]
        )
    
    with col2:
        exit_reason_filter = st.multiselect(
            "Filter by Exit Reason",
            options=sorted(trade_df['Exit Reason'].unique()),
            default=[]
        )
    
    with col3:
        show_only_winners = st.checkbox("Show only winning trades")
    
    # Apply filters
    filtered_df = trade_df.copy()
    
    if symbol_filter:
        filtered_df = filtered_df[filtered_df['Symbol'].isin(symbol_filter)]
    
    if exit_reason_filter:
        filtered_df = filtered_df[filtered_df['Exit Reason'].isin(exit_reason_filter)]
    
    if show_only_winners:
        # Extract numeric value from percentage string for filtering
        filtered_df = filtered_df[filtered_df['Return %'].str.replace('%', '').astype(float) > 0]
    
    # Display table
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Trade Data as CSV",
        data=csv,
        file_name=f"gap_trades_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()