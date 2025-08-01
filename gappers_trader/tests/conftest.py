"""Pytest configuration and shared fixtures."""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from gappers import DataFeed, GapParams, SignalGenerator, UniverseBuilder
from gappers.backtest import TradeResult


@pytest.fixture(scope="session")
def temp_data_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_config(temp_data_dir, monkeypatch):
    """Mock configuration for testing."""
    monkeypatch.setenv("DATA_PATH", str(temp_data_dir))
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("CACHE_EXPIRY_HOURS", "1")
    
    # Import config after setting environment variables
    from gappers.config import Config
    return Config()


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    def create_symbol_data(symbol: str, base_price: float = 100.0):
        np.random.seed(hash(symbol) % 2**32)  # Deterministic but different per symbol
        
        # Generate realistic price movements
        returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% daily return, 2% volatility
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Add some randomness to OHLC
        open_prices = prices * (1 + np.random.normal(0, 0.005, len(dates)))
        high_prices = np.maximum(open_prices, prices) * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
        low_prices = np.minimum(open_prices, prices) * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
        close_prices = prices
        
        # Volume with some correlation to price movements
        volumes = np.random.lognormal(12, 0.5, len(dates))  # ~160k average volume
        volumes = volumes * (1 + 0.5 * np.abs(returns))  # Higher volume on big moves
        
        return pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes.astype(int),
            'symbol': symbol
        }, index=dates)
    
    return {
        'AAPL': create_symbol_data('AAPL', 150.0),
        'MSFT': create_symbol_data('MSFT', 250.0),
        'GOOGL': create_symbol_data('GOOGL', 2500.0),
        'TSLA': create_symbol_data('TSLA', 200.0),
        'SPY': create_symbol_data('SPY', 400.0),
    }


@pytest.fixture
def sample_intraday_data():
    """Generate sample intraday (1-minute) data for testing."""
    # Single trading day
    date = datetime(2023, 6, 15)
    
    # Market hours: 9:30 AM to 4:00 PM ET (390 minutes)
    start_time = date.replace(hour=9, minute=30)
    end_time = date.replace(hour=16, minute=0)
    
    minutes = pd.date_range(start_time, end_time, freq='1min')
    
    def create_intraday_symbol(symbol: str, gap_pct: float = 0.03):
        np.random.seed(hash(symbol) % 2**32)
        
        # Start with gap
        prev_close = 100.0
        open_price = prev_close * (1 + gap_pct)
        
        # Random walk for intraday
        returns = np.random.normal(0, 0.001, len(minutes))  # Small intraday moves
        prices = open_price * np.exp(np.cumsum(returns))
        
        # Create OHLC bars
        data = []
        for i, (minute, price) in enumerate(zip(minutes, prices)):
            if i == 0:
                open_bar = open_price
            else:
                open_bar = prices[i-1]  # Previous close
            
            close_bar = price
            high_bar = max(open_bar, close_bar) * (1 + np.abs(np.random.normal(0, 0.002)))
            low_bar = min(open_bar, close_bar) * (1 - np.abs(np.random.normal(0, 0.002)))
            volume_bar = np.random.randint(100, 1000)
            
            data.append({
                'open': open_bar,
                'high': high_bar,
                'low': low_bar,
                'close': close_bar,
                'volume': volume_bar,
                'symbol': symbol
            })
        
        return pd.DataFrame(data, index=minutes)
    
    return {
        'AAPL': create_intraday_symbol('AAPL', 0.05),
        'TSLA': create_intraday_symbol('TSLA', 0.08),
        'SPY': create_intraday_symbol('SPY', 0.02),
    }


@pytest.fixture
def sample_gap_data():
    """Generate sample gap data for testing."""
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX']
    date = datetime(2023, 6, 15)
    
    gaps = []
    for i, symbol in enumerate(symbols):
        gap_pct = 0.08 - (i * 0.01)  # Decreasing gaps
        previous_close = 100.0 + i * 50
        current_open = previous_close * (1 + gap_pct)
        
        gaps.append({
            'symbol': symbol,
            'date': date.strftime('%Y-%m-%d'),
            'gap_pct': gap_pct,
            'gap_dollars': current_open - previous_close,
            'previous_close': previous_close,
            'current_open': current_open,
            'current_high': current_open * 1.02,
            'current_low': current_open * 0.98,
            'current_close': current_open * 1.01,
            'previous_volume': 1000000,
            'current_volume': 1500000,
            'volume_ratio': 1.5,
            'atr_14': previous_close * 0.02,
            'rsi_14': 60.0,
            'sector': 'Technology',
            'rank': i + 1
        })
    
    return pd.DataFrame(gaps)


@pytest.fixture
def sample_trades():
    """Generate sample trade results for testing."""
    trades = []
    base_date = datetime(2023, 6, 15, 9, 30)
    
    for i in range(20):
        entry_date = base_date + timedelta(days=i)
        hold_hours = np.random.uniform(1, 6)
        exit_date = entry_date + timedelta(hours=hold_hours)
        
        entry_price = 100 + np.random.uniform(-10, 10)
        return_pct = np.random.normal(0.02, 0.05)  # 2% average return, 5% volatility
        exit_price = entry_price * (1 + return_pct)
        
        shares = 100
        pnl_gross = (exit_price - entry_price) * shares
        commission = 0.005 * shares * 2  # Buy and sell
        pnl_net = pnl_gross - commission
        
        exit_reasons = ['profit_target', 'stop_loss', 'time_limit', 'eod']
        exit_reason = np.random.choice(exit_reasons)
        
        trade = TradeResult(
            symbol=f'SYM{i:02d}',
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            position_size=shares,
            pnl_gross=pnl_gross,
            pnl_net=pnl_net,
            return_pct=return_pct,
            hold_time_hours=hold_hours,
            exit_reason=exit_reason,
            gap_pct=np.random.uniform(0.02, 0.08),
            rank=i + 1,
        )
        trades.append(trade)
    
    return trades


@pytest.fixture
def default_gap_params():
    """Default gap trading parameters for testing."""
    return GapParams(
        profit_target=0.05,
        stop_loss=0.02,
        max_hold_time_hours=6,
        top_k=10,
        min_gap_pct=0.02,
        max_gap_pct=0.30,
        position_size=10000,
        max_positions=10,
        sector_diversification=True,
        max_per_sector=3,
        commission_per_share=0.005,
        slippage_bps=10,
    )


@pytest.fixture
def mock_data_feed(sample_ohlcv_data, sample_intraday_data):
    """Mock data feed that returns sample data."""
    class MockDataFeed:
        def __init__(self):
            self.daily_data = sample_ohlcv_data
            self.intraday_data = sample_intraday_data
        
        def download(self, symbols, start, end, interval='1d', **kwargs):
            if interval == '1d':
                return {symbol: df.loc[start:end] for symbol, df in self.daily_data.items() 
                       if symbol in symbols and not df.loc[start:end].empty}
            elif interval == '1m':
                return {symbol: df for symbol, df in self.intraday_data.items() 
                       if symbol in symbols}
            else:
                return {}
        
        def get_splits_dividends(self, symbol, start, end):
            return {"splits": pd.DataFrame(), "dividends": pd.DataFrame()}
        
        def clear_cache(self, older_than_days=None):
            return 0
    
    return MockDataFeed()


@pytest.fixture
def mock_universe_builder(mock_data_feed):
    """Mock universe builder for testing."""
    class MockUniverseBuilder:
        def __init__(self, data_feed):
            self.data_feed = data_feed
        
        def build_universe(self, date=None, **kwargs):
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
            return pd.DataFrame({
                'symbol': symbols,
                'price': [150, 250, 2500, 200, 400],
                'median_dollar_volume': [1e9, 8e8, 5e8, 1.2e9, 2e9],
                'avg_volume': [50e6, 30e6, 1e6, 25e6, 100e6],
                'date': [date.strftime('%Y-%m-%d')] * len(symbols) if date else ['2023-06-15'] * len(symbols),
                'sector': ['Technology'] * len(symbols),
                'market_cap': [3e12, 2e12, 1.5e12, 800e9, 0],  # SPY is ETF
                'float_shares': [16e9, 7.5e9, 13e9, 3.2e9, 900e6]
            })
    
    return MockUniverseBuilder(mock_data_feed)


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Clean up matplotlib figures after each test."""
    yield
    # Clean up any matplotlib figures
    import matplotlib.pyplot as plt
    plt.close('all')