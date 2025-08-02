"""SIMPLE gap trading backtest - no complex vectorbt, just basic logic."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

class SimpleGapTrader:
    """Simple gap trading strategy."""
    
    def __init__(self):
        # Top 30 most liquid stocks only
        self.symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX",
            "JPM", "BAC", "WFC", "V", "MA", "BRK-B",
            "WMT", "HD", "PG", "JNJ", "UNH", "MCD", "DIS", "NKE",
            "XOM", "CVX", "CAT", "VZ", "T", "CMCSA",
            "SPY", "QQQ"
        ]
    
    def run_backtest(self, start_date: datetime, end_date: datetime, params: dict):
        """Run simple backtest."""
        logger.info(f"Running simple backtest from {start_date} to {end_date}")
        
        results = {
            'trades': [],
            'portfolio_values': pd.DataFrame(),
            'total_return': 0.0,
            'win_rate': 0.0,
            'num_trades': 0
        }
        
        try:
            # Get data for all symbols
            data = {}
            for symbol in self.symbols[:10]:  # Just top 10 for speed
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_date - timedelta(days=5), end=end_date + timedelta(days=1))
                    if not hist.empty:
                        data[symbol] = hist
                        logger.info(f"Downloaded {symbol}: {len(hist)} days")
                except Exception as e:
                    logger.warning(f"Failed to get {symbol}: {e}")
            
            # Get all trading dates from the first stock to iterate through
            if not data:
                logger.warning("No data available for backtesting")
                return results
                
            # Use dates from the first symbol as trading calendar
            first_symbol = list(data.keys())[0]
            trading_dates = data[first_symbol].index
            
            # Convert to timezone-naive for comparison
            if trading_dates.tz is not None:
                trading_dates = trading_dates.tz_convert(None)
            
            trading_dates = trading_dates[(trading_dates >= start_date) & (trading_dates <= end_date)]
            
            logger.info(f"Processing {len(trading_dates)} trading days")
            
            total_pnl = 0
            trades = []
            
            for current_date in trading_dates:
                # Find gaps for this day
                gaps = []
                
                for symbol, hist in data.items():
                    try:
                        if current_date in hist.index:
                            current_row = hist.loc[current_date]
                            # Get previous trading day
                            prev_dates = hist.index[hist.index < current_date]
                            if len(prev_dates) > 0:
                                prev_date = prev_dates[-1]
                                prev_row = hist.loc[prev_date]
                                
                                # Calculate gap
                                gap_pct = (current_row['Open'] - prev_row['Close']) / prev_row['Close']
                                
                                # DEBUG: Log all gaps > 0.5%
                                if abs(gap_pct) > 0.005:
                                    logger.info(f"{current_date.date()} {symbol}: {gap_pct:.3%} gap (${prev_row['Close']:.2f} -> ${current_row['Open']:.2f})")
                                
                                min_gap = params.get('min_gap', 0.02)  # Use parameter from UI
                                if gap_pct > min_gap:  # Use dynamic threshold
                                    gaps.append({
                                        'symbol': symbol,
                                        'gap_pct': gap_pct,
                                        'open_price': current_row['Open'],
                                        'high': current_row['High'],
                                        'low': current_row['Low'],
                                        'close': current_row['Close']
                                    })
                                    logger.info(f"QUALIFYING GAP: {symbol} {gap_pct:.3%}")
                    except Exception as e:
                        continue
                
                logger.info(f"{current_date.date()}: Found {len(gaps)} qualifying gaps (min: {params.get('min_gap', 0.02):.1%})")
                
                # Trade top 3 gaps
                gaps.sort(key=lambda x: x['gap_pct'], reverse=True)
                for gap in gaps[:3]:
                    # Simple trade simulation
                    entry_price = gap['open_price']
                    profit_target = entry_price * (1 + params.get('profit_target', 0.05))
                    stop_loss = entry_price * (1 - params.get('stop_loss', 0.02))
                    
                    # Check if profit target or stop hit during the day
                    if gap['high'] >= profit_target:
                        exit_price = profit_target
                        exit_reason = 'profit_target'
                    elif gap['low'] <= stop_loss:
                        exit_price = stop_loss
                        exit_reason = 'stop_loss'
                    else:
                        exit_price = gap['close']
                        exit_reason = 'time_exit'
                    
                    # Calculate P&L
                    position_size = params.get('position_size', 10000)
                    shares = position_size / entry_price
                    pnl = shares * (exit_price - entry_price)
                    pnl_pct = (exit_price - entry_price) / entry_price
                    
                    total_pnl += pnl
                    
                    trades.append({
                        'symbol': gap['symbol'],
                        'date': current_date,
                        'gap_pct': gap['gap_pct'],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct
                    })
                    
                    logger.info(f"TRADE: {gap['symbol']} {gap['gap_pct']:.3%} gap -> {pnl_pct:.2%} return (${pnl:.2f})")
            
            # Calculate results
            winning_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0
            
            results.update({
                'trades': trades,
                'total_return': total_pnl,
                'win_rate': win_rate,
                'num_trades': len(trades),
                'avg_return_per_trade': sum(t['pnl_pct'] for t in trades) / len(trades) if trades else 0
            })
            
            logger.info(f"Backtest complete: {len(trades)} trades, {win_rate:.1%} win rate, ${total_pnl:.2f} total P&L")
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            
        return results