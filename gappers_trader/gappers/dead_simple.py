"""DEAD SIMPLE: Buy biggest movers at open, sell at targets."""

import logging
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

class DeadSimpleTrader:
    """Dead simple momentum trading."""
    
    def __init__(self):
        # Just the most liquid stocks
        self.symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "SPY", "QQQ"]
    
    def run_backtest(self, start_date: datetime, end_date: datetime, params: dict):
        """Dead simple backtest."""
        logger.info(f"Running DEAD SIMPLE backtest from {start_date} to {end_date}")
        
        results = {'trades': [], 'total_return': 0.0, 'win_rate': 0.0, 'num_trades': 0}
        
        try:
            # Get data for all symbols
            data = {}
            for symbol in self.symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_date - timedelta(days=10), end=end_date + timedelta(days=1))
                    if not hist.empty:
                        # Remove timezone to avoid issues
                        hist.index = hist.index.tz_localize(None) if hist.index.tz else hist.index
                        data[symbol] = hist
                        logger.info(f"Got {symbol}: {len(hist)} days")
                except Exception as e:
                    logger.warning(f"Failed {symbol}: {e}")
            
            if not data:
                logger.error("No data available!")
                return results
            
            # Get trading dates
            first_symbol = list(data.keys())[0]
            trading_dates = data[first_symbol].index
            trading_dates = trading_dates[(trading_dates >= start_date) & (trading_dates <= end_date)]
            
            logger.info(f"Processing {len(trading_dates)} days")
            
            total_pnl = 0
            trades = []
            
            for date in trading_dates:
                date_str = date.strftime('%Y-%m-%d')
                
                # Find biggest movers for this day
                movers = []
                for symbol, hist in data.items():
                    try:
                        if date not in hist.index:
                            continue
                            
                        today = hist.loc[date]
                        
                        # Find previous trading day
                        prev_dates = hist.index[hist.index < date]
                        if len(prev_dates) == 0:
                            continue
                        yesterday = hist.loc[prev_dates[-1]]
                        
                        # Calculate overnight move (gap)
                        gap = (today['Open'] - yesterday['Close']) / yesterday['Close']
                        
                        # Calculate intraday move 
                        intraday = (today['Close'] - today['Open']) / today['Open']
                        
                        logger.info(f"{date_str} {symbol}: Gap={gap:.2%}, Intraday={intraday:.2%}, Open=${today['Open']:.2f}")
                        
                        if gap > 0.005:  # Any gap > 0.5%
                            movers.append({
                                'symbol': symbol,
                                'gap': gap,
                                'open': today['Open'],
                                'high': today['High'],
                                'low': today['Low'],
                                'close': today['Close'],
                                'intraday': intraday
                            })
                    except Exception as e:
                        continue
                
                # Sort by gap size and take top 3
                movers.sort(key=lambda x: x['gap'], reverse=True)
                logger.info(f"{date_str}: Found {len(movers)} movers")
                
                for mover in movers[:3]:  # Trade top 3
                    # Simple trade: buy at open
                    entry = mover['open']
                    profit_target = entry * 1.03  # 3% profit target
                    stop_loss = entry * 0.98      # 2% stop loss
                    
                    # Check what happened during the day
                    if mover['high'] >= profit_target:
                        exit_price = profit_target
                        exit_reason = 'profit'
                    elif mover['low'] <= stop_loss:
                        exit_price = stop_loss  
                        exit_reason = 'stop'
                    else:
                        exit_price = mover['close']
                        exit_reason = 'close'
                    
                    # Calculate P&L
                    position_size = 10000
                    shares = position_size / entry
                    pnl = shares * (exit_price - entry)
                    pnl_pct = (exit_price - entry) / entry
                    
                    total_pnl += pnl
                    
                    trade = {
                        'symbol': mover['symbol'],
                        'date': date,
                        'gap_pct': mover['gap'],
                        'entry_price': entry,
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct
                    }
                    trades.append(trade)
                    
                    logger.info(f"TRADE: {mover['symbol']} gap={mover['gap']:.2%} -> {pnl_pct:.2%} (${pnl:.0f}) [{exit_reason}]")
            
            # Create portfolio value timeline
            portfolio_values = []
            running_balance = 100000  # Start with $100k
            
            for date in trading_dates:
                # Add daily P&L
                daily_pnl = sum(t['pnl'] for t in trades if t['date'].date() == date.date())
                running_balance += daily_pnl
                
                portfolio_values.append({
                    'date': date,
                    'portfolio_value': running_balance,
                    'daily_pnl': daily_pnl
                })
            
            portfolio_df = pd.DataFrame(portfolio_values)
            if not portfolio_df.empty:
                portfolio_df.set_index('date', inplace=True)
            
            # Summary
            winning_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0
            total_return_pct = (running_balance - 100000) / 100000 if trades else 0
            
            results.update({
                'trades': trades,
                'portfolio_values': portfolio_df,
                'total_return': total_pnl,
                'total_return_pct': total_return_pct,
                'final_portfolio_value': running_balance,
                'win_rate': win_rate,
                'num_trades': len(trades),
                'avg_return_per_trade': sum(t['pnl_pct'] for t in trades) / len(trades) if trades else 0
            })
            
            logger.info(f"DONE: {len(trades)} trades, {win_rate:.1%} win rate, ${total_pnl:.0f} total P&L")
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            
        return results