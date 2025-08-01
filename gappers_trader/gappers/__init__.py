"""Production-grade overnight gap trading system."""

__version__ = "0.1.0"
__author__ = "GapRunner Team"

from gappers.analytics import PerformanceAnalyzer
from gappers.backtest import Backtester, GapParams
from gappers.datafeed import DataFeed
from gappers.live import LiveTrader
from gappers.signals import SignalGenerator
from gappers.universe import UniverseBuilder

__all__ = [
    "DataFeed",
    "UniverseBuilder",
    "SignalGenerator",
    "Backtester",
    "GapParams",
    "LiveTrader",
    "PerformanceAnalyzer",
]