"""MT4 Universal Backtesting Framework.

A high-performance backtesting system for trading strategies.
"""

__version__ = "1.0.0"
__author__ = "MT4 Backtester Team"

from .core.engine import Backtester
from .core.strategy import Strategy
from .core.result import BacktestResult
from .data.manager import DataManager
from .optimization.optimizer import Optimizer

__all__ = [
    "Backtester",
    "Strategy",
    "BacktestResult",
    "DataManager",
    "Optimizer",
]