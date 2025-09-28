"""Backtest result and trade record data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    """Record of a single trade."""

    ticket: int
    symbol: str
    order_type: str
    open_time: datetime
    close_time: datetime
    open_price: float
    close_price: float
    volume: float
    commission: float
    swap: float
    pnl: float
    comment: str = ""
    magic_number: int = 0


@dataclass
class BacktestResult:
    """Complete backtest results with performance metrics."""

    # Basic info
    symbol: str
    start_date: datetime
    end_date: datetime

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # Financial metrics
    initial_balance: float = 10000.0
    final_balance: float = 10000.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trade records
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)

    # Strategy parameters
    strategy_params: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame for analysis."""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                'ticket': t.ticket,
                'symbol': t.symbol,
                'type': t.order_type,
                'open_time': t.open_time,
                'close_time': t.close_time,
                'open_price': t.open_price,
                'close_price': t.close_price,
                'volume': t.volume,
                'commission': t.commission,
                'swap': t.swap,
                'pnl': t.pnl,
                'comment': t.comment,
                'magic': t.magic_number,
            }
            for t in self.trades
        ])

    def get_equity_curve_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()

        df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        df.set_index('timestamp', inplace=True)
        return df

    def calculate_metrics(self) -> None:
        """Calculate all performance metrics."""
        if not self.trades:
            return

        # Calculate basic metrics
        self.final_balance = self.initial_balance + self.net_profit

        # Calculate Sharpe ratio
        if len(self.trades) > 1:
            trade_returns = [t.pnl / self.initial_balance for t in self.trades]
            if np.std(trade_returns) > 0:
                self.sharpe_ratio = (
                    np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252)
                )

        # Calculate maximum drawdown from equity curve
        if self.equity_curve:
            equity_values = np.array([e[1] for e in self.equity_curve])
            running_max = np.maximum.accumulate(equity_values)
            drawdown = (running_max - equity_values) / running_max * 100
            self.max_drawdown = np.max(drawdown)

    def summary(self) -> str:
        """Generate a text summary of results."""
        return f"""
Backtest Results Summary
========================
Symbol: {self.symbol}
Period: {self.start_date.date()} to {self.end_date.date()}

Performance Metrics:
-------------------
Net Profit: ${self.net_profit:.2f}
Total Trades: {self.total_trades}
Win Rate: {self.win_rate:.1%}
Profit Factor: {self.profit_factor:.2f}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Max Drawdown: {self.max_drawdown:.1%}

Trade Statistics:
----------------
Winning Trades: {self.winning_trades}
Losing Trades: {self.losing_trades}
Gross Profit: ${self.gross_profit:.2f}
Gross Loss: ${self.gross_loss:.2f}

Final Balance: ${self.final_balance:.2f}
Return: {((self.final_balance - self.initial_balance) / self.initial_balance * 100):.1f}%
"""