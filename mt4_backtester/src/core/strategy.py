"""Base strategy class and interfaces for trading strategies.

This module provides the abstract base class that all trading strategies
must inherit from, ensuring a consistent interface for the backtesting engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger


class OrderType(Enum):
    """Order type enumeration."""

    BUY = "BUY"
    SELL = "SELL"
    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"
    BUY_STOP = "BUY_STOP"
    SELL_STOP = "SELL_STOP"


@dataclass
class Order:
    """Represents a trading order."""

    order_type: OrderType
    volume: float
    price: Optional[float] = None  # None for market orders
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    magic_number: int = 0
    comment: str = ""
    timestamp: Optional[datetime] = None

    @property
    def is_market_order(self) -> bool:
        """Check if this is a market order."""
        return self.order_type in [OrderType.BUY, OrderType.SELL]

    @property
    def is_pending_order(self) -> bool:
        """Check if this is a pending order."""
        return not self.is_market_order


@dataclass
class Position:
    """Represents an open position."""

    ticket: int
    symbol: str
    order_type: OrderType
    volume: float
    open_price: float
    open_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    commission: float = 0.0
    swap: float = 0.0
    profit: float = 0.0
    magic_number: int = 0
    comment: str = ""


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    All trading strategies must inherit from this class and implement
    the required abstract methods. The strategy receives market data
    and indicators through the market_state object, which ensures
    point-in-time access to prevent look-ahead bias.

    Attributes:
        symbol: Trading symbol
        balance: Current account balance
        equity: Current account equity
        positions: List of open positions
        pending_orders: Queue of orders to be processed
    """

    def __init__(self):
        """Initialize the strategy."""
        self.symbol: Optional[str] = None
        self.balance: float = 0.0
        self.equity: float = 0.0
        self.positions: List[Position] = []
        self.pending_orders: List[Order] = []
        self.order_history: List[Order] = []
        self.trade_count: int = 0
        self._initialized: bool = False

    def initialize(self, symbol: str, initial_balance: float) -> None:
        """Initialize strategy with trading parameters.

        Called once before the backtest starts.

        Args:
            symbol: Trading symbol
            initial_balance: Starting account balance
        """
        self.symbol = symbol
        self.balance = initial_balance
        self.equity = initial_balance
        self._initialized = True
        logger.info(
            f"Strategy initialized for {symbol} with balance ${initial_balance:.2f}"
        )

        # Call user's custom initialization
        self.on_init()

    @abstractmethod
    def on_init(self) -> None:
        """Custom initialization logic.

        Override this method to perform any strategy-specific
        initialization such as setting parameters or loading models.
        """
        pass

    @abstractmethod
    def on_tick(self, tick: Any, market: Any) -> None:
        """Process a market tick.

        This is the main method where trading logic is implemented.
        Called for every tick when using tick data.

        Args:
            tick: Current market tick with bid/ask prices
            market: Market state with access to indicators and data
        """
        pass

    def on_bar(self, timeframe: str, bar: Any, market: Any) -> None:
        """Process a completed bar.

        Called when a new bar is completed. Override this method
        if your strategy operates on bar data rather than ticks.

        Args:
            timeframe: Timeframe of the bar (e.g., 'M5', 'H1')
            bar: Completed bar with OHLCV data
            market: Market state with access to indicators
        """
        pass

    def on_order_filled(self, order: Order) -> None:
        """Handle order fill event.

        Called when an order is filled by the broker.

        Args:
            order: Filled order details
        """
        self.trade_count += 1
        logger.debug(f"Order filled: {order}")

    def on_position_closed(self, position: Position) -> None:
        """Handle position close event.

        Called when a position is closed (by stop loss, take profit, or manual close).

        Args:
            position: Closed position details
        """
        logger.debug(
            f"Position closed: {position.ticket} P&L: ${position.profit:.2f}"
        )

    @abstractmethod
    def get_required_indicators(self) -> List[Tuple[str, str, Dict]]:
        """Get list of required indicators for pre-calculation.

        Returns a list of tuples containing:
        - Indicator type (e.g., 'SMA', 'RSI')
        - Timeframe (e.g., 'H1', 'M15')
        - Parameters dict (e.g., {'period': 200})

        Example:
            return [
                ('SMA', 'H1', {'period': 200}),
                ('RSI', 'M15', {'period': 14}),
                ('MACD', 'H4', {'fast': 12, 'slow': 26, 'signal': 9})
            ]

        Returns:
            List of indicator specifications
        """
        pass

    def get_optimization_params(self) -> Dict[str, Any]:
        """Get parameter ranges for optimization.

        Override this method to specify which parameters can be
        optimized and their valid ranges.

        Example:
            return {
                'ma_period': range(50, 500, 50),
                'rsi_period': range(10, 30, 2),
                'stop_loss_pips': range(10, 100, 10),
            }

        Returns:
            Dictionary of parameter names and their ranges
        """
        return {}

    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters.

        Returns:
            Dictionary of current parameter values
        """
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }

    # Trading Operations

    def buy(
        self,
        volume: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "",
    ) -> None:
        """Open a buy position.

        Args:
            volume: Position size in lots
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Order comment
        """
        order = Order(
            order_type=OrderType.BUY,
            volume=volume,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment,
        )
        self._submit_order(order)

    def sell(
        self,
        volume: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "",
    ) -> None:
        """Open a sell position.

        Args:
            volume: Position size in lots
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Order comment
        """
        order = Order(
            order_type=OrderType.SELL,
            volume=volume,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment,
        )
        self._submit_order(order)

    def buy_limit(
        self,
        volume: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "",
    ) -> None:
        """Place a buy limit order.

        Args:
            volume: Position size in lots
            price: Limit price
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Order comment
        """
        order = Order(
            order_type=OrderType.BUY_LIMIT,
            volume=volume,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment,
        )
        self._submit_order(order)

    def sell_limit(
        self,
        volume: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "",
    ) -> None:
        """Place a sell limit order.

        Args:
            volume: Position size in lots
            price: Limit price
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Order comment
        """
        order = Order(
            order_type=OrderType.SELL_LIMIT,
            volume=volume,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=comment,
        )
        self._submit_order(order)

    def close_all_positions(self) -> None:
        """Close all open positions."""
        for position in self.positions[:]:  # Copy list to avoid modification during iteration
            self.close_position(position.ticket)

    def close_position(self, ticket: int) -> None:
        """Close a specific position.

        Args:
            ticket: Position ticket number
        """
        # This will be handled by the order manager
        # We just mark it for closing
        logger.debug(f"Closing position {ticket}")

    def cancel_all_pending_orders(self) -> None:
        """Cancel all pending orders."""
        self.pending_orders.clear()

    def _submit_order(self, order: Order) -> None:
        """Submit an order to the queue.

        Args:
            order: Order to submit
        """
        if not self._initialized:
            raise RuntimeError("Strategy not initialized")

        self.pending_orders.append(order)
        self.order_history.append(order)

    def has_pending_orders(self) -> bool:
        """Check if there are pending orders to process.

        Returns:
            True if there are pending orders
        """
        return len(self.pending_orders) > 0

    def get_next_order(self) -> Optional[Order]:
        """Get the next order from the queue.

        Returns:
            Next order or None if queue is empty
        """
        if self.pending_orders:
            return self.pending_orders.pop(0)
        return None

    def get_open_positions_count(self) -> int:
        """Get the number of open positions.

        Returns:
            Number of open positions
        """
        return len(self.positions)

    def get_position_by_ticket(self, ticket: int) -> Optional[Position]:
        """Get a position by its ticket number.

        Args:
            ticket: Position ticket

        Returns:
            Position if found, None otherwise
        """
        for position in self.positions:
            if position.ticket == ticket:
                return position
        return None

    def update_account_info(
        self, balance: float, equity: float, positions: List[Position]
    ) -> None:
        """Update account information.

        Called by the engine to update account state.

        Args:
            balance: Current balance
            equity: Current equity
            positions: List of open positions
        """
        self.balance = balance
        self.equity = equity
        self.positions = positions