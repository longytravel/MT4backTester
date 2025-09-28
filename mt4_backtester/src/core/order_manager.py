"""Order management and position tracking system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .strategy import Order, OrderType, Position
from .costs import CostModel


@dataclass
class Fill:
    """Record of an order fill."""

    ticket: int
    symbol: str
    order_type: OrderType
    volume: float
    price: float
    timestamp: datetime
    slippage: float = 0.0
    commission: float = 0.0
    comment: str = ""
    magic_number: int = 0


class OrderManager:
    """Manages order execution and position tracking.

    Handles market and pending order execution, position management,
    stop loss/take profit monitoring, and P&L calculation.
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        leverage: int = 100,
        commission: float = 7.0,  # Deprecated, use cost_model
        slippage_points: int = 3,
        symbol: str = 'GBPAUD',
    ):
        """Initialize order manager.

        Args:
            initial_balance: Starting account balance
            leverage: Account leverage
            commission: Commission per trade (deprecated, kept for compatibility)
            slippage_points: Typical slippage in points
            symbol: Trading symbol for cost calculations
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.commission = commission  # Kept for backward compatibility
        self.slippage_points = slippage_points
        self.symbol = symbol

        # Initialize cost model
        self.cost_model = CostModel()

        # Position tracking
        self.positions: List[Position] = []
        self.pending_orders: List[Order] = []
        self.closed_positions: List[Position] = []

        # Trade tracking
        self.next_ticket = 1
        self.total_commission = 0.0
        self.total_swap = 0.0

        # Current prices
        self.current_bid = 0.0
        self.current_ask = 0.0

    def update_prices(self, bid: float, ask: float) -> None:
        """Update current market prices.

        Args:
            bid: Current bid price
            ask: Current ask price
        """
        self.current_bid = bid
        self.current_ask = ask

        # Update floating P&L for open positions
        self._update_position_pnl()

    def execute_market_order(self, order: Order, tick: Any) -> Fill:
        """Execute a market order immediately.

        Args:
            order: Market order to execute
            tick: Current market tick

        Returns:
            Fill record
        """
        # Determine fill price with slippage
        if order.order_type == OrderType.BUY:
            fill_price = tick.ask + self._calculate_slippage()
        else:  # SELL
            fill_price = tick.bid - self._calculate_slippage()

        # Calculate actual commission based on lot size
        symbol = tick.symbol if hasattr(tick, 'symbol') else self.symbol
        costs = self.cost_model.get_costs(symbol)
        actual_commission = costs.calculate_commission(order.volume)

        # Create position
        position = Position(
            ticket=self.next_ticket,
            symbol=symbol,
            order_type=order.order_type,
            volume=order.volume,
            open_price=fill_price,
            open_time=tick.timestamp,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            commission=actual_commission,
            magic_number=order.magic_number,
            comment=order.comment,
        )

        self.positions.append(position)
        self.next_ticket += 1

        # Update balance for commission
        self.balance -= actual_commission
        self.total_commission += actual_commission

        # Create fill record
        fill = Fill(
            ticket=position.ticket,
            symbol=position.symbol,
            order_type=order.order_type,
            volume=order.volume,
            price=fill_price,
            timestamp=tick.timestamp,
            slippage=self.slippage_points * self._get_point_value(),
            commission=actual_commission,
            comment=order.comment,
            magic_number=order.magic_number,
        )

        logger.debug(
            f"Executed {order.order_type.value} order: "
            f"{order.volume} lots at {fill_price:.5f}"
        )

        return fill

    def add_pending_order(self, order: Order) -> None:
        """Add a pending order.

        Args:
            order: Pending order to add
        """
        if not order.is_pending_order:
            raise ValueError("Not a pending order")

        self.pending_orders.append(order)
        logger.debug(f"Added pending {order.order_type.value} at {order.price:.5f}")

    def check_pending_orders(self, tick: Any) -> List[Fill]:
        """Check if any pending orders should be filled.

        Args:
            tick: Current market tick

        Returns:
            List of fills for triggered orders
        """
        fills = []
        remaining_orders = []

        for order in self.pending_orders:
            if self._should_fill_pending(order, tick):
                # Convert to position
                position = Position(
                    ticket=self.next_ticket,
                    symbol=tick.symbol if hasattr(tick, 'symbol') else '',
                    order_type=OrderType.BUY if 'BUY' in order.order_type.value else OrderType.SELL,
                    volume=order.volume,
                    open_price=order.price,
                    open_time=tick.timestamp,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    commission=self.commission,
                    magic_number=order.magic_number,
                    comment=order.comment,
                )

                self.positions.append(position)
                self.next_ticket += 1

                # Update balance
                self.balance -= self.commission
                self.total_commission += self.commission

                # Create fill
                fill = Fill(
                    ticket=position.ticket,
                    symbol=position.symbol,
                    order_type=position.order_type,
                    volume=order.volume,
                    price=order.price,
                    timestamp=tick.timestamp,
                    commission=self.commission,
                    comment=order.comment,
                    magic_number=order.magic_number,
                )
                fills.append(fill)

                logger.debug(f"Filled pending order at {order.price:.5f}")
            else:
                remaining_orders.append(order)

        self.pending_orders = remaining_orders
        return fills

    def check_stops(self, tick: Any) -> List[Position]:
        """Check stop loss and take profit levels.

        Args:
            tick: Current market tick

        Returns:
            List of closed positions
        """
        closed = []
        remaining = []

        for position in self.positions:
            should_close = False
            close_price = 0.0

            if position.order_type == OrderType.BUY:
                # Check stop loss
                if position.stop_loss and tick.bid <= position.stop_loss:
                    should_close = True
                    close_price = tick.bid
                # Check take profit
                elif position.take_profit and tick.bid >= position.take_profit:
                    should_close = True
                    close_price = tick.bid
            else:  # SELL
                # Check stop loss
                if position.stop_loss and tick.ask >= position.stop_loss:
                    should_close = True
                    close_price = tick.ask
                # Check take profit
                elif position.take_profit and tick.ask <= position.take_profit:
                    should_close = True
                    close_price = tick.ask

            if should_close:
                # Calculate P&L
                if position.order_type == OrderType.BUY:
                    pnl = (close_price - position.open_price) * position.volume * self._get_lot_size()
                else:
                    pnl = (position.open_price - close_price) * position.volume * self._get_lot_size()

                position.profit = pnl - position.commission
                self.balance += pnl

                closed.append(position)
                self.closed_positions.append(position)

                logger.debug(f"Closed position {position.ticket} at {close_price:.5f}, P&L: ${pnl:.2f}")
            else:
                remaining.append(position)

        self.positions = remaining
        return closed

    def close_position(self, ticket: int, price: float) -> Optional[Position]:
        """Close a specific position.

        Args:
            ticket: Position ticket to close
            price: Closing price

        Returns:
            Closed position or None if not found
        """
        for i, position in enumerate(self.positions):
            if position.ticket == ticket:
                # Calculate P&L
                if position.order_type == OrderType.BUY:
                    pnl = (price - position.open_price) * position.volume * self._get_lot_size()
                else:
                    pnl = (position.open_price - price) * position.volume * self._get_lot_size()

                position.profit = pnl - position.commission
                self.balance += pnl

                # Remove from open positions
                closed = self.positions.pop(i)
                self.closed_positions.append(closed)

                return closed

        return None

    def get_equity(self, bid: float, ask: float) -> float:
        """Calculate current account equity.

        Args:
            bid: Current bid price
            ask: Current ask price

        Returns:
            Current equity value
        """
        equity = self.balance

        # Add floating P&L
        for position in self.positions:
            if position.order_type == OrderType.BUY:
                floating_pnl = (bid - position.open_price) * position.volume * self._get_lot_size()
            else:
                floating_pnl = (position.open_price - ask) * position.volume * self._get_lot_size()

            equity += floating_pnl

        return equity

    @property
    def free_margin(self) -> float:
        """Calculate free margin available for trading.

        Returns:
            Free margin amount
        """
        used_margin = sum(
            self.calculate_required_margin_for_position(p)
            for p in self.positions
        )
        return self.balance * self.leverage - used_margin

    def calculate_required_margin(self, order: Order) -> float:
        """Calculate margin required for an order.

        Args:
            order: Order to check

        Returns:
            Required margin amount
        """
        lot_value = order.volume * self._get_lot_size()
        return lot_value / self.leverage

    def calculate_required_margin_for_position(self, position: Position) -> float:
        """Calculate margin used by a position.

        Args:
            position: Position to check

        Returns:
            Used margin amount
        """
        lot_value = position.volume * self._get_lot_size()
        return lot_value / self.leverage

    def _should_fill_pending(self, order: Order, tick: Any) -> bool:
        """Check if a pending order should be filled.

        Args:
            order: Pending order
            tick: Current tick

        Returns:
            True if order should be filled
        """
        if order.order_type == OrderType.BUY_LIMIT:
            return tick.ask <= order.price
        elif order.order_type == OrderType.SELL_LIMIT:
            return tick.bid >= order.price
        elif order.order_type == OrderType.BUY_STOP:
            return tick.ask >= order.price
        elif order.order_type == OrderType.SELL_STOP:
            return tick.bid <= order.price

        return False

    def _update_position_pnl(self) -> None:
        """Update floating P&L for all open positions."""
        for position in self.positions:
            if position.order_type == OrderType.BUY:
                position.profit = (
                    (self.current_bid - position.open_price) *
                    position.volume * self._get_lot_size() -
                    position.commission
                )
            else:
                position.profit = (
                    (position.open_price - self.current_ask) *
                    position.volume * self._get_lot_size() -
                    position.commission
                )

    def _calculate_slippage(self) -> float:
        """Calculate slippage amount.

        Returns:
            Slippage in price units
        """
        return self.slippage_points * self._get_point_value()

    def _get_point_value(self) -> float:
        """Get point value for the current symbol.

        Returns:
            Point value (e.g., 0.00001 for 5-digit forex)
        """
        # Simplified - should be based on actual symbol
        return 0.00001

    def _get_lot_size(self) -> float:
        """Get standard lot size.

        Returns:
            Lot size (typically 100,000 for forex)
        """
        return 100000

    def get_open_positions_count(self) -> int:
        """Get number of open positions.

        Returns:
            Number of open positions
        """
        return len(self.positions)

    def get_total_volume(self) -> float:
        """Get total volume of open positions.

        Returns:
            Total volume in lots
        """
        return sum(p.volume for p in self.positions)

    def reset(self) -> None:
        """Reset order manager to initial state."""
        self.balance = self.initial_balance
        self.positions.clear()
        self.pending_orders.clear()
        self.closed_positions.clear()
        self.next_ticket = 1
        self.total_commission = 0.0
        self.total_swap = 0.0