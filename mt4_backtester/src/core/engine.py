"""Core backtesting engine with universal architecture.

This module provides the main backtesting engine that processes
tick/bar data through strategies while maintaining strict point-in-time
data access to prevent look-ahead bias.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from numba import jit

from ..data.manager import DataManager, MarketState
from ..indicators.registry import IndicatorRegistry
from .order_manager import OrderManager
from .result import BacktestResult, TradeRecord
from .strategy import Strategy


@dataclass
class EngineConfig:
    """Configuration for the backtesting engine."""

    initial_balance: float = 10000.0
    leverage: int = 100
    commission: float = 7.0  # USD per trade
    slippage_points: int = 3
    max_spread_points: int = 50
    use_tick_data: bool = True
    enable_multi_timeframe: bool = True
    cache_indicators: bool = True
    point_in_time: bool = True  # Prevent look-ahead bias
    parallel_processing: bool = False
    log_level: str = "INFO"


class Backtester:
    """High-performance universal backtesting engine.

    This engine processes historical data through trading strategies
    while maintaining realistic market conditions and preventing
    look-ahead bias through point-in-time data access.

    Attributes:
        config: Engine configuration
        data_manager: Handles all data operations
        indicator_registry: Manages indicator calculations
        order_manager: Handles order execution and position tracking
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        """Initialize the backtesting engine.

        Args:
            config: Engine configuration, uses defaults if None
        """
        self.config = config or EngineConfig()
        self.data_manager = DataManager()
        self.indicator_registry = IndicatorRegistry()
        self.order_manager = OrderManager(
            initial_balance=self.config.initial_balance,
            leverage=self.config.leverage,
            commission=self.config.commission,
        )

        # Performance metrics
        self._tick_count = 0
        self._start_time = 0.0

        # Setup logging
        logger.remove()
        logger.add(
            lambda msg: print(msg, end=""),
            level=self.config.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        )

    def run(
        self,
        strategy: Strategy,
        symbol: str,
        start: Union[str, datetime],
        end: Union[str, datetime],
        data_source: str = "auto",
        timeframe: Optional[str] = None,
    ) -> BacktestResult:
        """Run a backtest for the given strategy.

        Args:
            strategy: Trading strategy to test
            symbol: Trading symbol (e.g., 'EURUSD')
            start: Start date for backtest
            end: End date for backtest
            data_source: 'tick', 'bar', or 'auto'
            timeframe: Primary timeframe if using bar data

        Returns:
            BacktestResult containing performance metrics and trade history
        """
        logger.info(f"Starting backtest for {symbol} from {start} to {end}")
        self._start_time = time.time()

        # Load data
        data = self.data_manager.load_data(
            symbol=symbol,
            start=start,
            end=end,
            data_source=data_source,
            timeframe=timeframe,
        )

        # Pre-calculate indicators if caching is enabled
        if self.config.cache_indicators:
            self._precalculate_indicators(strategy, data)

        # Initialize strategy
        strategy.initialize(
            symbol=symbol,
            initial_balance=self.config.initial_balance,
        )

        # Process data through strategy
        if self.config.use_tick_data and data.has_ticks:
            result = self._process_tick_data(strategy, data)
        else:
            result = self._process_bar_data(strategy, data)

        # Calculate final metrics
        result = self._finalize_result(result, strategy)

        elapsed = time.time() - self._start_time
        logger.info(
            f"Backtest completed in {elapsed:.2f}s "
            f"({self._tick_count / elapsed:.0f} ticks/sec)"
        )

        return result

    def _process_tick_data(
        self, strategy: Strategy, data: Any
    ) -> BacktestResult:
        """Process tick-by-tick data through the strategy.

        Args:
            strategy: Trading strategy
            data: Market data

        Returns:
            Backtest result with partial metrics
        """
        result = BacktestResult(
            symbol=data.symbol,
            start_date=data.start_date,
            end_date=data.end_date,
        )

        for tick_index, tick in enumerate(data.ticks):
            self._tick_count += 1

            # Create point-in-time market state
            market_state = self._create_market_state(
                data, tick_index, tick.timestamp
            )

            # Update order manager with current prices
            self.order_manager.update_prices(tick.bid, tick.ask)

            # Process pending orders
            fills = self.order_manager.check_pending_orders(tick)
            for fill in fills:
                strategy.on_order_filled(fill)
                result.trades.append(fill)

            # Check stop loss and take profit
            closes = self.order_manager.check_stops(tick)
            for close in closes:
                strategy.on_position_closed(close)
                result.trades.append(close)

            # Execute strategy logic
            strategy.on_tick(tick, market_state)

            # Process any new orders from strategy
            self._process_strategy_orders(strategy, tick)

            # Update equity curve
            equity = self.order_manager.get_equity(tick.bid, tick.ask)
            result.equity_curve.append((tick.timestamp, equity))

            # Check for equity stop
            if self._check_equity_stop(equity):
                logger.warning(f"Equity stop triggered at {equity:.2f}")
                break

        return result

    def _process_bar_data(
        self, strategy: Strategy, data: Any
    ) -> BacktestResult:
        """Process bar data through the strategy.

        Args:
            strategy: Trading strategy
            data: Market data

        Returns:
            Backtest result with partial metrics
        """
        result = BacktestResult(
            symbol=data.symbol,
            start_date=data.start_date,
            end_date=data.end_date,
        )

        for bar_index, bar in enumerate(data.bars):
            # Simulate intra-bar movement for realistic fills
            ticks = self._simulate_bar_ticks(bar)

            for tick in ticks:
                self._tick_count += 1

                market_state = self._create_market_state(
                    data, bar_index, bar.timestamp
                )

                # Process similar to tick data
                self.order_manager.update_prices(tick.bid, tick.ask)

                fills = self.order_manager.check_pending_orders(tick)
                for fill in fills:
                    strategy.on_order_filled(fill)
                    result.trades.append(fill)

                closes = self.order_manager.check_stops(tick)
                for close in closes:
                    strategy.on_position_closed(close)
                    result.trades.append(close)

            # Call strategy's on_bar method
            strategy.on_bar(data.timeframe, bar, market_state)

            # Process orders
            self._process_strategy_orders(strategy, ticks[-1])

            # Update equity
            equity = self.order_manager.get_equity(bar.close, bar.close)
            result.equity_curve.append((bar.timestamp, equity))

            if self._check_equity_stop(equity):
                break

        return result

    def _precalculate_indicators(
        self, strategy: Strategy, data: Any
    ) -> None:
        """Pre-calculate all indicators for performance.

        This calculates indicators once upfront for speed, but ensures
        point-in-time access during backtesting to prevent look-ahead bias.

        Args:
            strategy: Trading strategy
            data: Market data
        """
        logger.info("Pre-calculating indicators for optimal performance...")

        required_indicators = strategy.get_required_indicators()

        for indicator_spec in required_indicators:
            indicator_type, timeframe, params = indicator_spec

            # Get or aggregate data to required timeframe
            tf_data = self.data_manager.get_timeframe_data(
                data, timeframe
            )

            # Calculate indicator
            indicator_values = self.indicator_registry.calculate(
                indicator_type, tf_data, params
            )

            # Store in point-in-time cache
            self.indicator_registry.cache_indicator(
                symbol=data.symbol,
                timeframe=timeframe,
                indicator_type=indicator_type,
                params=params,
                values=indicator_values,
            )

    def _create_market_state(
        self, data: Any, index: int, timestamp: datetime
    ) -> MarketState:
        """Create point-in-time market state for strategy.

        Args:
            data: Market data
            index: Current data index
            timestamp: Current timestamp

        Returns:
            Market state with point-in-time data access
        """
        return MarketState(
            timestamp=timestamp,
            data=data,
            current_index=index,
            indicator_cache=self.indicator_registry.cache,
            point_in_time=self.config.point_in_time,
        )

    def _simulate_bar_ticks(self, bar: Any) -> List[Any]:
        """Simulate realistic tick movement within a bar.

        Creates a realistic path through OHLC values to ensure
        proper order fill simulation.

        Args:
            bar: Bar data

        Returns:
            List of simulated ticks
        """
        from ..data.tick import Tick

        ticks = []
        spread = self.data_manager.get_typical_spread(bar.symbol)

        # Create realistic price path: Open -> High/Low -> Close
        if bar.close > bar.open:  # Bullish bar
            prices = [bar.open, bar.low, bar.high, bar.close]
        else:  # Bearish bar
            prices = [bar.open, bar.high, bar.low, bar.close]

        for price in prices:
            tick = Tick(
                timestamp=bar.timestamp,
                bid=price - spread / 2,
                ask=price + spread / 2,
                volume=bar.volume / 4,
            )
            ticks.append(tick)

        return ticks

    def _process_strategy_orders(
        self, strategy: Strategy, current_tick: Any
    ) -> None:
        """Process any pending orders from the strategy.

        Args:
            strategy: Trading strategy
            current_tick: Current market tick
        """
        while strategy.has_pending_orders():
            order = strategy.get_next_order()

            # Validate order
            if not self._validate_order(order):
                logger.warning(f"Invalid order rejected: {order}")
                continue

            # Submit to order manager
            if order.is_market_order:
                fill = self.order_manager.execute_market_order(
                    order, current_tick
                )
                strategy.on_order_filled(fill)
            else:
                self.order_manager.add_pending_order(order)

    def _validate_order(self, order: Any) -> bool:
        """Validate an order before execution.

        Args:
            order: Order to validate

        Returns:
            True if order is valid
        """
        # Check lot size
        if order.volume <= 0:
            return False

        # Check margin requirements
        required_margin = self.order_manager.calculate_required_margin(
            order
        )
        if required_margin > self.order_manager.free_margin:
            logger.warning("Insufficient margin for order")
            return False

        return True

    def _check_equity_stop(self, equity: float) -> bool:
        """Check if equity stop has been triggered.

        Args:
            equity: Current account equity

        Returns:
            True if equity stop triggered
        """
        drawdown_pct = (
            (self.config.initial_balance - equity)
            / self.config.initial_balance
            * 100
        )
        return drawdown_pct > 50  # 50% drawdown stop

    def _finalize_result(
        self, result: BacktestResult, strategy: Strategy
    ) -> BacktestResult:
        """Calculate final performance metrics.

        Args:
            result: Partial backtest result
            strategy: Trading strategy

        Returns:
            Complete backtest result with all metrics
        """
        if not result.trades:
            logger.warning("No trades executed during backtest")
            return result

        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame([t.__dict__ for t in result.trades])

        # Calculate metrics
        result.total_trades = len(trades_df)
        result.winning_trades = len(trades_df[trades_df['pnl'] > 0])
        result.losing_trades = len(trades_df[trades_df['pnl'] < 0])

        if result.total_trades > 0:
            result.win_rate = result.winning_trades / result.total_trades

        result.gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        result.gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        result.net_profit = result.gross_profit - result.gross_loss

        if result.gross_loss > 0:
            result.profit_factor = result.gross_profit / result.gross_loss

        # Calculate max drawdown
        equity_array = np.array([e[1] for e in result.equity_curve])
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (running_max - equity_array) / running_max * 100
        result.max_drawdown = np.max(drawdown)

        # Calculate Sharpe ratio
        if len(trades_df) > 1:
            returns = trades_df['pnl'] / self.config.initial_balance
            result.sharpe_ratio = (
                np.mean(returns) / np.std(returns) * np.sqrt(252)
            )

        # Add strategy parameters to result
        result.strategy_params = strategy.get_parameters()

        return result


@jit(nopython=True)
def _calculate_drawdown_numba(equity_curve: np.ndarray) -> float:
    """Calculate maximum drawdown using Numba for speed.

    Args:
        equity_curve: Array of equity values

    Returns:
        Maximum drawdown percentage
    """
    running_max = equity_curve[0]
    max_dd = 0.0

    for value in equity_curve:
        if value > running_max:
            running_max = value

        dd = (running_max - value) / running_max * 100
        if dd > max_dd:
            max_dd = dd

    return max_dd