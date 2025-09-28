"""FairPrice Grid Trading Strategy implementation.

This example demonstrates how to implement the FairPrice EA using
the universal backtesting framework. It shows how strategies can use
multiple timeframes and indicators dynamically.
"""

from typing import Any, Dict, List, Optional, Tuple

from mt4_backtester import Strategy
from mt4_backtester.core.strategy import Order, OrderType


class FairPriceStrategy(Strategy):
    """FairPrice grid trading strategy with universal indicator support.

    This strategy demonstrates:
    - Dynamic indicator selection (SMA, RSI, MACD, etc.)
    - Multi-timeframe analysis
    - Grid order management
    - Flexible parameter optimization
    """

    def __init__(
        self,
        # Core parameters
        ma_period: int = 200,
        ma_timeframe: str = 'H1',
        lot_size: float = 0.01,
        magic_number: int = 12345,

        # Grid parameters
        initial_trigger_pips: int = 100,
        pending_order_count: int = 10,
        pending_order_range_pips: int = 50,

        # Filter parameters (universal)
        use_filter: bool = True,
        filter_indicator: str = 'SMA',  # Can be SMA, RSI, MACD, ADX, etc.
        filter_timeframe: str = 'H4',
        filter_period: int = 800,
        filter_threshold: float = 50.0,  # For RSI/ADX type indicators

        # Risk management
        equity_stop_percent: float = 5.0,
        close_at_ma: bool = True,
    ):
        """Initialize FairPrice strategy with configurable parameters.

        Args:
            ma_period: Period for fair value MA
            ma_timeframe: Timeframe for fair value MA
            lot_size: Position size
            magic_number: EA identifier
            initial_trigger_pips: Pips from MA to trigger grid
            pending_order_count: Number of grid orders
            pending_order_range_pips: Total range for grid orders
            use_filter: Whether to use trend filter
            filter_indicator: Type of filter indicator
            filter_timeframe: Timeframe for filter
            filter_period: Period for filter indicator
            filter_threshold: Threshold for non-MA filters
            equity_stop_percent: Equity drawdown stop
            close_at_ma: Close all when price returns to MA
        """
        super().__init__()

        # Store parameters
        self.ma_period = ma_period
        self.ma_timeframe = ma_timeframe
        self.lot_size = lot_size
        self.magic_number = magic_number

        self.initial_trigger_pips = initial_trigger_pips
        self.pending_order_count = pending_order_count
        self.pending_order_range_pips = pending_order_range_pips

        self.use_filter = use_filter
        self.filter_indicator = filter_indicator
        self.filter_timeframe = filter_timeframe
        self.filter_period = filter_period
        self.filter_threshold = filter_threshold

        self.equity_stop_percent = equity_stop_percent
        self.close_at_ma = close_at_ma

        # Internal state
        self.grid_active = False
        self.initial_equity = 0.0
        self.pip_size = 0.0001  # Will be set based on symbol

    def on_init(self) -> None:
        """Initialize strategy state."""
        self.initial_equity = self.balance

        # Determine pip size based on symbol
        if 'JPY' in self.symbol:
            self.pip_size = 0.01
        else:
            self.pip_size = 0.0001

    def get_required_indicators(self) -> List[Tuple[str, str, Dict]]:
        """Get list of required indicators for pre-calculation.

        Dynamically generates indicator requirements based on
        strategy parameters.
        """
        indicators = [
            # Main fair value MA
            ('SMA', self.ma_timeframe, {'period': self.ma_period}),
        ]

        # Add filter indicator if enabled
        if self.use_filter:
            if self.filter_indicator in ['SMA', 'EMA']:
                indicators.append((
                    self.filter_indicator,
                    self.filter_timeframe,
                    {'period': self.filter_period}
                ))
            elif self.filter_indicator == 'RSI':
                indicators.append((
                    'RSI',
                    self.filter_timeframe,
                    {'period': self.filter_period}
                ))
            elif self.filter_indicator == 'MACD':
                indicators.append((
                    'MACD',
                    self.filter_timeframe,
                    {
                        'fast': 12,
                        'slow': 26,
                        'signal': 9
                    }
                ))
            elif self.filter_indicator == 'ADX':
                indicators.append((
                    'ADX',
                    self.filter_timeframe,
                    {'period': self.filter_period}
                ))

        return indicators

    def on_tick(self, tick: Any, market: Any) -> None:
        """Process each market tick.

        Args:
            tick: Current market tick
            market: Market state with indicator access
        """
        # Check equity stop
        if self._check_equity_stop():
            self.close_all_positions()
            self.cancel_all_pending_orders()
            self.grid_active = False
            return

        # Get fair value MA
        fair_value = market.indicator(
            'SMA',
            self.ma_timeframe,
            {'period': self.ma_period}
        )

        # Check filter condition
        filter_passed = self._check_filter(tick, market)

        # Calculate distance from fair value
        distance_pips = abs(tick.bid - fair_value) / self.pip_size

        # Check if we should close at MA
        if self.close_at_ma and self.grid_active:
            if distance_pips < 10:  # Within 10 pips of MA
                self.close_all_positions()
                self.cancel_all_pending_orders()
                self.grid_active = False
                return

        # Check if we should activate grid
        if not self.grid_active and filter_passed:
            if distance_pips >= self.initial_trigger_pips:
                self._activate_grid(tick, fair_value)

    def _check_filter(self, tick: Any, market: Any) -> bool:
        """Check if filter conditions are met.

        Args:
            tick: Current market tick
            market: Market state

        Returns:
            True if filter passes or disabled
        """
        if not self.use_filter:
            return True

        if self.filter_indicator in ['SMA', 'EMA']:
            # Trend filter - price above/below MA
            filter_value = market.indicator(
                self.filter_indicator,
                self.filter_timeframe,
                {'period': self.filter_period}
            )
            return tick.bid > filter_value  # Bullish bias

        elif self.filter_indicator == 'RSI':
            # Momentum filter
            rsi_value = market.indicator(
                'RSI',
                self.filter_timeframe,
                {'period': self.filter_period}
            )
            if tick.bid > market.indicator('SMA', self.ma_timeframe, {'period': self.ma_period}):
                # Price above MA, look for oversold
                return rsi_value < (100 - self.filter_threshold)
            else:
                # Price below MA, look for overbought
                return rsi_value > self.filter_threshold

        elif self.filter_indicator == 'MACD':
            # MACD filter
            macd_values = market.indicator(
                'MACD',
                self.filter_timeframe,
                {'fast': 12, 'slow': 26, 'signal': 9}
            )
            # macd_values[2] is histogram
            return macd_values[2] > 0  # Bullish momentum

        elif self.filter_indicator == 'ADX':
            # Trend strength filter
            adx_value = market.indicator(
                'ADX',
                self.filter_timeframe,
                {'period': self.filter_period}
            )
            return adx_value > self.filter_threshold  # Strong trend

        return True

    def _activate_grid(self, tick: Any, fair_value: float) -> None:
        """Activate grid trading with pending orders.

        Args:
            tick: Current market tick
            fair_value: Current fair value (MA)
        """
        self.grid_active = True

        # Determine direction
        if tick.bid > fair_value:
            # Price above MA - place sell limit orders
            self._place_sell_grid(tick)
        else:
            # Price below MA - place buy limit orders
            self._place_buy_grid(tick)

    def _place_buy_grid(self, tick: Any) -> None:
        """Place buy limit grid orders.

        Args:
            tick: Current market tick
        """
        # Open initial market order
        self.buy(self.lot_size, comment=f"FP_Initial_{self.magic_number}")

        # Calculate grid levels
        grid_step = (self.pending_order_range_pips / self.pending_order_count) * self.pip_size

        for i in range(self.pending_order_count):
            price = tick.bid - (i + 1) * grid_step
            self.buy_limit(
                self.lot_size,
                price,
                comment=f"FP_Grid_{i}_{self.magic_number}"
            )

    def _place_sell_grid(self, tick: Any) -> None:
        """Place sell limit grid orders.

        Args:
            tick: Current market tick
        """
        # Open initial market order
        self.sell(self.lot_size, comment=f"FP_Initial_{self.magic_number}")

        # Calculate grid levels
        grid_step = (self.pending_order_range_pips / self.pending_order_count) * self.pip_size

        for i in range(self.pending_order_count):
            price = tick.ask + (i + 1) * grid_step
            self.sell_limit(
                self.lot_size,
                price,
                comment=f"FP_Grid_{i}_{self.magic_number}"
            )

    def _check_equity_stop(self) -> bool:
        """Check if equity stop has been triggered.

        Returns:
            True if equity stop triggered
        """
        if self.initial_equity <= 0:
            return False

        drawdown_percent = ((self.initial_equity - self.equity) /
                           self.initial_equity) * 100

        return drawdown_percent >= self.equity_stop_percent

    def get_optimization_params(self) -> Dict[str, Any]:
        """Get parameter ranges for optimization.

        This allows the optimizer to test different configurations
        including different indicator types and timeframes.
        """
        return {
            # Core parameters
            'ma_period': range(50, 500, 50),
            'ma_timeframe': ['M15', 'M30', 'H1', 'H4'],

            # Grid parameters
            'initial_trigger_pips': range(50, 200, 25),
            'pending_order_count': range(5, 20, 5),
            'pending_order_range_pips': range(30, 100, 10),

            # Universal filter parameters
            'use_filter': [True, False],
            'filter_indicator': ['SMA', 'RSI', 'MACD', 'ADX'],
            'filter_timeframe': ['H1', 'H4', 'D1'],
            'filter_period': range(14, 200, 14),
            'filter_threshold': range(30, 70, 10),

            # Risk parameters
            'equity_stop_percent': [3.0, 5.0, 10.0],
            'close_at_ma': [True, False],
        }


# Example usage
if __name__ == "__main__":
    from mt4_backtester import Backtester

    # Create strategy with RSI filter instead of slow MA
    strategy = FairPriceStrategy(
        ma_period=200,
        ma_timeframe='H1',
        use_filter=True,
        filter_indicator='RSI',  # Using RSI instead of slow MA
        filter_timeframe='H4',
        filter_period=14,
        filter_threshold=30,  # Buy when RSI < 30 (oversold)
    )

    # Run backtest
    backtester = Backtester()
    result = backtester.run(
        strategy=strategy,
        symbol='GBPAUD',
        start='2023-01-01',
        end='2023-12-31'
    )

    print(f"Net Profit: ${result.net_profit:.2f}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")