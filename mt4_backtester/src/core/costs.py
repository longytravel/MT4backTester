"""Cost model for accurate commission and spread calculations.

This module handles all trading costs including commission, spread,
and slippage based on real broker data.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TradingCosts:
    """Container for trading costs."""

    symbol: str
    spread_pips: float
    commission_per_lot: float  # Per standard lot
    total_cost_per_lot: float  # Per standard lot

    def calculate_commission(self, lot_size: float) -> float:
        """Calculate commission for given lot size.

        Args:
            lot_size: Size in lots (0.01 = micro lot, 1.0 = standard lot)

        Returns:
            Commission in USD
        """
        return self.commission_per_lot * lot_size

    def calculate_spread_cost(self, lot_size: float, pip_value: float = 10.0) -> float:
        """Calculate spread cost for given lot size.

        Args:
            lot_size: Size in lots
            pip_value: Value of 1 pip for 1 standard lot (default $10)

        Returns:
            Spread cost in USD
        """
        return self.spread_pips * lot_size * pip_value

    def calculate_total_cost(self, lot_size: float, pip_value: float = 10.0) -> float:
        """Calculate total trading cost (commission + spread).

        Args:
            lot_size: Size in lots
            pip_value: Value of 1 pip for 1 standard lot

        Returns:
            Total cost in USD
        """
        return self.calculate_commission(lot_size) + self.calculate_spread_cost(lot_size, pip_value)


class CostModel:
    """Manages trading costs from forex_costs.csv."""

    def __init__(self, costs_file: Optional[str] = None):
        """Initialize cost model.

        Args:
            costs_file: Path to forex_costs.csv
        """
        self.costs: Dict[str, TradingCosts] = {}

        if costs_file is None:
            # Try to find forex_costs.csv
            search_paths = [
                Path("forex_costs.csv"),
                Path("../forex_costs.csv"),
                Path("../../forex_costs.csv"),
            ]

            for path in search_paths:
                if path.exists():
                    costs_file = str(path)
                    break

        if costs_file and Path(costs_file).exists():
            self._load_costs(costs_file)
        else:
            self._use_defaults()

    def _load_costs(self, costs_file: str) -> None:
        """Load costs from CSV file.

        Args:
            costs_file: Path to forex_costs.csv
        """
        df = pd.read_csv(costs_file)

        for _, row in df.iterrows():
            # Convert EUR/USD to EURUSD format
            symbol = row['pair'].replace('/', '')

            self.costs[symbol] = TradingCosts(
                symbol=symbol,
                spread_pips=row['spread_pips'],
                commission_per_lot=row['commission_usd'],
                total_cost_per_lot=row['total_cost_usd']
            )

            # Also store lowercase version
            self.costs[symbol.lower()] = self.costs[symbol]

        print(f"Loaded costs for {len(self.costs) // 2} currency pairs")

    def _use_defaults(self) -> None:
        """Use default costs if file not found."""
        defaults = {
            'EURUSD': (0.2, 7.0, 9.0),
            'GBPUSD': (0.3, 7.0, 10.0),
            'USDJPY': (0.2, 7.0, 9.0),
            'GBPAUD': (1.0, 7.0, 17.0),
            'AUDUSD': (0.3, 7.0, 10.0),
        }

        for symbol, (spread, commission, total) in defaults.items():
            self.costs[symbol] = TradingCosts(
                symbol=symbol,
                spread_pips=spread,
                commission_per_lot=commission,
                total_cost_per_lot=total
            )
            self.costs[symbol.lower()] = self.costs[symbol]

    def get_costs(self, symbol: str) -> TradingCosts:
        """Get trading costs for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'GBPAUD' or 'gbpaud')

        Returns:
            TradingCosts for the symbol
        """
        # Try exact match first
        if symbol in self.costs:
            return self.costs[symbol]

        # Try uppercase
        if symbol.upper() in self.costs:
            return self.costs[symbol.upper()]

        # Try lowercase
        if symbol.lower() in self.costs:
            return self.costs[symbol.lower()]

        # Return default if not found
        return TradingCosts(
            symbol=symbol,
            spread_pips=1.0,
            commission_per_lot=7.0,
            total_cost_per_lot=17.0
        )

    def calculate_trade_cost(
        self,
        symbol: str,
        lot_size: float,
        include_spread: bool = True
    ) -> Tuple[float, float, float]:
        """Calculate all costs for a trade.

        Args:
            symbol: Trading symbol
            lot_size: Size in lots (0.01 = micro lot)
            include_spread: Whether to include spread cost

        Returns:
            Tuple of (commission, spread_cost, total_cost)
        """
        costs = self.get_costs(symbol)

        commission = costs.calculate_commission(lot_size)
        spread_cost = costs.calculate_spread_cost(lot_size) if include_spread else 0.0
        total = commission + spread_cost

        return commission, spread_cost, total


# Example usage
if __name__ == "__main__":
    # Load cost model
    model = CostModel()

    # Get costs for GBPAUD
    gbpaud_costs = model.get_costs('GBPAUD')
    print(f"GBPAUD Costs:")
    print(f"  Spread: {gbpaud_costs.spread_pips} pips")
    print(f"  Commission: ${gbpaud_costs.commission_per_lot} per lot")

    # Calculate for 0.01 lots
    lot_size = 0.01
    commission = gbpaud_costs.calculate_commission(lot_size)
    spread = gbpaud_costs.calculate_spread_cost(lot_size)

    print(f"\nFor {lot_size} lots:")
    print(f"  Commission: ${commission:.2f}")
    print(f"  Spread cost: ${spread:.2f}")
    print(f"  Total cost: ${commission + spread:.2f}")