"""Example script to run FairPrice backtest and optimization.

This script demonstrates how to use the MT4 Backtester framework
for both single backtests and parameter optimization.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mt4_backtester.core.engine import Backtester, EngineConfig
from mt4_backtester.optimization.optimizer import Optimizer, OptimizationConfig
from examples.fairprice_strategy import FairPriceStrategy


def run_single_backtest():
    """Run a single backtest with specific parameters."""
    print("=" * 60)
    print("Running Single Backtest")
    print("=" * 60)

    # Create strategy with specific parameters
    strategy = FairPriceStrategy(
        ma_period=200,
        ma_timeframe='H1',
        use_filter=True,
        filter_indicator='RSI',  # Using RSI instead of slow MA
        filter_timeframe='H4',
        filter_period=14,
        filter_threshold=30,  # Buy when RSI < 30 (oversold)
        initial_trigger_pips=100,
        pending_order_count=10,
        pending_order_range_pips=50,
        equity_stop_percent=5.0,
    )

    # Configure and run backtest
    config = EngineConfig(
        initial_balance=10000,
        leverage=100,
        commission=7.0,
        use_tick_data=False,  # Use bar data for speed
        cache_indicators=True,
    )

    backtester = Backtester(config)

    # Use available data - adjust dates as needed
    result = backtester.run(
        strategy=strategy,
        symbol='GBPAUD',
        start='2023-01-01',
        end='2023-06-30',
        data_source='bar',
        timeframe='H1',
    )

    # Print results
    print(result.summary())

    return result


def run_optimization():
    """Run parameter optimization to find best settings."""
    print("=" * 60)
    print("Running Parameter Optimization")
    print("=" * 60)

    # Configure optimization
    opt_config = OptimizationConfig(
        method="random",  # Use random search for quick demo
        n_trials=50,  # Reduced for demo
        n_jobs=4,  # Use 4 CPU cores
        objective="sharpe",  # Optimize for Sharpe ratio
        show_progress=True,
        save_results=True,
    )

    # Configure backtesting engine
    engine_config = EngineConfig(
        initial_balance=10000,
        leverage=100,
        commission=7.0,
        use_tick_data=False,
        cache_indicators=True,
    )

    # Create optimizer
    optimizer = Optimizer(opt_config, engine_config)

    # Define parameter search space
    param_space = {
        'ma_period': range(100, 400, 50),
        'ma_timeframe': ['H1', 'H4'],
        'initial_trigger_pips': range(50, 200, 25),
        'pending_order_count': range(5, 20, 5),
        'pending_order_range_pips': range(30, 100, 20),
        'use_filter': [True, False],
        'filter_indicator': ['SMA', 'RSI', 'MACD'],
        'filter_period': range(14, 100, 14),
        'filter_threshold': range(30, 70, 10),
        'equity_stop_percent': [3.0, 5.0, 10.0],
    }

    # Run optimization
    result = optimizer.optimize(
        strategy_class=FairPriceStrategy,
        symbol='GBPAUD',
        start='2023-01-01',
        end='2023-06-30',
        param_space=param_space,
        data_source='bar',
        timeframe='H1',
    )

    # Print best parameters
    print("\n" + "=" * 60)
    print("Optimization Results")
    print("=" * 60)
    print(f"Best Score (Sharpe Ratio): {result.best_score:.3f}")
    print(f"Total Trials: {result.total_trials}")
    print(f"Optimization Time: {result.optimization_time:.1f} seconds")
    print("\nBest Parameters:")
    for param, value in result.best_params.items():
        print(f"  {param}: {value}")

    # Export to MT4 .set file
    from mt4_backtester.optimization.optimizer import export_to_mt4_set

    export_to_mt4_set(
        result.best_params,
        "optimized_fairprice.set",
        ea_name="FairPrice"
    )
    print("\nExported optimized parameters to: optimized_fairprice.set")

    return result


def main():
    """Main entry point."""
    print("\nMT4 Universal Backtester - Example Script")
    print("=========================================\n")

    # Check if data exists
    data_path = Path(__file__).parent.parent / "fxData"
    if not data_path.exists():
        print("Warning: Data directory not found at", data_path)
        print("Please ensure your data files are in the correct location")
        print("Expected structure:")
        print("  ../fxData/GBPAUD_60min.csv")
        print("  ../fxData/GBPAUD_240min.csv")
        print("  etc.")
        return

    # Run single backtest
    try:
        backtest_result = run_single_backtest()
    except Exception as e:
        print(f"Error running backtest: {e}")
        print("Please check your data files and paths")
        return

    # Ask user if they want to run optimization
    print("\n" + "=" * 60)
    response = input("Run parameter optimization? (y/n): ").lower()

    if response == 'y':
        try:
            optimization_result = run_optimization()
        except Exception as e:
            print(f"Error running optimization: {e}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()