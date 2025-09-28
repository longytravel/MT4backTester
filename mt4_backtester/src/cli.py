"""Command-line interface for MT4 Backtester."""

import click
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mt4_backtester.core.engine import Backtester, EngineConfig
from mt4_backtester.optimization.optimizer import Optimizer, OptimizationConfig, export_to_mt4_set


@click.group()
def cli():
    """MT4 Universal Backtester - High-performance trading strategy tester."""
    pass


@cli.command()
@click.option('--strategy', '-s', required=True, help='Strategy module path (e.g., examples.fairprice_strategy.FairPriceStrategy)')
@click.option('--symbol', '-sym', required=True, help='Trading symbol (e.g., EURUSD)')
@click.option('--start', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end', required=True, help='End date (YYYY-MM-DD)')
@click.option('--balance', default=10000, help='Initial balance')
@click.option('--leverage', default=100, help='Account leverage')
@click.option('--commission', default=7.0, help='Commission per trade')
@click.option('--data-source', default='auto', type=click.Choice(['tick', 'bar', 'auto']), help='Data source type')
@click.option('--timeframe', default='H1', help='Primary timeframe')
def backtest(strategy, symbol, start, end, balance, leverage, commission, data_source, timeframe):
    """Run a single backtest with specified parameters."""
    click.echo(f"Running backtest for {symbol} from {start} to {end}")

    # Import strategy class dynamically
    module_path, class_name = strategy.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    strategy_class = getattr(module, class_name)

    # Create strategy instance
    strategy_instance = strategy_class()

    # Configure engine
    config = EngineConfig(
        initial_balance=balance,
        leverage=leverage,
        commission=commission,
        use_tick_data=(data_source == 'tick'),
        cache_indicators=True,
    )

    # Run backtest
    backtester = Backtester(config)
    result = backtester.run(
        strategy=strategy_instance,
        symbol=symbol,
        start=start,
        end=end,
        data_source=data_source,
        timeframe=timeframe,
    )

    # Display results
    click.echo("\n" + "=" * 60)
    click.echo("BACKTEST RESULTS")
    click.echo("=" * 60)
    click.echo(f"Net Profit: ${result.net_profit:.2f}")
    click.echo(f"Total Trades: {result.total_trades}")
    click.echo(f"Win Rate: {result.win_rate:.1%}")
    click.echo(f"Profit Factor: {result.profit_factor:.2f}")
    click.echo(f"Max Drawdown: {result.max_drawdown:.1%}")
    click.echo(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")


@cli.command()
@click.option('--strategy', '-s', required=True, help='Strategy module path')
@click.option('--symbol', '-sym', required=True, help='Trading symbol')
@click.option('--start', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end', required=True, help='End date (YYYY-MM-DD)')
@click.option('--method', default='optuna', type=click.Choice(['optuna', 'grid', 'random', 'genetic']), help='Optimization method')
@click.option('--trials', default=100, help='Number of trials')
@click.option('--objective', default='sharpe', type=click.Choice(['sharpe', 'profit', 'win_rate', 'profit_factor']), help='Optimization objective')
@click.option('--jobs', default=-1, help='Number of parallel jobs (-1 for all cores)')
@click.option('--output', '-o', help='Output file for optimized parameters (.set)')
def optimize(strategy, symbol, start, end, method, trials, objective, jobs, output):
    """Run parameter optimization for a strategy."""
    click.echo(f"Running {method} optimization for {symbol}")
    click.echo(f"Objective: maximize {objective}")
    click.echo(f"Trials: {trials}")

    # Import strategy class
    module_path, class_name = strategy.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    strategy_class = getattr(module, class_name)

    # Configure optimization
    opt_config = OptimizationConfig(
        method=method,
        n_trials=trials,
        n_jobs=jobs,
        objective=objective,
        show_progress=True,
        save_results=True,
    )

    # Run optimization
    optimizer = Optimizer(opt_config)
    result = optimizer.optimize(
        strategy_class=strategy_class,
        symbol=symbol,
        start=start,
        end=end,
        data_source='auto',
    )

    # Display results
    click.echo("\n" + "=" * 60)
    click.echo("OPTIMIZATION RESULTS")
    click.echo("=" * 60)
    click.echo(f"Best Score ({objective}): {result.best_score:.4f}")
    click.echo(f"Total Trials: {result.total_trials}")
    click.echo(f"Time: {result.optimization_time:.1f} seconds")
    click.echo("\nBest Parameters:")
    for param, value in result.best_params.items():
        click.echo(f"  {param}: {value}")

    # Export if requested
    if output:
        export_to_mt4_set(result.best_params, output)
        click.echo(f"\nExported to: {output}")


@cli.command()
@click.argument('input_file')
@click.argument('output_file')
@click.option('--format', default='parquet', type=click.Choice(['parquet', 'hdf5', 'feather']), help='Output format')
def convert_data(input_file, output_file, format):
    """Convert CSV data to optimized format for faster loading."""
    import pandas as pd

    click.echo(f"Converting {input_file} to {format} format...")

    # Read CSV
    df = pd.read_csv(input_file, parse_dates=['datetime'])

    # Save in optimized format
    if format == 'parquet':
        df.to_parquet(output_file)
    elif format == 'hdf5':
        df.to_hdf(output_file, key='data')
    elif format == 'feather':
        df.to_feather(output_file)

    # Show size comparison
    input_size = Path(input_file).stat().st_size / 1024 / 1024
    output_size = Path(output_file).stat().st_size / 1024 / 1024
    compression = (1 - output_size / input_size) * 100

    click.echo(f"✅ Conversion complete!")
    click.echo(f"Original size: {input_size:.2f} MB")
    click.echo(f"New size: {output_size:.2f} MB")
    click.echo(f"Compression: {compression:.1f}%")


@cli.command()
def list_strategies():
    """List available example strategies."""
    click.echo("\nAvailable Example Strategies:")
    click.echo("=" * 40)
    click.echo("1. examples.fairprice_strategy.FairPriceStrategy")
    click.echo("   - Grid trading with universal indicator support")
    click.echo("   - Supports SMA, RSI, MACD, ADX filters")
    click.echo("\nTo use a strategy, provide its full module path:")
    click.echo("mt4-backtest -s examples.fairprice_strategy.FairPriceStrategy ...")


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == '__main__':
    main()