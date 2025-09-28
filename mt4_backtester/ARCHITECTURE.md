# MT4 Backtester Architecture

## Overview

This universal backtesting framework is designed with **clean architecture** principles, emphasizing modularity, performance, and extensibility. The system can handle ANY trading strategy, not just specific implementations.

## Core Design Principles

### 1. **Universal Strategy Interface**
- Strategies inherit from abstract `Strategy` base class
- No hardcoded strategy logic in core engine
- Supports any indicator combination dynamically

### 2. **Point-in-Time Data Access**
- Pre-calculates indicators for speed
- Ensures no look-ahead bias during execution
- `MarketState` object controls data visibility

### 3. **Plugin Architecture**
- Indicators registered dynamically
- New indicators added without modifying core
- Strategies can request ANY indicator combination

### 4. **Multi-Timeframe Support**
- Automatic timeframe aggregation
- Strategies can access any timeframe
- Efficient caching across timeframes

## System Components

### Core Engine (`src/core/`)

#### `engine.py` - Backtesting Engine
- Main event loop processing ticks/bars
- Manages strategy execution lifecycle
- Coordinates all subsystems
- Prevents look-ahead bias

#### `strategy.py` - Strategy Base Class
- Abstract interface all strategies implement
- Provides trading operations (buy/sell/close)
- Manages pending orders queue
- Defines required indicators dynamically

#### `order_manager.py` - Order & Position Management
- Executes market and pending orders
- Tracks open positions and P&L
- Monitors stop loss/take profit
- Calculates margin and equity

#### `result.py` - Results & Metrics
- Stores trade history
- Calculates performance metrics
- Generates equity curves
- Exports results for analysis

### Data Layer (`src/data/`)

#### `manager.py` - Data Management
- Loads tick and bar data
- Handles multiple data formats
- Aggregates timeframes on-demand
- Manages spread models

**Key Features:**
- Dual data source support (tick + bar)
- Automatic tick synthesis from bars
- Efficient caching system
- Point-in-time market state

### Indicators (`src/indicators/`)

#### `registry.py` - Indicator Plugin System
- Dynamic indicator registration
- Numba-optimized calculations
- Caching for performance
- Extensible for custom indicators

**Built-in Indicators:**
- Moving Averages (SMA, EMA)
- Oscillators (RSI, Stochastic)
- Trend (MACD, ADX)
- Volatility (Bollinger Bands, ATR)

### Optimization (`src/optimization/`)

#### `optimizer.py` - Parameter Optimization
- Multiple algorithms (Optuna, Grid, Random, Genetic)
- Parallel processing across CPU cores
- Flexible objective functions
- MT4 .set file generation

**Optimization Methods:**
- **Optuna**: Bayesian optimization with pruning
- **Grid Search**: Exhaustive parameter testing
- **Random Search**: Monte Carlo sampling
- **Genetic Algorithm**: Evolutionary optimization

## Data Flow

```
1. Data Loading
   CSV/Parquet → DataManager → MarketData

2. Indicator Calculation
   MarketData → IndicatorRegistry → Cached Values

3. Strategy Execution
   Market Tick → Engine → Strategy.on_tick()
                    ↓
                OrderManager → Position Updates

4. Results Generation
   Trades → BacktestResult → Performance Metrics
```

## Performance Optimizations

### 1. **Smart Caching**
- Pre-calculated indicators
- Reusable across backtests
- Memory-mapped data files

### 2. **Numba JIT Compilation**
- Critical loops compiled to machine code
- 100x speedup for indicator calculations
- No Python overhead in hot paths

### 3. **Vectorized Operations**
- NumPy arrays for bulk processing
- Pandas for efficient time series
- Minimal Python loops

### 4. **Parallel Processing**
- Multi-core optimization
- Distributed backtesting with Ray
- Async I/O for data loading

## Extensibility Points

### Adding New Indicators

```python
class MyIndicator(Indicator):
    def calculate(self, data):
        # Your calculation
        return result

registry.register('MY_IND', MyIndicator)
```

### Creating Custom Strategies

```python
class MyStrategy(Strategy):
    def on_tick(self, tick, market):
        # Your trading logic
        indicator_value = market.indicator('MY_IND', 'H1', params)
```

### Custom Optimization Objectives

```python
def custom_objective(result):
    return result.sharpe_ratio * result.profit_factor

optimizer = Optimizer()
optimizer.config.objective = custom_objective
```

## Best Practices

1. **Always use point-in-time data access**
2. **Pre-calculate indicators when possible**
3. **Implement proper position sizing**
4. **Test with both tick and bar data**
5. **Validate against MT4 results**

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock dependencies
- Verify calculations

### Integration Tests
- End-to-end backtesting
- Compare with known results
- Stress test with edge cases

### Performance Tests
- Benchmark against MT4
- Profile bottlenecks
- Optimize critical paths

## Future Enhancements

1. **Machine Learning Integration**
   - Feature engineering from indicators
   - Strategy parameter prediction
   - Market regime detection

2. **Real-time Trading**
   - Live data feed integration
   - Order execution APIs
   - Risk management systems

3. **Cloud Deployment**
   - Kubernetes orchestration
   - Distributed backtesting
   - Web-based UI

## Development Guidelines

1. **Code Style**: Follow PEP 8, use Black formatter
2. **Type Hints**: Add type annotations for clarity
3. **Documentation**: Docstrings for all public methods
4. **Testing**: Minimum 80% test coverage
5. **Performance**: Profile before optimizing

## Conclusion

This architecture provides a solid foundation for high-performance backtesting while maintaining flexibility for any trading strategy. The modular design allows easy extension without modifying core components, and the optimization framework ensures strategies can be fine-tuned efficiently.