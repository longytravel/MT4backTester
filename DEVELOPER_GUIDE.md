# Developer Guide - MT4 Backtester

## System Architecture

### Core Design Principles

1. **Single Data Source**: Use tick data for everything
2. **Lazy Processing**: Process data only when needed
3. **Aggressive Caching**: Cache everything that's expensive
4. **Background Work**: Load data while user interacts with UI

### Key Innovation: Tick-Aware Execution

Traditional backtesting uses bar data for both signals AND execution. This system:
- Uses rolled-up bars for strategy signals (indicators)
- Uses original tick data for order execution
- Maintains perfect time synchronization between both

## Code Structure

```
fairPrice/
├── dashboard.py                        # Main Streamlit dashboard
├── run_backtest.py                     # Example standalone backtest
├── mt4_backtester/
│   └── src/
│       ├── core/
│       │   ├── engine.py              # Backtesting engine
│       │   ├── strategy.py            # Strategy base class
│       │   ├── order_manager.py       # Order execution logic
│       │   ├── costs.py               # Commission/spread model
│       │   └── indicators.py          # Technical indicators
│       └── data/
│           ├── tick_aware_manager.py  # Main data manager
│           └── __init__.py
```

## Key Classes

### TickAwareDataManager

```python
class TickAwareDataManager:
    def get_data_with_ticks(symbol, start, end, strategy_timeframe):
        # Returns tuple: (bars_df, tick_df)
        # bars_df: Rolled up to strategy timeframe
        # tick_df: Original tick data
```

**Important Methods:**
- `_load_tick_data()`: Loads from tickStoryData folders
- `_rollup_ticks_to_bars()`: Converts ticks to OHLCV bars
- `_start_preloading()`: Background thread for performance

### Dashboard State Management

Uses Streamlit session_state for:
- `data_manager`: TickAwareDataManager instance
- `cost_model`: CostModel instance
- `initial_balance`: Starting capital
- Strategy parameters: `ma_period`, `trigger_pips`, etc.

## Data Pipeline

### 1. Data Discovery
```python
# Searches these paths automatically:
data_paths = {
    'fxdata': Path('fxData'),
    'tickstory': Path('tickStoryData'),
    'dukascopy': Path('.dukascopy-cache'),
}
```

### 2. Tick Processing
```python
# Raw ticks → Strategy timeframe bars
tick_df.resample('1H').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'count'  # Tick count as volume
})
```

### 3. Caching Layers
1. **Memory Cache**: `_tick_cache`, `_bars_cache` dicts
2. **Disk Cache**: `.backtester_cache/*.parquet` files
3. **Cache Key**: MD5 hash of (symbol, start, end, timeframe)

## Performance Optimizations

### Background Preloading
```python
def _start_preloading(self):
    # Runs in separate thread on startup
    # Loads first symbol alphabetically (AUDCAD)
    # Loads last 3 months first, then full range
```

### Memory Optimization
```python
# Downcast floats: float64 → float32
df[float_cols] = df[float_cols].astype('float32')

# Optimize integers based on range
if df[col].max() < 255:
    df[col] = df[col].astype('uint8')
```

### Chunked Processing
For massive tick files:
- Process monthly chunks
- Stream through data
- Keep only necessary columns

## Adding New Features

### Adding a New Strategy

1. Create strategy class in `mt4_backtester/src/strategies/`
2. Inherit from `Strategy` base class
3. Implement `on_bar()` and `on_tick()` methods
4. Add to dashboard strategy selector

### Adding New Indicators

1. Add to `mt4_backtester/src/core/indicators.py`
2. Use pandas/numpy for vectorized calculations
3. Cache results in strategy timeframe bars

### Adding New Data Source

1. Update `data_paths` in TickAwareDataManager
2. Implement loader in `_load_tick_data()`
3. Map to standard format: datetime, bid, ask

## Testing

### Unit Tests (TODO)
```python
# Test tick rollup accuracy
def test_tick_to_bar_conversion():
    # Verify OHLC values match expected

# Test commission calculation
def test_commission_scaling():
    # Verify $7 per standard lot scales correctly
```

### Integration Tests
```bash
# Test with sample data
python run_backtest.py

# Test dashboard
streamlit run dashboard.py
```

## Common Issues & Solutions

### Issue: ModuleNotFoundError
**Solution**: Paths are added dynamically
```python
sys.path.insert(0, str(Path(__file__).parent / "mt4_backtester" / "src" / "data"))
```

### Issue: Memory Error with Large Files
**Solution**: Process in chunks
```python
for chunk in pd.read_csv(file, chunksize=1000000):
    process(chunk)
```

### Issue: Slow First Load
**Solution**: This is normal - tick processing
- Use Quick Test mode (last 3 months)
- Let background preload complete
- Cache makes subsequent runs 107x faster

## Database Schema (Cache)

### Parquet File Structure
```
.backtester_cache/
├── {cache_key}.parquet  # Processed bar data
└── tick_{cache_key}.parquet  # Tick data (future)
```

### Cache Key Generation
```python
def _get_cache_key(symbol, start, end, timeframe):
    key_string = f"{symbol}_{start}_{end}_{timeframe}"
    return hashlib.md5(key_string.encode()).hexdigest()
```

## API Reference

### TickAwareDataManager

```python
manager = TickAwareDataManager(cache_dir=".backtester_cache")

# Get bars and ticks
bars, ticks = manager.get_data_with_ticks(
    symbol="GBPAUD",
    start="2023-01-01",
    end="2023-12-31",
    strategy_timeframe="H1"
)

# Get available symbols
symbols = manager.get_available_symbols()

# Get date range for symbol
start, end = manager.get_date_range("GBPAUD")

# Clear cache
manager.clear_cache()
```

### CostModel

```python
cost_model = CostModel("forex_costs.csv")

# Get costs for symbol
costs = cost_model.get_costs("GBPAUD")

# Calculate commission for lot size
commission = costs.calculate_commission(0.01)  # Returns 0.07
```

## Future Architecture Considerations

1. **Multi-Processing**: Process symbols in parallel
2. **Distributed Cache**: Redis for multi-user setup
3. **Real-Time Data**: WebSocket integration
4. **Cloud Storage**: S3 for tick data
5. **GPU Acceleration**: RAPIDS for massive datasets

## Contributing

1. Keep tick-aware architecture
2. Maintain aggressive caching
3. Add tests for new features
4. Update this guide with changes
5. Use type hints for clarity

## Debug Mode

Set environment variable:
```bash
export DEBUG=1
```

Enables verbose logging:
- Data loading steps
- Cache hits/misses
- Processing times
- Memory usage