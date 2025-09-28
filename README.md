# MT4 Backtester - High-Performance Trading System

A fast, tick-aware backtesting system that replaces MT4's slow tester with 100-1000x speed improvements.

## Key Features

- **Tick-Aware Execution**: Uses actual tick data for realistic order fills
- **Strategy Timeframe Rollup**: Automatically converts tick data to any timeframe (M15, M30, H1, H4, D1) for indicator calculations
- **Smart Caching**: 107x faster on subsequent runs through intelligent data caching
- **Background Preloading**: Starts loading data immediately when dashboard opens
- **Real Commission Scaling**: Correctly calculates commission based on lot size ($7 per standard lot, $0.07 for 0.01 lots)

## Quick Start

```bash
# Run the dashboard
streamlit run dashboard.py
# OR
run_dashboard.bat

# Run example backtest
python run_backtest.py
```

## Architecture

### Data Flow
```
TickStory Data (tickStoryData/GBPAUD/)
    ↓
TickAwareDataManager
    ├── Loads tick data
    ├── Rolls up to strategy timeframe (H1, etc.)
    └── Caches for speed
    ↓
Backtester
    ├── Uses bars for strategy signals (MA crossovers, etc.)
    └── Uses ticks for realistic execution
```

### Core Components

1. **TickAwareDataManager** (`mt4_backtester/src/data/tick_aware_manager.py`)
   - Loads tick data from existing folders
   - Rolls up to strategy timeframes
   - Returns both bars and ticks for backtesting
   - Background preloading for performance

2. **Dashboard** (`dashboard.py`)
   - Tab 1: Data Selection - Choose symbol, strategy timeframe, test mode
   - Tab 2: Strategy Settings - Configure FairPrice parameters
   - Tab 3: Run Backtest - Execute with tick-aware system
   - Tab 4: Results - View performance metrics
   - Tab 5: Export - Save results as CSV or MT4 .set files

3. **Cost Model** (`mt4_backtester/src/core/costs.py`)
   - Loads real forex costs from `forex_costs.csv`
   - Correctly scales commission by lot size
   - Handles spread and slippage

## Data Structure

```
fairPrice/
├── tickStoryData/         # Your tick data (from TickStory)
│   └── GBPAUD/           # Tick CSV files
├── fxData/               # Bar data (CSV files)
├── .backtester_cache/    # Cached processed data
└── forex_costs.csv       # Commission and spread data
```

## How It Works

### 1. Tick Data Loading
The system automatically finds your tick data in `tickStoryData/` folders. No preprocessing required.

### 2. Strategy Timeframe
When you select "H1" as strategy timeframe:
- Tick data is rolled up to H1 bars
- Indicators (MA, RSI, etc.) calculated on H1 bars
- Original ticks kept for execution

### 3. Execution
- Strategy signals generated from bars (e.g., price crosses MA on H1)
- Orders executed using tick data for realistic fills
- Commission calculated correctly based on lot size

### 4. Performance
- First run: Processes tick data and caches
- Subsequent runs: 107x faster from cache
- Background preloading: Data loads while you configure settings

## Configuration

### Test Modes
- **Quick Test**: Last 3 months of data for rapid testing
- **Full Range**: All available data (2022-present for GBPAUD)

### Strategy Parameters (FairPrice Grid)
- **MA Period**: Moving average period (default 200)
- **Trigger Distance**: Pips from MA to trigger entry (default 100)
- **Grid Orders**: Number of grid levels
- **Lot Size**: Position size (scales commission correctly)

## Adding New Data

1. **Tick Data**: Place in `tickStoryData/SYMBOL/` folder
2. **Bar Data**: Place in `fxData/` folder
3. System auto-detects and processes on first use

## Performance Tips

1. Use Quick Test mode for rapid iteration
2. Let background preloading complete before running
3. Cache is cleared with button in dashboard
4. Tick data processing happens only once per date range

## Technical Details

### Memory Optimization
- Uses float32 instead of float64
- Chunks large tick files
- Streams data instead of loading all at once

### Caching Strategy
- Parquet format for efficient storage
- Separate caches for tick and bar data
- Cache key includes symbol, dates, timeframe

### Thread Safety
- Background preloading in separate thread
- Session state for Streamlit compatibility
- Atomic cache operations

## Future Enhancements

1. Support for more tick data formats (.bi5)
2. Multi-symbol portfolio backtesting
3. Walk-forward optimization
4. Monte Carlo analysis
5. More strategy templates

## Requirements

```
streamlit>=1.28.0
plotly>=5.17.0
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0
```

## Troubleshooting

### "No tick data found"
- Ensure tick data is in `tickStoryData/SYMBOL/` folder
- Check for CSV files in the directory

### "Initial balance not defined"
- Go to Strategy Settings tab first
- Set initial balance before running backtest

### Slow first run
- Normal - processing tick data
- Subsequent runs will be 107x faster
- Use Quick Test mode for faster results

## Contact

For issues or questions, check the GitHub repository:
https://github.com/longytravel/MT4backTester