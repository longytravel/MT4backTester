"""Tick-aware data manager for realistic backtesting.

This manager handles tick data properly:
- Loads tick data for execution
- Rolls up to strategy timeframe for indicators
- Returns both for coordinated backtesting
"""

import hashlib
import pickle
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class TickAwareDataManager:
    """Smart data manager that uses tick data for both strategy and execution."""

    def __init__(self, cache_dir: str = ".backtester_cache"):
        """Initialize tick-aware data manager."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Data paths
        self.data_paths = {
            'fxdata': Path('fxData'),
            'tickstory': Path('tickStoryData'),
            'dukascopy': Path('.dukascopy-cache'),
        }

        # Memory caches
        self._tick_cache = {}  # Raw tick data
        self._bars_cache = {}  # Rolled-up bars
        self._preload_thread = None

        # Start background preloading
        self._start_preloading()

    def get_data_with_ticks(
        self,
        symbol: str,
        start: Union[str, datetime],
        end: Union[str, datetime],
        strategy_timeframe: str = 'H1'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get both strategy bars and execution ticks.

        Args:
            symbol: Trading symbol
            start: Start date
            end: End date
            strategy_timeframe: Timeframe for strategy indicators (M15, M30, H1, etc.)

        Returns:
            Tuple of (strategy_bars, execution_ticks)
        """
        # Convert dates
        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)

        # Load tick data
        tick_df = self._load_tick_data(symbol, start, end)

        # Roll up to strategy timeframe
        if tick_df is not None and not tick_df.empty:
            strategy_bars = self._rollup_ticks_to_bars(tick_df, strategy_timeframe)
            return strategy_bars, tick_df
        else:
            # Fall back to CSV bar data if no ticks
            strategy_bars = self._load_csv_bars(symbol, start, end, strategy_timeframe)
            return strategy_bars, pd.DataFrame()

    def _load_tick_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> Optional[pd.DataFrame]:
        """Load tick data from TickStory directory."""

        # Check cache first
        cache_key = f"{symbol}_TICK_{start}_{end}"
        if cache_key in self._tick_cache:
            print(f"[TickAware] Using cached tick data for {symbol}")
            return self._tick_cache[cache_key]

        # Find tick directory
        tick_path = self.data_paths['tickstory'] / symbol.upper()
        if not tick_path.exists():
            tick_path = self.data_paths['tickstory'] / symbol
            if not tick_path.exists():
                print(f"[TickAware] No tick data found for {symbol}")
                return None

        print(f"[TickAware] Loading tick data for {symbol} from {tick_path}")

        # Load all CSV files in tick directory
        all_ticks = []
        csv_files = list(tick_path.rglob("*.csv"))

        if not csv_files:
            print(f"[TickAware] No CSV files found in {tick_path}")
            return None

        for csv_file in csv_files:
            try:
                # Read tick data
                df = pd.read_csv(csv_file, parse_dates=[0])

                # Ensure proper columns
                if len(df.columns) >= 2:
                    if len(df.columns) == 2:
                        df.columns = ['datetime', 'price']
                        df['bid'] = df['price']
                        df['ask'] = df['price']
                    elif len(df.columns) >= 3:
                        df.columns = ['datetime', 'bid', 'ask'] + list(df.columns[3:])

                all_ticks.append(df)
            except Exception as e:
                print(f"[TickAware] Error loading {csv_file}: {e}")
                continue

        if all_ticks:
            # Combine all tick data
            tick_df = pd.concat(all_ticks, ignore_index=True)

            # Sort by time
            tick_df.sort_values('datetime', inplace=True)

            # Filter date range
            tick_df = tick_df[(tick_df['datetime'] >= start) & (tick_df['datetime'] <= end)]

            # Cache for later
            self._tick_cache[cache_key] = tick_df

            print(f"[TickAware] Loaded {len(tick_df)} ticks for {symbol}")
            return tick_df

        return None

    def _rollup_ticks_to_bars(
        self,
        tick_df: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:
        """Roll up tick data to strategy timeframe bars."""

        # Check cache
        tick_hash = hashlib.md5(pd.util.hash_pandas_object(tick_df).values).hexdigest()[:8]
        cache_key = f"bars_{tick_hash}_{timeframe}"

        if cache_key in self._bars_cache:
            print(f"[TickAware] Using cached {timeframe} bars")
            return self._bars_cache[cache_key]

        print(f"[TickAware] Rolling up ticks to {timeframe} bars")

        # Map timeframe to pandas resample rule
        timeframe_map = {
            'M1': '1T',
            'M5': '5T',
            'M15': '15T',
            'M30': '30T',
            'H1': '1H',
            'H4': '4H',
            'D1': '1D',
        }

        if timeframe not in timeframe_map:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        rule = timeframe_map[timeframe]

        # Set datetime as index
        tick_df = tick_df.copy()
        tick_df.set_index('datetime', inplace=True)

        # Use bid price for OHLC
        price_col = 'bid' if 'bid' in tick_df.columns else tick_df.columns[0]

        # Resample to create bars
        bars = tick_df[price_col].resample(rule).agg([
            ('open', 'first'),
            ('high', 'max'),
            ('low', 'min'),
            ('close', 'last'),
            ('volume', 'count')  # Tick count as volume
        ]).dropna()

        # Reset index
        bars.reset_index(inplace=True)

        # Cache result
        self._bars_cache[cache_key] = bars

        print(f"[TickAware] Created {len(bars)} {timeframe} bars from ticks")
        return bars

    def _load_csv_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str
    ) -> pd.DataFrame:
        """Load bar data from CSV files (fallback when no tick data)."""

        # Map timeframe to file pattern
        timeframe_map = {
            'M15': 'm15',
            'M30': 'm30',
            'H1': 'h1',
            'H4': 'h4',
            'D1': 'd1',
        }

        tf_pattern = timeframe_map.get(timeframe, 'h1')

        # Find matching CSV file
        pattern = f"{symbol.lower()}-{tf_pattern}-*.csv"
        csv_files = list(self.data_paths['fxdata'].glob(pattern))

        if csv_files:
            # Load first matching file
            df = pd.read_csv(csv_files[0], parse_dates=['datetime'])

            # Filter date range
            df = df[(df['datetime'] >= start) & (df['datetime'] <= end)]

            print(f"[TickAware] Loaded {len(df)} bars from CSV for {symbol} {timeframe}")
            return df

        # Return empty if no data
        return pd.DataFrame()

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        symbols = set()

        # Check tick data
        if self.data_paths['tickstory'].exists():
            for item in self.data_paths['tickstory'].iterdir():
                if item.is_dir():
                    symbols.add(item.name.upper())

        # Check CSV data
        if self.data_paths['fxdata'].exists():
            for csv_file in self.data_paths['fxdata'].glob("*.csv"):
                parts = csv_file.stem.split('-')
                if parts:
                    symbols.add(parts[0].upper())

        return sorted(list(symbols))

    def get_date_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """Get available date range for symbol."""

        # Try tick data first
        tick_path = self.data_paths['tickstory'] / symbol.upper()
        if tick_path.exists():
            # Sample first and last file to get range
            csv_files = sorted(tick_path.rglob("*.csv"))
            if csv_files:
                try:
                    # Read first few lines of first file
                    first_df = pd.read_csv(csv_files[0], nrows=1, parse_dates=[0])
                    start_date = pd.to_datetime(first_df.iloc[0, 0])

                    # Read last few lines of last file
                    last_df = pd.read_csv(csv_files[-1], parse_dates=[0])
                    end_date = pd.to_datetime(last_df.iloc[-1, 0])

                    return start_date, end_date
                except:
                    pass

        # Default range if can't determine
        return datetime(2022, 1, 1), datetime.now()

    def _start_preloading(self):
        """Start background thread to preload first symbol."""

        def preload():
            symbols = self.get_available_symbols()
            if symbols:
                first_symbol = symbols[0]  # First alphabetically
                print(f"[TickAware] Background preloading {first_symbol}")

                # Get date range
                start, end = self.get_date_range(first_symbol)

                # Load last 3 months first (Quick Test)
                quick_start = max(start, end - timedelta(days=90))
                self._load_tick_data(first_symbol, quick_start, end)

                # Then load full range
                self._load_tick_data(first_symbol, start, end)

                print(f"[TickAware] Preloaded {first_symbol} in background")

        # Start in background thread
        self._preload_thread = threading.Thread(target=preload, daemon=True)
        self._preload_thread.start()

    def clear_cache(self):
        """Clear all memory caches."""
        self._tick_cache.clear()
        self._bars_cache.clear()
        print("[TickAware] Cache cleared")