"""Indicator plugin system with dynamic registration.

This module provides a flexible plugin system for technical indicators,
allowing new indicators to be added without modifying core code.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from numba import jit


class Indicator(ABC):
    """Abstract base class for all technical indicators."""

    def __init__(self, **params):
        """Initialize indicator with parameters.

        Args:
            **params: Indicator-specific parameters
        """
        self.params = params
        self.name = self.__class__.__name__

    @abstractmethod
    def calculate(self, data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Calculate indicator values.

        Args:
            data: Price data (typically close prices)

        Returns:
            Array of indicator values
        """
        pass

    def validate_params(self) -> bool:
        """Validate indicator parameters.

        Returns:
            True if parameters are valid
        """
        return True

    def get_cache_key(self, symbol: str, timeframe: str) -> str:
        """Generate unique cache key for this indicator.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Unique cache key
        """
        param_str = '_'.join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{symbol}_{timeframe}_{self.name}_{param_str}"


class IndicatorRegistry:
    """Registry for managing and caching technical indicators."""

    def __init__(self):
        """Initialize the indicator registry."""
        self._indicators: Dict[str, type[Indicator]] = {}
        self.cache: Dict[str, np.ndarray] = {}
        self._register_default_indicators()

    def register(
        self,
        name: str,
        indicator_class: type[Indicator],
        override: bool = False
    ) -> None:
        """Register a new indicator.

        Args:
            name: Indicator name (e.g., 'SMA', 'RSI')
            indicator_class: Indicator class
            override: Whether to override existing indicator
        """
        if name in self._indicators and not override:
            logger.warning(
                f"Indicator {name} already registered. Use override=True to replace."
            )
            return

        self._indicators[name] = indicator_class
        logger.info(f"Registered indicator: {name}")

    def calculate(
        self,
        indicator_type: str,
        data: Union[np.ndarray, pd.DataFrame],
        params: Dict[str, Any]
    ) -> np.ndarray:
        """Calculate indicator values.

        Args:
            indicator_type: Type of indicator
            data: Market data
            params: Indicator parameters

        Returns:
            Array of indicator values
        """
        if indicator_type not in self._indicators:
            raise ValueError(f"Unknown indicator type: {indicator_type}")

        indicator_class = self._indicators[indicator_type]
        indicator = indicator_class(**params)

        if not indicator.validate_params():
            raise ValueError(f"Invalid parameters for {indicator_type}: {params}")

        # Extract price data if DataFrame
        if isinstance(data, pd.DataFrame):
            if 'close' in data.columns:
                price_data = data['close'].values
            else:
                price_data = data.iloc[:, 0].values
        else:
            price_data = data

        return indicator.calculate(price_data)

    def cache_indicator(
        self,
        symbol: str,
        timeframe: str,
        indicator_type: str,
        params: Dict[str, Any],
        values: np.ndarray
    ) -> None:
        """Cache calculated indicator values.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            indicator_type: Type of indicator
            params: Indicator parameters
            values: Calculated values
        """
        cache_key = self._generate_cache_key(
            symbol, timeframe, indicator_type, params
        )
        self.cache[cache_key] = values
        logger.debug(f"Cached indicator: {cache_key}")

    def get_cached(
        self,
        symbol: str,
        timeframe: str,
        indicator_type: str,
        params: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Get cached indicator values.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            indicator_type: Type of indicator
            params: Indicator parameters

        Returns:
            Cached values or None if not found
        """
        cache_key = self._generate_cache_key(
            symbol, timeframe, indicator_type, params
        )
        return self.cache.get(cache_key)

    def clear_cache(self) -> None:
        """Clear all cached indicators."""
        self.cache.clear()
        logger.info("Indicator cache cleared")

    def list_indicators(self) -> List[str]:
        """Get list of registered indicators.

        Returns:
            List of indicator names
        """
        return list(self._indicators.keys())

    def _generate_cache_key(
        self,
        symbol: str,
        timeframe: str,
        indicator_type: str,
        params: Dict[str, Any]
    ) -> str:
        """Generate unique cache key.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            indicator_type: Type of indicator
            params: Indicator parameters

        Returns:
            Unique cache key
        """
        param_hash = hashlib.md5(
            str(sorted(params.items())).encode()
        ).hexdigest()[:8]
        return f"{symbol}_{timeframe}_{indicator_type}_{param_hash}"

    def _register_default_indicators(self) -> None:
        """Register default set of indicators."""
        self.register('SMA', SMA)
        self.register('EMA', EMA)
        self.register('RSI', RSI)
        self.register('MACD', MACD)
        self.register('BB', BollingerBands)
        self.register('ATR', ATR)
        self.register('ADX', ADX)
        self.register('STOCH', Stochastic)


# Default Indicator Implementations

class SMA(Indicator):
    """Simple Moving Average indicator."""

    def __init__(self, period: int = 20):
        """Initialize SMA.

        Args:
            period: Averaging period
        """
        super().__init__(period=period)

    def calculate(self, data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Calculate SMA values."""
        return calculate_sma_numba(data, self.params['period'])

    def validate_params(self) -> bool:
        """Validate SMA parameters."""
        return self.params.get('period', 0) > 0


class EMA(Indicator):
    """Exponential Moving Average indicator."""

    def __init__(self, period: int = 20):
        """Initialize EMA.

        Args:
            period: Averaging period
        """
        super().__init__(period=period)

    def calculate(self, data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Calculate EMA values."""
        return calculate_ema_numba(data, self.params['period'])

    def validate_params(self) -> bool:
        """Validate EMA parameters."""
        return self.params.get('period', 0) > 0


class RSI(Indicator):
    """Relative Strength Index indicator."""

    def __init__(self, period: int = 14):
        """Initialize RSI.

        Args:
            period: RSI period
        """
        super().__init__(period=period)

    def calculate(self, data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Calculate RSI values."""
        return calculate_rsi_numba(data, self.params['period'])

    def validate_params(self) -> bool:
        """Validate RSI parameters."""
        return 1 < self.params.get('period', 0) < 100


class MACD(Indicator):
    """Moving Average Convergence Divergence indicator."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """Initialize MACD.

        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
        """
        super().__init__(fast=fast, slow=slow, signal=signal)

    def calculate(self, data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Calculate MACD values.

        Returns:
            Array with shape (n, 3) containing [macd, signal, histogram]
        """
        return calculate_macd_numba(
            data,
            self.params['fast'],
            self.params['slow'],
            self.params['signal']
        )

    def validate_params(self) -> bool:
        """Validate MACD parameters."""
        return (
            self.params.get('fast', 0) > 0 and
            self.params.get('slow', 0) > self.params.get('fast', 0) and
            self.params.get('signal', 0) > 0
        )


class BollingerBands(Indicator):
    """Bollinger Bands indicator."""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """Initialize Bollinger Bands.

        Args:
            period: SMA period
            std_dev: Standard deviation multiplier
        """
        super().__init__(period=period, std_dev=std_dev)

    def calculate(self, data: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Calculate Bollinger Bands.

        Returns:
            Array with shape (n, 3) containing [upper, middle, lower]
        """
        return calculate_bollinger_bands_numba(
            data,
            self.params['period'],
            self.params['std_dev']
        )

    def validate_params(self) -> bool:
        """Validate Bollinger Bands parameters."""
        return (
            self.params.get('period', 0) > 1 and
            self.params.get('std_dev', 0) > 0
        )


class ATR(Indicator):
    """Average True Range indicator."""

    def __init__(self, period: int = 14):
        """Initialize ATR.

        Args:
            period: ATR period
        """
        super().__init__(period=period)

    def calculate(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Calculate ATR values.

        Note: Requires OHLC data as DataFrame
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("ATR requires OHLC data as DataFrame")

        return calculate_atr_numba(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            self.params['period']
        )


class ADX(Indicator):
    """Average Directional Index indicator."""

    def __init__(self, period: int = 14):
        """Initialize ADX.

        Args:
            period: ADX period
        """
        super().__init__(period=period)

    def calculate(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Calculate ADX values.

        Note: Requires OHLC data as DataFrame
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("ADX requires OHLC data as DataFrame")

        return calculate_adx_numba(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            self.params['period']
        )


class Stochastic(Indicator):
    """Stochastic Oscillator indicator."""

    def __init__(self, k_period: int = 14, d_period: int = 3):
        """Initialize Stochastic.

        Args:
            k_period: %K period
            d_period: %D period (SMA of %K)
        """
        super().__init__(k_period=k_period, d_period=d_period)

    def calculate(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Calculate Stochastic values.

        Returns:
            Array with shape (n, 2) containing [%K, %D]
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Stochastic requires OHLC data as DataFrame")

        return calculate_stochastic_numba(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            self.params['k_period'],
            self.params['d_period']
        )


# Numba-optimized calculation functions

@jit(nopython=True)
def calculate_sma_numba(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average using Numba.

    Args:
        data: Price data
        period: SMA period

    Returns:
        Array of SMA values
    """
    n = len(data)
    sma = np.empty(n)
    sma[:period - 1] = np.nan

    for i in range(period - 1, n):
        sma[i] = np.mean(data[i - period + 1:i + 1])

    return sma


@jit(nopython=True)
def calculate_ema_numba(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average using Numba.

    Args:
        data: Price data
        period: EMA period

    Returns:
        Array of EMA values
    """
    n = len(data)
    ema = np.empty(n)
    ema[:period - 1] = np.nan

    # Calculate initial SMA
    ema[period - 1] = np.mean(data[:period])

    # Calculate EMA
    multiplier = 2.0 / (period + 1.0)
    for i in range(period, n):
        ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]

    return ema


@jit(nopython=True)
def calculate_rsi_numba(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate Relative Strength Index using Numba.

    Args:
        data: Price data
        period: RSI period

    Returns:
        Array of RSI values
    """
    n = len(data)
    rsi = np.empty(n)
    rsi[:period] = np.nan

    # Calculate price changes
    deltas = np.diff(data)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Calculate initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Calculate RSI
    for i in range(period, n):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))

    return rsi


@jit(nopython=True)
def calculate_macd_numba(
    data: np.ndarray,
    fast: int,
    slow: int,
    signal: int
) -> np.ndarray:
    """Calculate MACD using Numba.

    Args:
        data: Price data
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period

    Returns:
        Array with shape (n, 3) containing [macd, signal, histogram]
    """
    fast_ema = calculate_ema_numba(data, fast)
    slow_ema = calculate_ema_numba(data, slow)

    macd_line = fast_ema - slow_ema

    # Calculate signal line (EMA of MACD)
    signal_line = np.empty(len(data))
    signal_line[:slow + signal - 1] = np.nan

    # Initial signal value
    signal_line[slow + signal - 1] = np.mean(macd_line[slow - 1:slow + signal - 1])

    # Calculate signal EMA
    multiplier = 2.0 / (signal + 1.0)
    for i in range(slow + signal, len(data)):
        signal_line[i] = (macd_line[i] - signal_line[i - 1]) * multiplier + signal_line[i - 1]

    # Calculate histogram
    histogram = macd_line - signal_line

    # Stack results
    result = np.column_stack((macd_line, signal_line, histogram))
    return result


@jit(nopython=True)
def calculate_bollinger_bands_numba(
    data: np.ndarray,
    period: int,
    std_dev: float
) -> np.ndarray:
    """Calculate Bollinger Bands using Numba.

    Args:
        data: Price data
        period: SMA period
        std_dev: Standard deviation multiplier

    Returns:
        Array with shape (n, 3) containing [upper, middle, lower]
    """
    n = len(data)
    middle = calculate_sma_numba(data, period)

    upper = np.empty(n)
    lower = np.empty(n)

    for i in range(period - 1, n):
        std = np.std(data[i - period + 1:i + 1])
        upper[i] = middle[i] + std_dev * std
        lower[i] = middle[i] - std_dev * std

    upper[:period - 1] = np.nan
    lower[:period - 1] = np.nan

    result = np.column_stack((upper, middle, lower))
    return result


@jit(nopython=True)
def calculate_atr_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int
) -> np.ndarray:
    """Calculate Average True Range using Numba.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period

    Returns:
        Array of ATR values
    """
    n = len(high)
    tr = np.empty(n)

    # Calculate True Range
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

    # Calculate ATR as EMA of TR
    atr = np.empty(n)
    atr[:period - 1] = np.nan
    atr[period - 1] = np.mean(tr[:period])

    for i in range(period, n):
        atr[i] = ((atr[i - 1] * (period - 1)) + tr[i]) / period

    return atr


@jit(nopython=True)
def calculate_adx_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int
) -> np.ndarray:
    """Calculate Average Directional Index using Numba.

    Note: Simplified implementation for performance

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period

    Returns:
        Array of ADX values
    """
    n = len(high)
    adx = np.empty(n)

    # Calculate directional movement
    plus_dm = np.empty(n)
    minus_dm = np.empty(n)

    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]

        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        else:
            plus_dm[i] = 0

        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
        else:
            minus_dm[i] = 0

    plus_dm[0] = 0
    minus_dm[0] = 0

    # Calculate ATR
    atr = calculate_atr_numba(high, low, close, period)

    # Calculate DI
    plus_di = 100 * (calculate_ema_numba(plus_dm, period) / atr)
    minus_di = 100 * (calculate_ema_numba(minus_dm, period) / atr)

    # Calculate DX
    dx = np.empty(n)
    for i in range(n):
        if plus_di[i] + minus_di[i] == 0:
            dx[i] = 0
        else:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])

    # Calculate ADX as EMA of DX
    adx = calculate_ema_numba(dx, period)

    return adx


@jit(nopython=True)
def calculate_stochastic_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int,
    d_period: int
) -> np.ndarray:
    """Calculate Stochastic Oscillator using Numba.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period
        d_period: %D period

    Returns:
        Array with shape (n, 2) containing [%K, %D]
    """
    n = len(high)
    k = np.empty(n)
    d = np.empty(n)

    # Calculate %K
    for i in range(k_period - 1, n):
        highest = np.max(high[i - k_period + 1:i + 1])
        lowest = np.min(low[i - k_period + 1:i + 1])

        if highest == lowest:
            k[i] = 50  # Neutral value when no range
        else:
            k[i] = 100 * (close[i] - lowest) / (highest - lowest)

    k[:k_period - 1] = np.nan

    # Calculate %D as SMA of %K
    d = calculate_sma_numba(k, d_period)

    result = np.column_stack((k, d))
    return result