#!/usr/bin/env python3
"""
Technical Indicators Library

Reusable indicators and filters for trading strategies.
All functions work on pandas DataFrames with OHLCV data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


# ============================================================================
# VOLATILITY INDICATORS
# ============================================================================

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    ATR measures volatility by decomposing the entire range of price movement.
    Useful for setting dynamic stop losses based on stock's natural volatility.

    Args:
        df: DataFrame with 'High', 'Low', 'Close' columns
        period: Lookback period (default: 14)

    Returns:
        Series with ATR values

    Example:
        df['atr_14'] = calculate_atr(df, period=14)
        dynamic_stop = entry_price - (2 * df['atr_14'].iloc[-1])
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    # True Range is the maximum of:
    # 1. Current High - Current Low
    # 2. abs(Current High - Previous Close)
    # 3. abs(Current Low - Previous Close)
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is the moving average of True Range
    atr = true_range.rolling(window=period, min_periods=period).mean()

    return atr


def calculate_adr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Average Daily Range (ADR) as percentage.

    ADR measures the average percentage movement from high to low.
    Used to filter stocks with sufficient volatility for momentum trading.

    Args:
        df: DataFrame with 'High', 'Low', 'Close' columns
        period: Lookback period (default: 20)

    Returns:
        Series with ADR as percentage (e.g., 2.5 means 2.5% average range)

    Example:
        df['adr_pct'] = calculate_adr(df, period=20)
        # Filter: Only trade stocks with ADR > 2%
        signals = signals & (df['adr_pct'] > 2.0)
    """
    daily_range = df['High'] - df['Low']
    adr = daily_range.rolling(window=period, min_periods=period).mean()
    adr_pct = (adr / df['Close']) * 100

    return adr_pct


def calculate_historical_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Historical Volatility (annualized standard deviation of returns).

    Args:
        df: DataFrame with 'Close' column
        period: Lookback period (default: 20)

    Returns:
        Series with annualized volatility as percentage

    Example:
        df['volatility'] = calculate_historical_volatility(df, period=20)
    """
    returns = df['Close'].pct_change()
    volatility = returns.rolling(window=period, min_periods=period).std() * np.sqrt(252) * 100

    return volatility


# ============================================================================
# MOMENTUM INDICATORS
# ============================================================================

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    RSI measures the speed and magnitude of price changes.
    Values: 0-100 (>70 overbought, <30 oversold)

    Args:
        df: DataFrame with 'Close' column
        period: Lookback period (default: 14)

    Returns:
        Series with RSI values (0-100)

    Example:
        df['rsi'] = calculate_rsi(df, period=14)
        # Avoid overbought: signals & (df['rsi'] < 70)
    """
    close = df['Close']
    delta = close.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def consecutive_up_days(df: pd.DataFrame, n_days: int = 2) -> pd.Series:
    """
    Check if price has been up for N consecutive days.

    Useful for confirming sustained momentum before entry.

    Args:
        df: DataFrame with 'Close' column
        n_days: Number of consecutive up days required (default: 2)

    Returns:
        Boolean Series (True if last N days were all up)

    Example:
        df['momentum_confirmed'] = consecutive_up_days(df, n_days=2)
        # Require 2 consecutive up days
        signals = signals & df['momentum_confirmed']
    """
    returns = df['Close'].pct_change()

    # Check if all of the last n_days were positive
    result = pd.Series(False, index=df.index)

    for i in range(n_days):
        if i == 0:
            result = returns > 0
        else:
            result = result & (returns.shift(i) > 0)

    return result


def price_above_ma(df: pd.DataFrame, period: int = 50) -> pd.Series:
    """
    Check if price is above its moving average.

    Confirms overall uptrend before taking long positions.

    Args:
        df: DataFrame with 'Close' column
        period: Moving average period (default: 50)

    Returns:
        Boolean Series (True if price > MA)

    Example:
        df['uptrend'] = price_above_ma(df, period=50)
        # Only long when in uptrend
        signals = signals & df['uptrend']
    """
    ma = df['Close'].rolling(window=period, min_periods=period).mean()
    return df['Close'] > ma


# ============================================================================
# VOLUME INDICATORS
# ============================================================================

def calculate_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate current volume relative to average volume.

    Args:
        df: DataFrame with 'Volume' column
        period: Lookback period (default: 20)

    Returns:
        Series with volume ratio (e.g., 1.5 means 1.5x average volume)

    Example:
        df['volume_ratio'] = calculate_volume_ratio(df, period=20)
        # Volume spike: volume_ratio > 1.5
    """
    avg_volume = df['Volume'].rolling(window=period, min_periods=period).mean()
    volume_ratio = df['Volume'] / avg_volume

    return volume_ratio


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).

    OBV measures buying and selling pressure as a cumulative indicator.

    Args:
        df: DataFrame with 'Close' and 'Volume' columns

    Returns:
        Series with OBV values

    Example:
        df['obv'] = calculate_obv(df)
        df['obv_ma'] = df['obv'].rolling(20).mean()
        # OBV rising: df['obv'] > df['obv_ma']
    """
    price_change = df['Close'].diff()
    obv = (np.sign(price_change) * df['Volume']).fillna(0).cumsum()

    return obv


# ============================================================================
# FILTERING FUNCTIONS
# ============================================================================

def filter_by_volatility(
    df: pd.DataFrame,
    min_adr: float = 2.0,
    period: int = 20
) -> pd.Series:
    """
    Filter stocks by minimum Average Daily Range.

    Excludes low-volatility stocks unsuitable for momentum trading.

    Args:
        df: DataFrame with OHLC columns
        min_adr: Minimum ADR percentage (default: 2.0%)
        period: Lookback period (default: 20)

    Returns:
        Boolean Series (True if stock meets volatility requirement)

    Example:
        df['volatile_enough'] = filter_by_volatility(df, min_adr=2.0)
        signals = signals & df['volatile_enough']
    """
    adr_pct = calculate_adr(df, period=period)
    return adr_pct >= min_adr


def filter_by_price_range(
    df: pd.DataFrame,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None
) -> pd.Series:
    """
    Filter stocks by price range.

    Excludes very low-priced (penny stocks) or very high-priced stocks.

    Args:
        df: DataFrame with 'Close' column
        min_price: Minimum price (default: None)
        max_price: Maximum price (default: None)

    Returns:
        Boolean Series (True if price within range)

    Example:
        # Avoid penny stocks and very expensive stocks
        df['good_price'] = filter_by_price_range(df, min_price=10, max_price=500)
        signals = signals & df['good_price']
    """
    result = pd.Series(True, index=df.index)

    if min_price is not None:
        result = result & (df['Close'] >= min_price)

    if max_price is not None:
        result = result & (df['Close'] <= max_price)

    return result


def filter_by_liquidity(
    df: pd.DataFrame,
    min_dollar_volume: float = 10_000_000,
    period: int = 20
) -> pd.Series:
    """
    Filter stocks by minimum average dollar volume.

    Ensures sufficient liquidity for trading.

    Args:
        df: DataFrame with 'Close' and 'Volume' columns
        min_dollar_volume: Minimum average daily dollar volume (default: $10M)
        period: Lookback period (default: 20)

    Returns:
        Boolean Series (True if stock meets liquidity requirement)

    Example:
        df['liquid'] = filter_by_liquidity(df, min_dollar_volume=10_000_000)
        signals = signals & df['liquid']
    """
    dollar_volume = df['Close'] * df['Volume']
    avg_dollar_volume = dollar_volume.rolling(window=period, min_periods=period).mean()

    return avg_dollar_volume >= min_dollar_volume


# ============================================================================
# BLACKLIST/WHITELIST MANAGEMENT
# ============================================================================

class SymbolFilter:
    """
    Manage symbol blacklists and whitelists for strategy filtering.

    Example:
        # In strategy file
        from helpers.indicators import SymbolFilter

        # Create filter with defensive stocks blacklisted
        symbol_filter = SymbolFilter(
            blacklist=['VZ', 'KO', 'PG', 'JNJ', 'PEP', 'ABT'],
            reason="Low volatility defensive stocks"
        )

        # Apply to signals
        df['allowed'] = df['symbol'].apply(symbol_filter.is_allowed)
        signals = signals & df['allowed']
    """

    # Predefined blacklists
    DEFENSIVE_STOCKS = ['VZ', 'KO', 'PG', 'JNJ', 'PEP', 'ABT', 'MCD', 'WMT']
    UTILITIES = ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL']
    LOW_MOMENTUM = DEFENSIVE_STOCKS + UTILITIES

    def __init__(
        self,
        blacklist: Optional[List[str]] = None,
        whitelist: Optional[List[str]] = None,
        reason: str = ""
    ):
        """
        Initialize symbol filter.

        Args:
            blacklist: List of symbols to exclude
            whitelist: List of symbols to include (if set, only these allowed)
            reason: Optional description of why symbols are filtered
        """
        self.blacklist = set(blacklist or [])
        self.whitelist = set(whitelist or []) if whitelist else None
        self.reason = reason

    def is_allowed(self, symbol: str) -> bool:
        """Check if symbol is allowed to trade."""
        # If whitelist exists, only allow whitelisted symbols
        if self.whitelist is not None:
            return symbol in self.whitelist

        # Otherwise, allow everything except blacklisted
        return symbol not in self.blacklist

    def filter_dataframe(self, df: pd.DataFrame, symbol_col: str = 'symbol') -> pd.Series:
        """
        Apply filter to DataFrame.

        Args:
            df: DataFrame with symbol column
            symbol_col: Name of symbol column (default: 'symbol')

        Returns:
            Boolean Series indicating which rows are allowed
        """
        return df[symbol_col].apply(self.is_allowed)


# ============================================================================
# STOP LOSS CALCULATORS
# ============================================================================

def calculate_dynamic_stop(
    entry_price: float,
    atr: float,
    direction: str = "long",
    atr_multiplier: float = 2.0
) -> float:
    """
    Calculate dynamic stop loss based on ATR.

    Stop loss adapts to stock's volatility rather than using fixed percentage.

    Args:
        entry_price: Entry price of position
        atr: Current ATR value
        direction: "long" or "short"
        atr_multiplier: How many ATRs away to place stop (default: 2.0)

    Returns:
        Stop loss price

    Example:
        # In strategy's should_exit() function
        atr_value = get_current_atr(symbol)
        stop_price = calculate_dynamic_stop(
            entry_price=entry_price,
            atr=atr_value,
            direction="long",
            atr_multiplier=2.0
        )

        if current_price <= stop_price:
            return True, "stop_loss"
    """
    if direction == "long":
        return entry_price - (atr_multiplier * atr)
    else:  # short
        return entry_price + (atr_multiplier * atr)


def calculate_percent_stop(
    entry_price: float,
    direction: str = "long",
    stop_pct: float = 0.005
) -> float:
    """
    Calculate fixed percentage stop loss.

    Args:
        entry_price: Entry price of position
        direction: "long" or "short"
        stop_pct: Stop loss percentage (default: 0.005 = 0.5%)

    Returns:
        Stop loss price

    Example:
        stop_price = calculate_percent_stop(entry_price=100, direction="long", stop_pct=0.0075)
        # Returns 99.25 (0.75% below entry)
    """
    if direction == "long":
        return entry_price * (1 - stop_pct)
    else:  # short
        return entry_price * (1 + stop_pct)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def annotate_all_indicators(
    df: pd.DataFrame,
    atr_period: int = 14,
    adr_period: int = 20,
    rsi_period: int = 14,
    volume_period: int = 20,
    ma_period: int = 50
) -> pd.DataFrame:
    """
    Add all common indicators to a DataFrame.

    Convenience function to add multiple indicators at once.

    Args:
        df: DataFrame with OHLCV columns
        atr_period: ATR lookback period
        adr_period: ADR lookback period
        rsi_period: RSI lookback period
        volume_period: Volume ratio lookback period
        ma_period: Moving average period

    Returns:
        DataFrame with all indicators added

    Example:
        df = annotate_all_indicators(df)
        # Now has: atr, adr_pct, rsi, volume_ratio, ma, price_above_ma
    """
    df = df.copy()

    # Volatility indicators
    df['atr'] = calculate_atr(df, period=atr_period)
    df['adr_pct'] = calculate_adr(df, period=adr_period)
    df['volatility'] = calculate_historical_volatility(df, period=adr_period)

    # Momentum indicators
    df['rsi'] = calculate_rsi(df, period=rsi_period)
    df['ma'] = df['Close'].rolling(window=ma_period, min_periods=ma_period).mean()
    df['price_above_ma'] = df['Close'] > df['ma']

    # Volume indicators
    df['volume_ratio'] = calculate_volume_ratio(df, period=volume_period)
    df['obv'] = calculate_obv(df)

    return df


def print_indicator_summary(df: pd.DataFrame, symbol: str = "") -> None:
    """
    Print summary statistics for indicators.

    Useful for debugging and understanding indicator values.

    Args:
        df: DataFrame with indicators
        symbol: Optional symbol name for display
    """
    title = f"Indicator Summary for {symbol}" if symbol else "Indicator Summary"
    print("=" * 60)
    print(title)
    print("=" * 60)

    indicators = ['atr', 'adr_pct', 'rsi', 'volume_ratio', 'volatility']

    for ind in indicators:
        if ind in df.columns:
            values = df[ind].dropna()
            if len(values) > 0:
                print(f"{ind:20s} | Mean: {values.mean():8.2f} | "
                      f"Min: {values.min():8.2f} | Max: {values.max():8.2f}")

    print("=" * 60)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of indicators library.
    """
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100)
    sample_data = pd.DataFrame({
        'date': dates,
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 105,
        'Low': np.random.randn(100).cumsum() + 95,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 100)
    })

    # Add all indicators
    sample_data = annotate_all_indicators(sample_data)

    # Print summary
    print_indicator_summary(sample_data, symbol="SAMPLE")

    # Example: Filter for tradeable conditions
    tradeable = (
        filter_by_volatility(sample_data, min_adr=2.0) &
        filter_by_liquidity(sample_data, min_dollar_volume=5_000_000) &
        price_above_ma(sample_data, period=50)
    )

    print(f"\nTradeable days: {tradeable.sum()} out of {len(tradeable)}")
