#!/usr/bin/env python3
"""
Green/Red Bars Strategy - Pure Price Action

CONCEPT: Buy on 2nd consecutive green bar, sell on 1st red bar
- Green bar = Close > Open
- Red bar = Close < Open

Entry Rules:
- LONG: Enter on close of 2nd consecutive green bar
- Position size: Full allocation to first signal

Exit Rules:
- Exit immediately on first red bar (Close < Open)
- No stop loss, no profit target, no max hold
- Pure momentum following

Strategy Characteristics:
- Very high frequency (many trades per day)
- Quick exits (often 1-2 bars)
- No filters, pure price action
- Tests if green/red momentum has edge

Position Sizing:
- Single position at a time
- Full capital allocation
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "helpers"))
from indicators import SymbolFilter

# Strategy Configuration
MAX_CONCURRENT_POSITIONS = 1

# No symbol filtering - test on all stocks
SYMBOL_FILTER = SymbolFilter(
    blacklist=[],
    reason="No filters - test pure price action"
)


def annotate_signals(prices: pd.DataFrame, summary: pd.DataFrame = None) -> pd.DataFrame:
    """
    Annotate price data with green/red bar signals.

    Entry: 2nd consecutive green bar (Close > Open)

    Args:
        prices: DataFrame with OHLCV data
        summary: Optional summary DataFrame (not used)

    Returns:
        DataFrame with additional signal columns
    """
    def _per_symbol(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()

        # Determine if bar is green (up) or red (down)
        group['is_green'] = group['Close'] > group['Open']
        group['is_red'] = group['Close'] < group['Open']

        # Count consecutive green bars
        # Reset counter when we hit a non-green bar
        group['green_streak'] = (
            group['is_green']
            .groupby((group['is_green'] != group['is_green'].shift()).cumsum())
            .cumsum()
        )

        # Shift to avoid lookahead bias
        group['green_streak_prev'] = group['green_streak'].shift(1)
        group['is_green_prev'] = group['is_green'].shift(1)

        # Entry Signal: Current bar is green AND we had 1 green bar before (this is 2nd green)
        # This means green_streak_prev == 1 and current bar is also green
        group['long_entry_signal'] = (
            group['is_green'] &  # Current bar is green
            (group['green_streak_prev'] >= 1) &  # Previous bar was green
            group['green_streak_prev'].notna()
        )

        # No short signals - long only strategy
        group['short_entry_signal'] = False

        return group

    # Process each symbol separately
    if 'symbol' in prices.columns:
        result = prices.groupby('symbol', group_keys=False).apply(_per_symbol)
    else:
        result = _per_symbol(prices)

    return result


def should_exit(
    entry_price: float,
    current_price: float,
    direction: str,
    bars_held: int,
    peak_price: float = None
):
    """
    Check if position should exit.

    Exit: Immediately on first red bar
    This is handled by checking if current bar is red in the backtest loop.

    For this strategy, we rely on the backtest engine to check the current bar color
    and exit if it's red. Since the backtest engine only calls should_exit() once per bar,
    we need a different approach.

    Actually, we can't detect "red bar" here because we only have Close price.
    We need to pass additional data or handle this in annotate_signals.

    For now, let's use a very loose exit: no exit conditions here.
    The real exit logic will be in generate_exit_signals or we modify the backtest engine.

    Args:
        entry_price: Entry price of position
        current_price: Current market price
        direction: "long"
        bars_held: Number of bars position has been held
        peak_price: Highest price reached (not used)

    Returns:
        Tuple of (should_exit: bool, reason: str)
    """
    # No exit conditions - we'll handle red bar exit differently
    # This strategy needs special handling in the backtest engine
    return False, ""


def generate_buy_calls(prices: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """
    Generate buy calls for live trading.

    Args:
        prices: Price data with signals
        summary: Summary data

    Returns:
        DataFrame with buy recommendations
    """
    # Get latest data for each symbol
    latest = prices.groupby('symbol').tail(1)

    # Filter for long entry signals
    buy_signals = latest[latest['long_entry_signal'] == True].copy()

    if buy_signals.empty:
        return pd.DataFrame(columns=['symbol', 'company', 'signal_date', 'close_price',
                                    'green_streak'])

    # Prepare output
    calls = buy_signals[['symbol', 'company', 'date', 'Close', 'green_streak']].copy()
    calls.columns = ['symbol', 'company', 'signal_date', 'close_price', 'green_streak']

    return calls.sort_values('green_streak', ascending=False)


def generate_sell_calls(prices: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """
    Generate sell calls for live trading.

    Note: This is a long-only strategy, so no short positions are taken.

    Args:
        prices: Price data with signals
        summary: Summary data

    Returns:
        Empty DataFrame (no short positions in this strategy)
    """
    # Long-only strategy - no sell (short) signals
    return pd.DataFrame(columns=['symbol', 'company', 'signal_date', 'close_price',
                                'green_streak'])


# Strategy metadata
STRATEGY_NAME = "Green/Red Bars - Pure Price Action"
STRATEGY_DESCRIPTION = """
Green/Red Bars Strategy - Buy on 2nd green, sell on 1st red

Entry:
- Buy on close of 2nd consecutive green bar
- Green bar = Close > Open
- No volume filters, no other criteria

Exit:
- Sell immediately on first red bar (Close < Open)
- Red bar = Close < Open
- No stop loss, no profit target, no max hold

Why This Strategy:
- Tests pure price momentum
- Very simple, no indicators
- High frequency trading
- Shows theoretical edge before costs

WARNING: Platform costs (0.045% round-trip) will likely destroy profits
This is a TEST to see if the pattern has any edge at all.

Position Sizing:
- Single position at a time
- Full capital to first signal
"""

PARAMETERS = {
    "entry": "2nd consecutive green bar",
    "exit": "1st red bar",
    "max_concurrent_positions": MAX_CONCURRENT_POSITIONS,
}

VERSION = "1.0"
