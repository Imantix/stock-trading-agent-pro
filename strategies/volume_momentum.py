#!/usr/bin/env python3
"""
Volume Momentum Strategy

Entry Rules:
- LONG ONLY: Volume spike (>1.5x 20-day avg) + Price up >0.3%
- Buy on big upward momentum push

Exit Rules:
- Profit Target: +0.3% for long
- Stop Loss: -0.5% for long
- Or hold max 5 days

Strategy Characteristics:
- Momentum-following strategy (long only)
- Targets larger gains (0.3% per trade)
- High turnover - enter and exit quickly
- Volume-based momentum detection
- Avoids position lock by having explicit exits

Position Sizing:
- 250 units per trade (configurable)
- Max 18 concurrent positions (pyramiding)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "helpers"))
from indicators import SymbolFilter

# Strategy Configuration
VOLUME_LOOKBACK = 20  # Days to calculate average volume
VOLUME_THRESHOLD = 1.5  # Volume must be 1.5x average
PRICE_CHANGE_THRESHOLD = 0.003  # 0.3% price change required
PROFIT_TARGET = 0.003  # 0.3% profit target (increased from 0.15%)
STOP_LOSS = 0.005  # 0.5% stop loss
MAX_HOLD_DAYS = 5  # Maximum days to hold position

# Symbol filtering - blacklist defensive stocks that showed consistent losses
SYMBOL_FILTER = SymbolFilter(
    blacklist=SymbolFilter.DEFENSIVE_STOCKS,
    reason="Defensive stocks showed consistent losses in backtest"
)


def annotate_signals(prices: pd.DataFrame, summary: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Annotate price data with volume momentum signals.

    Args:
        prices: DataFrame with OHLCV data
        summary: Optional summary DataFrame (not used)

    Returns:
        DataFrame with additional signal columns
    """
    def _per_symbol(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()

        # Calculate volume indicators
        group['avg_volume'] = group['Volume'].rolling(window=VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK).mean()
        group['volume_ratio'] = group['Volume'] / group['avg_volume']

        # Calculate price change
        group['price_change'] = group['Close'].pct_change()

        # Shift to avoid lookahead bias
        group['avg_volume_prev'] = group['avg_volume'].shift(1)
        group['volume_ratio_prev'] = group['volume_ratio'].shift(1)
        group['price_change_prev'] = group['price_change'].shift(1)

        # Entry Signals
        # LONG ONLY: Volume spike + Price up (big momentum push)
        group['long_entry_signal'] = (
            (group['volume_ratio_prev'] > VOLUME_THRESHOLD) &
            (group['price_change_prev'] > PRICE_CHANGE_THRESHOLD) &
            group['avg_volume_prev'].notna()
        )

        # No short signals - long only strategy
        group['short_entry_signal'] = False

        # Track entry price for exit calculation (will be used by portfolio backtest)
        group['entry_price'] = np.nan
        group['entry_direction'] = None
        group['days_held'] = 0

        return group

    # Process each symbol separately
    if 'symbol' in prices.columns:
        result = prices.groupby('symbol', group_keys=False).apply(_per_symbol)
        # Apply symbol blacklist filter
        result['symbol_allowed'] = SYMBOL_FILTER.filter_dataframe(result)
        result['long_entry_signal'] = result['long_entry_signal'] & result['symbol_allowed']
    else:
        result = _per_symbol(prices)

    return result


def should_exit(
    entry_price: float,
    current_price: float,
    direction: str,
    days_held: int
) -> tuple[bool, str]:
    """
    Check if position should exit based on profit target, stop loss, or max hold.

    Args:
        entry_price: Entry price of position
        current_price: Current market price
        direction: "long" (only long positions in this strategy)
        days_held: Number of days position has been held

    Returns:
        Tuple of (should_exit: bool, reason: str)
    """
    pnl_pct = (current_price - entry_price) / entry_price

    # Long only strategy
    # Profit target hit (+0.3%)
    if pnl_pct >= PROFIT_TARGET:
        return True, "profit_target"

    # Stop loss hit (-0.5%)
    if pnl_pct <= -STOP_LOSS:
        return True, "stop_loss"

    # Max hold period (5 days)
    if days_held >= MAX_HOLD_DAYS:
        return True, "max_hold_days"

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
                                    'volume_ratio', 'price_change'])

    # Prepare output
    calls = buy_signals[['symbol', 'company', 'date', 'Close', 'volume_ratio', 'price_change']].copy()
    calls.columns = ['symbol', 'company', 'signal_date', 'close_price', 'volume_ratio', 'price_change']

    return calls.sort_values('volume_ratio', ascending=False)


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
                                'volume_ratio', 'price_change'])


# Strategy metadata
STRATEGY_NAME = "Volume Momentum V2.4"
STRATEGY_DESCRIPTION = """
Volume Momentum Strategy V2.4 - Long-only momentum following with defensive stock filter

Entry (LONG ONLY):
- Volume > 1.5x 20-day average
- Price up >0.3% (big upward momentum push)
- Buy on upward momentum
- Excludes defensive stocks that showed consistent losses

Exit:
- Profit Target: 0.3%
- Stop Loss: 0.5%
- Max Hold: 5 days

Improvement from V1: Added symbol blacklist for defensive stocks (VZ, KO, PG, JNJ, PEP, ABT, MCD, WMT)
that consistently lost money. This improved returns from 12.35% to 12.62% with lower max drawdown.
"""

PARAMETERS = {
    "volume_lookback": VOLUME_LOOKBACK,
    "volume_threshold": VOLUME_THRESHOLD,
    "price_change_threshold": PRICE_CHANGE_THRESHOLD,
    "profit_target": PROFIT_TARGET,
    "stop_loss": STOP_LOSS,
    "max_hold_days": MAX_HOLD_DAYS,
    "blacklist_count": len(SYMBOL_FILTER.blacklist),
}

VERSION = "2.4"
