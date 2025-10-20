#!/usr/bin/env python3
"""
Cumulative Momentum Strategy V3.0 - Peak Exit Strategy

NEW APPROACH: Enter on momentum, exit at peaks!

Entry Rules (EARLY MOMENTUM ENTRY):
- LONG: Enter when momentum is building (not at troughs!)
- Volume spike + Price momentum + Cumulative volume rising
- Goal: Enter EARLY in the upward move, not try to predict troughs

Exit Rules (INTELLIGENT PEAK DETECTION):
- Track cumulative upward movement since entry
- Exit when price reverses from recent high after significant cumulative up move
- Dynamic exits based on how much we've captured
- Stop Loss: -0.4%
- Max Hold: 60 bars (ride the full wave)

Strategy Characteristics:
- Fewer trades (high selectivity on entry)
- Enter early on momentum building
- Exit intelligently at peaks (not fixed profit targets)
- Captures full upward moves (0.75% avg per analysis)
- Reduces platform costs

Key Innovation vs V2.0:
- V2.0 FAILED: Tried to enter AT troughs (0% win rate)
- V3.0: Enter on MOMENTUM, exit at PEAKS
- Uses cumulative up move tracking to detect peak reversals
- No fixed profit targets - dynamic based on move captured

Position Sizing:
- Single position at a time
- Full capital allocation to best momentum signal
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "helpers"))
from indicators import SymbolFilter

# Strategy Configuration - V3.0 Peak Exit
LOOKBACK_BARS = 20  # Bars to calculate indicators
MOMENTUM_LOOKBACK = 5  # Recent bars for momentum score

# Entry parameters (momentum-based, not trough-based)
VOLUME_THRESHOLD = 1.3  # Lower than volume_momentum (1.5x) to enter earlier
MIN_PRICE_MOVE = 0.002  # Minimum 0.2% price move (enter earlier)
MOMENTUM_SCORE_THRESHOLD = 0.8  # Lower threshold to enter on building momentum

# Exit parameters (peak detection, not fixed targets)
STOP_LOSS = 0.004  # 0.4% stop loss
STOP_LOSS_TOLERANCE_BARS = 5  # Number of consecutive bars below stop loss before exiting
MIN_MOVE_FOR_PEAK_EXIT = 0.005  # Must be up 0.5% before peak exit activates
PEAK_REVERSAL_PCT = 0.002  # 0.2% drop from recent high = peak detected
MAX_HOLD_BARS = 60  # Maximum 60 bars to hold
MAX_CONCURRENT_POSITIONS = 1  # Single position at a time

# Symbol filtering - blacklist defensive stocks
SYMBOL_FILTER = SymbolFilter(
    blacklist=SymbolFilter.DEFENSIVE_STOCKS,
    reason="Defensive stocks showed consistent losses"
)


def annotate_signals(prices: pd.DataFrame, summary: pd.DataFrame = None) -> pd.DataFrame:
    """
    Annotate price data with momentum-based entry signals.

    V3.0 APPROACH: Enter on building momentum, NOT at troughs!

    Entry criteria:
    1. Volume spike (above threshold)
    2. Positive price momentum
    3. Rising cumulative volume (institutional buying)
    4. Momentum score above threshold

    Args:
        prices: DataFrame with OHLCV data
        summary: Optional summary DataFrame (not used)

    Returns:
        DataFrame with additional signal columns
    """
    def _per_symbol(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()

        # Calculate basic indicators
        group['price_change'] = group['Close'].pct_change()
        group['avg_volume'] = group['Volume'].rolling(window=LOOKBACK_BARS, min_periods=LOOKBACK_BARS).mean()
        group['volume_ratio'] = group['Volume'] / group['avg_volume']

        # Cumulative Volume Pressure (OBV-like indicator)
        group['obv_direction'] = np.sign(group['price_change'])
        group['volume_weighted'] = group['Volume'] * group['obv_direction']
        group['cumulative_volume'] = group['volume_weighted'].cumsum()
        group['cumulative_volume_slope'] = group['cumulative_volume'].diff(MOMENTUM_LOOKBACK)

        # Momentum Score = Recent price change + Recent volume pressure
        group['price_momentum'] = group['price_change'].rolling(MOMENTUM_LOOKBACK).sum() * 100
        group['volume_momentum'] = group['volume_ratio'].rolling(MOMENTUM_LOOKBACK).mean()
        group['momentum_score'] = group['price_momentum'] + group['volume_momentum']

        # Calculate recent price move (for entry)
        group['recent_price_move'] = group['price_change'].rolling(MOMENTUM_LOOKBACK).sum()

        # Shift to avoid lookahead bias
        group['volume_ratio_prev'] = group['volume_ratio'].shift(1)
        group['cumulative_volume_slope_prev'] = group['cumulative_volume_slope'].shift(1)
        group['momentum_score_prev'] = group['momentum_score'].shift(1)
        group['recent_price_move_prev'] = group['recent_price_move'].shift(1)
        group['price_change_prev'] = group['price_change'].shift(1)

        # Entry Signals - MOMENTUM-BASED (Early Entry)
        # Enter when momentum is BUILDING, not at troughs
        # Must meet ALL criteria:
        # 1. Volume spike (above threshold)
        # 2. Positive price movement (upward momentum)
        # 3. Rising cumulative volume (institutional buying)
        # 4. Momentum score above threshold
        group['long_entry_signal'] = (
            (group['volume_ratio_prev'] > VOLUME_THRESHOLD) &  # Volume spike
            (group['recent_price_move_prev'] > MIN_PRICE_MOVE) &  # Positive price move
            (group['price_change_prev'] > 0) &  # Current bar positive
            (group['cumulative_volume_slope_prev'] > 0) &  # Cumulative volume building
            (group['momentum_score_prev'] > MOMENTUM_SCORE_THRESHOLD) &  # Momentum building
            group['cumulative_volume_slope_prev'].notna() &
            group['avg_volume'].notna()
        )

        # No short signals - long only strategy
        group['short_entry_signal'] = False

        # Track entry price for exit calculation
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
    bars_held: int,
    peak_price: float = None,
    bars_below_stop_loss: int = 0
):
    """
    Check if position should exit - with PEAK DETECTION and STOP LOSS TOLERANCE.

    V3.0 EXIT STRATEGY:
    - No fixed profit targets
    - Exit when price reverses from peak (intelligent peak detection)
    - Only trigger peak exit after minimum move captured
    - Stop loss with 5-bar tolerance (breathing room)
    - Max hold as safety net

    Args:
        entry_price: Entry price of position
        current_price: Current market price
        direction: "long" (only long positions in this strategy)
        bars_held: Number of bars position has been held
        peak_price: Highest price reached since entry (for peak detection)
        bars_below_stop_loss: Number of consecutive bars below stop loss

    Returns:
        Tuple of (should_exit: bool, reason: str)
    """
    pnl_pct = (current_price - entry_price) / entry_price

    # Stop loss hit (-0.4%) with 5-bar tolerance
    # Only exit if below stop loss for 5+ consecutive bars
    # This gives trades breathing room to recover from temporary dips
    if pnl_pct <= -STOP_LOSS:
        if bars_below_stop_loss >= STOP_LOSS_TOLERANCE_BARS:
            return True, "stop_loss"
        # Still below stop loss but within tolerance - don't exit yet
        return False, ""

    # PEAK DETECTION EXIT
    # Only activate after we've captured minimum move (0.5%)
    # This ensures we don't exit too early on small wiggles
    if peak_price is not None and pnl_pct >= MIN_MOVE_FOR_PEAK_EXIT:
        # Calculate drop from peak
        drop_from_peak = (peak_price - current_price) / peak_price

        # If price has dropped PEAK_REVERSAL_PCT (0.2%) from peak, exit
        # This indicates we've hit the peak and it's reversing
        if drop_from_peak >= PEAK_REVERSAL_PCT:
            return True, "peak_detected"

    # Max hold period (60 bars = 60 minutes)
    if bars_held >= MAX_HOLD_BARS:
        return True, "max_hold_bars"

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
                                    'momentum_score', 'cumulative_volume_slope'])

    # Prepare output
    calls = buy_signals[['symbol', 'company', 'date', 'Close', 'momentum_score', 'cumulative_volume_slope']].copy()
    calls.columns = ['symbol', 'company', 'signal_date', 'close_price', 'momentum_score', 'cumulative_volume_slope']

    return calls.sort_values('momentum_score', ascending=False)


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
                                'momentum_score', 'cumulative_volume_slope'])


# Strategy metadata
STRATEGY_NAME = "Cumulative Momentum V3.0 - Peak Exit Strategy"
STRATEGY_DESCRIPTION = """
Cumulative Momentum Strategy V3.0 - Enter on momentum, exit at peaks

NEW APPROACH (learned from V2.0 failure):
V2.0 tried to enter AT troughs = 0% win rate (too hard to predict)
V3.0 enters on BUILDING MOMENTUM, exits at DETECTED PEAKS

Entry (MOMENTUM-BASED - Early Entry):
Enter when momentum is BUILDING (not at troughs):
1. Volume spike >1.3x average (enter earlier than baseline 1.5x)
2. Positive price move >0.2% over 5 bars
3. Current bar moving up
4. Rising cumulative volume (institutional buying)
5. Momentum score >0.8 (building momentum)

Exit (INTELLIGENT PEAK DETECTION):
- NO fixed profit targets
- Wait for minimum 0.5% gain
- Exit when price drops 0.2% from peak (peak reversal detected)
- Stop Loss: 0.4%
- Max Hold: 60 bars

Why This Works:
- Analysis showed 167 trough→peak moves avg 0.75%
- 53% reach 0.6%+, 13% reach 1.0%+
- Peak detection captures MORE of each move
- Fewer trades (high selectivity) = lower platform costs

Position Sizing:
- Single position at a time
- Full capital to best momentum signal

V2.0 → V3.0 Changes:
1. ENTRY: Stop trying to predict troughs → Enter on momentum building
2. EXIT: Fixed targets → Dynamic peak detection
3. GOAL: Capture full upward moves instead of guessing bottoms
"""

PARAMETERS = {
    "lookback_bars": LOOKBACK_BARS,
    "momentum_lookback": MOMENTUM_LOOKBACK,
    "volume_threshold": VOLUME_THRESHOLD,
    "min_price_move": MIN_PRICE_MOVE,
    "momentum_score_threshold": MOMENTUM_SCORE_THRESHOLD,
    "stop_loss": STOP_LOSS,
    "min_move_for_peak_exit": MIN_MOVE_FOR_PEAK_EXIT,
    "peak_reversal_pct": PEAK_REVERSAL_PCT,
    "max_hold_bars": MAX_HOLD_BARS,
    "blacklist_count": len(SYMBOL_FILTER.blacklist),
}

VERSION = "3.0"
