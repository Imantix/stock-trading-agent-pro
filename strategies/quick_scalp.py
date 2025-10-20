#!/usr/bin/env python3
"""
Quick Scalp Strategy - Maximum Returns with Minimal Trades

PHILOSOPHY:
Starting fresh with ₹500,000 - how to maximize returns while minimizing platform costs?

KEY INSIGHT:
- Platform cost is 0.045% round-trip
- To earn 10% return with 0.5% targets → need 22 winning trades
- To earn 10% return with 0.8% targets → need only 14 winning trades
- QUALITY over QUANTITY: fewer, bigger wins

STRATEGY:
Entry on EXTREME setups only (not just any momentum):
- Volume surge >3.0x average (extreme, not mild 1.5x)
- Price move >0.5% in single bar (significant gap, not tiny 0.3%)
- Time-of-day filter (avoid midday chop - first 90min OR last 60min)
- Trend confirmation (2+ consecutive positive bars)

Exit FAST and DISCIPLINED:
- Profit Target: 0.8% (net 0.755% after 0.045% cost)
- Stop Loss: 0.25% (tight - cut losers fast)
- Time Stop: 10 bars max (don't wait around)
- Risk/Reward: 3.2:1 (excellent)

Position Sizing:
- Single position at a time (all-in on best setup)
- Full capital allocation to highest-conviction trades only

Expected Performance:
- 1-2 trades/day @ 50% win rate → 4-9% return over 21 days
- Focus on QUALITY setups, not trade count
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "helpers"))
from indicators import SymbolFilter

# Strategy Configuration
LOOKBACK_BARS = 20  # Bars to calculate average volume

# Entry parameters (VERY selective)
VOLUME_THRESHOLD = 3.0  # Extreme volume (>3x average, not 1.5x)
PRICE_MOVE_THRESHOLD = 0.005  # 0.5% single-bar move (significant, not 0.3%)
TREND_BARS = 2  # Require 2+ consecutive up bars for trend confirmation

# Time-of-day filter (avoid midday chop)
TRADING_START_TIME = "09:15"
MORNING_CUTOFF_TIME = "10:45"  # First 90 minutes
AFTERNOON_START_TIME = "14:30"  # Last 60 minutes
TRADING_END_TIME = "15:30"

# Exit parameters (fast and disciplined)
PROFIT_TARGET = 0.008  # 0.8% profit (net 0.755% after 0.045% cost)
STOP_LOSS = 0.0025  # 0.25% stop loss (tight)
MAX_HOLD_BARS = 10  # Maximum 10 bars (10 minutes)
MAX_CONCURRENT_POSITIONS = 1  # Single position only

# Symbol filtering - avoid defensive/slow movers
SYMBOL_FILTER = SymbolFilter(
    blacklist=SymbolFilter.DEFENSIVE_STOCKS,
    reason="Defensive stocks lack the volatility needed for quick scalps"
)


def annotate_signals(prices: pd.DataFrame, summary: pd.DataFrame = None) -> pd.DataFrame:
    """
    Annotate price data with quick scalp signals.

    VERY selective entry criteria:
    1. Extreme volume (>3x average)
    2. Significant price move (>0.5% in single bar)
    3. Time-of-day filter (morning burst OR afternoon surge)
    4. Trend confirmation (2+ consecutive up bars)

    Args:
        prices: DataFrame with OHLCV data
        summary: Optional summary DataFrame (not used)

    Returns:
        DataFrame with additional signal columns
    """
    def _per_symbol(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()

        # Calculate volume indicators
        group['avg_volume'] = group['Volume'].rolling(window=LOOKBACK_BARS, min_periods=LOOKBACK_BARS).mean()
        group['volume_ratio'] = group['Volume'] / group['avg_volume']

        # Calculate price change (single bar)
        group['price_change'] = group['Close'].pct_change()

        # Trend confirmation: count consecutive up bars
        group['up_bar'] = (group['Close'] > group['Close'].shift(1)).astype(int)
        group['consecutive_up'] = group['up_bar'].rolling(TREND_BARS).sum()

        # Time-of-day filter
        if 'datetime' in group.columns:
            group['time'] = pd.to_datetime(group['datetime']).dt.time
            group['morning_session'] = (
                (group['time'] >= pd.to_datetime(TRADING_START_TIME).time()) &
                (group['time'] <= pd.to_datetime(MORNING_CUTOFF_TIME).time())
            )
            group['afternoon_session'] = (
                (group['time'] >= pd.to_datetime(AFTERNOON_START_TIME).time()) &
                (group['time'] <= pd.to_datetime(TRADING_END_TIME).time())
            )
            group['favorable_time'] = group['morning_session'] | group['afternoon_session']
        else:
            group['favorable_time'] = True  # Default to True if no datetime

        # Shift to avoid lookahead bias
        group['volume_ratio_prev'] = group['volume_ratio'].shift(1)
        group['price_change_prev'] = group['price_change'].shift(1)
        group['consecutive_up_prev'] = group['consecutive_up'].shift(1)
        group['favorable_time_prev'] = group['favorable_time'].shift(1)

        # LONG ENTRY SIGNAL (very selective)
        # ALL conditions must be met:
        # 1. Extreme volume (>3x average)
        # 2. Significant price move (>0.5%)
        # 3. Trend confirmation (2+ consecutive up bars)
        # 4. Favorable time of day
        group['long_entry_signal'] = (
            (group['volume_ratio_prev'] > VOLUME_THRESHOLD) &  # Extreme volume
            (group['price_change_prev'] > PRICE_MOVE_THRESHOLD) &  # Significant move
            (group['consecutive_up_prev'] >= TREND_BARS) &  # Trend confirmed
            (group['favorable_time_prev'] == True) &  # Good time of day
            group['avg_volume'].notna()
        )

        # No short signals - long only
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
    peak_price: float = None
):
    """
    Check if position should exit based on profit target, stop loss, or time stop.

    Fast and disciplined exits:
    - Profit Target: 0.8% (take profit quickly)
    - Stop Loss: 0.25% (cut losers fast)
    - Time Stop: 10 bars (don't wait around)

    Args:
        entry_price: Entry price of position
        current_price: Current market price
        direction: "long" (only long positions in this strategy)
        bars_held: Number of bars position has been held
        peak_price: Highest price reached since entry (not used in this strategy)

    Returns:
        Tuple of (should_exit: bool, reason: str)
    """
    pnl_pct = (current_price - entry_price) / entry_price

    # Profit target hit (+0.8%)
    if pnl_pct >= PROFIT_TARGET:
        return True, "profit_target"

    # Stop loss hit (-0.25%)
    if pnl_pct <= -STOP_LOSS:
        return True, "stop_loss"

    # Time stop (10 bars maximum)
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
STRATEGY_NAME = "Quick Scalp V1.0 - Maximum Returns, Minimal Trades"
STRATEGY_DESCRIPTION = """
Quick Scalp Strategy - Designed independently to maximize returns with minimal platform costs

PHILOSOPHY:
Start with ₹500,000 and think: how to maximize returns while minimizing costs?

ANSWER: QUALITY over QUANTITY
- Fewer trades = lower platform costs
- Bigger targets = fewer trades needed
- Extreme setups only = higher win rate

ENTRY (Ultra-Selective):
- Volume >3.0x average (extreme surge, not mild)
- Price >0.5% single-bar move (significant gap)
- 2+ consecutive up bars (trend confirmed)
- Time filter: First 90min OR last 60min (avoid midday chop)

EXIT (Fast & Disciplined):
- Profit Target: 0.8% (net 0.755% after 0.045% cost)
- Stop Loss: 0.25% (cut losers fast)
- Max Hold: 10 bars (10 minutes max)
- Risk/Reward: 3.2:1

POSITION SIZING:
- Single position at a time
- All-in on best setup only

EXPECTED PERFORMANCE:
- 1-2 trades/day @ 50% win rate → 4-9% return (21 days)
- Focus on QUALITY, not quantity

KEY ADVANTAGES OVER EXISTING STRATEGIES:
1. Higher profit targets (0.8% vs 0.6%) → fewer trades needed
2. Tighter stop loss (0.25% vs 0.3%) → cut losers faster
3. More selective entry (3x vol vs 1.5x) → higher quality setups
4. Time-of-day filter → avoid midday noise
5. Trend confirmation → higher probability setups
"""

PARAMETERS = {
    "lookback_bars": LOOKBACK_BARS,
    "volume_threshold": VOLUME_THRESHOLD,
    "price_move_threshold": PRICE_MOVE_THRESHOLD,
    "trend_bars": TREND_BARS,
    "profit_target": PROFIT_TARGET,
    "stop_loss": STOP_LOSS,
    "max_hold_bars": MAX_HOLD_BARS,
    "max_concurrent_positions": MAX_CONCURRENT_POSITIONS,
    "blacklist_count": len(SYMBOL_FILTER.blacklist),
}

VERSION = "1.0"
