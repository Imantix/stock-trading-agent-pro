#!/usr/bin/env python3
"""
Volume Momentum Strategy

Entry Rules:
- LONG ONLY: Volume spike (>1.5x 20-bar avg) + Price up >0.3%
- Require cumulative intraday momentum (>0.2% from day open)
- Buy on big upward momentum push while day trend is positive

Exit Rules:
- Profit Target: +0.6% for long (optimized for platform costs)
- Stop Loss: -0.3% for long (tighter to cut false signals faster)
- Or hold max 15 bars (15 minutes on 1m data)

Strategy Characteristics:
- Momentum-following strategy (long only)
- Targets 0.6% gains per trade (optimized for 0.0225% platform cost)
- Medium hold time (up to 15 minutes)
- Volume-based momentum detection
- Avoids position lock by having explicit exits

Position Sizing:
- ALL-IN on up to three concurrent positions
- Full capital allocation to highest volume ratio signals
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "helpers"))
from indicators import SymbolFilter

# Strategy Configuration
VOLUME_LOOKBACK = 20  # Bars to calculate average volume
VOLUME_THRESHOLD = 1.5  # Volume must be 1.5x average (Config 1 baseline)
PRICE_CHANGE_THRESHOLD = 0.003  # 0.3% price change required (single bar momentum)
DAY_MOMENTUM_THRESHOLD = 0.002  # Require 0.2% day move in our favor (cumulative momentum)
PROFIT_TARGET = 0.006  # 0.6% profit target (optimized for 0.0225% platform cost)
STOP_LOSS = 0.003  # 0.3% stop loss (tighter to cut false momentum signals faster)
MAX_HOLD_BARS = 15  # Maximum bars to hold position (15 mins on 1m data)
MAX_CONCURRENT_POSITIONS = 3  # Maximum number of concurrent positions (allow more trades per day)
PEAK_PROFIT_TRIGGER = 0.004  # Require 0.4% gain before peak trailing activates
TRAILING_STOP_PCT = 0.003  # 0.3% trailing stop from peak once trigger met
MAX_DAILY_POSITIONS = 3  # Cap total new entries per trading day

# BSE30-specific blacklist based on backtest performance and sector rules
# Banking Sector - Full blacklist (user preference: avoid banking volatility)
BSE30_BANKING_SECTOR = [
    'AXISBANK.BO',     # Axis Bank
    'HDFCBANK.BO',     # HDFC Bank
    'ICICIBANK.BO',    # ICICI Bank
    'KOTAKBANK.BO',    # Kotak Mahindra Bank
    'SBIN.BO',         # State Bank of India
]

# Other poor performers (consistent losses: PnL < -500 AND Win Rate <= 25%)
BSE30_POOR_PERFORMERS = [
    'HCLTECH.BO',      # IT - Lost ₹6,229 (0% win rate)
    'LT.BO',           # Construction - Lost ₹6,171 (0% win rate)
    'ASIANPAINT.BO',   # Consumer Durables - Lost ₹4,644 (0% win rate)
    'TITAN.BO',        # Consumer Durables - Lost ₹2,543 (0% win rate)
    'INFY.BO',         # IT - Lost ₹2,107 (0% win rate)
    'ULTRACEMCO.BO',   # Cement - Lost ₹1,870 (0% win rate)
    'BHARTIARTL.BO',   # Telecom - Lost ₹1,173 (0% win rate)
    'BAJFINANCE.BO',   # Finance - Lost ₹661 (0% win rate)
    'NTPC.BO',         # Power - Lost ₹650 (0% win rate)
    'POWERGRID.BO',    # Power - Lost ₹538 (25% win rate)
]

# Combined blacklist
BSE30_BLACKLIST = BSE30_BANKING_SECTOR + BSE30_POOR_PERFORMERS

# Best performers to prioritize (optional - can be used with whitelist mode)
BSE30_TOP_PERFORMERS = [
    'HINDUNILVR.BO',   # FMCG - Made ₹4,091 (50% win rate)
    'BEL.BO',          # Defence - Made ₹2,716 (40% win rate)
    'TECHM.BO',        # IT - Made ₹2,470 (50% win rate) - Keep this IT stock!
    'BAJAJFINSV.BO',   # Finance - Made ₹1,154 (33% win rate)
    'TATASTEEL.BO',    # Metals - Made ₹856 (43% win rate)
    'MARUTI.BO',       # Auto - Made ₹696 (25% win rate)
    'ITC.BO',          # FMCG - Made ₹5 (50% win rate)
]

# Symbol filtering - blacklist banking sector + poor performers
SYMBOL_FILTER = SymbolFilter(
    blacklist=BSE30_BLACKLIST,
    reason="Banking sector (user preference) + BSE30 stocks with consistent losses"
)


def annotate_signals(prices: pd.DataFrame, summary: pd.DataFrame = None) -> pd.DataFrame:
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

        # Track day-level momentum so that we only trade when the session is trending up
        if 'date' in group.columns:
            day_open = group.groupby('date')['Open'].transform('first')
            group['day_momentum'] = (group['Close'] / day_open) - 1
        else:
            # Fallback if date column missing (should not happen with standard pipeline)
            group['day_momentum'] = group['Close'] / group['Close'].iloc[0] - 1

        # Shift to avoid lookahead bias
        group['avg_volume_prev'] = group['avg_volume'].shift(1)
        group['volume_ratio_prev'] = group['volume_ratio'].shift(1)
        group['price_change_prev'] = group['price_change'].shift(1)
        group['day_momentum_prev'] = group['day_momentum'].shift(1)

        # Entry Signals
        # LONG ONLY: Volume spike + Price up (big momentum push)
        group['long_entry_signal'] = (
            (group['volume_ratio_prev'] > VOLUME_THRESHOLD) &
            (group['price_change_prev'] > PRICE_CHANGE_THRESHOLD) &
            (group['day_momentum_prev'] > DAY_MOMENTUM_THRESHOLD) &
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
    bars_held: int,
    peak_price: float = None
):
    """
    Check if position should exit based on profit target, stop loss, trailing stop, or max hold.

    Args:
        entry_price: Entry price of position
        current_price: Current market price
        direction: "long" (only long positions in this strategy)
        bars_held: Number of bars position has been held
        peak_price: Highest price reached since entry (for trailing stop)

    Returns:
        Tuple of (should_exit: bool, reason: str)
    """
    pnl_pct = (current_price - entry_price) / entry_price

    # Long only strategy
    # Profit target hit (+0.6%)
    if pnl_pct >= PROFIT_TARGET:
        return True, "profit_target"

    # Stop loss hit (-0.3%) - tighter stop to cut false signals faster
    if pnl_pct <= -STOP_LOSS:
        return True, "stop_loss"

    # Trailing stop: if we had gains and price drops from peak, lock in profit
    if peak_price is not None and peak_price > entry_price:
        peak_gain = (peak_price - entry_price) / entry_price
        if peak_gain >= PEAK_PROFIT_TRIGGER:
            drop_from_peak = (peak_price - current_price) / peak_price
            # If price dropped more than trailing stop threshold from peak, exit (momentum fading)
            if drop_from_peak >= TRAILING_STOP_PCT:
                return True, "trailing_stop"

    # Max hold period (15 bars)
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
STRATEGY_NAME = "Volume Momentum V3.3 - No Banking"
STRATEGY_DESCRIPTION = """
Volume Momentum Strategy V3.3 - BSE30 optimized with banking sector excluded

Entry (LONG ONLY):
- Volume > 1.5x 20-bar average
- Price up >0.3% (big upward momentum push)
- Day momentum >0.2% from open (session trend filter)
- Up to three concurrent positions (ranked by volume_ratio)
- Excludes ALL banking sector stocks (5 banks: Axis, HDFC, ICICI, Kotak, SBI)
- Excludes 10 other poor performers (PnL < -500, WinRate <= 25%)
- Includes Tech Mahindra (profitable IT stock) while excluding INFY & HCLTECH

Exit:
- Profit Target: 0.6% (optimized for platform costs)
- Stop Loss: 0.3% (tighter to cut false signals faster)
- Peak lock-in: once trade is up 0.4%, exit on 0.3% pullback from high
- Max Hold: 15 bars (15 minutes on 1m data)

Position Sizing:
- Up to three concurrent positions (ALL-IN per signal)
- Full capital allocation to best signals

BSE30 Blacklist (15 stocks total):
- Banking Sector (5): Axis Bank, HDFC Bank, ICICI Bank, Kotak Bank, SBI
- IT (2): Infosys, HCLTech (keeping only Tech Mahindra)
- Construction (1): L&T
- Consumer/Materials (3): Asian Paints, Titan, UltraTech Cement
- Finance (1): Bajaj Finance
- Telecom (1): Bharti Airtel
- Power/Utilities (2): NTPC, Power Grid

Eligible stocks (15 remaining from BSE30):
- FMCG: HUL, ITC
- Defence: BEL
- IT: Tech Mahindra (only)
- Finance: Bajaj Finserv
- Auto: Maruti, M&M, Tata Motors
- Metals: Tata Steel
- Energy: Reliance Industries, Adani Ports
- Pharma: Sun Pharma
- Services: TCS, Trent, Eternal

Version highlights:
- Complete banking sector exclusion (user preference)
- Focus on non-banking stocks with strong momentum characteristics
- 15 stocks blacklisted, 15 eligible for trading
"""

PARAMETERS = {
    "volume_lookback": VOLUME_LOOKBACK,
    "volume_threshold": VOLUME_THRESHOLD,
    "price_change_threshold": PRICE_CHANGE_THRESHOLD,
    "day_momentum_threshold": DAY_MOMENTUM_THRESHOLD,
    "profit_target": PROFIT_TARGET,
    "stop_loss": STOP_LOSS,
    "peak_profit_trigger": PEAK_PROFIT_TRIGGER,
    "trailing_stop_pct": TRAILING_STOP_PCT,
    "max_hold_bars": MAX_HOLD_BARS,
    "max_concurrent_positions": MAX_CONCURRENT_POSITIONS,
    "max_daily_positions": MAX_DAILY_POSITIONS,
    "blacklist_count": len(SYMBOL_FILTER.blacklist),
    "banking_sector_blacklisted": True,
    "blacklisted_symbols": BSE30_BLACKLIST,
}

VERSION = "3.3"
