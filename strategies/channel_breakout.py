#!/usr/bin/env python3
"""
Channel Breakout Strategy

Entry Rules:
- Enter LONG when price breaks above the upper channel (highest high over length period)
- Entry at next day's Open price
- Only if we have available cash

Exit Rules:
- Exit when price breaks below the lower channel (lowest low over length period)
- Exit at next day's Open price
- Only for current holdings

Configuration:
- Channel Length: 10 bars (lookback period)
- Order Size: 250 units per trade (fixed position sizing)
- Pyramiding: Up to 18 concurrent positions allowed
- Commission: 0%
- Slippage: 0 ticks (ideal fills)

Position Sizing:
- Fixed quantity per trade: 250 units
- Allow multiple positions (pyramiding up to 18 entries)
"""
from __future__ import annotations

import math
from typing import List

import pandas as pd


# Strategy Configuration
CHANNEL_LENGTH = 10
FIXED_ORDER_SIZE = 250
MAX_PYRAMIDING = 18


def annotate_signals(prices: pd.DataFrame, summary: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Annotate price data with long/short entry signals based on channel breakout.

    Implements the Pine Script strategy behavior:
    - strategy.entry("ChBrkLE", strategy.long, stop=upBound + syminfo.mintick)
    - strategy.entry("ChBrkSE", strategy.short, stop=downBound - syminfo.mintick)

    This means positions automatically reverse between long and short.

    Args:
        prices: DataFrame with columns [symbol, date, Open, High, Low, Close]
        summary: Optional DataFrame with backtest metrics [symbol, avg_trade_return, win_rate]

    Returns:
        DataFrame with additional columns [long_entry_signal, short_entry_signal, upper_channel, lower_channel]
        and optional [avg_return, win_rate]
    """
    def _per_symbol(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()

        # Calculate channel bounds using rolling window
        # upBound = ta.highest(high, length)
        # downBound = ta.lowest(low, length)
        group["upper_channel"] = group["High"].rolling(window=CHANNEL_LENGTH, min_periods=CHANNEL_LENGTH).max()
        group["lower_channel"] = group["Low"].rolling(window=CHANNEL_LENGTH, min_periods=CHANNEL_LENGTH).min()

        # Shift channel bounds by 1 to avoid lookahead bias
        # We use the channel calculated up to yesterday to make today's decision
        group["upper_channel_prev"] = group["upper_channel"].shift(1)
        group["lower_channel_prev"] = group["lower_channel"].shift(1)

        # Long Entry Signal: High breaks above previous upper channel
        # strategy.entry("ChBrkLE", strategy.long, stop=upBound + syminfo.mintick)
        # When this triggers: Enter LONG or reverse from SHORT to LONG
        group["long_entry_signal"] = (
            (group["High"] > group["upper_channel_prev"]) &
            group["upper_channel_prev"].notna()
        )

        # Short Entry Signal: Low breaks below previous lower channel
        # strategy.entry("ChBrkSE", strategy.short, stop=downBound - syminfo.mintick)
        # When this triggers: Enter SHORT or reverse from LONG to SHORT
        group["short_entry_signal"] = (
            (group["Low"] < group["lower_channel_prev"]) &
            group["lower_channel_prev"].notna()
        )

        # For backward compatibility, also set buy_signal and sell_signal
        # but note that sell_signal now means SHORT ENTRY, not exit!
        group["buy_signal"] = group["long_entry_signal"]
        group["sell_signal"] = group["short_entry_signal"]

        return group

    frames = [_per_symbol(group) for _, group in prices.groupby("symbol", sort=False)]
    result = pd.concat(frames, ignore_index=True)

    # Merge historical performance metrics if provided
    if summary is not None and not summary.empty:
        performance_cols = ["symbol", "avg_trade_return", "win_rate"]
        available_cols = [col for col in performance_cols if col in summary.columns]
        if len(available_cols) > 1:  # At least symbol + one metric
            result = result.merge(
                summary[available_cols],
                on="symbol",
                how="left"
            )
            # Rename for clarity
            if "avg_trade_return" in result.columns:
                result = result.rename(columns={"avg_trade_return": "avg_return"})

    return result


def generate_buy_calls(
    prices: pd.DataFrame,
    candidate_symbols: List[str],
    current_holdings: List[str],
    available_cash: float,
    per_trade_budget: float | None = None,
) -> List[dict]:
    """
    Generate buy calls based on channel breakout signals.

    Args:
        prices: Historical price data with annotated signals
        candidate_symbols: List of symbols to consider (e.g., top performers)
        current_holdings: List of symbols already held
        available_cash: Available cash for trading
        per_trade_budget: Optional fixed budget per trade (defaults to available_cash)

    Returns:
        List of buy call dictionaries with symbol, quantity, estimated_cost
    """
    if per_trade_budget is None:
        per_trade_budget = available_cash

    calls: List[dict] = []
    remaining_cash = available_cash

    # Count current positions for pyramiding limit
    position_count = len(current_holdings)

    # Determine time column (datetime for intraday, date for daily)
    time_col = "datetime" if "datetime" in prices.columns else "date"

    for symbol in candidate_symbols:
        # Check pyramiding limit (max 18 positions)
        if position_count >= MAX_PYRAMIDING:
            break

        # Get price data for symbol
        df = prices[prices["symbol"] == symbol].sort_values(time_col).copy()
        if df.empty or len(df) < CHANNEL_LENGTH + 1:
            continue

        # Get the most recent signal
        last_row = df.iloc[-1]

        # Check for buy signal
        if not last_row["buy_signal"]:
            continue

        # Use fixed order size (250 units) as specified in the strategy
        quantity = FIXED_ORDER_SIZE
        reference_price = float(last_row["Close"])
        estimated_cost = quantity * reference_price

        # Check if we have enough cash
        if estimated_cost > remaining_cash:
            # Scale down to available cash if not enough
            quantity = math.floor(remaining_cash / reference_price)
            if quantity < 1:
                continue
            estimated_cost = quantity * reference_price

        calls.append({
            "symbol": symbol,
            "company": last_row["company"],
            "reference_price": reference_price,
            "quantity": quantity,
            "estimated_cost": estimated_cost,
            "upper_channel": float(last_row["upper_channel_prev"]) if pd.notna(last_row["upper_channel_prev"]) else None,
        })

        remaining_cash -= estimated_cost
        position_count += 1

        if remaining_cash <= 0:
            break

    return calls


def generate_sell_calls(
    prices: pd.DataFrame,
    current_holdings: List[dict],
) -> List[dict]:
    """
    Generate sell calls based on channel breakout signals.

    Args:
        prices: Historical price data with annotated signals
        current_holdings: List of position dicts with symbol, quantity, avg_price

    Returns:
        List of sell call dictionaries with symbol, quantity, reference_price
    """
    calls: List[dict] = []

    # Determine time column (datetime for intraday, date for daily)
    time_col = "datetime" if "datetime" in prices.columns else "date"

    for position in current_holdings:
        symbol = position["symbol"]
        quantity = position["quantity"]

        # Get price data for symbol
        df = prices[prices["symbol"] == symbol].sort_values(time_col).copy()
        if df.empty or len(df) < CHANNEL_LENGTH + 1:
            continue

        # Get the most recent signal
        last_row = df.iloc[-1]

        # Check for sell signal
        if not last_row["sell_signal"]:
            continue

        reference_price = float(last_row["Close"])

        calls.append({
            "symbol": symbol,
            "company": last_row["company"],
            "quantity": quantity,
            "reference_price": reference_price,
            "estimated_proceeds": quantity * reference_price,
            "lower_channel": float(last_row["lower_channel_prev"]) if pd.notna(last_row["lower_channel_prev"]) else None,
        })

    return calls


def get_strategy_description() -> dict:
    """Return strategy metadata."""
    return {
        "name": "Channel Breakout",
        "type": "Breakout / Momentum",
        "timeframe": "Daily",
        "channel_length": CHANNEL_LENGTH,
        "order_size": FIXED_ORDER_SIZE,
        "max_pyramiding": MAX_PYRAMIDING,
        "entry_rules": [
            f"Price breaks above upper channel (highest high over {CHANNEL_LENGTH} bars)",
            "Entry at next day's Open price",
            f"Fixed position size: {FIXED_ORDER_SIZE} units per trade",
            f"Allow up to {MAX_PYRAMIDING} concurrent positions (pyramiding)"
        ],
        "exit_rules": [
            f"Price breaks below lower channel (lowest low over {CHANNEL_LENGTH} bars)",
            "Exit at next day's Open price",
            "Only for current holdings"
        ],
        "position_sizing": f"Fixed order size: {FIXED_ORDER_SIZE} units per trade",
        "commission": "0%",
        "slippage": "0 ticks",
    }
