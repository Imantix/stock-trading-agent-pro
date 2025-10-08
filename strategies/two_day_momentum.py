#!/usr/bin/env python3
"""
Two-Day Momentum Strategy

Entry Rules:
- Buy when a stock shows two consecutive positive days (Close > Open)
- Entry at next day's Open price
- Only if we have available cash

Exit Rules:
- Sell when a stock shows two consecutive negative days (Close < Open)
- Exit at next day's Open price
- Only for current holdings

Position Sizing:
- Equal weight allocation across trades
- Budget per trade = available_cash / number_of_signals
"""
from __future__ import annotations

import math
from typing import List

import pandas as pd


def annotate_signals(prices: pd.DataFrame, summary: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Annotate price data with buy/sell signals.

    Args:
        prices: DataFrame with columns [symbol, date, Open, Close]
        summary: Optional DataFrame with backtest metrics [symbol, avg_trade_return, win_rate]

    Returns:
        DataFrame with additional columns [buy_signal, sell_signal] and optional [avg_return, win_rate]
    """
    def _per_symbol(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        positive = group["Close"] > group["Open"]
        negative = group["Close"] < group["Open"]
        group["buy_signal"] = positive & positive.shift(1, fill_value=False)
        group["sell_signal"] = negative & negative.shift(1, fill_value=False)
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
    Generate buy calls based on two-day momentum signals.

    Args:
        prices: Historical price data
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

    for symbol in candidate_symbols:
        # Skip if already holding
        if symbol in current_holdings:
            continue

        # Get price data for symbol
        df = prices[prices["symbol"] == symbol].sort_values("date").copy()
        if df.empty or len(df) < 2:
            continue

        # Check for buy signal (two consecutive positive days)
        df["positive"] = df["Close"] > df["Open"]
        last_two = df.tail(2)

        if len(last_two) < 2:
            continue

        has_buy_signal = (
            bool(last_two.iloc[-1]["positive"]) and
            bool(last_two.iloc[-2]["positive"])
        )

        if not has_buy_signal:
            continue

        # Calculate position size
        reference_price = float(last_two.iloc[-1]["Close"])
        budget = min(per_trade_budget, remaining_cash)
        quantity = math.floor(budget / reference_price)

        if quantity < 1:
            continue

        estimated_cost = quantity * reference_price

        calls.append({
            "symbol": symbol,
            "company": last_two.iloc[-1]["company"],
            "reference_price": reference_price,
            "quantity": quantity,
            "estimated_cost": estimated_cost,
        })

        remaining_cash -= estimated_cost

        if remaining_cash <= 0:
            break

    return calls


def generate_sell_calls(
    prices: pd.DataFrame,
    current_holdings: List[dict],
) -> List[dict]:
    """
    Generate sell calls based on two-day momentum signals.

    Args:
        prices: Historical price data
        current_holdings: List of position dicts with symbol, quantity, avg_price

    Returns:
        List of sell call dictionaries with symbol, quantity, reference_price
    """
    calls: List[dict] = []

    for position in current_holdings:
        symbol = position["symbol"]
        quantity = position["quantity"]

        # Get price data for symbol
        df = prices[prices["symbol"] == symbol].sort_values("date").copy()
        if df.empty or len(df) < 2:
            continue

        # Check for sell signal (two consecutive negative days)
        df["negative"] = df["Close"] < df["Open"]
        last_two = df.tail(2)

        if len(last_two) < 2:
            continue

        has_sell_signal = (
            bool(last_two.iloc[-1]["negative"]) and
            bool(last_two.iloc[-2]["negative"])
        )

        if not has_sell_signal:
            continue

        reference_price = float(last_two.iloc[-1]["Close"])

        calls.append({
            "symbol": symbol,
            "company": last_two.iloc[-1]["company"],
            "quantity": quantity,
            "reference_price": reference_price,
            "estimated_proceeds": quantity * reference_price,
        })

    return calls


def get_strategy_description() -> dict:
    """Return strategy metadata."""
    return {
        "name": "Two-Day Momentum",
        "type": "Trend Following",
        "timeframe": "Daily",
        "entry_rules": [
            "Two consecutive positive days (Close > Open)",
            "Entry at next day's Open price",
            "Only when cash is available"
        ],
        "exit_rules": [
            "Two consecutive negative days (Close < Open)",
            "Exit at next day's Open price",
            "Only for current holdings"
        ],
        "position_sizing": "Equal weight allocation based on available cash",
    }
