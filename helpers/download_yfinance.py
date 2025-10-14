#!/usr/bin/env python3
"""
Download price data using Yahoo Finance (yfinance).

Supports both daily and intraday data for any index (primarily S&P 500).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import INDICES, TIMEFRAME_MAP


def load_constituents(path: Path) -> pd.DataFrame:
    """Load the constituents file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Constituents file not found at {path}. "
            "The file should exist in the data/ directory."
        )
    return pd.read_csv(path)


def fetch_yfinance_data(
    symbol: str,
    interval: str = "1d",
    period_days: int = 365,
) -> pd.DataFrame:
    """
    Fetch price data from Yahoo Finance for a single symbol.

    Args:
        symbol: Ticker symbol (e.g., "AAPL", "MSFT")
        interval: Data interval - "1d", "5m", "15m", "30m", "1h"
        period_days: Number of days of historical data

    Returns:
        DataFrame with columns: [datetime/date, Open, High, Low, Close, Volume]
    """
    # Calculate start and end dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)

    try:
        # Download data using yfinance
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=False  # Keep raw prices
        )

        if df.empty:
            return pd.DataFrame()

        # Reset index to get datetime as a column
        df = df.reset_index()

        # Rename columns to match our standard format
        if interval == "1d":
            df = df.rename(columns={"Date": "date"})
        else:
            df = df.rename(columns={"Datetime": "datetime"})
            df["date"] = df["datetime"].dt.date

        # Select and order columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        time_col = "date" if interval == "1d" else "datetime"

        df = df[[time_col, "date"] + required_cols] if interval != "1d" else df[[time_col] + required_cols]

        # Add Adj Close for compatibility
        df["Adj Close"] = df["Close"]

        return df

    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
        return pd.DataFrame()


def download_all_prices(
    constituents: pd.DataFrame,
    interval: str = "1d",
    period_days: int = 365,
) -> pd.DataFrame:
    """Download price data for all constituents."""
    frames: List[pd.DataFrame] = []

    total = len(constituents)
    for idx, row in constituents.iterrows():
        symbol = row["Symbol"]
        company = row["Company"]

        print(f"[{idx+1}/{total}] Downloading {symbol} ({company})...", end=" ")

        df = fetch_yfinance_data(symbol, interval, period_days)
        if df.empty:
            print("no data")
            continue

        # Add metadata columns
        df.insert(0, "symbol", symbol)
        df.insert(1, "company", company)
        df.insert(2, "sector", row.get("Sector", "Unknown"))

        frames.append(df)
        print(f"{len(df)} rows")
        time.sleep(0.1)  # Rate limiting

    if not frames:
        raise RuntimeError("No price data retrieved for any constituent")

    return pd.concat(frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=str,
        default="sp500",
        choices=list(INDICES.keys()),
        help="Index to download (default: sp500)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="daily",
        choices=list(TIMEFRAME_MAP.keys()),
        help="Timeframe: daily, 5m, 15m, 1h (default: daily)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: auto-generated from index/timeframe)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of days to download (overrides default for timeframe)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Get index configuration
    index_config = INDICES[args.index]
    timeframe_config = TIMEFRAME_MAP[args.timeframe]

    # Get paths from config
    from config import get_data_paths
    paths = get_data_paths(
        provider=index_config.provider,
        index=args.index,
        strategy="channel_breakout",  # dummy strategy for path generation
        timeframe=args.timeframe
    )

    # Determine output path (allow override)
    output_path = args.output if args.output else paths["prices"]

    # Determine period
    period_days = args.days if args.days else timeframe_config["period_days"]

    # Load constituents (use config path)
    constituents_path = paths["constituents"]
    print(f"Loading constituents from {constituents_path}")
    constituents = load_constituents(constituents_path)
    print(f"Found {len(constituents)} constituents\n")

    # Download data
    interval = timeframe_config["interval"]
    print(f"Downloading {args.timeframe} data ({interval}) for {index_config.name}")
    print(f"Period: {period_days} days")
    print("=" * 60)

    prices = download_all_prices(constituents, interval, period_days)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_path, index=False)

    # Summary
    time_col = "datetime" if args.timeframe != "daily" else "date"
    print(f"\n✓ Saved {len(prices)} price records to {output_path}")
    print(f"  Symbols: {prices['symbol'].nunique()}")
    print(f"  Period: {prices[time_col].min()} to {prices[time_col].max()}")
    print(f"  Timeframe: {args.timeframe} ({interval})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
