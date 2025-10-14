#!/usr/bin/env python3
"""
Download 5-minute intraday prices for BSE-30 constituents.

By default, downloads differential updates (only missing data since last download).
Supports full refresh with --full flag.

Note: Upstox typically provides 30 days of intraday data.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
import requests

UPSTOX_API_BASE = "https://api.upstox.com/v2"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)
DEFAULT_DAYS = 30  # Upstox typically provides 30 days of intraday data


def get_access_token(arg_token: str | None) -> str:
    token = arg_token or os.getenv("UPSTOX_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Provide an access token via --access-token or UPSTOX_ACCESS_TOKEN env var")
    return token


def load_constituents(path: Path) -> pd.DataFrame:
    """Load the pre-configured BSE-30 constituents file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Constituents file not found at {path}. "
            "The file should be committed in the repository."
        )
    return pd.read_csv(path)


def fetch_upstox_intraday_candles(
    instrument_key: str,
    start: date,
    end: date,
    token: str,
    interval: str = "5minute"
) -> pd.DataFrame:
    """
    Fetch intraday candles from Upstox for a single instrument.

    Args:
        instrument_key: Upstox instrument key (e.g., "BSE_EQ|INE062A01020")
        start: Start date for data
        end: End date for data
        token: Upstox access token
        interval: Candle interval - "1minute", "5minute", "10minute", "15minute", "30minute", "60minute"

    Returns:
        DataFrame with columns: [timestamp, datetime, Open, High, Low, Close, Volume]
    """
    url = (
        f"{UPSTOX_API_BASE}/historical-candle/{instrument_key}/{interval}/"
        f"{end.isoformat()}/{start.isoformat()}"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 401:
        raise RuntimeError("Unauthorized: verify the Upstox access token")
    resp.raise_for_status()

    payload = resp.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"Upstox error for {instrument_key}: {payload}")

    candles = payload.get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(
        candles,
        columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "OpenInterest"],
    )
    # Convert timestamp to datetime
    df["datetime"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df["date"] = df["datetime"].dt.date
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def download_all_intraday_prices(
    constituents: pd.DataFrame,
    start: date,
    end: date,
    token: str,
    interval: str = "5minute"
) -> pd.DataFrame:
    """Download intraday price data for all constituents."""
    frames: List[pd.DataFrame] = []

    for idx, row in constituents.iterrows():
        instrument_key = row["InstrumentKey"]
        symbol = row["TickerYahoo"]
        company = row["Company"]
        bse_code = row["BseCode"]

        print(f"Downloading {symbol} ({company}) {interval}...", end=" ")

        df = fetch_upstox_intraday_candles(instrument_key, start, end, token, interval)
        if df.empty:
            print("no data")
            continue

        df.insert(0, "symbol", symbol)
        df.insert(1, "company", company)
        df.insert(2, "bse_code", str(bse_code))
        df["Adj Close"] = df["Close"].astype(float)

        df = df[[
            "symbol",
            "company",
            "bse_code",
            "datetime",
            "date",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
        ]]

        frames.append(df)
        print(f"{len(df)} rows")
        time.sleep(0.2)  # Rate limiting

    if not frames:
        raise RuntimeError("No intraday price data retrieved for any constituent")

    return pd.concat(frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--constituents",
        type=Path,
        default=Path("data/constituents/bse30_constituents.csv"),
        help="Path to BSE-30 constituents file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prices/upstox/bse30/bse30_5min_prices.csv"),
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--access-token",
        type=str,
        default=None,
        help="Upstox access token (or use UPSTOX_ACCESS_TOKEN env var)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Number of trailing days for initial download (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="5minute",
        choices=["1minute", "5minute", "10minute", "15minute", "30minute", "60minute"],
        help="Candle interval (default: 5minute)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force full refresh instead of differential update",
    )
    return parser.parse_args()


def determine_date_range(output_path: Path, days: int, full_refresh: bool = False) -> tuple[date, date]:
    """
    Determine the date range to download.

    If existing data exists and not doing full refresh:
    - Start from day after last date in existing data
    - Go to today

    Otherwise:
    - Start from today - days
    - Go to today
    """
    end_date = date.today()

    # If full refresh requested, ignore existing data
    if full_refresh:
        start_date = end_date - timedelta(days=max(days, 1))
        print(f"Full refresh: downloading {days} days of data")
        return start_date, end_date

    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path, parse_dates=["datetime"])
            if not existing_df.empty:
                last_datetime = existing_df["datetime"].max()
                last_date = last_datetime.date()
                # Start from day after last date
                start_date = last_date + timedelta(days=1)
                print(f"Found existing data up to {last_datetime}")
                print(f"Will download differential update from {start_date} to {end_date}")
                return start_date, end_date
        except Exception as e:
            print(f"Could not read existing data: {e}")
            print("Will download full dataset")

    # Default: download specified number of days
    start_date = end_date - timedelta(days=max(days, 1))
    return start_date, end_date


def merge_with_existing(new_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Merge new data with existing data, removing duplicates."""
    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path, parse_dates=["datetime"])
            if not existing_df.empty:
                # Combine and remove duplicates
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["symbol", "datetime"], keep="last")
                combined = combined.sort_values(["symbol", "datetime"]).reset_index(drop=True)
                print(f"Merged with existing data: {len(existing_df)} old + {len(new_df)} new = {len(combined)} total")
                return combined
        except Exception as e:
            print(f"Could not merge with existing data: {e}")

    return new_df


def main() -> int:
    args = parse_args()
    access_token = get_access_token(args.access_token)

    print(f"Loading constituents from {args.constituents}")
    constituents = load_constituents(args.constituents)
    print(f"Found {len(constituents)} constituents\n")

    # Determine date range (differential or full)
    start_date, end_date = determine_date_range(args.output, args.days, args.full)

    # Check if there's anything to download
    if start_date > end_date:
        print("✓ Data is already up to date. Nothing to download.")
        return 0

    print(f"Downloading {args.interval} prices from {start_date} to {end_date}")
    new_prices = download_all_intraday_prices(constituents, start_date, end_date, access_token, args.interval)

    if new_prices.empty:
        print("\n✓ No new data available")
        return 0

    # Merge with existing data
    final_prices = merge_with_existing(new_prices, args.output)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_prices.to_csv(args.output, index=False)

    print(f"\n✓ Saved {len(final_prices)} intraday price records to {args.output}")
    print(f"   Interval: {args.interval}")
    print(f"   Date range: {final_prices['datetime'].min()} to {final_prices['datetime'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
