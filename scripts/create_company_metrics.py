#!/usr/bin/env python3
"""Create company metrics CSV with Yahoo Finance data for the last closing date."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

# USD to INR conversion rate (approximate)
USD_TO_INR = 83.0


def load_sector_mapping(companies_file: Path) -> Dict[int, str]:
    """Load sector mapping from companies.json."""
    with open(companies_file) as f:
        data = json.load(f)

    # Create sector ID to name mapping
    sectors = {
        1: "Automobiles",
        2: "Aviation",
        3: "Chemicals",
        4: "Commodities",
        5: "Construction",
        6: "Construction Materials",
        7: "Consumer Goods",
        8: "Food & Beverages",
        9: "Financial Services",
        10: "Healthcare",
        11: "Industrials",
        12: "Insurance",
        13: "Oil & Gas",
        14: "Logistics",
        15: "Technology",
        16: "Logistics & Transportation",
        17: "Ports",
        18: "Utilities",
    }

    # Map company ID to sector
    company_to_sector = {}
    for company in data["companies"]:
        sector_id = company.get("sectorId")
        company_to_sector[company["id"]] = sectors.get(sector_id, "Other")

    return company_to_sector


def get_yahoo_ticker(nse_symbol: str) -> str:
    """Convert NSE symbol to Yahoo Finance ticker."""
    # Yahoo Finance uses .NS suffix for NSE stocks
    return f"{nse_symbol}.NS"


def fetch_company_metrics(
    yahoo_ticker: str, nse_symbol: str
) -> Optional[Dict]:
    """Fetch company metrics from Yahoo Finance."""
    try:
        ticker = yf.Ticker(yahoo_ticker)

        # Get historical data for price changes
        end_date = datetime.now()
        start_date = end_date - timedelta(days=400)  # Get more data for 1yr calculation
        hist = ticker.history(start=start_date, end=end_date)

        if hist.empty:
            print(f"  ⚠ No historical data for {nse_symbol}")
            return None

        # Get info
        info = ticker.info

        # Get latest price and date
        latest_close = hist['Close'].iloc[-1]
        latest_date = hist.index[-1].strftime('%Y-%m-%d')

        # Calculate price changes
        def get_price_change(days_back: int) -> Optional[float]:
            if len(hist) > days_back:
                old_price = hist['Close'].iloc[-days_back - 1]
                return ((latest_close - old_price) / old_price) * 100
            return None

        price_change_1d = get_price_change(1)
        price_change_1m = get_price_change(21)  # ~21 trading days in a month
        price_change_3m = get_price_change(63)  # ~63 trading days in 3 months
        price_change_6m = get_price_change(126)  # ~126 trading days in 6 months
        price_change_1yr = get_price_change(252)  # ~252 trading days in a year

        # Get volume data (last 20 days average)
        avg_volume = hist['Volume'].tail(20).mean()
        avg_value_traded = (hist['Close'] * hist['Volume']).tail(20).mean()

        # Get market cap
        market_cap_inr = info.get('marketCap', None)
        if market_cap_inr is None:
            # Try to calculate from shares outstanding and price
            shares = info.get('sharesOutstanding', None)
            if shares:
                market_cap_inr = shares * latest_close

        market_cap_usd = market_cap_inr / USD_TO_INR if market_cap_inr else None

        return {
            'price_date': latest_date,
            'price': latest_close,
            'market_cap_inr': market_cap_inr,
            'market_cap_usd': market_cap_usd,
            'avg_daily_volume': avg_volume,
            'avg_daily_value': avg_value_traded,
            'price_change_1d': price_change_1d,
            'price_change_1m': price_change_1m,
            'price_change_3m': price_change_3m,
            'price_change_6m': price_change_6m,
            'price_change_1yr': price_change_1yr,
        }

    except Exception as e:
        print(f"  ✗ Error fetching {nse_symbol}: {e}")
        return None


def fetch_bse30_metrics() -> Dict:
    """Fetch BSE-30 (SENSEX) index metrics."""
    try:
        ticker = yf.Ticker("^BSESN")

        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=400)
        hist = ticker.history(start=start_date, end=end_date)

        if hist.empty:
            return {}

        latest_close = hist['Close'].iloc[-1]
        latest_date = hist.index[-1].strftime('%Y-%m-%d')

        # Calculate price changes
        def get_price_change(days_back: int) -> Optional[float]:
            if len(hist) > days_back:
                old_price = hist['Close'].iloc[-days_back - 1]
                return ((latest_close - old_price) / old_price) * 100
            return None

        avg_volume = hist['Volume'].tail(20).mean()
        avg_value_traded = (hist['Close'] * hist['Volume']).tail(20).mean()

        return {
            'bloomberg_code': 'SENSEX Index',
            'nse_symbol': 'BSE-30',
            'upstox_code': 'BSE_INDEX|SENSEX',
            'company_name': 'BSE SENSEX',
            'sector': 'Index',
            'price_date': latest_date,
            'price': latest_close,
            'market_cap_inr': None,
            'market_cap_usd': None,
            'avg_daily_volume': avg_volume,
            'avg_daily_value': avg_value_traded,
            'price_change_1d': get_price_change(1),
            'price_change_1m': get_price_change(21),
            'price_change_3m': get_price_change(63),
            'price_change_6m': get_price_change(126),
            'price_change_1yr': get_price_change(252),
        }

    except Exception as e:
        print(f"Error fetching BSE-30: {e}")
        return {}


def create_metrics_csv(
    mapping_file: Path,
    companies_file: Path,
    output_file: Path,
) -> None:
    """Create company metrics CSV."""
    # Load mapping
    mapping_df = pd.read_csv(mapping_file)
    mapping_df = mapping_df[mapping_df["matched"].str.contains("Yes", na=False)]

    # Load sector mapping
    sector_mapping = load_sector_mapping(companies_file)

    print(f"Fetching metrics for {len(mapping_df)} companies...")

    results = []

    # Process each company
    for idx, row in mapping_df.iterrows():
        company_id = row["id"]
        company_name = row["company_name"]
        nse_symbol = row["nse_symbol"]
        bloomberg_code = row["bloomberg_code"]
        upstox_code = row["upstox_code"]
        sector = sector_mapping.get(company_id, "Other")

        print(f"[{idx + 1}/{len(mapping_df)}] {company_name} ({nse_symbol})...")

        yahoo_ticker = get_yahoo_ticker(nse_symbol)
        metrics = fetch_company_metrics(yahoo_ticker, nse_symbol)

        if metrics:
            results.append({
                'Bloomberg Code': bloomberg_code,
                'NSE Ticker Code': nse_symbol,
                'Upstox Code': upstox_code,
                'Company Name': company_name,
                'Sector': sector,
                'Price Date': metrics['price_date'],
                'Price': metrics['price'],
                'Mkt Cap (INR)': metrics['market_cap_inr'],
                'Mkt Cap (USD)': metrics['market_cap_usd'],
                'Avg Daily Volume Traded': metrics['avg_daily_volume'],
                'Avg Daily Value Traded': metrics['avg_daily_value'],
                'Price Change 1d': metrics['price_change_1d'],
                'Price Change 1m': metrics['price_change_1m'],
                'Price Change 3m': metrics['price_change_3m'],
                'Price Change 6m': metrics['price_change_6m'],
                'Price Change 1yr': metrics['price_change_1yr'],
            })
            print(f"  ✓ Price: ₹{metrics['price']:.2f}, Mkt Cap: ₹{metrics['market_cap_inr']:,.0f}" if metrics['market_cap_inr'] else f"  ✓ Price: ₹{metrics['price']:.2f}")

        # Be polite to Yahoo Finance API
        time.sleep(0.5)

    # Add BSE-30 index
    print("\nFetching BSE-30 (SENSEX) metrics...")
    bse30_metrics = fetch_bse30_metrics()
    if bse30_metrics:
        results.append({
            'Bloomberg Code': bse30_metrics['bloomberg_code'],
            'NSE Ticker Code': bse30_metrics['nse_symbol'],
            'Upstox Code': bse30_metrics['upstox_code'],
            'Company Name': bse30_metrics['company_name'],
            'Sector': bse30_metrics['sector'],
            'Price Date': bse30_metrics['price_date'],
            'Price': bse30_metrics['price'],
            'Mkt Cap (INR)': bse30_metrics['market_cap_inr'],
            'Mkt Cap (USD)': bse30_metrics['market_cap_usd'],
            'Avg Daily Volume Traded': bse30_metrics['avg_daily_volume'],
            'Avg Daily Value Traded': bse30_metrics['avg_daily_value'],
            'Price Change 1d': bse30_metrics['price_change_1d'],
            'Price Change 1m': bse30_metrics['price_change_1m'],
            'Price Change 3m': bse30_metrics['price_change_3m'],
            'Price Change 6m': bse30_metrics['price_change_6m'],
            'Price Change 1yr': bse30_metrics['price_change_1yr'],
        })
        print(f"  ✓ SENSEX: {bse30_metrics['price']:.2f}")

    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print(f"Saved metrics for {len(df)} companies to: {output_file}")
    print(f"Columns: {', '.join(df.columns)}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=Path("data/company_code_mapping.csv"),
        help="Path to company mapping CSV file",
    )
    parser.add_argument(
        "--companies-file",
        type=Path,
        default=Path("companies.json"),
        help="Path to companies.json file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/company_metrics.csv"),
        help="Path to output CSV file",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    if not args.mapping_file.exists():
        print(f"Error: Mapping file not found: {args.mapping_file}")
        return 1

    if not args.companies_file.exists():
        print(f"Error: Companies file not found: {args.companies_file}")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    create_metrics_csv(args.mapping_file, args.companies_file, args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
