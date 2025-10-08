#!/usr/bin/env python3
"""Create mapping of company codes (Upstox, BSE, Bloomberg) for all companies in companies.json."""
from __future__ import annotations

import argparse
import json
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

NSE_EQUITY_MASTER_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
BSE_EQUITY_MASTER_URL = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)


def fetch_nse_master_data() -> pd.DataFrame:
    """Fetch NSE equity master data."""
    print("Fetching NSE master data...")
    headers = {"User-Agent": USER_AGENT, "Accept": "text/csv"}
    resp = requests.get(NSE_EQUITY_MASTER_URL, headers=headers, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text))
    df.columns = df.columns.str.strip()

    # Clean up data
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
    df["NAME OF COMPANY"] = df["NAME OF COMPANY"].astype(str).str.strip()
    df["ISIN NUMBER"] = df["ISIN NUMBER"].astype(str).str.strip()

    print(f"Loaded {len(df)} NSE symbols")
    return df


def normalize_company_name(name: str) -> str:
    """Normalize company name for matching."""
    # Remove common suffixes and normalize
    name = name.lower()
    name = name.replace(" ltd", "").replace(" limited", "")
    name = name.replace(".", "").replace(",", "")
    name = name.replace("  ", " ").strip()
    return name


def find_best_nse_match(company_name: str, nse_df: pd.DataFrame) -> Optional[Dict]:
    """Find best NSE match for a company name."""
    normalized_search = normalize_company_name(company_name)

    # Try exact match first
    for _, row in nse_df.iterrows():
        normalized_nse = normalize_company_name(row["NAME OF COMPANY"])
        if normalized_search == normalized_nse:
            return {
                "nse_symbol": row["SYMBOL"],
                "isin": row["ISIN NUMBER"],
                "official_name": row["NAME OF COMPANY"],
            }

    # Try partial match (company name contains or is contained in NSE name)
    for _, row in nse_df.iterrows():
        normalized_nse = normalize_company_name(row["NAME OF COMPANY"])
        if normalized_search in normalized_nse or normalized_nse in normalized_search:
            return {
                "nse_symbol": row["SYMBOL"],
                "isin": row["ISIN NUMBER"],
                "official_name": row["NAME OF COMPANY"],
            }

    return None


def fetch_bse_code_from_nse_symbol(nse_symbol: str) -> Optional[str]:
    """Try to fetch BSE code using NSE symbol (basic approach)."""
    # This is a simplified approach - in reality, you might need a proper mapping
    # For now, we'll return None and manually fill later if needed
    return None


def create_company_mapping(companies_file: Path, output_file: Path) -> pd.DataFrame:
    """Create mapping of all company codes."""
    # Load companies
    with open(companies_file) as f:
        data = json.load(f)
    companies = data["companies"]

    print(f"Processing {len(companies)} companies...")

    # Fetch NSE master data
    nse_df = fetch_nse_master_data()

    # Build mapping
    results = []
    matched = 0
    unmatched = []

    for company in companies:
        company_id = company["id"]
        company_name = company["name"]

        print(f"Processing: {company_name}")

        # Find NSE match
        nse_match = find_best_nse_match(company_name, nse_df)

        if nse_match:
            matched += 1
            nse_symbol = nse_match["nse_symbol"]
            isin = nse_match["isin"]
            official_name = nse_match["official_name"]

            # Create codes
            upstox_code = f"NSE_EQ|{isin}"
            bloomberg_code = f"{nse_symbol} IN Equity"
            bse_code = fetch_bse_code_from_nse_symbol(nse_symbol) or ""

            results.append({
                "id": company_id,
                "company_name": company_name,
                "nse_official_name": official_name,
                "nse_symbol": nse_symbol,
                "isin": isin,
                "upstox_code": upstox_code,
                "bse_code": bse_code,
                "bloomberg_code": bloomberg_code,
                "matched": "Yes",
            })
        else:
            unmatched.append(company_name)
            results.append({
                "id": company_id,
                "company_name": company_name,
                "nse_official_name": "",
                "nse_symbol": "",
                "isin": "",
                "upstox_code": "",
                "bse_code": "",
                "bloomberg_code": "",
                "matched": "No",
            })

        time.sleep(0.01)  # Small delay to avoid overwhelming

    # Create DataFrame
    mapping_df = pd.DataFrame(results)

    # Save to CSV
    mapping_df.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print(f"Mapping complete!")
    print(f"Total companies: {len(companies)}")
    print(f"Matched: {matched}")
    print(f"Unmatched: {len(unmatched)}")
    print(f"\nUnmatched companies:")
    for name in unmatched:
        print(f"  - {name}")
    print(f"\nMapping saved to: {output_file}")

    return mapping_df


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--companies-file",
        type=Path,
        default=Path("companies.json"),
        help="Path to companies.json file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/company_code_mapping.csv"),
        help="Path to output CSV file",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    if not args.companies_file.exists():
        print(f"Error: {args.companies_file} not found")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    create_company_mapping(args.companies_file, args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
