#!/usr/bin/env python3
"""
Configuration for multi-provider, multi-index trading system.

Supported configurations:
- Provider: Upstox, YFinance
- Index: BSE-30, S&P 500
- Timeframe: daily, 5m, 15m, 1h
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class IndexConfig:
    """Configuration for a specific index."""
    name: str
    provider: str  # "upstox" or "yfinance"
    constituents_file: str  # relative to data/
    symbols_count: int
    description: str


# Index configurations
INDICES: Dict[str, IndexConfig] = {
    "bse30": IndexConfig(
        name="BSE-30",
        provider="upstox",
        constituents_file="bse30_constituents.csv",
        symbols_count=30,
        description="Bombay Stock Exchange Top 30 (India)"
    ),
    "niftysmallcap100": IndexConfig(
        name="NIFTY Smallcap 100",
        provider="upstox",
        constituents_file="niftysmallcap100_constituents.csv",
        symbols_count=100,
        description="NSE liquid small-cap basket"
    ),
    "niftysmallcap100_liquid": IndexConfig(
        name="NIFTY Smallcap 100 (Liquid)",
        provider="upstox",
        constituents_file="niftysmallcap100_liquid_constituents.csv",
        symbols_count=34,
        description="Top liquidity slice of NSE small caps"
    ),
    "sp500": IndexConfig(
        name="S&P 500",
        provider="yfinance",
        constituents_file="sp500_constituents.csv",
        symbols_count=500,
        description="Standard & Poor's 500 (USA)"
    ),
}

# Timeframe mappings
TIMEFRAME_MAP = {
    "daily": {"interval": "1d", "period_days": 365},
    "5m": {"interval": "5m", "period_days": 60},  # YFinance limit
    "15m": {"interval": "15m", "period_days": 60},
    "1h": {"interval": "1h", "period_days": 730},
}

# Directory structure:
#
# data/
#   constituents/              # Stock lists (input)
#     bse30_constituents.csv
#     sp500_constituents.csv
#   prices/                    # Raw price data (input)
#     {provider}/
#       {index}/
#         {index}_daily_prices.csv
#         {index}_5m_prices.csv
#   backtest_results/          # Backtest outputs (working data)
#     {provider}/
#       {index}/
#         {index}_summary_{strategy}_{timeframe}.csv
#         portfolio_history.csv
#         portfolio_trade_log.csv
#
# reports/
#   {provider}/
#     {index}/
#       backtest_report_{strategy}_{timeframe}.md
#       portfolio_backtest_report.md

DATA_DIR = Path("data")
CONSTITUENTS_DIR = DATA_DIR / "constituents"
PRICES_DIR = DATA_DIR / "prices"
BACKTEST_RESULTS_DIR = DATA_DIR / "backtest_results"
REPORTS_DIR = Path("reports")


def get_data_paths(provider: str, index: str, strategy: str, timeframe: str) -> dict:
    """
    Get all file paths for a given configuration.

    Args:
        provider: Data provider name (upstox, yfinance)
        index: Index name (bse30, sp500)
        strategy: Strategy name (two_day_momentum, channel_breakout)
        timeframe: Timeframe (daily, 5m, 15m, 1h)

    Returns:
        Dictionary with all relevant file paths
    """
    # Prices directory (input data)
    prices_dir = PRICES_DIR / provider / index
    prices_dir.mkdir(parents=True, exist_ok=True)

    # Backtest results directory (output data)
    results_dir = BACKTEST_RESULTS_DIR / provider / index
    results_dir.mkdir(parents=True, exist_ok=True)

    # Reports directory
    reports_dir = REPORTS_DIR / provider / index
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Determine price file suffix
    if timeframe == "daily":
        price_suffix = "daily"
    else:
        price_suffix = timeframe

    return {
        "prices_dir": prices_dir,
        "results_dir": results_dir,
        "reports_dir": reports_dir,
        "constituents": CONSTITUENTS_DIR / INDICES[index].constituents_file,
        "prices": prices_dir / f"{index}_{price_suffix}_prices.csv",
        "summary": results_dir / f"{index}_summary_{strategy}_{timeframe}.csv",
        "report": reports_dir / f"backtest_report_{strategy}_{timeframe}.md",
        "portfolio_report": reports_dir / f"portfolio_backtest_report.md",
        "portfolio_state": results_dir / f"portfolio_state_{strategy}_{timeframe}.json",
        "portfolio_history": results_dir / "portfolio_history.csv",
        "portfolio_trade_log": results_dir / "portfolio_trade_log.csv",
        # Legacy: for backward compatibility
        "index_dir": results_dir,
    }


def list_available_configs() -> List[dict]:
    """List all available provider/index combinations."""
    configs = []
    for index_key, index_config in INDICES.items():
        for timeframe in TIMEFRAME_MAP.keys():
            configs.append({
                "index": index_key,
                "provider": index_config.provider,
                "timeframe": timeframe,
                "description": f"{index_config.name} - {timeframe} ({index_config.provider})"
            })
    return configs
