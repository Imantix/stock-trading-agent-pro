#!/usr/bin/env python3
"""Backtesting framework for trading strategies on BSE-30 daily prices."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

# Import strategy
sys.path.insert(0, str(Path(__file__).parent.parent / "strategies"))
from two_day_momentum import annotate_signals


@dataclass
class Trade:
    symbol: str
    company: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float = 1.0
    timeframe: str = "daily"  # "daily" or "intraday"
    direction: str = "long"  # "long" or "short"

    @property
    def pnl(self) -> float:
        """Calculate P&L considering position direction."""
        if self.direction == "long":
            return (self.exit_price - self.entry_price) * self.shares
        else:  # short
            return (self.entry_price - self.exit_price) * self.shares

    @property
    def return_pct(self) -> float:
        """Calculate return percentage considering position direction."""
        if self.direction == "long":
            return (self.exit_price / self.entry_price - 1.0) * 100.0
        else:  # short
            return (self.entry_price / self.exit_price - 1.0) * 100.0

    @property
    def holding_days(self) -> int:
        return int((self.exit_date - self.entry_date).days)

    @property
    def holding_duration(self) -> timedelta:
        """Return holding duration as timedelta (works for both daily and intraday)."""
        return self.exit_date - self.entry_date


def load_prices(path: Path, timeframe: str = "daily") -> pd.DataFrame:
    """
    Load price data from CSV file.

    Args:
        path: Path to CSV file
        timeframe: Either "daily" or "intraday" (default: "daily")

    Returns:
        DataFrame sorted by symbol and date/datetime
    """
    if timeframe == "intraday":
        df = pd.read_csv(path, parse_dates=["datetime"])
        required = {"symbol", "company", "datetime", "Open", "Close", "High", "Low"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Price file is missing required columns: {sorted(missing)}")
        return df.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    else:
        df = pd.read_csv(path, parse_dates=["date"])
        required = {"symbol", "company", "date", "Open", "Close"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Price file is missing required columns: {sorted(missing)}")
        return df.sort_values(["symbol", "date"]).reset_index(drop=True)


# annotate_signals is imported from the strategy module


def generate_trades(prices: pd.DataFrame, timeframe: str = "daily") -> List[Trade]:
    """
    Generate trades from annotated price data with long/short entry signals.

    Implements TradingView strategy.entry() behavior with automatic position reversals:
    - When long_entry_signal fires: Enter LONG or reverse from SHORT to LONG
    - When short_entry_signal fires: Enter SHORT or reverse from LONG to SHORT
    - Positions automatically reverse (matching Pine Script behavior)

    Args:
        prices: DataFrame with long_entry_signal and short_entry_signal columns
        timeframe: Either "daily" or "intraday" (default: "daily")

    Returns:
        List of Trade objects
    """
    trades: List[Trade] = []
    time_col = "datetime" if timeframe == "intraday" else "date"

    for symbol, group in prices.groupby("symbol"):
        group = group.sort_values(time_col).reset_index(drop=True)
        rows = list(group.itertuples(index=False))
        company = group["company"].iloc[0]

        # Track current position: None, "long", or "short"
        position: Optional[str] = None
        entry_price: Optional[float] = None
        entry_date: Optional[pd.Timestamp] = None

        for idx in range(len(rows) - 1):
            row = rows[idx]
            next_row = rows[idx + 1]

            # Check for long entry signal
            long_signal = getattr(row, "long_entry_signal", False) or getattr(row, "buy_signal", False)
            # Check for short entry signal
            short_signal = getattr(row, "short_entry_signal", False) or getattr(row, "sell_signal", False)

            # LONG ENTRY (or reversal from SHORT to LONG)
            if long_signal:
                # If currently SHORT, close short position first
                if position == "short":
                    exit_price = float(next_row.Open)
                    exit_date = pd.Timestamp(getattr(next_row, time_col))
                    trades.append(
                        Trade(
                            symbol=symbol,
                            company=company,
                            entry_date=entry_date,
                            exit_date=exit_date,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            timeframe=timeframe,
                            direction="short",
                        )
                    )

                # Enter LONG position (or switch from SHORT to LONG)
                if position != "long":  # Only enter if not already long
                    position = "long"
                    entry_price = float(next_row.Open)
                    entry_date = pd.Timestamp(getattr(next_row, time_col))

            # SHORT ENTRY (or reversal from LONG to SHORT)
            elif short_signal:
                # If currently LONG, close long position first
                if position == "long":
                    exit_price = float(next_row.Open)
                    exit_date = pd.Timestamp(getattr(next_row, time_col))
                    trades.append(
                        Trade(
                            symbol=symbol,
                            company=company,
                            entry_date=entry_date,
                            exit_date=exit_date,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            timeframe=timeframe,
                            direction="long",
                        )
                    )

                # Enter SHORT position (or switch from LONG to SHORT)
                if position != "short":  # Only enter if not already short
                    position = "short"
                    entry_price = float(next_row.Open)
                    entry_date = pd.Timestamp(getattr(next_row, time_col))

    return trades


def summarize_trades(trades: Iterable[Trade]) -> pd.DataFrame:
    """
    Convert list of Trade objects to a summary DataFrame.

    Returns:
        DataFrame with columns for trade details, PnL, returns, holding period, and direction
    """
    trades_df = pd.DataFrame(
        [
            {
                "symbol": t.symbol,
                "company": t.company,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "shares": t.shares,
                "pnl": t.pnl,
                "return_pct": t.return_pct,
                "holding_days": t.holding_days,
                "holding_duration": str(t.holding_duration) if hasattr(t, 'holding_duration') else None,
                "timeframe": t.timeframe if hasattr(t, 'timeframe') else "daily",
                "direction": t.direction if hasattr(t, 'direction') else "long",
            }
            for t in trades
        ]
    )
    if trades_df.empty:
        return trades_df
    return trades_df.sort_values(["symbol", "entry_date"]).reset_index(drop=True)


def aggregate_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df

    grouped = trades_df.groupby("symbol")
    summary = grouped.agg(
        company=("company", "first"),
        trades=("pnl", "size"),
        net_pnl=("pnl", "sum"),
        avg_trade_return=("return_pct", "mean"),
        win_rate=("return_pct", lambda s: (s > 0).mean() * 100.0),
    ).reset_index()
    return summary


def overall_metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {
            "total_trades": 0,
            "net_pnl": 0.0,
            "win_rate": np.nan,
            "avg_return_pct": np.nan,
            "compound_return_pct": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown_pct": np.nan,
        }

    total_trades = int(len(trades_df))
    net_pnl = float(trades_df["pnl"].sum())
    win_rate = float((trades_df["return_pct"] > 0).mean() * 100.0)
    avg_return = float(trades_df["return_pct"].mean())
    log_returns = np.log1p(trades_df["return_pct"] / 100.0)
    compound_return = float((np.exp(log_returns.sum()) - 1.0) * 100.0)

    # Calculate Sharpe Ratio (assuming 252 trading days, 6% risk-free rate)
    returns_std = trades_df["return_pct"].std()
    if returns_std > 0:
        # Annualize: avg daily return * sqrt(252) / std
        sharpe_ratio = float((avg_return - 0.02) / returns_std * np.sqrt(252))
    else:
        sharpe_ratio = 0.0

    # Calculate Maximum Drawdown
    trades_sorted = trades_df.sort_values("entry_date").copy()
    trades_sorted["cumulative_return"] = (1 + trades_sorted["return_pct"] / 100.0).cumprod()
    trades_sorted["cumulative_max"] = trades_sorted["cumulative_return"].cummax()
    trades_sorted["drawdown"] = (
        trades_sorted["cumulative_return"] / trades_sorted["cumulative_max"] - 1.0
    ) * 100.0
    max_drawdown = float(trades_sorted["drawdown"].min())

    return {
        "total_trades": total_trades,
        "net_pnl": net_pnl,
        "win_rate": win_rate,
        "avg_return_pct": avg_return,
        "compound_return_pct": compound_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_pct": max_drawdown,
    }


def generate_markdown_report(
    metrics: dict,
    summary_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate a detailed Markdown backtest report."""
    start_date = trades_df["entry_date"].min().date() if not trades_df.empty else "N/A"
    end_date = trades_df["exit_date"].max().date() if not trades_df.empty else "N/A"

    # Calculate long/short breakdown
    long_trades = trades_df[trades_df["direction"] == "long"] if "direction" in trades_df.columns else trades_df
    short_trades = trades_df[trades_df["direction"] == "short"] if "direction" in trades_df.columns else pd.DataFrame()

    long_count = len(long_trades)
    short_count = len(short_trades)
    long_win_rate = (long_trades["return_pct"] > 0).mean() * 100 if long_count > 0 else 0
    short_win_rate = (short_trades["return_pct"] > 0).mean() * 100 if short_count > 0 else 0

    report = f"""# Channel Breakout Strategy - Backtest Report

**Generated:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
**Period:** {start_date} to {end_date}
**Universe:** BSE-30 (30 stocks)

---

## Overall Performance

| Metric | Value |
|--------|-------|
| **Total Trades** | {metrics['total_trades']} |
| **Long Trades** | {long_count} ({long_win_rate:.1f}% win rate) |
| **Short Trades** | {short_count} ({short_win_rate:.1f}% win rate) |
| **Net P&L** | ₹{metrics['net_pnl']:,.2f} |
| **Win Rate** | {metrics['win_rate']:.2f}% |
| **Average Return** | {metrics['avg_return_pct']:.2f}% |
| **Compound Return** | {metrics['compound_return_pct']:.2f}% |
| **Sharpe Ratio** | {metrics['sharpe_ratio']:.2f} |
| **Maximum Drawdown** | {metrics['max_drawdown_pct']:.2f}% |

---

## Top Performers (by Net P&L)

"""
    top_10 = summary_df.sort_values("net_pnl", ascending=False).head(10)
    report += "| Rank | Symbol | Company | Trades | Net P&L | Avg Return | Win Rate |\n"
    report += "|------|--------|---------|--------|---------|------------|----------|\n"

    for idx, row in enumerate(top_10.itertuples(), 1):
        report += (
            f"| {idx} | {row.symbol} | {row.company} | {int(row.trades)} | "
            f"₹{row.net_pnl:,.2f} | {row.avg_trade_return:.2f}% | {row.win_rate:.2f}% |\n"
        )

    report += "\n---\n\n## Bottom Performers (by Net P&L)\n\n"
    bottom_10 = summary_df.sort_values("net_pnl").head(10)
    report += "| Rank | Symbol | Company | Trades | Net P&L | Avg Return | Win Rate |\n"
    report += "|------|--------|---------|--------|---------|------------|----------|\n"

    for idx, row in enumerate(bottom_10.itertuples(), 1):
        report += (
            f"| {idx} | {row.symbol} | {row.company} | {int(row.trades)} | "
            f"₹{row.net_pnl:,.2f} | {row.avg_trade_return:.2f}% | {row.win_rate:.2f}% |\n"
        )

    report += "\n---\n\n## Strategy Description\n\n"
    report += "**Channel Breakout Strategy (Long/Short with Reversals)**\n\n"
    report += "**Long Entry Rules:**\n"
    report += "- Price breaks above upper channel (10-bar highest high)\n"
    report += "- Entry at next bar's Open price\n"
    report += "- If currently SHORT, automatically reverse to LONG\n\n"
    report += "**Short Entry Rules:**\n"
    report += "- Price breaks below lower channel (10-bar lowest low)\n"
    report += "- Entry at next bar's Open price\n"
    report += "- If currently LONG, automatically reverse to SHORT\n\n"
    report += "**Position Sizing:**\n"
    report += "- 1 share per trade (for backtesting purposes)\n"
    report += "- Fixed order size: 250 units (in live trading)\n"
    report += "- Max pyramiding: 18 concurrent positions\n\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prices",
        type=Path,
        default=Path("data/bse30_daily_prices.csv"),
        help="Path to the CSV file with daily prices",
    )
    parser.add_argument(
        "--trades-output",
        type=Path,
        default=None,
        help="Optional path to write the trades log as CSV",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional path to write the per-symbol summary as CSV",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="Optional path to write the Markdown report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    prices = load_prices(args.prices)
    prices = annotate_signals(prices)
    trades = generate_trades(prices)
    trades_df = summarize_trades(trades)

    if trades_df.empty:
        print("No trades were generated by the strategy.")
        return 0

    summary_df = aggregate_summary(trades_df)
    metrics = overall_metrics(trades_df)

    print("Overall metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\nTop symbols by net PnL:")
    top = summary_df.sort_values("net_pnl", ascending=False).head(10)
    for _, row in top.iterrows():
        print(
            f"  {row.symbol:>12} | trades={int(row.trades):3d} | net_pnl={row.net_pnl:9.2f} | "
            f"avg_return={row.avg_trade_return:6.2f}% | win_rate={row.win_rate:6.2f}%"
        )

    if args.trades_output is not None:
        trades_path = args.trades_output
        trades_path.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(trades_path, index=False)
        print(f"\nSaved trades log to {trades_path}")

    if args.summary_output is not None:
        summary_path = args.summary_output
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved per-symbol summary to {summary_path}")

    if args.report_output is not None:
        report_path = args.report_output
        generate_markdown_report(metrics, summary_df, trades_df, report_path)
        print(f"Saved Markdown report to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
