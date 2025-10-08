#!/usr/bin/env python3
"""Backtesting framework for trading strategies on BSE-30 daily prices."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def return_pct(self) -> float:
        return (self.exit_price / self.entry_price - 1.0) * 100.0

    @property
    def holding_days(self) -> int:
        return int((self.exit_date - self.entry_date).days)


def load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"symbol", "company", "date", "Open", "Close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Price file is missing required columns: {sorted(missing)}")
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


# annotate_signals is imported from the strategy module


def generate_trades(prices: pd.DataFrame) -> List[Trade]:
    trades: List[Trade] = []

    for symbol, group in prices.groupby("symbol"):
        group = group.sort_values("date").reset_index(drop=True)
        rows = list(group.itertuples(index=False))
        company = group["company"].iloc[0]
        holding = False
        entry_price: Optional[float] = None
        entry_date: Optional[pd.Timestamp] = None

        for idx in range(len(rows) - 1):
            row = rows[idx]
            next_row = rows[idx + 1]

            if not holding and getattr(row, "buy_signal"):
                holding = True
                entry_price = float(next_row.Open)
                entry_date = pd.Timestamp(next_row.date)
            elif holding and getattr(row, "sell_signal"):
                exit_price = float(next_row.Open)
                exit_date = pd.Timestamp(next_row.date)
                trades.append(
                    Trade(
                        symbol=symbol,
                        company=company,
                        entry_date=entry_date,
                        exit_date=exit_date,
                        entry_price=entry_price,
                        exit_price=exit_price,
                    )
                )
                holding = False
                entry_price = None
                entry_date = None

    return trades


def summarize_trades(trades: Iterable[Trade]) -> pd.DataFrame:
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

    report = f"""# Two-Day Momentum Strategy - Backtest Report

**Generated:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
**Period:** {start_date} to {end_date}
**Universe:** BSE-30 (30 stocks)

---

## Overall Performance

| Metric | Value |
|--------|-------|
| **Total Trades** | {metrics['total_trades']} |
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
    report += "**Entry Rules:**\n"
    report += "- Buy when a stock shows two consecutive positive days (Close > Open)\n"
    report += "- Entry at next day's Open price\n\n"
    report += "**Exit Rules:**\n"
    report += "- Sell when a stock shows two consecutive negative days (Close < Open)\n"
    report += "- Exit at next day's Open price\n\n"
    report += "**Position Sizing:**\n"
    report += "- 1 share per trade (for backtesting purposes)\n\n"

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
