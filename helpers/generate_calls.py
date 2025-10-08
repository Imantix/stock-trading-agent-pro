#!/usr/bin/env python3
"""Generate today's trade calls and snapshot current holdings for the momentum strategy."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Import backtest helper and strategy
sys.path.insert(0, str(Path(__file__).parent))
import backtest as bt

sys.path.insert(0, str(Path(__file__).parent.parent / "strategies"))
from two_day_momentum import generate_buy_calls, generate_sell_calls

DEFAULT_INVESTMENT = 100_000.0
DEFAULT_TOP_N = 5
PORTFOLIO_STATE_PATH = Path("data/portfolio_state.json")


def load_portfolio_state(path: Path, initial_cash: float) -> Dict:
    if path.exists():
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"cash": initial_cash, "positions": []}
    path.write_text(json.dumps(state, indent=2))
    return state


def save_portfolio_state(path: Path, state: Dict) -> None:
    path.write_text(json.dumps(state, indent=2))


def compute_holdings(state: Dict, prices: pd.DataFrame) -> List[dict]:
    latest = (
        prices.sort_values("date")
        .groupby("symbol")
        .tail(1)
        .set_index("symbol")
    )

    holdings: List[dict] = []
    for position in state.get("positions", []):
        symbol = position["symbol"]
        if symbol not in latest.index:
            continue
        row = latest.loc[symbol]
        current_price = float(row["Close"])
        avg_price = float(position["avg_price"])
        quantity = float(position["quantity"])
        holdings.append(
            {
                "type": "EQUITY",
                "symbol": symbol,
                "company": row["company"],
                "quantity": quantity,
                "avg_price": avg_price,
                "current_price": current_price,
                "market_value": quantity * current_price,
                "unrealized_pnl": (current_price - avg_price) * quantity,
                "order_ids": ",".join(map(str, position.get("order_ids", []))),
            }
        )

    holdings.append(
        {
            "type": "CASH",
            "symbol": "CASH",
            "company": "Cash Balance",
            "quantity": state.get("cash", 0.0),
            "avg_price": None,
            "current_price": None,
            "market_value": state.get("cash", 0.0),
            "unrealized_pnl": 0.0,
            "order_ids": "",
        }
    )
    return holdings


def compute_buy_calls(
    prices: pd.DataFrame,
    top_symbols: List[str],
    state: Dict,
    per_trade_budget: float,
) -> List[dict]:
    """Wrapper for strategy's generate_buy_calls with date annotation."""
    available_cash = state.get("cash", 0.0)
    existing_symbols = [pos["symbol"] for pos in state.get("positions", [])]

    # Use strategy's portfolio-aware buy call generation
    calls = generate_buy_calls(
        prices=prices,
        candidate_symbols=top_symbols,
        current_holdings=existing_symbols,
        available_cash=available_cash,
        per_trade_budget=per_trade_budget,
    )

    # Add planned trade date to each call
    latest_date = prices["date"].max()
    planned_trade_date = (latest_date + timedelta(days=1)).date()

    for call in calls:
        call["planned_trade_date"] = planned_trade_date
        call["budget"] = min(per_trade_budget, available_cash)

    return calls


def compute_sell_calls(
    prices: pd.DataFrame,
    state: Dict,
) -> List[dict]:
    """Wrapper for strategy's generate_sell_calls with date annotation."""
    current_positions = state.get("positions", [])

    # Use strategy's portfolio-aware sell call generation
    calls = generate_sell_calls(
        prices=prices,
        current_holdings=current_positions,
    )

    # Add planned trade date to each call
    latest_date = prices["date"].max()
    planned_trade_date = (latest_date + timedelta(days=1)).date()

    for call in calls:
        call["planned_trade_date"] = planned_trade_date

    return calls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=Path("data/bse30_daily_prices.csv"))
    parser.add_argument("--summary", type=Path, default=Path("data/bse30_summary.csv"))
    parser.add_argument("--investment", type=float, default=DEFAULT_INVESTMENT)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--calls-out", type=Path, default=None)
    parser.add_argument("--holdings-out", type=Path, default=Path("data/current_holdings.csv"))
    parser.add_argument("--portfolio-state", type=Path, default=PORTFOLIO_STATE_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    prices = bt.load_prices(args.prices)
    summary = pd.read_csv(args.summary).sort_values("net_pnl", ascending=False)
    top_symbols = summary.head(args.top_n)["symbol"].tolist()

    state = load_portfolio_state(args.portfolio_state, args.investment)

    holdings = compute_holdings(state, prices)
    calls = compute_buy_calls(prices, top_symbols, state, args.investment)

    latest_date = prices["date"].max().date()
    planned_trade_date = (prices["date"].max() + timedelta(days=1)).date()

    if args.calls_out is None:
        args.calls_out = Path(f"data/daily_calls_{planned_trade_date.isoformat()}.csv")

    args.calls_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(calls).to_csv(args.calls_out, index=False)

    args.holdings_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(holdings).to_csv(args.holdings_out, index=False)

    save_portfolio_state(args.portfolio_state, state)

    print(
        f"Generated {len(calls)} call(s) for {planned_trade_date}, saved to {args.calls_out}"
    )
    print(
        f"Captured holdings snapshot (including cash) as of {latest_date}, saved to {args.holdings_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
