#!/usr/bin/env python3
"""Execute today's trade calls against the Upstox sandbox API and update portfolio state."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import pandas as pd
import requests

SANDBOX_BASE_URL = "https://api-sandbox.upstox.com/v3"
PORTFOLIO_STATE_PATH = Path("data/portfolio_state.json")


def load_portfolio_state(path: Path) -> Dict:
    if not path.exists():
        raise RuntimeError(
            f"Portfolio state not found at {path}. Run generate_daily_calls.py first to initialise it."
        )
    return json.loads(path.read_text())


def save_portfolio_state(path: Path, state: Dict) -> None:
    path.write_text(json.dumps(state, indent=2))


def load_instrument_token_map(constituents_path: Path) -> Dict[str, str]:
    df = pd.read_csv(constituents_path)
    return dict(zip(df["TickerYahoo"], df["InstrumentKey"]))


def place_order(access_token: str, payload: Dict) -> Dict:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        f"{SANDBOX_BASE_URL}/order/place",
        headers=headers,
        json=payload,
        timeout=15,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Order placement failed (status {resp.status_code}): {resp.text}"
        )
    return resp.json()


def update_portfolio_after_buy(
    state: Dict, symbol: str, quantity: int, cost: float, order_ids: list[str]
) -> None:
    state["cash"] = state.get("cash", 0.0) - cost
    positions = state.setdefault("positions", [])
    for pos in positions:
        if pos["symbol"] == symbol:
            total_qty = pos["quantity"] + quantity
            if total_qty <= 0:
                positions.remove(pos)
            else:
                avg_cost = (
                    pos["avg_price"] * pos["quantity"] + cost
                ) / total_qty
                pos["quantity"] = total_qty
                pos["avg_price"] = avg_cost
                existing_orders = pos.setdefault("order_ids", [])
                existing_orders.extend(order_ids)
            return
    positions.append(
        {
            "symbol": symbol,
            "quantity": quantity,
            "avg_price": cost / quantity,
            "order_ids": order_ids,
        }
    )


def update_portfolio_after_sell(
    state: Dict, symbol: str, quantity: int, proceeds: float
) -> None:
    """Update portfolio state after selling a position."""
    state["cash"] = state.get("cash", 0.0) + proceeds
    positions = state.setdefault("positions", [])

    for pos in positions:
        if pos["symbol"] == symbol:
            remaining_qty = pos["quantity"] - quantity
            if remaining_qty <= 0:
                # Completely sold out, remove position
                positions.remove(pos)
            else:
                # Partially sold, update quantity
                pos["quantity"] = remaining_qty
            return

    # If we get here, position wasn't found (shouldn't happen)
    # Log a warning or raise an error
    print(f"  ⚠ Warning: Position {symbol} not found in portfolio state")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls-file", type=Path, required=True)
    parser.add_argument("--constituents", type=Path, default=Path("data/bse30_constituents.csv"))
    parser.add_argument("--portfolio-state", type=Path, default=PORTFOLIO_STATE_PATH)
    parser.add_argument("--access-token", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    calls = pd.read_csv(args.calls_file)
    if calls.empty:
        print("No trades to execute today.")
        return 0

    state = load_portfolio_state(args.portfolio_state)
    instrument_map = load_instrument_token_map(args.constituents)

    access_token = args.access_token or os.getenv("UPSTOX_SANDBOX_ACCESS_TOKEN") or os.getenv("UPSTOX_ACCESS_TOKEN")
    if not access_token:
        raise RuntimeError("Upstox access token not provided")

    print(f"Executing {len(calls)} trade call(s) from {args.calls_file} against Upstox sandbox...")

    for _, row in calls.iterrows():
        symbol = row["symbol"]
        instrument_key = instrument_map.get(symbol)
        if not instrument_key:
            raise RuntimeError(f"Instrument key not found for {symbol}")
        quantity = int(row["quantity"])
        estimated_cost = float(row["estimated_cost"])

        payload = {
            "quantity": quantity,
            "product": "D",
            "validity": "DAY",
            "price": 0,
            "tag": "momentum-signal",
            "slice": False,
            "instrument_token": instrument_key,
            "order_type": "MARKET",
            "transaction_type": "BUY",
            "disclosed_quantity": 0,
            "trigger_price": 0,
            "is_amo": False,
        }

        response = place_order(access_token, payload)
        status = response.get("status")
        order_ids = response.get("data", {}).get("order_ids", [])
        print(
            f"  Placed BUY {quantity} x {symbol} (instrument {instrument_key})",
            f"status={status}, order_ids={order_ids}",
        )
        update_portfolio_after_buy(state, symbol, quantity, estimated_cost, order_ids)

    save_portfolio_state(args.portfolio_state, state)
    print(f"Updated portfolio state saved to {args.portfolio_state}")
    print(f"Remaining cash: ₹{state.get('cash', 0.0):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
