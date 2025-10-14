#!/usr/bin/env python3
"""
Stock Trading Agent - Single entry point for all trading operations.

Default mode (daily trading):
1. Download latest BSE-30 prices
2. Generate trade calls from pre-computed backtest summary
3. Execute calls via Upstox API

Backtest mode (--backtest):
1. Download latest BSE-30 prices
2. Run backtest and generate performance report

This script can be scheduled as a cron job to run daily.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path

# Import the main functions from each module
sys.path.insert(0, str(Path(__file__).parent / "helpers"))

import pandas as pd

import backtest as bt
import generate_calls as calls
import execute_calls as execute
from download_prices import main as download_prices
from download_intraday_prices import main as download_intraday_prices

# Import config for multi-provider support
sys.path.insert(0, str(Path(__file__).parent))
from config import INDICES, get_data_paths


DEFAULT_INVESTMENT = 100_000.0
DEFAULT_TOP_N = 5
DATA_DIR = Path("data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Strategy to use (e.g., two_day_momentum, channel_breakout)",
    )
    parser.add_argument(
        "--index",
        type=str,
        default="bse30",
        choices=["bse30", "sp500"],
        help="Index to trade (default: bse30)",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest mode instead of daily trading",
    )
    parser.add_argument(
        "--portfolio",
        action="store_true",
        help="Run portfolio-level backtest with realistic capital allocation (requires --backtest)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1_000_000.0,
        help="Initial capital for portfolio backtest (default: $1,000,000)",
    )
    parser.add_argument(
        "--position-size",
        type=int,
        default=250,
        help="Position size in units per trade for portfolio backtest (default: 250)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=18,
        help="Maximum concurrent positions for portfolio backtest (default: 18)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="daily",
        choices=["daily", "5m", "15m", "1h"],
        help="Timeframe for data (default: daily)",
    )
    parser.add_argument(
        "--investment",
        type=float,
        default=DEFAULT_INVESTMENT,
        help="Total investment capital (default: 100,000)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of top stocks to trade (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate calls but don't execute trades",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading prices (use existing data)",
    )
    parser.add_argument(
        "--access-token",
        type=str,
        default=None,
        help="Upstox access token (or use UPSTOX_ACCESS_TOKEN env var)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load strategy module dynamically
    sys.path.insert(0, str(Path(__file__).parent / "strategies"))
    try:
        strategy_module = __import__(args.strategy)
    except ImportError:
        print(f"✗ Error: Strategy '{args.strategy}' not found in strategies/")
        print(f"  Available strategies: {', '.join([f.stem for f in (Path(__file__).parent / 'strategies').glob('*.py') if f.stem != '__init__'])}")
        return 1

    # Get index configuration
    index_config = INDICES[args.index]
    provider = index_config.provider

    # Get file paths using new config system
    paths = get_data_paths(provider, args.index, args.strategy, args.timeframe)
    prices_file = paths["prices"]
    summary_file = paths["summary"]
    report_file = paths["report"]
    constituents_file = paths["constituents"]
    portfolio_state_file = paths["portfolio_state"]

    # Determine if intraday
    is_intraday = args.timeframe != "daily"

    # BACKTEST MODE
    if args.backtest:
        print("=" * 60)
        print(f"BACKTEST MODE: {index_config.name} ({args.timeframe.upper()})")
        print(f"Provider: {provider}")
        print("=" * 60)

        # Step 1: Download latest prices
        if not args.skip_download:
            print(f"\nSTEP 1: Downloading {index_config.name} {args.timeframe} data via {provider}...")
            print("=" * 60)
            try:
                if provider == "yfinance":
                    # Use Yahoo Finance downloader
                    original_argv = sys.argv[:]
                    sys.argv = [
                        "download_yfinance.py",
                        "--index", args.index,
                        "--timeframe", args.timeframe,
                        "--output", str(prices_file),
                    ]
                    sys.path.insert(0, str(Path(__file__).parent / "helpers"))
                    from download_yfinance import main as download_yfinance
                    download_yfinance()
                    sys.argv = original_argv
                elif provider == "upstox":
                    # Use Upstox downloader
                    if is_intraday:
                        interval_map = {"5m": "5minute", "15m": "15minute", "1h": "60minute"}
                        interval = interval_map.get(args.timeframe, "5minute")
                        original_argv = sys.argv[:]
                        sys.argv = [
                            "download_intraday_prices.py",
                            "--interval", interval,
                            "--output", str(prices_file),
                        ]
                        download_intraday_prices()
                        sys.argv = original_argv
                    else:
                        download_prices()
                print(f"✓ Prices downloaded to {prices_file}\n")
            except Exception as e:
                print(f"✗ Error downloading prices: {e}")
                import traceback
                traceback.print_exc()
                return 1
        else:
            print(f"\nSkipping price download (using existing data: {prices_file})\n")

        # Step 2: Run backtest
        print("=" * 60)
        if args.portfolio:
            print(f"STEP 2: Running PORTFOLIO backtest with {args.strategy} strategy ({args.timeframe} data)...")
            print(f"Initial Capital: ${args.initial_capital:,.2f}")
            print(f"Position Size: {args.position_size} units per trade")
            print(f"Max Positions: {args.max_positions}")
        else:
            print(f"STEP 2: Running PER-SYMBOL backtest with {args.strategy} strategy ({args.timeframe} data)...")
        print("=" * 60)

        try:
            annotate_signals = getattr(strategy_module, 'annotate_signals')
            timeframe_mode = "intraday" if is_intraday else "daily"

            if args.portfolio:
                # Portfolio-level backtest with realistic capital allocation
                from backtest_portfolio import PortfolioBacktest, calculate_metrics, generate_consolidated_report

                # Load and annotate prices
                prices = bt.load_prices(prices_file, timeframe=timeframe_mode)
                prices = annotate_signals(prices)

                # Run portfolio backtest
                print(f"\nRunning portfolio backtest...")
                portfolio_bt = PortfolioBacktest(
                    initial_capital=args.initial_capital,
                    position_size=args.position_size,
                    max_positions=args.max_positions
                )

                portfolio_history = portfolio_bt.run_backtest(
                    prices=prices,
                    strategy_module=strategy_module,
                    timeframe=timeframe_mode
                )

                print(f"Backtest completed! Processed {len(portfolio_history)} trading days\n")

                # Calculate metrics
                print("Calculating performance metrics...")
                metrics = calculate_metrics(portfolio_history, portfolio_bt.trade_log)

                # Print summary
                print("=" * 60)
                print("PORTFOLIO PERFORMANCE SUMMARY")
                print("=" * 60)
                print(f"Initial Capital:     ${metrics['initial_capital']:,.2f}")
                print(f"Final Capital:       ${metrics['final_capital']:,.2f}")
                total_pnl = metrics['final_capital'] - metrics['initial_capital']
                print(f"Total P&L:           ${total_pnl:,.2f} ({metrics['total_return_pct']:.2f}%)")
                print(f"CAGR:                {metrics['cagr_pct']:.2f}%")
                print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
                print(f"Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%")
                print(f"Total Trades:        {metrics['total_trades']}")
                print(f"Win Rate:            {metrics['win_rate_pct']:.2f}%")
                print(f"Profit Factor:       {metrics['profit_factor']:.2f}")
                print("=" * 60)
                print()

                # Add portfolio configuration to metrics
                metrics['position_size'] = args.position_size
                metrics['max_positions'] = args.max_positions

                # Generate report
                portfolio_report = paths["portfolio_report"]
                print(f"Generating consolidated report: {portfolio_report}")
                generate_consolidated_report(
                    metrics=metrics,
                    results=portfolio_history,
                    trade_log=portfolio_bt.trade_log,
                    output_path=portfolio_report,
                    strategy_module=strategy_module,
                    index_name=index_config.name,
                    timeframe=args.timeframe.capitalize()
                )
                print(f"✓ Report saved to: {portfolio_report}\n")

                # Save portfolio data
                portfolio_history.to_csv(paths["portfolio_history"], index=False)
                print(f"✓ Portfolio history saved to: {paths['portfolio_history']}")

                trade_log_df = pd.DataFrame(portfolio_bt.trade_log)
                trade_log_df.to_csv(paths["portfolio_trade_log"], index=False)
                print(f"✓ Trade log saved to: {paths['portfolio_trade_log']}")

            else:
                # Per-symbol backtest (original behavior)
                prices = bt.load_prices(prices_file, timeframe=timeframe_mode)
                prices = annotate_signals(prices)
                trades = bt.generate_trades(prices, timeframe=timeframe_mode)
                trades_df = bt.summarize_trades(trades)

                if trades_df.empty:
                    print("✗ No trades were generated by the strategy")
                    return 1

                summary_df = bt.aggregate_summary(trades_df)
                metrics = bt.overall_metrics(trades_df)

                # Save summary
                summary_df.to_csv(summary_file, index=False)
                print(f"✓ Summary saved to {summary_file}")

                # Generate report
                bt.generate_markdown_report(metrics, summary_df, trades_df, report_file)
                print(f"✓ Report saved to {report_file}")

            print("\n" + "=" * 60)
            print("BACKTEST COMPLETE")
            print("=" * 60)
            if args.portfolio:
                print(f"\nView report: {paths['portfolio_report']}")
            else:
                print(f"\nView report: {report_file}")

            return 0

        except Exception as e:
            print(f"✗ Error running backtest: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # DAILY TRADING MODE (default)
    # Step 1: Download latest prices
    if not args.skip_download:
        print("=" * 60)
        print(f"STEP 1: Downloading latest BSE-30 {args.timeframe} prices...")
        print("=" * 60)
        try:
            if is_intraday:
                # Map timeframe to interval
                interval_map = {"5m": "5minute", "15m": "15minute", "1h": "60minute"}
                interval = interval_map.get(args.timeframe, "5minute")

                # Call intraday downloader with interval parameter
                original_argv = sys.argv[:]
                sys.argv = [
                    "download_intraday_prices.py",
                    "--interval", interval,
                    "--output", str(prices_file),
                ]
                download_intraday_prices()
                sys.argv = original_argv
            else:
                download_prices()
            print(f"✓ Prices downloaded to {prices_file}\n")
        except Exception as e:
            print(f"✗ Error downloading prices: {e}")
            return 1
    else:
        print("Skipping price download (using existing data)\n")

    # Step 2: Load pre-computed backtest summary
    print("=" * 60)
    print("STEP 2: Loading backtest summary...")
    print("=" * 60)

    if not summary_file.exists():
        print(f"✗ Error: Backtest summary not found at {summary_file}")
        print(f"  Please run: python agent.py --strategy {args.strategy} --backtest")
        return 1

    try:
        summary_df = pd.read_csv(summary_file)
        print(f"✓ Loaded summary with {len(summary_df)} symbols\n")
    except Exception as e:
        print(f"✗ Error loading summary: {e}")
        return 1

    # Step 3: Load prices for call generation
    try:
        if is_intraday:
            prices = pd.read_csv(prices_file, parse_dates=["datetime"])
        else:
            prices = pd.read_csv(prices_file, parse_dates=["date"])
    except Exception as e:
        print(f"✗ Error loading prices: {e}")
        return 1

    # Step 3: Generate trade calls (both buy and sell)
    print("=" * 60)
    print("STEP 3: Generating trade calls...")
    print("=" * 60)
    try:
        top_symbols = summary_df.sort_values("net_pnl", ascending=False).head(args.top_n)["symbol"].tolist()
        state = calls.load_portfolio_state(portfolio_state_file, args.investment)
        holdings = calls.compute_holdings(state, prices)

        # Generate sell calls for current holdings
        sell_calls = calls.compute_sell_calls(prices, state, strategy_module)

        # Generate buy calls for top performers
        buy_calls = calls.compute_buy_calls(prices, top_symbols, state, args.investment, strategy_module)

        if not sell_calls and not buy_calls:
            print("✓ No new trade calls to execute")
            return 0

        # Display sell calls
        if sell_calls:
            print(f"✓ Generated {len(sell_calls)} SELL call(s):")
            for call in sell_calls:
                print(f"  - {call['symbol']}: SELL {call['quantity']} @ ~₹{call['reference_price']:.2f}")
            print()

        # Display buy calls
        if buy_calls:
            print(f"✓ Generated {len(buy_calls)} BUY call(s):")
            for call in buy_calls:
                print(f"  - {call['symbol']}: BUY {call['quantity']} @ ~₹{call['reference_price']:.2f}")
            print()

        # Save all calls for reference
        if is_intraday:
            latest_datetime = prices["datetime"].max()
            planned_date = latest_datetime.date()
        else:
            latest_date = prices["date"].max()
            planned_date = (latest_date + timedelta(days=1)).date()

        if sell_calls:
            sell_calls_file = DATA_DIR / f"daily_sell_calls_{planned_date.isoformat()}.csv"
            pd.DataFrame(sell_calls).to_csv(sell_calls_file, index=False)
            print(f"✓ Sell calls saved to {sell_calls_file}")

        if buy_calls:
            buy_calls_file = DATA_DIR / f"daily_buy_calls_{planned_date.isoformat()}.csv"
            pd.DataFrame(buy_calls).to_csv(buy_calls_file, index=False)
            print(f"✓ Buy calls saved to {buy_calls_file}")
        print()

    except Exception as e:
        print(f"✗ Error generating calls: {e}")
        return 1

    # Step 4: Execute trades (if not dry run)
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN MODE: Skipping trade execution")
        print("=" * 60)
        return 0

    print("=" * 60)
    print("STEP 4: Executing trades via Upstox API...")
    print("=" * 60)

    try:
        access_token = args.access_token or os.getenv("UPSTOX_ACCESS_TOKEN")
        if not access_token:
            print("✗ Error: Upstox access token not provided")
            print("  Set UPSTOX_ACCESS_TOKEN environment variable or use --access-token")
            return 1

        if not constituents_file.exists():
            print(f"✗ Error: Constituents file not found at {constituents_file}")
            print("  Please ensure you have the BSE-30 constituents CSV with InstrumentKey mapping")
            return 1

        # Load instrument mapping
        instrument_map = execute.load_instrument_token_map(constituents_file)

        # Execute SELL orders first (to free up cash)
        if sell_calls:
            print("Executing SELL orders...")
            for call in sell_calls:
                symbol = call["symbol"]
                instrument_key = instrument_map.get(symbol)
                if not instrument_key:
                    print(f"  ✗ Skipping {symbol}: instrument key not found")
                    continue

                quantity = int(call["quantity"])
                estimated_proceeds = float(call["estimated_proceeds"])

                payload = {
                    "quantity": quantity,
                    "product": "D",
                    "validity": "DAY",
                    "price": 0,
                    "tag": "momentum-signal",
                    "slice": False,
                    "instrument_token": instrument_key,
                    "order_type": "MARKET",
                    "transaction_type": "SELL",
                    "disclosed_quantity": 0,
                    "trigger_price": 0,
                    "is_amo": False,
                }

                response = execute.place_order(access_token, payload)
                status = response.get("status")
                order_ids = response.get("data", {}).get("order_ids", [])

                print(f"  ✓ {symbol}: SELL {quantity} - status={status}, order_ids={order_ids}")

                # Update portfolio state (remove position, add cash)
                execute.update_portfolio_after_sell(state, symbol, quantity, estimated_proceeds)

            print()

        # Execute BUY orders
        if buy_calls:
            print("Executing BUY orders...")
            for call in buy_calls:
                symbol = call["symbol"]
                instrument_key = instrument_map.get(symbol)
                if not instrument_key:
                    print(f"  ✗ Skipping {symbol}: instrument key not found")
                    continue

                quantity = int(call["quantity"])
                estimated_cost = float(call["estimated_cost"])

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

                response = execute.place_order(access_token, payload)
                status = response.get("status")
                order_ids = response.get("data", {}).get("order_ids", [])

                print(f"  ✓ {symbol}: BUY {quantity} - status={status}, order_ids={order_ids}")

                # Update portfolio state
                execute.update_portfolio_after_buy(state, symbol, quantity, estimated_cost, order_ids)

            print()

        # Save updated portfolio state
        execute.save_portfolio_state(portfolio_state_file, state)
        print(f"✓ Portfolio state updated. Remaining cash: ₹{state.get('cash', 0.0):.2f}")
        print("=" * 60)
        print("TRADING PIPELINE COMPLETE")
        print("=" * 60)

    except Exception as e:
        print(f"✗ Error executing trades: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
