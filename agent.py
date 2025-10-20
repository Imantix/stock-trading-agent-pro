#!/usr/bin/env python3
"""
Stock Trading Agent - Single entry point for all trading operations.

Default mode (daily trading):
1. Download latest BSE-30 prices
2. Generate trade calls from pre-computed backtest summary
3. Execute calls via Upstox API

Backtest mode (--action backtest):
1. Download latest BSE-30 prices
2. Run backtest and generate performance report

Analysis modes:
- analyze-1pct: Analyze 1% profit target viability
- analyze-price: Analyze price movements for optimal targets
- optimize-costs: Optimize strategy with platform costs
- test-thresholds: Test different momentum thresholds
- realtime-backtest: Run real-time simulation backtest
- realtime-backtest-oneday: Run real-time backtest for one day

This script can be scheduled as a cron job to run daily.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta, date
from pathlib import Path
import numpy as np

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
        "--action",
        type=str,
        default="trade",
        choices=[
            "trade",  # Daily trading (default)
            "backtest",  # Backtesting
            "analyze-1pct",  # Analyze 1% profit target
            "analyze-price",  # Analyze price movements
            "optimize-costs",  # Optimize with platform costs
            "test-thresholds",  # Test momentum thresholds
            "realtime-backtest",  # Real-time simulation backtest
            "realtime-backtest-oneday",  # Real-time backtest (one day)
        ],
        help="Action to perform (default: trade)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=False,  # Not required for all actions
        help="Strategy to use (e.g., two_day_momentum, channel_breakout, volume_momentum)",
    )
    parser.add_argument(
        "--index",
        type=str,
        default="bse30",
        choices=list(INDICES.keys()),
        help="Index to trade (default: bse30)",
    )
    parser.add_argument(
        "--portfolio",
        action="store_true",
        help="Run portfolio-level backtest with realistic capital allocation (for backtest action)",
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
        choices=["daily", "1m", "5m", "15m", "1h"],
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
    # Legacy support for --backtest flag
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="DEPRECATED: Use --action backtest instead",
    )

    args = parser.parse_args()

    # Handle legacy --backtest flag
    if args.backtest and args.action == "trade":
        args.action = "backtest"
        print("Note: --backtest is deprecated. Use --action backtest instead.")

    return args


# ============================================================================
# ANALYSIS AND TESTING FUNCTIONS
# ============================================================================

def analyze_1pct_target(prices_file: Path) -> None:
    """
    Analyze viability of 1% profit target with 0.4% stop loss.
    From analyze_1pct_target.py
    """
    sys.path.insert(0, str(Path(__file__).parent / "strategies"))
    import volume_momentum as vm

    print("=" * 80)
    print("1% PROFIT TARGET vs 0.4% STOP LOSS ANALYSIS")
    print("=" * 80)
    print()

    # Load data
    prices = bt.load_prices(prices_file, timeframe='intraday')
    print(f"Loaded {len(prices):,} bars")
    print()

    # Annotate signals
    print("Finding volume momentum signals...")
    annotated = vm.annotate_signals(prices)
    signals = annotated[annotated['long_entry_signal'] == True].copy()
    signals = signals.sort_values(['symbol', 'datetime']).reset_index(drop=True)

    print(f"Found {len(signals)} signals")
    print()

    # Simulate trades with 1% target / 0.4% stop
    print("Simulating trades with 1% profit target and 0.4% stop loss...")
    print("(holding until one of them hits, no time limit)")
    print()

    results = []
    for idx, signal in signals.iterrows():
        symbol = signal['symbol']
        entry_time = signal['datetime']
        entry_price = signal['Close']

        # Get future bars for this symbol after entry
        future = annotated[
            (annotated['symbol'] == symbol) &
            (annotated['datetime'] > entry_time)
        ].head(100)  # Look ahead max 100 bars

        if len(future) == 0:
            continue

        # Simulate exit
        bars_held = None
        exit_price = None
        exit_reason = None

        for i, bar in enumerate(future.itertuples(), 1):
            high = bar.High
            low = bar.Low
            close = bar.Close

            max_gain = (high - entry_price) / entry_price
            max_loss = (low - entry_price) / entry_price

            # Check if profit target hit
            if max_gain >= 0.01:  # 1%
                exit_price = entry_price * 1.01
                exit_reason = 'profit_target'
                bars_held = i
                break

            # Check if stop loss hit
            if max_loss <= -0.004:  # 0.4%
                exit_price = entry_price * 0.996
                exit_reason = 'stop_loss'
                bars_held = i
                break

            # If last bar and neither hit, exit at close
            if i >= len(future):
                exit_price = close
                exit_reason = 'max_bars'
                bars_held = i
                break

        if bars_held is not None:
            pnl = exit_price - entry_price
            pnl_pct = (pnl / entry_price) * 100

            results.append({
                'symbol': symbol,
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'bars_held': bars_held,
                'exit_reason': exit_reason,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
            })

    df = pd.DataFrame(results)

    if df.empty:
        print("No trades could be simulated")
        return

    print(f"Simulated {len(df)} trades")
    print()

    print("=" * 80)
    print("RESULTS: 1% Target vs 0.4% Stop Loss")
    print("=" * 80)
    print()

    # Exit reason breakdown
    exit_counts = df['exit_reason'].value_counts()
    for reason, count in exit_counts.items():
        pct = count / len(df) * 100
        avg_pnl = df[df['exit_reason'] == reason]['pnl_pct'].mean()
        print(f"  {reason:20s}: {count:4d} trades ({pct:5.1f}%) | Avg P&L: {avg_pnl:+.2f}%")
    print()

    # Win rate
    winners = df[df['pnl'] > 0]
    losers = df[df['pnl'] < 0]
    win_rate = len(winners) / len(df) * 100

    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Winners: {len(winners)} trades")
    print(f"Losers: {len(losers)} trades")
    print()

    # P&L stats
    print("P&L Statistics:")
    print(f"  Total P&L: {df['pnl_pct'].sum():+.2f}%")
    print(f"  Average P&L: {df['pnl_pct'].mean():+.2f}%")
    print(f"  Median P&L: {df['pnl_pct'].median():+.2f}%")
    print(f"  Best trade: {df['pnl_pct'].max():+.2f}%")
    print(f"  Worst trade: {df['pnl_pct'].min():+.2f}%")
    print()

    # Expected value
    profit_target_rate = (df['exit_reason'] == 'profit_target').sum() / len(df)
    stop_loss_rate = (df['exit_reason'] == 'stop_loss').sum() / len(df)
    timeout_rate = (df['exit_reason'] == 'max_bars').sum() / len(df)

    avg_timeout_pnl = df[df['exit_reason'] == 'max_bars']['pnl_pct'].mean() if timeout_rate > 0 else 0
    expected_value = (
        1.0 * profit_target_rate +
        -0.4 * stop_loss_rate +
        avg_timeout_pnl * timeout_rate / 100
    )

    print(f"Expected Value per trade: {expected_value:+.3f}%")
    print()

    # Save results
    output_file = Path("data/analysis/1pct_target_simulation.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")
    print()


def analyze_price_movements(prices_file: Path) -> None:
    """
    Analyze 1-minute price movements to determine optimal profit targets.
    From analyze_price_movements.py
    """
    print("=" * 80)
    print("PRICE MOVEMENT ANALYSIS - Finding Optimal Profit Target")
    print("=" * 80)
    print()

    # Load 1-minute data
    prices = bt.load_prices(prices_file, timeframe='intraday')

    print(f"Loaded {len(prices):,} 1-minute bars")
    print(f"Date range: {prices['datetime'].min()} to {prices['datetime'].max()}")
    print(f"Symbols: {prices['symbol'].nunique()}")
    print()

    # Calculate bar-to-bar price changes
    prices = prices.sort_values(['symbol', 'datetime'])
    prices['price_change_pct'] = prices.groupby('symbol')['Close'].pct_change() * 100

    # For each bar, calculate how high price goes in next N bars
    def analyze_forward_moves(group, max_bars=10):
        """Calculate max price reached in next N bars."""
        group = group.copy()

        for n in [1, 3, 5, 10]:
            # Max high in next N bars
            group[f'max_high_{n}bars'] = group['High'].shift(-n).rolling(window=n, min_periods=1).max()
            group[f'max_gain_{n}bars'] = ((group[f'max_high_{n}bars'] - group['Close']) / group['Close']) * 100

            # Min low in next N bars
            group[f'min_low_{n}bars'] = group['Low'].shift(-n).rolling(window=n, min_periods=1).min()
            group[f'max_loss_{n}bars'] = ((group[f'min_low_{n}bars'] - group['Close']) / group['Close']) * 100

        return group

    print("Calculating forward price movements (this may take a moment)...")
    prices = prices.groupby('symbol', group_keys=False).apply(analyze_forward_moves)

    # Filter out NaN values
    prices = prices.dropna(subset=['max_gain_1bars', 'max_gain_5bars'])

    print()
    print("=" * 80)
    print("SINGLE BAR (1-MINUTE) PRICE CHANGES")
    print("=" * 80)
    print()

    # Analyze single-bar moves
    changes = prices['price_change_pct'].dropna()
    print(f"Total bars analyzed: {len(changes):,}")
    print()
    print("Price change distribution:")
    print(f"  Mean: {changes.mean():.3f}%")
    print(f"  Median: {changes.median():.3f}%")
    print(f"  Std Dev: {changes.std():.3f}%")
    print()

    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("Percentiles:")
    for p in percentiles:
        val = changes.quantile(p/100)
        print(f"  {p}th percentile: {val:+.3f}%")
    print()

    # Analyze forward moves
    print("=" * 80)
    print("FORWARD-LOOKING ANALYSIS")
    print("=" * 80)

    for n_bars in [1, 3, 5, 10]:
        col = f'max_gain_{n_bars}bars'
        gains = prices[col].dropna()

        print(f"\nNext {n_bars} bars (= {n_bars} minutes):")
        print(f"  Mean max gain: {gains.mean():.3f}%")
        print(f"  Median max gain: {gains.median():.3f}%")

        # How often do we hit different profit targets?
        profit_targets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
        print(f"  Probability of hitting profit target within {n_bars} bars:")
        for target in profit_targets:
            hit_rate = (gains >= target).sum() / len(gains) * 100
            print(f"    {target:>4.1f}% target: {hit_rate:5.1f}% of the time")

    print()
    print("=" * 80)
    print("RECOMMENDED PROFIT TARGETS")
    print("=" * 80)
    print()

    # Calculate optimal target based on 5-bar holding period
    gains_5bar = prices['max_gain_5bars'].dropna()

    print("Based on 5-bar (5-minute) holding period:")
    print()
    print("Target | Hit Rate | Expected Value")
    print("-------|----------|---------------")

    for target in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        hit_rate = (gains_5bar >= target).sum() / len(gains_5bar)
        # Expected value = profit_if_hit × hit_rate - loss_if_miss × (1-hit_rate)
        # Assuming 0.5% stop loss
        expected = (target * hit_rate) - (0.5 * (1 - hit_rate))
        print(f"{target:>5.1f}% | {hit_rate*100:>7.1f}% | {expected:>+6.3f}%")

    print()


def optimize_for_platform_costs(prices_file: Path, platform_cost: float = 0.000225) -> None:
    """
    Optimize strategy parameters accounting for platform costs.
    From optimize_for_platform_costs.py
    """
    sys.path.insert(0, str(Path(__file__).parent / "strategies"))

    print("=" * 80)
    print(f"STRATEGY OPTIMIZATION WITH {platform_cost*100:.4f}% PLATFORM COST")
    print("=" * 80)
    print()

    # Load data
    prices = bt.load_prices(prices_file, timeframe='intraday')
    print(f"Loaded {len(prices):,} bars")
    print(f"Testing different parameter combinations...")
    print()

    # Test different configurations
    print("=" * 80)
    print("PARAMETER OPTIMIZATION")
    print("=" * 80)
    print()

    print(f"{'Profit':>6} | {'Stop':>5} | {'Hold':>4} | {'Trades':>6} | {'Win%':>5} | {'Avg':>7} | {'Total':>8} | Result")
    print("-" * 80)

    results = []

    # Test different profit targets (must be > platform cost to cover costs)
    profit_targets = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008]  # 0.3% to 0.8%
    stop_losses = [0.003, 0.004, 0.005, 0.006]  # 0.3% to 0.6%
    hold_periods = [3, 5, 7, 10, 15]  # bars

    for profit in profit_targets:
        for stop in stop_losses:
            for hold in hold_periods:
                # Skip if stop loss >= profit target (bad risk/reward)
                if stop >= profit:
                    continue

                # Simulate strategy
                total_pnl, win_rate, avg_trade, trades = simulate_strategy(
                    prices, profit, stop, hold, platform_cost
                )

                result = "✓ PROFIT" if total_pnl > 0 else "✗ LOSS"

                print(f"{profit*100:>5.1f}% | {stop*100:>5.1f}% | {hold:>4d} | "
                      f"{trades:>6d} | {win_rate:>5.1f} | {avg_trade*100:>+6.2f}% | "
                      f"{total_pnl*100:>+7.2f}% | {result}")

                results.append({
                    'profit_target': profit,
                    'stop_loss': stop,
                    'max_hold_bars': hold,
                    'total_pnl': total_pnl,
                    'win_rate': win_rate,
                    'avg_trade': avg_trade,
                    'trades': trades,
                })

    print()
    print("=" * 80)
    print("TOP 10 CONFIGURATIONS (by total P&L)")
    print("=" * 80)
    print()

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('total_pnl', ascending=False)

    print(f"{'Rank':>4} | {'Profit':>6} | {'Stop':>5} | {'Hold':>4} | {'Trades':>6} | "
          f"{'Win%':>5} | {'Avg':>7} | {'Total':>8}")
    print("-" * 80)

    for i, row in df_results.head(10).iterrows():
        print(f"{df_results.index.get_loc(i)+1:>4d} | "
              f"{row['profit_target']*100:>5.1f}% | "
              f"{row['stop_loss']*100:>5.1f}% | "
              f"{int(row['max_hold_bars']):>4d} | "
              f"{int(row['trades']):>6d} | "
              f"{row['win_rate']:>5.1f} | "
              f"{row['avg_trade']*100:>+6.2f}% | "
              f"{row['total_pnl']*100:>+7.2f}%")

    print()
    best = df_results.iloc[0]
    print(f"RECOMMENDED: Profit={best['profit_target']*100:.1f}%, Stop={best['stop_loss']*100:.1f}%, Hold={int(best['max_hold_bars'])} bars")
    print()


def simulate_strategy(prices, profit_target, stop_loss, max_hold_bars, platform_cost=0.000225):
    """
    Helper function to simulate strategy with different parameters.
    """
    import volume_momentum as vm

    # Temporarily override strategy parameters
    original_profit = vm.PROFIT_TARGET
    original_stop = vm.STOP_LOSS
    original_hold = vm.MAX_HOLD_BARS

    vm.PROFIT_TARGET = profit_target
    vm.STOP_LOSS = stop_loss
    vm.MAX_HOLD_BARS = max_hold_bars

    # Annotate signals
    annotated = vm.annotate_signals(prices)
    signals = annotated[annotated['long_entry_signal'] == True].copy()
    signals = signals.sort_values(['symbol', 'datetime'])

    trades = []

    for idx, signal in signals.iterrows():
        symbol = signal['symbol']
        entry_time = signal['datetime']
        entry_price = signal['Close']

        # Get future bars
        future = annotated[
            (annotated['symbol'] == symbol) &
            (annotated['datetime'] > entry_time)
        ].head(max_hold_bars)

        if len(future) == 0:
            continue

        # Simulate exit
        exit_price = None
        exit_reason = None
        bars_held = 0

        for i, bar in enumerate(future.itertuples(), 1):
            high = bar.High
            low = bar.Low
            close = bar.Close

            max_gain = (high - entry_price) / entry_price
            max_loss = (low - entry_price) / entry_price

            # Check profit target
            if max_gain >= profit_target:
                exit_price = entry_price * (1 + profit_target)
                exit_reason = 'profit_target'
                bars_held = i
                break

            # Check stop loss
            if max_loss <= -stop_loss:
                exit_price = entry_price * (1 - stop_loss)
                exit_reason = 'stop_loss'
                bars_held = i
                break

            # Max hold
            if i >= max_hold_bars:
                exit_price = close
                exit_reason = 'max_hold'
                bars_held = i
                break

        if exit_price:
            pnl_pct = (exit_price - entry_price) / entry_price
            # Subtract platform cost
            pnl_pct_after_cost = pnl_pct - platform_cost

            trades.append({
                'symbol': symbol,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'pnl_pct_after_cost': pnl_pct_after_cost,
                'exit_reason': exit_reason,
                'bars_held': bars_held,
            })

    # Restore original parameters
    vm.PROFIT_TARGET = original_profit
    vm.STOP_LOSS = original_stop
    vm.MAX_HOLD_BARS = original_hold

    if not trades:
        return 0, 0, 0, 0

    df = pd.DataFrame(trades)

    total_pnl = df['pnl_pct_after_cost'].sum()
    win_rate = (df['pnl_pct_after_cost'] > 0).sum() / len(df) * 100
    avg_trade = df['pnl_pct_after_cost'].mean()

    return total_pnl, win_rate, avg_trade, len(df)


def test_momentum_thresholds(prices_file: Path) -> None:
    """
    Test different MOMENTUM_SCORE_THRESHOLD values for Cumulative Momentum strategy.
    From test_momentum_thresholds.py
    """
    sys.path.insert(0, str(Path(__file__).parent / "strategies"))
    from backtest_realtime import RealtimeBacktest
    import cumulative_momentum as cm
    import volume_momentum as vm

    print('=' * 80)
    print('OPTIMIZING MOMENTUM_SCORE_THRESHOLD')
    print('=' * 80)
    print()

    # Load data
    prices = bt.load_prices(prices_file, timeframe='intraday')

    # Use Oct 16 data for quick testing
    oct16 = prices[prices['datetime'].dt.date == date(2025, 10, 16)].copy()

    if oct16.empty:
        print("No data for Oct 16, 2025. Using all available data.")
        oct16 = prices

    print(f'Testing on {len(oct16):,} bars')
    print(f'Symbols: {oct16["symbol"].nunique()}')
    print()

    # First run Volume Momentum baseline
    print('BASELINE: VOLUME MOMENTUM')
    print('-' * 40)
    backtest_vm = RealtimeBacktest(
        initial_capital=500_000.0,
        position_size=250,
        max_positions=1,
        lookback_bars=20,
        execution_delay_ms=100.0,
        platform_cost_pct=0.0225,
    )

    results_vm = backtest_vm.run(prices=oct16, strategy_module=vm, verbose=False)

    if not results_vm.empty:
        vm_pnl = results_vm['pnl'].sum()
        vm_trades = len(results_vm)
        vm_win_rate = (results_vm['pnl'] > 0).sum() / vm_trades * 100
        print(f'Trades: {vm_trades}')
        print(f'P&L: ₹{vm_pnl:,.2f}')
        print(f'Win Rate: {vm_win_rate:.1f}%')
    else:
        vm_pnl = 0
        vm_trades = 0
        vm_win_rate = 0
        print('No trades generated')

    print()
    print('TESTING DIFFERENT THRESHOLDS')
    print('-' * 40)

    # Test different thresholds
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    results_summary = []

    original_threshold = cm.MOMENTUM_SCORE_THRESHOLD

    for threshold in thresholds:
        # Modify threshold
        cm.MOMENTUM_SCORE_THRESHOLD = threshold

        # Run backtest
        backtest = RealtimeBacktest(
            initial_capital=500_000.0,
            position_size=250,
            max_positions=1,
            lookback_bars=20,
            execution_delay_ms=100.0,
            platform_cost_pct=0.0225,
        )

        results = backtest.run(prices=oct16, strategy_module=cm, verbose=False)

        if not results.empty:
            total_pnl = results['pnl'].sum()
            win_rate = (results['pnl'] > 0).sum() / len(results) * 100
            improvement = total_pnl - vm_pnl

            print(f"Threshold {threshold:.1f}: {len(results):2d} trades, "
                  f"₹{total_pnl:8,.0f} P&L, {win_rate:5.1f}% win rate, "
                  f"Improvement: {'+'if improvement>=0 else ''}₹{improvement:,.0f}")

            results_summary.append({
                'threshold': threshold,
                'trades': len(results),
                'pnl': total_pnl,
                'win_rate': win_rate,
                'improvement': improvement,
            })
        else:
            print(f"Threshold {threshold:.1f}: No trades generated")
            results_summary.append({
                'threshold': threshold,
                'trades': 0,
                'pnl': 0,
                'win_rate': 0,
                'improvement': -vm_pnl,
            })

    # Restore original threshold
    cm.MOMENTUM_SCORE_THRESHOLD = original_threshold

    print()
    if results_summary:
        best = max(results_summary, key=lambda x: x['pnl'])
        if best['trades'] > 0:
            print(f"BEST THRESHOLD: {best['threshold']:.1f}")
            print(f"  Trades: {best['trades']}")
            print(f"  P&L: ₹{best['pnl']:,.2f}")
            print(f"  Improvement vs baseline: ₹{best['improvement']:,.2f}")


def realtime_backtest(prices_file: Path, one_day: bool = False) -> None:
    """
    Run real-time simulation backtest.
    From test_realtime_backtest.py and test_realtime_backtest_oneday.py
    """
    sys.path.insert(0, str(Path(__file__).parent / "strategies"))
    from backtest_realtime import RealtimeBacktest
    import volume_momentum as vm

    print("=" * 80)
    if one_day:
        print("REAL-TIME SIMULATION BACKTEST (ONE DAY)")
    else:
        print("REAL-TIME SIMULATION BACKTEST")
    print("=" * 80)
    print()
    print("This backtest simulates live trading by:")
    print("  1. Processing data interval-by-interval")
    print("  2. Scanning for signals on each new bar")
    print("  3. Generating orders when conditions are met")
    print("  4. Executing orders with realistic delays")
    print("  5. Logging all timing metrics")
    print()

    # Load 1-minute data
    print("Loading 1-minute price data...")
    prices = bt.load_prices(prices_file, timeframe='intraday')

    if one_day:
        # Take only the first trading day
        prices['date'] = pd.to_datetime(prices['datetime']).dt.date
        first_day = sorted(prices['date'].unique())[0]
        prices = prices[prices['date'] == first_day].copy()
        print(f"✓ Loaded {len(prices):,} 1-minute bars for {first_day}")
    else:
        print(f"✓ Loaded {len(prices):,} 1-minute bars")

    print(f"  Date range: {prices['datetime'].min()} to {prices['datetime'].max()}")
    print(f"  Symbols: {prices['symbol'].nunique()}")
    print()

    # Initialize backtest
    max_positions = getattr(vm, 'MAX_CONCURRENT_POSITIONS', 1)
    backtest = RealtimeBacktest(
        initial_capital=1_000_000.0 if one_day else 500_000.0,
        position_size=250,
        max_positions=18 if one_day else max_positions,
        lookback_bars=20,
        execution_delay_ms=100.0,
    )

    # Run backtest
    results = backtest.run(
        prices=prices,
        strategy_module=vm,
        verbose=True
    )

    if results.empty:
        print("No trades were generated")
        return

    # Display results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    print(f"Total trades: {len(results)}")
    print(f"Total P&L: ${results['pnl'].sum():,.2f}")
    print(f"Win rate: {(results['pnl'] > 0).sum() / len(results) * 100:.1f}%")
    print()

    print("TIMING SUMMARY:")
    print(f"  Average scan time: {results['scan_time_ms'].mean():.2f} ms")
    print(f"  Average order generation: {results['order_gen_time_ms'].mean():.2f} ms")
    print(f"  Average execution time: {results['execution_time_ms'].mean():.2f} ms")
    print(f"  Average total latency: {results['total_latency_ms'].mean():.2f} ms")
    print(f"  Scans/second capacity: {1000 / results['total_latency_ms'].mean():.1f}")
    print()

    # Save results
    if one_day:
        output_dir = Path("data/backtest_results/upstox/bse30/realtime_oneday")
        results_file = output_dir / "realtime_trades_oneday.csv"
        report_file = Path("reports/upstox/bse30/realtime_backtest_oneday_report.md")
    else:
        output_dir = Path("data/backtest_results/upstox/bse30/realtime")
        results_file = output_dir / "realtime_trades.csv"
        report_file = Path("reports/upstox/bse30/realtime_backtest_report.md")

    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_file, index=False)
    print(f"✓ Trade results saved to: {results_file}")

    # Generate report
    backtest.generate_report(report_file)

    print()
    print("=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)


def main() -> int:
    args = parse_args()

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # ANALYSIS AND TESTING ACTIONS
    # ========================================================================

    # These actions don't require a strategy parameter
    analysis_actions = {
        "analyze-1pct": analyze_1pct_target,
        "analyze-price": analyze_price_movements,
        "optimize-costs": optimize_for_platform_costs,
        "test-thresholds": test_momentum_thresholds,
        "realtime-backtest": lambda p: realtime_backtest(p, one_day=False),
        "realtime-backtest-oneday": lambda p: realtime_backtest(p, one_day=True),
    }

    if args.action in analysis_actions:
        # For analysis actions, use default 1-minute data file
        prices_file = Path("data/prices/upstox/bse30/bse30_1m_prices.csv")

        if not prices_file.exists():
            print(f"✗ Error: {prices_file} not found")
            print("Please download 1-minute data first:")
            print("  python helpers/download_intraday_prices.py --interval 1minute")
            return 1

        # Run the analysis function
        analysis_actions[args.action](prices_file)
        return 0

    # ========================================================================
    # TRADING AND BACKTESTING ACTIONS (require strategy)
    # ========================================================================

    if not args.strategy:
        print(f"✗ Error: --strategy is required for {args.action} action")
        return 1

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
    if args.action == "backtest":
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
                        interval_map = {"1m": "1minute", "5m": "5minute", "15m": "15minute", "1h": "60minute"}
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
                        original_argv = sys.argv[:]
                        sys.argv = [
                            "download_prices.py",
                            "--output", str(prices_file),
                        ]
                        download_prices()
                        sys.argv = original_argv
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
                original_argv = sys.argv[:]
                sys.argv = [
                    "download_prices.py",
                    "--output", str(prices_file),
                ]
                download_prices()
                sys.argv = original_argv
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
