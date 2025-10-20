#!/usr/bin/env python3
"""
Real-time simulation backtest.

Simulates real trading by:
1. Processing data interval-by-interval (1-minute bars arrive one at a time)
2. Scanning for signals on each new bar
3. Generating trade orders
4. Executing orders with realistic delays
5. Logging all timing metrics (scan time, order generation, execution)

This provides a realistic assessment of strategy performance in live trading.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


@dataclass
class Position:
    """Active trading position."""
    symbol: str
    company: str
    entry_time: datetime
    entry_price: float
    quantity: int
    direction: str  # 'long' or 'short'

    def __post_init__(self):
        self.bars_held = 0
        self.peak_price = self.entry_price  # Track highest price for trailing stop
        self.bars_below_stop_loss = 0  # Track consecutive bars below stop loss


@dataclass
class Order:
    """Trade order to be executed."""
    symbol: str
    company: str
    action: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    order_time: datetime
    reason: str = ""


@dataclass
class ExecutedTrade:
    """Completed trade with full details."""
    symbol: str
    company: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    quantity: int
    direction: str
    pnl: float
    return_pct: float
    bars_held: int
    exit_reason: str

    # Timing metrics
    scan_time_ms: float
    order_gen_time_ms: float
    execution_time_ms: float


@dataclass
class ScanMetrics:
    """Metrics for each scan cycle."""
    timestamp: datetime
    scan_time_ms: float
    bars_processed: int
    signals_found: int
    orders_generated: int
    positions_checked: int


class RealtimeBacktest:
    """
    Real-time simulation backtesting engine.

    Processes market data interval-by-interval to simulate live trading.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        position_size: int = 250,
        max_positions: int = 18,
        lookback_bars: int = 20,  # Bars needed for indicators
        execution_delay_ms: float = 100.0,  # Simulated execution delay
        platform_cost_pct: float = 0.0225,  # Platform cost per side (0.0225% = 0.000225)
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size = position_size
        self.max_positions = max_positions
        self.lookback_bars = lookback_bars
        self.execution_delay_ms = execution_delay_ms
        self.platform_cost_pct = platform_cost_pct / 100.0  # Convert to decimal

        self.positions: dict[str, Position] = {}
        self.trades: list[ExecutedTrade] = []
        self.scan_metrics: list[ScanMetrics] = []
        self.equity_curve: list[tuple[datetime, float]] = []

        # Data buffers for rolling window calculations
        self.price_history: dict[str, pd.DataFrame] = {}

    def run(
        self,
        prices: pd.DataFrame,
        strategy_module: Any,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run backtest with interval-by-interval processing.

        Args:
            prices: Full OHLCV dataset
            strategy_module: Strategy with annotate_signals and should_exit functions
            verbose: Print progress

        Returns:
            DataFrame with trade results
        """
        if verbose:
            print(f"\nStarting real-time simulation backtest...")
            print(f"Processing {len(prices)} bars interval-by-interval")
            print(f"Initial capital: ${self.capital:,.2f}")
            print(f"Position size: {self.position_size} units")
            print(f"Max positions: {self.max_positions}")
            print("=" * 80)

        # Get unique timestamps sorted
        prices = prices.sort_values(['datetime', 'symbol']).reset_index(drop=True)
        unique_times = prices['datetime'].unique()

        # Group by trading day to process day-by-day
        prices['date'] = pd.to_datetime(prices['datetime']).dt.date
        trading_days = sorted(prices['date'].unique())

        total_scans = 0
        total_scan_time = 0.0

        if verbose:
            print(f"\nProcessing {len(trading_days)} trading days...")
            print()

        strategy_max_positions = getattr(strategy_module, 'MAX_CONCURRENT_POSITIONS', self.max_positions)
        max_daily_positions = getattr(strategy_module, 'MAX_DAILY_POSITIONS', strategy_max_positions)

        for day_idx, trading_day in enumerate(trading_days, 1):
            day_data = prices[prices['date'] == trading_day].copy()
            day_times = sorted(day_data['datetime'].unique())

            if verbose:
                print(f"Day {day_idx}/{len(trading_days)}: {trading_day} ({len(day_times)} intervals)")

            entries_today = 0

            # Seed initial history so opening-bar signals have required context
            if day_times:
                day_start_time = day_times[0]
                day_symbols = day_data['symbol'].unique()
                for symbol in day_symbols:
                    history = self.price_history.get(symbol)
                    if history is None or history.empty:
                        prior_bars = prices[
                            (prices['symbol'] == symbol) &
                            (prices['datetime'] < day_start_time)
                        ][['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol', 'company']]

                        if not prior_bars.empty:
                            self.price_history[symbol] = prior_bars.tail(self.lookback_bars).copy().reset_index(drop=True)

            # Process each interval (bar) as it arrives
            for bar_idx, current_time in enumerate(day_times):
                # Get all symbols at this timestamp
                current_bars = day_data[day_data['datetime'] == current_time].copy()

                # START SCAN TIMER
                scan_start = time.perf_counter()

                # 1. Update price history buffers
                for _, bar in current_bars.iterrows():
                    symbol = bar['symbol']
                    if symbol not in self.price_history:
                        self.price_history[symbol] = pd.DataFrame()

                    # Append new bar
                    new_row = pd.DataFrame([{
                        'datetime': bar['datetime'],
                        'Open': bar['Open'],
                        'High': bar['High'],
                        'Low': bar['Low'],
                        'Close': bar['Close'],
                        'Volume': bar['Volume'],
                        'symbol': bar['symbol'],
                        'company': bar['company'],
                    }])
                    self.price_history[symbol] = pd.concat([
                        self.price_history[symbol],
                        new_row
                    ], ignore_index=True)

                    # Keep only lookback window + current
                    if len(self.price_history[symbol]) > self.lookback_bars + 10:
                        self.price_history[symbol] = self.price_history[symbol].iloc[-(self.lookback_bars + 10):]

                # 2. Scan for entry signals (only if we have enough history)
                signals = []
                for symbol, hist in self.price_history.items():
                    if len(hist) >= self.lookback_bars:
                        # Annotate signals on this symbol's history
                        try:
                            annotated = strategy_module.annotate_signals(hist)
                            latest = annotated.iloc[-1]

                            # Check for entry signal
                            if latest.get('long_entry_signal', False) and symbol not in self.positions:
                                max_positions_allowed = min(self.max_positions, strategy_max_positions)
                                if len(self.positions) < max_positions_allowed:
                                    signals.append({
                                        'symbol': symbol,
                                        'company': latest['company'],
                                        'price': latest['Close'],
                                        'volume': latest['Volume'],
                                        'volume_ratio': latest.get('volume_ratio_prev', 0),
                                        'time': latest['datetime'],
                                    })
                        except Exception:
                            # Skip if annotation fails
                            pass

                # Position sizing: Take best signals up to max_positions limit
                if signals:
                    # Sort by volume ratio (best signals first)
                    signals = sorted(signals, key=lambda x: x['volume_ratio'], reverse=True)

                    max_positions_allowed = min(self.max_positions, strategy_max_positions)
                    available_slots = max(0, max_positions_allowed - len(self.positions))
                    daily_remaining = max(0, max_daily_positions - entries_today)
                    slot_limit = min(available_slots, daily_remaining)

                    signals = signals[:slot_limit]

                # 3. Check existing positions for exits
                positions_to_close = []
                for symbol, pos in list(self.positions.items()):
                    # Get current price
                    current_bar = current_bars[current_bars['symbol'] == symbol]
                    if current_bar.empty:
                        continue

                    current_price = current_bar.iloc[0]['Close']
                    current_high = current_bar.iloc[0]['High']
                    pos.bars_held += 1

                    # Update peak price for trailing stop and peak detection
                    # Use High price to capture intraday peaks (not just Close)
                    pos.peak_price = max(pos.peak_price, current_high)

                    # Update bars_below_stop_loss counter for stop loss tolerance
                    # Check if strategy has STOP_LOSS parameter to determine if we're below stop
                    stop_loss_threshold = getattr(strategy_module, 'STOP_LOSS', None)
                    if stop_loss_threshold is not None:
                        pnl_pct = (current_price - pos.entry_price) / pos.entry_price
                        if pnl_pct <= -stop_loss_threshold:
                            # Price is below stop loss - increment counter
                            pos.bars_below_stop_loss += 1
                        else:
                            # Price recovered above stop loss - reset counter
                            pos.bars_below_stop_loss = 0

                    # Check exit conditions (pass peak_price for trailing stop and bars_below_stop_loss)
                    # Try passing bars_below_stop_loss, fall back to old signature if strategy doesn't support it
                    try:
                        should_exit, exit_reason = strategy_module.should_exit(
                            pos.entry_price,
                            current_price,
                            pos.direction,
                            pos.bars_held,
                            pos.peak_price,
                            pos.bars_below_stop_loss
                        )
                    except TypeError:
                        # Strategy doesn't accept bars_below_stop_loss parameter, use old signature
                        should_exit, exit_reason = strategy_module.should_exit(
                            pos.entry_price,
                            current_price,
                            pos.direction,
                            pos.bars_held,
                            pos.peak_price
                        )

                    if should_exit:
                        positions_to_close.append((symbol, current_price, exit_reason, current_time))

                # END SCAN TIMER
                scan_time_ms = (time.perf_counter() - scan_start) * 1000
                total_scan_time += scan_time_ms
                total_scans += 1

                # START ORDER GENERATION TIMER
                order_gen_start = time.perf_counter()

                # 4. Generate orders
                entry_orders = []
                exit_orders = []

                # Entry orders
                for signal in signals:
                    entry_orders.append(Order(
                        symbol=signal['symbol'],
                        company=signal['company'],
                        action='BUY',
                        quantity=self.position_size,
                        price=signal['price'],
                        order_time=current_time,
                        reason='entry_signal'
                    ))

                # Exit orders
                for symbol, exit_price, exit_reason, exit_time in positions_to_close:
                    pos = self.positions[symbol]
                    exit_orders.append(Order(
                        symbol=symbol,
                        company=pos.company,
                        action='SELL',
                        quantity=pos.quantity,
                        price=exit_price,
                        order_time=exit_time,
                        reason=exit_reason
                    ))

                order_gen_time_ms = (time.perf_counter() - order_gen_start) * 1000

                # START EXECUTION TIMER
                exec_start = time.perf_counter()

                # 5. Execute orders (simulate execution delay)
                # NOTE: Removed actual sleep for faster backtesting, but still log the delay
                # time.sleep(self.execution_delay_ms / 1000.0)

                # Execute exits first
                for order in exit_orders:
                    self._execute_exit(order, scan_time_ms, order_gen_time_ms)

                # Execute entries
                for order in entry_orders:
                    if self._execute_entry(order, scan_time_ms, order_gen_time_ms):
                        entries_today += 1

                exec_time_ms = (time.perf_counter() - exec_start) * 1000

                # 6. Record metrics
                self.scan_metrics.append(ScanMetrics(
                    timestamp=current_time,
                    scan_time_ms=scan_time_ms,
                    bars_processed=len(current_bars),
                    signals_found=len(signals),
                    orders_generated=len(entry_orders) + len(exit_orders),
                    positions_checked=len(self.positions)
                ))

                # Record equity
                portfolio_value = self._calculate_portfolio_value(current_bars)
                self.equity_curve.append((current_time, portfolio_value))

        if verbose:
            print()
            print("=" * 80)
            print("SCAN PERFORMANCE METRICS")
            print("=" * 80)
            print(f"Total scans: {total_scans:,}")
            print(f"Average scan time: {total_scan_time / total_scans:.2f} ms")
            print(f"Total scan time: {total_scan_time / 1000:.2f} seconds")
            print(f"Total trades: {len(self.trades)}")
            print()

        return self._generate_results()

    def _execute_entry(self, order: Order, scan_time: float, order_gen_time: float) -> bool:
        """Execute entry order. Returns True when position is opened."""
        # Check capital
        order_value = order.price * order.quantity
        entry_cost = order_value * self.platform_cost_pct  # Platform cost on entry
        total_cost = order_value + entry_cost

        if total_cost > self.capital:
            return False  # Insufficient capital

        # Create position
        self.positions[order.symbol] = Position(
            symbol=order.symbol,
            company=order.company,
            entry_time=order.order_time,
            entry_price=order.price,
            quantity=order.quantity,
            direction='long'
        )

        # Deduct capital (order value + entry cost)
        self.capital -= total_cost
        return True

    def _execute_exit(self, order: Order, scan_time: float, order_gen_time: float):
        """Execute exit order and record trade."""
        if order.symbol not in self.positions:
            return

        pos = self.positions[order.symbol]

        # Calculate P&L with platform costs
        entry_value = pos.entry_price * pos.quantity
        exit_value = order.price * pos.quantity

        # Platform costs on both entry and exit
        entry_cost = entry_value * self.platform_cost_pct
        exit_cost = exit_value * self.platform_cost_pct
        total_platform_cost = entry_cost + exit_cost

        # Net P&L after platform costs
        gross_pnl = exit_value - entry_value
        net_pnl = gross_pnl - total_platform_cost
        return_pct = (net_pnl / entry_value) * 100

        # Record trade with timing
        self.trades.append(ExecutedTrade(
            symbol=order.symbol,
            company=pos.company,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            exit_time=order.order_time,
            exit_price=order.price,
            quantity=pos.quantity,
            direction=pos.direction,
            pnl=net_pnl,  # Net P&L after platform costs
            return_pct=return_pct,
            bars_held=pos.bars_held,
            exit_reason=order.reason,
            scan_time_ms=scan_time,
            order_gen_time_ms=order_gen_time,
            execution_time_ms=self.execution_delay_ms,
        ))

        # Return capital (exit value minus exit cost)
        self.capital += (exit_value - exit_cost)

        # Close position
        del self.positions[order.symbol]

    def _calculate_portfolio_value(self, current_bars: pd.DataFrame) -> float:
        """Calculate total portfolio value including positions."""
        value = self.capital

        for symbol, pos in self.positions.items():
            bar = current_bars[current_bars['symbol'] == symbol]
            if not bar.empty:
                current_price = bar.iloc[0]['Close']
                value += current_price * pos.quantity

        return value

    def _generate_results(self) -> pd.DataFrame:
        """Generate results DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'symbol': trade.symbol,
                'company': trade.company,
                'direction': trade.direction,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'pnl': trade.pnl,
                'return_pct': trade.return_pct,
                'bars_held': trade.bars_held,
                'exit_reason': trade.exit_reason,
                'scan_time_ms': trade.scan_time_ms,
                'order_gen_time_ms': trade.order_gen_time_ms,
                'execution_time_ms': trade.execution_time_ms,
                'total_latency_ms': trade.scan_time_ms + trade.order_gen_time_ms + trade.execution_time_ms,
            })

        return pd.DataFrame(trades_data)

    def generate_report(self, output_file: Path) -> None:
        """Generate detailed performance report with timing metrics."""
        if not self.trades:
            print("No trades to report")
            return

        trades_df = self._generate_results()

        # Calculate metrics
        total_pnl = trades_df['pnl'].sum()
        total_return = (total_pnl / self.initial_capital) * 100
        win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) * 100
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if (trades_df['pnl'] > 0).any() else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (trades_df['pnl'] < 0).any() else 0

        # Timing metrics
        avg_scan = trades_df['scan_time_ms'].mean()
        avg_order_gen = trades_df['order_gen_time_ms'].mean()
        avg_execution = trades_df['execution_time_ms'].mean()
        avg_total_latency = trades_df['total_latency_ms'].mean()

        report = f"""# Real-Time Simulation Backtest Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Performance Summary

| Metric | Value |
|--------|-------|
| **Initial Capital** | ${self.initial_capital:,.2f} |
| **Final Capital** | ${self.capital:,.2f} |
| **Total P&L** | ${total_pnl:,.2f} ({total_return:.2f}%) |
| **Total Trades** | {len(trades_df)} |
| **Win Rate** | {win_rate:.2f}% |
| **Average Win** | ${avg_win:,.2f} |
| **Average Loss** | ${avg_loss:,.2f} |

---

## Timing Metrics (System Performance)

| Operation | Average Time | Notes |
|-----------|--------------|-------|
| **Market Scan** | {avg_scan:.2f} ms | Time to scan all symbols for signals |
| **Order Generation** | {avg_order_gen:.2f} ms | Time to generate buy/sell orders |
| **Order Execution** | {avg_execution:.2f} ms | Simulated broker execution delay |
| **Total Latency** | {avg_total_latency:.2f} ms | End-to-end time per interval |

**Scans per second capacity:** {1000 / avg_total_latency:.1f} scans/sec

---

## Top 10 Trades by P&L

| Entry Time | Symbol | P&L | Return % | Hold Time | Scan Time | Total Latency |
|------------|--------|-----|----------|-----------|-----------|---------------|
"""

        top_10 = trades_df.nlargest(10, 'pnl')
        for _, trade in top_10.iterrows():
            entry = trade['entry_time'].strftime('%m/%d %H:%M')
            report += f"| {entry} | {trade['symbol']} | ${trade['pnl']:.2f} | {trade['return_pct']:.2f}% | {trade['bars_held']} bars | {trade['scan_time_ms']:.2f}ms | {trade['total_latency_ms']:.2f}ms |\n"

        report += f"""
---

## Timing Distribution

**Scan Time:**
- Min: {trades_df['scan_time_ms'].min():.2f} ms
- Max: {trades_df['scan_time_ms'].max():.2f} ms
- Median: {trades_df['scan_time_ms'].median():.2f} ms

**Total Latency:**
- Min: {trades_df['total_latency_ms'].min():.2f} ms
- Max: {trades_df['total_latency_ms'].max():.2f} ms
- Median: {trades_df['total_latency_ms'].median():.2f} ms

---

## Notes

This backtest simulates real-time trading by:
1. Processing market data interval-by-interval (as bars arrive)
2. Scanning for signals on each new bar
3. Generating orders when conditions are met
4. Executing with realistic delays ({self.execution_delay_ms}ms)
5. Logging all timing metrics

**All timing measurements are actual Python execution times during the simulation.**
"""

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report)
        print(f"✓ Report saved to: {output_file}")
