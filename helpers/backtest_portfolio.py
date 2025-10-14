#!/usr/bin/env python3
"""
Portfolio-level backtesting with realistic capital allocation.

This module simulates actual trading with:
- Starting capital (e.g., $1,000,000)
- Fixed position sizing (250 units per trade)
- Max pyramiding (18 concurrent positions)
- Day-by-day portfolio tracking
- Proper P&L calculation with position reversals
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "strategies"))


@dataclass
class Position:
    """Represents a single position in the portfolio."""
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    entry_date: pd.Timestamp
    quantity: int = 250  # Fixed position size
    days_held: int = 0  # Track holding period


@dataclass
class PortfolioState:
    """Portfolio state at a point in time."""
    date: pd.Timestamp
    cash: float
    positions: Dict[str, Position]
    total_value: float

    def position_value(self, symbol: str, current_price: float) -> float:
        """Calculate current value of a position."""
        if symbol not in self.positions:
            return 0.0

        pos = self.positions[symbol]
        if pos.direction == "long":
            return pos.quantity * current_price
        else:  # short
            # Short P&L = (entry_price - current_price) * quantity
            # Value = initial_value + P&L
            initial_value = pos.quantity * pos.entry_price
            pnl = (pos.entry_price - current_price) * pos.quantity
            return initial_value + pnl


class PortfolioBacktest:
    """Portfolio-level backtesting engine."""

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        position_size: int = 250,
        max_positions: int = 18,
    ):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.max_positions = max_positions

        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.portfolio_history: List[PortfolioState] = []
        self.trade_log: List[Dict] = []

    def run_backtest(
        self,
        prices: pd.DataFrame,
        strategy_module,
        timeframe: str = "daily"
    ) -> pd.DataFrame:
        """
        Run portfolio backtest day by day.

        Args:
            prices: Price data with signals annotated
            strategy_module: Strategy module with annotate_signals
            timeframe: "daily" or "intraday"

        Returns:
            DataFrame with daily portfolio values and metrics
        """
        # Annotate signals
        prices = strategy_module.annotate_signals(prices)

        # Get time column
        time_col = "datetime" if timeframe == "intraday" else "date"

        # Group by date and process day by day
        dates = sorted(prices[time_col].unique())

        print(f"Running portfolio backtest from {dates[0]} to {dates[-1]}")
        print(f"Initial capital: ${self.initial_capital:,.2f}")
        print(f"Position size: {self.position_size} units")
        print(f"Max positions: {self.max_positions}")
        print("=" * 60)

        for i, current_date in enumerate(dates):
            if i == 0:
                # Skip first date (need previous bar for signals)
                continue

            # Get data up to current date for signal checking
            historical_data = prices[prices[time_col] <= current_date]

            # Process signals for this date
            self._process_date(historical_data, current_date, time_col)

            # Record portfolio state
            self._record_portfolio_state(current_date, historical_data, time_col)

        # Convert to DataFrame
        return self._generate_results()

    def _process_date(
        self,
        historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        time_col: str
    ):
        """Process all signals for a given date."""
        # Get latest price for each symbol
        latest_prices = historical_data.groupby('symbol').tail(1)

        # STEP 1: Check profit targets and stop losses for existing positions FIRST
        positions_to_close = []
        for symbol, position in list(self.positions.items()):
            # Get current price for this position
            symbol_row = latest_prices[latest_prices['symbol'] == symbol]
            if symbol_row.empty:
                continue

            current_price = float(symbol_row.iloc[0]['Close'])

            # Increment days held
            position.days_held += 1

            # Check if strategy has should_exit function
            try:
                strategy_module = sys.modules.get('volume_momentum') or sys.modules.get('channel_breakout')
                if strategy_module and hasattr(strategy_module, 'should_exit'):
                    should_close, reason = strategy_module.should_exit(
                        position.entry_price,
                        current_price,
                        position.direction,
                        position.days_held
                    )
                    if should_close:
                        positions_to_close.append((symbol, current_price, current_date, reason, symbol_row.iloc[0]))
            except Exception:
                pass  # Strategy doesn't have should_exit, use signals only

        # Close positions that hit profit/stop
        for symbol, price, date, reason, row in positions_to_close:
            self._close_position(symbol, price, date, reason)

        # STEP 2: Process entry signals for new positions
        for _, row in latest_prices.iterrows():
            symbol = row['symbol']

            # Check for long entry signal
            long_signal = row.get('long_entry_signal', False) or row.get('buy_signal', False)
            # Check for short entry signal
            short_signal = row.get('short_entry_signal', False) or row.get('sell_signal', False)

            current_price = float(row['Close'])

            # LONG ENTRY (or reversal from SHORT to LONG)
            if long_signal:
                self._enter_long(symbol, current_price, current_date, row)

            # SHORT ENTRY (or reversal from LONG to SHORT)
            elif short_signal:
                self._enter_short(symbol, current_price, current_date, row)

    def _enter_long(
        self,
        symbol: str,
        price: float,
        date: pd.Timestamp,
        row: pd.Series
    ):
        """Enter or reverse to long position."""
        # If currently short, close short first
        if symbol in self.positions and self.positions[symbol].direction == "short":
            self._close_position(symbol, price, date, "reversal_to_long")

        # Check if we're already long
        if symbol in self.positions and self.positions[symbol].direction == "long":
            return  # Already long, no action

        # Check max positions limit
        if len(self.positions) >= self.max_positions:
            return  # Max positions reached

        # Calculate required capital
        required_capital = self.position_size * price

        # Check if we have enough cash
        if required_capital > self.cash:
            # Scale down position size
            actual_size = int(self.cash / price)
            if actual_size < 1:
                return  # Not enough cash
        else:
            actual_size = self.position_size

        # Enter long position
        self.positions[symbol] = Position(
            symbol=symbol,
            direction="long",
            entry_price=price,
            entry_date=date,
            quantity=actual_size
        )

        # Deduct cash
        self.cash -= actual_size * price

        # Log trade
        self.trade_log.append({
            'date': date,
            'symbol': symbol,
            'action': 'LONG_ENTRY',
            'price': price,
            'quantity': actual_size,
            'value': actual_size * price,
            'company': row.get('company', 'Unknown')
        })

    def _enter_short(
        self,
        symbol: str,
        price: float,
        date: pd.Timestamp,
        row: pd.Series
    ):
        """Enter or reverse to short position."""
        # If currently long, close long first
        if symbol in self.positions and self.positions[symbol].direction == "long":
            self._close_position(symbol, price, date, "reversal_to_short")

        # Check if we're already short
        if symbol in self.positions and self.positions[symbol].direction == "short":
            return  # Already short, no action

        # Check max positions limit
        if len(self.positions) >= self.max_positions:
            return  # Max positions reached

        # For shorts, we need margin (assume 100% margin requirement)
        required_capital = self.position_size * price

        # Check if we have enough cash
        if required_capital > self.cash:
            # Scale down position size
            actual_size = int(self.cash / price)
            if actual_size < 1:
                return  # Not enough cash
        else:
            actual_size = self.position_size

        # Enter short position
        self.positions[symbol] = Position(
            symbol=symbol,
            direction="short",
            entry_price=price,
            entry_date=date,
            quantity=actual_size
        )

        # Deduct margin requirement
        self.cash -= actual_size * price

        # Log trade
        self.trade_log.append({
            'date': date,
            'symbol': symbol,
            'action': 'SHORT_ENTRY',
            'price': price,
            'quantity': actual_size,
            'value': actual_size * price,
            'company': row.get('company', 'Unknown')
        })

    def _close_position(
        self,
        symbol: str,
        price: float,
        date: pd.Timestamp,
        reason: str
    ):
        """Close an existing position."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        # Calculate P&L
        if pos.direction == "long":
            pnl = (price - pos.entry_price) * pos.quantity
            proceeds = pos.quantity * price
        else:  # short
            pnl = (pos.entry_price - price) * pos.quantity
            proceeds = (pos.quantity * pos.entry_price) + pnl

        # Add proceeds to cash
        self.cash += proceeds

        # Log trade
        self.trade_log.append({
            'date': date,
            'symbol': symbol,
            'action': f'{pos.direction.upper()}_EXIT',
            'price': price,
            'quantity': pos.quantity,
            'value': proceeds,
            'pnl': pnl,
            'return_pct': (pnl / (pos.quantity * pos.entry_price)) * 100,
            'holding_days': (date - pos.entry_date).days,
            'reason': reason
        })

        # Remove position
        del self.positions[symbol]

    def _record_portfolio_state(
        self,
        date: pd.Timestamp,
        historical_data: pd.DataFrame,
        time_col: str
    ):
        """Record current portfolio state."""
        # Get latest prices for position valuation
        latest_prices = historical_data.groupby('symbol').tail(1)
        price_dict = dict(zip(latest_prices['symbol'], latest_prices['Close']))

        # Calculate total portfolio value
        positions_value = sum(
            self.position_value(symbol, price_dict.get(symbol, pos.entry_price))
            for symbol, pos in self.positions.items()
        )

        total_value = self.cash + positions_value

        # Record state
        state = PortfolioState(
            date=date,
            cash=self.cash,
            positions=self.positions.copy(),
            total_value=total_value
        )

        self.portfolio_history.append(state)

    def position_value(self, symbol: str, current_price: float) -> float:
        """Calculate current value of a position."""
        if symbol not in self.positions:
            return 0.0

        pos = self.positions[symbol]
        if pos.direction == "long":
            return pos.quantity * current_price
        else:  # short
            initial_value = pos.quantity * pos.entry_price
            pnl = (pos.entry_price - current_price) * pos.quantity
            return initial_value + pnl

    def _generate_results(self) -> pd.DataFrame:
        """Generate results DataFrame with daily portfolio values."""
        if not self.portfolio_history:
            return pd.DataFrame()

        results = pd.DataFrame([
            {
                'date': state.date,
                'cash': state.cash,
                'positions_count': len(state.positions),
                'total_value': state.total_value,
                'return': (state.total_value / self.initial_capital - 1.0) * 100,
            }
            for state in self.portfolio_history
        ])

        # Calculate daily returns
        results['daily_return'] = results['total_value'].pct_change() * 100

        # Calculate cumulative max
        results['cumulative_max'] = results['total_value'].cummax()

        # Calculate drawdown
        results['drawdown'] = (results['total_value'] / results['cumulative_max'] - 1.0) * 100

        return results


def calculate_metrics(results: pd.DataFrame, trade_log: List[Dict]) -> Dict:
    """Calculate comprehensive portfolio metrics."""
    if results.empty:
        return {}

    # Time period
    start_date = results['date'].min()
    end_date = results['date'].max()
    days = (end_date - start_date).days
    years = days / 365.25

    # Returns
    initial_value = results['total_value'].iloc[0]
    final_value = results['total_value'].iloc[-1]
    total_return = (final_value / initial_value - 1.0) * 100

    # CAGR
    if years > 0:
        cagr = ((final_value / initial_value) ** (1 / years) - 1.0) * 100
    else:
        cagr = 0.0

    # Sharpe Ratio (annualized, assume 0% risk-free rate)
    daily_returns = results['daily_return'].dropna()
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max Drawdown
    max_dd = results['drawdown'].min()

    # Trade statistics
    trades_df = pd.DataFrame(trade_log)
    closed_trades = trades_df[trades_df['action'].str.contains('EXIT', na=False)]

    if len(closed_trades) > 0:
        total_trades = len(closed_trades)
        winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
        win_rate = (winning_trades / total_trades) * 100
        avg_win = closed_trades[closed_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = closed_trades[closed_trades['pnl'] <= 0]['pnl'].mean() if total_trades > winning_trades else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * (total_trades - winning_trades))) if avg_loss != 0 else 0
    else:
        total_trades = 0
        win_rate = 0
        profit_factor = 0

    return {
        'start_date': start_date,
        'end_date': end_date,
        'days': days,
        'years': years,
        'initial_capital': initial_value,
        'final_capital': final_value,
        'total_return_pct': total_return,
        'cagr_pct': cagr,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_dd,
        'total_trades': total_trades,
        'win_rate_pct': win_rate,
        'profit_factor': profit_factor,
    }


def generate_consolidated_report(
    metrics: Dict,
    results: pd.DataFrame,
    trade_log: List[Dict],
    output_path: Path,
    strategy_module=None,
    index_name: str = "S&P 500",
    timeframe: str = "Daily"
):
    """Generate consolidated markdown report."""
    trades_df = pd.DataFrame(trade_log)
    closed_trades = trades_df[trades_df['action'].str.contains('EXIT', na=False)]

    # Get strategy info from module if available
    strategy_name = getattr(strategy_module, 'STRATEGY_NAME', 'Unknown Strategy')
    strategy_params = getattr(strategy_module, 'PARAMETERS', {})

    # Top performing symbols
    if len(closed_trades) > 0:
        symbol_performance = closed_trades.groupby('symbol').agg({
            'pnl': ['sum', 'count', 'mean'],
            'return_pct': 'mean'
        }).round(2)
        symbol_performance.columns = ['total_pnl', 'trades', 'avg_pnl', 'avg_return_pct']
        symbol_performance = symbol_performance.sort_values('total_pnl', ascending=False)

        top_10 = symbol_performance.head(10)
        bottom_10 = symbol_performance.tail(10)
    else:
        top_10 = pd.DataFrame()
        bottom_10 = pd.DataFrame()

    report = f"""# {strategy_name} - Portfolio Backtest Report

**Generated:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
**Strategy:** {strategy_name}
**Index:** {index_name}
**Timeframe:** {timeframe}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Test Period** | {metrics['start_date'].strftime('%Y-%m-%d')} to {metrics['end_date'].strftime('%Y-%m-%d')} |
| **Duration** | {metrics['days']} days ({metrics['years']:.2f} years) |
| **Initial Capital** | ${metrics['initial_capital']:,.2f} |
| **Final Capital** | ${metrics['final_capital']:,.2f} |
| **Total Return** | {metrics['total_return_pct']:.2f}% |
| **CAGR** | {metrics['cagr_pct']:.2f}% |
| **Sharpe Ratio** | {metrics['sharpe_ratio']:.2f} |
| **Max Drawdown** | {metrics['max_drawdown_pct']:.2f}% |

---

## Trading Statistics

| Metric | Value |
|--------|-------|
| **Total Trades** | {metrics['total_trades']} |
| **Win Rate** | {metrics['win_rate_pct']:.2f}% |
| **Profit Factor** | {metrics['profit_factor']:.2f} |

---

## Top 10 Performing Symbols

| Rank | Symbol | Total P&L | Trades | Avg P&L | Avg Return % |
|------|--------|-----------|--------|---------|--------------|
"""

    if not top_10.empty:
        for idx, (symbol, row) in enumerate(top_10.iterrows(), 1):
            report += f"| {idx} | {symbol} | ${row['total_pnl']:,.2f} | {int(row['trades'])} | ${row['avg_pnl']:,.2f} | {row['avg_return_pct']:.2f}% |\n"

    report += "\n---\n\n## Bottom 10 Performing Symbols\n\n"
    report += "| Rank | Symbol | Total P&L | Trades | Avg P&L | Avg Return % |\n"
    report += "|------|--------|-----------|--------|---------|--------------||\n"

    if not bottom_10.empty:
        for idx, (symbol, row) in enumerate(bottom_10.iterrows(), 1):
            report += f"| {idx} | {symbol} | ${row['total_pnl']:,.2f} | {int(row['trades'])} | ${row['avg_pnl']:,.2f} | {row['avg_return_pct']:.2f}% |\n"

    report += f"""
---

## Portfolio Growth

**Starting Capital:** ${metrics['initial_capital']:,.2f}
**Ending Capital:** ${metrics['final_capital']:,.2f}
**Profit/Loss:** ${metrics['final_capital'] - metrics['initial_capital']:,.2f}

---

## Risk Metrics

- **Maximum Drawdown:** {metrics['max_drawdown_pct']:.2f}%
- **Sharpe Ratio:** {metrics['sharpe_ratio']:.2f}
- **Total Return:** {metrics['total_return_pct']:.2f}%
- **Annualized Return (CAGR):** {metrics['cagr_pct']:.2f}%

---

## Strategy Configuration

- **Position Size:** {metrics.get('position_size', 250)} units per trade
- **Max Positions:** {metrics.get('max_positions', 18)} (pyramiding)
"""

    # Add strategy-specific parameters if available
    if strategy_params:
        report += "\n### Strategy Parameters\n\n"
        for param, value in strategy_params.items():
            param_name = param.replace('_', ' ').title()
            report += f"- **{param_name}:** {value}\n"

    report += """
- **Commission:** 0% (not included in backtest)
- **Slippage:** 0 (not included in backtest)

---

## Notes

This backtest uses realistic portfolio simulation with:
- Fixed starting capital
- Fixed position sizing per trade
- Maximum concurrent position limit
- Day-by-day portfolio tracking
- Entry/exit rules based on strategy signals

**Important:** Real trading will include commissions, slippage, and execution delays not reflected here.
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)

    print(f"\n✓ Consolidated report saved to {output_path}")
