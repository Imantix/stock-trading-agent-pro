# Stock Trading Agent Pro

A professional algorithmic trading system with backtesting engine, portfolio simulation, and live trading capabilities. **All functionality is accessed through a single entry point: `agent.py`**

## Features

- **Single Entry Point**: All operations (trading, backtesting, analysis) through `agent.py`
- **Multiple Trading Strategies**: Six built-in strategies including Volume Momentum, Channel Breakout, Cumulative Momentum
- **Portfolio Backtesting**: Realistic simulation with capital allocation, position sizing, and risk management
- **Analysis Tools**: Built-in optimization, price analysis, and strategy testing tools
- **Performance Analytics**: CAGR, Sharpe Ratio, Max Drawdown, Win Rate tracking
- **Real-Time Simulation**: Bar-by-bar backtesting with execution timing metrics
- **Technical Indicators Library**: Reusable ATR, RSI, ADR, volume ratios, filters, and more
- **Live Trading**: Integration with Upstox API for paper/live trading
- **Multi-Market Support**: S&P 500 (via YFinance) and BSE-30/Nifty (via Upstox)

## Best Performing Strategy

**Volume Momentum V2.4** - Our flagship strategy

- **Return**: 12.62% (1 year backtest)
- **Sharpe Ratio**: 1.35
- **Max Drawdown**: -3.28%
- **Win Rate**: 58.09%
- **Total Trades**: 377

**How it works:**
- Enters on volume spikes (>1.5x average) + price momentum (>0.3%)
- Filters out defensive stocks unsuitable for momentum trading
- Exits at +0.3% profit target, -0.5% stop loss, or 5-day max hold

## Quick Usage Examples

```bash
# List all available strategies
python agent.py --strategy list

# Run backtest with Volume Momentum strategy
python agent.py --action backtest --strategy volume_momentum

# Analyze price movements for optimal profit targets
python agent.py --action analyze-price

# Optimize strategy for platform costs
python agent.py --action optimize-costs

# Run real-time simulation (one day)
python agent.py --action realtime-backtest-oneday

# Live trading with dry run
python agent.py --strategy volume_momentum --dry-run

# Live trading (requires Upstox token)
python agent.py --strategy volume_momentum
```

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd stock-trading-agent-pro

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run a Backtest

```bash
# Portfolio backtest (realistic simulation with $1M capital)
python agent.py --action backtest --strategy volume_momentum --portfolio

# Per-symbol backtest (for strategy analysis)
python agent.py --action backtest --strategy volume_momentum

# Real-time simulation backtest
python agent.py --action realtime-backtest

# List all available strategies (will show error with list)
python agent.py --strategy list
```

**Outputs:**
- `reports/yfinance/sp500/portfolio_backtest_report.md` - Portfolio performance report
- `data/backtest_results/yfinance/sp500/portfolio_history.csv` - Daily portfolio values
- `data/backtest_results/yfinance/sp500/trade_log.csv` - All trades executed

### 3. Live Trading Setup (Optional)

```bash
# Set up Upstox credentials
cp .env.example .env
# Edit .env and add your UPSTOX_ACCESS_TOKEN

# Dry run (no actual trades)
python agent.py --strategy volume_momentum --dry-run

# Live trading (default action is 'trade')
python agent.py --strategy volume_momentum
```

## Project Structure

```
.
├── agent.py                          # Single entry point for ALL operations
├── strategies/
│   ├── volume_momentum.py            # Volume Momentum V2.4 (best performer)
│   ├── channel_breakout.py           # Channel breakout strategy
│   ├── two_day_momentum.py           # Two-day momentum strategy
│   ├── cumulative_momentum.py        # Cumulative momentum with thresholds
│   ├── green_red_bars.py             # Green/red bar pattern strategy
│   └── quick_scalp.py                # Quick scalping strategy
├── helpers/
│   ├── backtest_portfolio.py         # Portfolio backtesting engine
│   ├── backtest_realtime.py          # Real-time simulation backtesting
│   ├── backtest.py                   # Per-symbol backtesting
│   ├── indicators.py                 # Technical indicators library
│   ├── download_prices.py            # Data fetching (YFinance/Upstox)
│   ├── generate_calls.py             # Trade signal generation
│   └── execute_calls.py              # Trade execution via Upstox API
├── data/
│   ├── yfinance/sp500/               # S&P 500 data and results
│   ├── upstox/bse30/                 # BSE-30 data and results
│   └── backtest_results/             # Portfolio backtest outputs
└── reports/                          # Generated performance reports
```

## Available Strategies

### Volume Momentum V2.4 (Recommended)
**Best risk-adjusted returns**

- Entry: Volume spike (>1.5x) + price up >0.3%
- Exit: +0.3% profit / -0.5% stop / 5-day max hold
- Filter: Excludes defensive stocks (VZ, KO, PG, JNJ, PEP, ABT, MCD, WMT)
- Timeframe: Daily or intraday (1m, 5m)
- Performance: 12.62% return, 1.35 Sharpe, -3.28% max DD

### Channel Breakout
**Volatility breakout strategy**

- Entry: Price breaks above upper channel
- Exit: Price falls below lower channel
- Timeframe: Daily or 5-minute intraday
- Use: Higher volatility environments

### Two-Day Momentum
**Simple trend-following**

- Entry: 2 consecutive green candles
- Exit: 2 consecutive red candles
- Timeframe: Daily
- Use: Strong trending markets

### Cumulative Momentum
**Threshold-based momentum scoring**

- Entry: Combined momentum score exceeds threshold
- Exit: Dynamic stop loss with trailing stops
- Timeframe: Intraday (1m, 5m)
- Use: Momentum bursts with risk management

### Green Red Bars
**Bar pattern recognition**

- Entry: Specific green/red bar patterns
- Exit: Pattern-based or time-based exits
- Timeframe: Daily or intraday
- Use: Pattern-based trading

### Quick Scalp
**Fast in-and-out scalping**

- Entry: Quick momentum signals
- Exit: Tight profit targets and stops
- Timeframe: 1-minute preferred
- Use: High-frequency scalping

## Technical Indicators Library

The `helpers/indicators.py` module provides reusable technical analysis tools:

**Volatility Indicators:**
- `calculate_atr()` - Average True Range
- `calculate_adr()` - Average Daily Range
- `calculate_historical_volatility()` - Annualized volatility
- `filter_by_volatility()` - Minimum volatility filter

**Momentum Indicators:**
- `calculate_rsi()` - Relative Strength Index
- `consecutive_up_days()` - Multi-day momentum confirmation
- `price_above_ma()` - Trend confirmation

**Volume Indicators:**
- `calculate_volume_ratio()` - Volume vs average
- `calculate_obv()` - On-Balance Volume

**Symbol Filtering:**
- `SymbolFilter` - Blacklist/whitelist management
- Predefined lists: `DEFENSIVE_STOCKS`, `UTILITIES`, `LOW_MOMENTUM`

**Stop Loss Calculators:**
- `calculate_dynamic_stop()` - ATR-based stops
- `calculate_percent_stop()` - Fixed percentage stops

**Usage Example:**
```python
from helpers.indicators import calculate_atr, SymbolFilter

# Add ATR for dynamic stops
df['atr'] = calculate_atr(df, period=14)

# Filter out defensive stocks
symbol_filter = SymbolFilter(blacklist=SymbolFilter.DEFENSIVE_STOCKS)
df['allowed'] = symbol_filter.filter_dataframe(df)
signals = signals & df['allowed']
```

## Portfolio Backtesting

The portfolio backtest simulates realistic trading with:
- **Starting Capital**: $1,000,000
- **Position Size**: 250 units per trade
- **Max Positions**: 18 concurrent (pyramiding)
- **Day-by-day tracking**: Portfolio value, cash, positions
- **Exit management**: Profit targets, stop losses, max hold periods

**Run Portfolio Backtest:**
```bash
python run_portfolio_backtest.py
```

**Customize Strategy:**
Edit `run_portfolio_backtest.py` to change strategy, date range, or capital.

## Command Line Options

```bash
python agent.py --help
```

**Action (Main Operation Mode):**
- `--action ACTION` - Operation to perform:
  - `trade` (default) - Live trading mode
  - `backtest` - Run backtesting
  - `analyze-1pct` - Analyze 1% profit target viability
  - `analyze-price` - Analyze price movements for optimal targets
  - `optimize-costs` - Optimize strategy with platform costs
  - `test-thresholds` - Test different momentum thresholds
  - `realtime-backtest` - Full real-time simulation
  - `realtime-backtest-oneday` - One-day real-time simulation

**Strategy:**
- `--strategy STRATEGY` - Strategy name (required for trade/backtest)
  - Available: `volume_momentum`, `channel_breakout`, `two_day_momentum`,
    `cumulative_momentum`, `green_red_bars`, `quick_scalp`

**Modes:**
- `--dry-run` - Generate trade signals without executing
- `--portfolio` - Run portfolio-level backtest (with --action backtest)

**Data:**
- `--index INDEX` - Market index: `bse30`, `sp500`, `niftysmallcap100` (default: `bse30`)
- `--timeframe TIMEFRAME` - Data timeframe: `daily`, `1m`, `5m`, `15m`, `1h` (default: `daily`)
- `--skip-download` - Use existing data without downloading

**Trading Parameters:**
- `--investment AMOUNT` - Total capital (default: ₹100,000)
- `--top-n N` - Number of stocks to trade (default: 5)
- `--position-size SIZE` - Units per trade for portfolio mode (default: 250)
- `--max-positions N` - Max concurrent positions (default: 18)
- `--initial-capital AMOUNT` - Starting capital for portfolio backtest (default: ₹1,000,000)
- `--access-token TOKEN` - Upstox API token

## Analysis and Optimization Tools

Agent.py now includes built-in analysis and optimization tools:

### 1. Analyze 1% Profit Target
Test viability of 1% profit targets vs 0.4% stop loss:
```bash
python agent.py --action analyze-1pct
```
- Simulates trades with different profit/stop combinations
- Calculates win rates and expected values
- Recommends optimal targets based on historical data

### 2. Price Movement Analysis
Analyze historical price movements to find optimal profit targets:
```bash
python agent.py --action analyze-price
```
- Studies 1-minute price change distributions
- Calculates probability of hitting various targets
- Recommends targets based on expected value

### 3. Platform Cost Optimization
Optimize strategy parameters accounting for transaction costs:
```bash
python agent.py --action optimize-costs
```
- Tests multiple parameter combinations
- Accounts for 0.0225% platform costs
- Finds configurations that remain profitable after costs

### 4. Momentum Threshold Testing
Test different threshold values for momentum strategies:
```bash
python agent.py --action test-thresholds
```
- Compares performance across threshold values
- Tests against baseline strategy
- Identifies optimal momentum scores

### 5. Real-Time Simulation
Simulate live trading with realistic execution:
```bash
# Full dataset
python agent.py --action realtime-backtest

# Single day (faster)
python agent.py --action realtime-backtest-oneday
```
- Processes data bar-by-bar like live trading
- Includes execution delays and timing metrics
- Measures scan times and capacity

## Performance Reports

Backtest reports are automatically generated with:
- Executive summary (return, Sharpe, max drawdown, CAGR)
- Trading statistics (total trades, win rate, profit factor)
- Top/bottom performing symbols
- Portfolio growth over time
- Risk metrics and strategy parameters

**Example:**
- `reports/yfinance/sp500/portfolio_backtest_report.md` - Latest portfolio backtest
- `reports/yfinance/sp500/loss_analysis_volume_momentum.md` - Strategy loss analysis

## Scheduling Automated Trading

### Linux/Mac (Cron)

**Daily trading (after market close):**
```bash
crontab -e

# Add this line (run at 3:30 PM weekdays)
30 15 * * 1-5 cd /path/to/stock-trading-agent-pro && venv/bin/python agent.py --strategy volume_momentum
```

**Weekly backtest refresh:**
```bash
# Run at 10 AM every Sunday
0 10 * * 0 cd /path/to/stock-trading-agent-pro && venv/bin/python agent.py --strategy volume_momentum --backtest
```

### Windows (Task Scheduler)

1. Open Task Scheduler
2. Create Basic Task
3. **Trigger**: Daily at 3:30 PM, weekdays only
4. **Action**: Start a program
   - Program: `C:\path\to\stock-trading-agent-pro\venv\Scripts\python.exe`
   - Arguments: `agent.py --strategy volume_momentum`
   - Start in: `C:\path\to\stock-trading-agent-pro`

## Transaction Costs

Backtests include realistic costs for Indian markets:
- Brokerage: ₹20 per trade leg
- STT: 0.025% on sell side
- Transaction charges: 0.00345%
- SEBI charges: 0.0001%
- Stamp duty: 0.003% on buy side
- GST: 18% on brokerage and charges

**Note**: For S&P 500 backtests, commissions/slippage can be configured or set to 0 for analysis.

## Creating Custom Strategies

Strategies must implement these functions:

```python
def annotate_signals(prices: pd.DataFrame, summary: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add entry/exit signals to price data"""
    # Your logic here
    prices['long_entry_signal'] = ...
    prices['short_entry_signal'] = ...
    return prices

def should_exit(entry_price: float, current_price: float, direction: str, days_held: int) -> tuple[bool, str]:
    """Determine if position should exit"""
    # Your exit logic
    if condition:
        return True, "reason"
    return False, ""

def generate_buy_calls(prices: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Generate buy signals for live trading"""
    return calls_df

def generate_sell_calls(prices: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Generate sell signals for live trading"""
    return calls_df

# Strategy metadata
STRATEGY_NAME = "Your Strategy Name"
STRATEGY_DESCRIPTION = """Strategy description"""
PARAMETERS = {...}
VERSION = "1.0"
```

Save as `strategies/your_strategy.py` and run:
```bash
python agent.py --strategy your_strategy --backtest
```

## Risk Warning

**For educational and research purposes only.**

- Trading involves substantial risk of loss
- Past performance does not guarantee future results
- Always test thoroughly in paper trading before using real capital
- The authors assume no liability for trading losses

## Dependencies

- Python 3.10+
- pandas, numpy - Data manipulation
- yfinance - S&P 500 data
- requests - API calls
- python-dotenv - Environment variables

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test your changes thoroughly
4. Submit a pull request

## License

[Specify your license]

## Contact

[Your contact information]
