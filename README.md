# Stock Trading Agent

An algorithmic trading system for Indian equities (BSE-30) implementing the **two-day momentum strategy** - a simple yet effective trend-following approach that buys after two consecutive positive days and sells after two consecutive negative days.

## Strategy: Two-Day Momentum

This agent implements the **two-day momentum strategy**, which:
- Identifies stocks showing consistent upward momentum (2 consecutive green candles)
- Enters positions at the next day's open
- Exits when momentum reverses (2 consecutive red candles)
- Includes realistic transaction costs for Indian markets
- Provides portfolio management and trade execution via Upstox API

## Features

- **Backtesting Engine**: Complete backtesting framework with realistic transaction costs
- **Live Trading**: Integration with Upstox sandbox API for paper trading
- **Portfolio Management**: Track positions, cash balance, and performance
- **Data Management**: Automated historical data fetching from Yahoo Finance

## Project Structure

```
.
├── agent.py                        # Main trading agent (cron-job ready)
├── strategies/
│   └── two_day_momentum.py        # Strategy logic (entry/exit rules)
├── helpers/
│   ├── download_prices.py         # Download price data (differential updates)
│   ├── backtest.py                # Backtesting framework
│   ├── generate_calls.py          # Generate trade signals
│   └── execute_calls.py           # Execute trades via Upstox API
├── data/
│   └── bse30_constituents.csv     # BSE-30 stocks with Upstox instrument keys
└── requirements.txt                # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stock-trading-agent
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Upstox access token
# Get your token from: https://upstox.com/developer/
```

## Usage

### Quick Start

**1. First Time Setup - Run Backtest:**

```bash
# Download historical prices (1 year initially, then differential updates)
python helpers/download_prices.py

# Run backtest
python helpers/backtest.py \
    --prices data/bse30_daily_prices.csv \
    --summary-output data/bse30_summary.csv \
    --report-output data/backtest_report.md
```

**Outputs:**
- [data/bse30_summary.csv](data/bse30_summary.csv) - Performance summary (used by trading pipeline)
- [data/backtest_report.md](data/backtest_report.md) - **Detailed markdown report** with Sharpe Ratio, Max Drawdown, top/bottom performers

Run this weekly/monthly to update stock rankings.

**2. Daily Trading - Run Trade Pipeline:**

```bash
# Dry run (no actual trades)
python agent.py --dry-run

# Live trading
python agent.py --investment 100000 --top-n 5
```

This will:
1. Download latest BSE-30 prices
2. Load pre-computed backtest summary
3. Generate trade calls for top performers
4. Execute trades via Upstox API

### Schedule as Cron Job

**Daily Trading (after market close):**
```bash
# Run at 3:30 PM on weekdays
30 15 * * 1-5 cd /path/to/stock-trading-agent && /path/to/.venv/bin/python agent.py
```

**Weekly Backtest Update (optional):**
```bash
# Run backtest every Sunday at 10 AM
0 10 * * 0 cd /path/to/stock-trading-agent && /path/to/.venv/bin/python helpers/backtest.py --prices data/bse30_daily_prices.csv --summary-output data/bse30_summary.csv --report-output data/backtest_report.md
```

### Advanced: Run Components Individually

If you need more control, you can run each step separately:

**1. Download Historical Data:**
```bash
# Differential update (default - only downloads new data)
python helpers/download_prices.py

# Or full refresh (1 year)
python helpers/download_prices.py --full
```

**2. Backtest Strategy:**
```bash
python helpers/backtest.py \
    --prices data/bse30_daily_prices.csv \
    --summary-output data/bse30_summary.csv \
    --report-output data/backtest_report.md
```

**Outputs:**
- [data/bse30_summary.csv](data/bse30_summary.csv) - Performance summary
- [data/backtest_report.md](data/backtest_report.md) - Detailed markdown report

**3. Generate Trade Signals:**
```bash
python helpers/generate_calls.py \
    --prices data/bse30_daily_prices.csv \
    --summary data/bse30_summary.csv \
    --investment 100000 \
    --top-n 5
```

**4. Execute Trades:**
```bash
python helpers/execute_calls.py \
    --calls-file data/daily_calls_<date>.csv \
    --constituents data/bse30_constituents.csv \
    --access-token $UPSTOX_ACCESS_TOKEN
```

### Command Line Options

```bash
python agent.py --help

Options:
  --investment AMOUNT    Total investment capital (default: 100,000)
  --top-n N             Number of top stocks to trade (default: 5)
  --dry-run             Generate calls but don't execute trades
  --skip-download       Skip downloading prices (use existing data)
  --access-token TOKEN  Upstox access token (or use env var)
```

## Strategy Details

### Two-Day Momentum
- Buys stocks after two consecutive positive days (green candles)
- Sells after two consecutive negative days (red candles)
- Simple, robust trend-following approach
- Works on daily timeframes with end-of-day execution
- Includes complete portfolio management and order execution

## Transaction Costs

The backtesting engine includes realistic Indian market costs:
- Brokerage: ₹20 per leg
- STT: 0.025% on sell leg (intraday)
- Transaction charges: 0.00345%
- SEBI charges: 0.0001%
- Stamp duty: 0.003% on buy leg
- GST: 18% on brokerage and transaction charges

## Data Format

### Daily Prices CSV
```
symbol,company,date,Open,High,Low,Close,Volume
```

### Intraday CSV
```
symbol,datetime,Open,High,Low,Close,Volume
```

## Risk Warning

This software is for educational and research purposes only. Trading stocks involves substantial risk of loss. Past performance does not guarantee future results. Always test strategies thoroughly in sandbox/paper trading before deploying with real capital.

## License

[Specify your license]

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

[Your contact information]
