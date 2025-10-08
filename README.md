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

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:

**Linux/Mac:**
```bash
cp .env.example .env
```

**Windows:**
```cmd
copy .env.example .env
```

Edit `.env` and add your Upstox access token from https://upstox.com/developer/

## Usage

All operations run through the single `agent.py` entry point.

### Quick Start

**1. First Time Setup - Run Backtest:**

```bash
python agent.py --strategy two_day_momentum --backtest
```

**Outputs:**
- [data/bse30_summary_two_day_momentum.csv](data/bse30_summary_two_day_momentum.csv) - Performance summary (used by daily trading)
- [data/backtest_report_two_day_momentum.md](data/backtest_report_two_day_momentum.md) - **Detailed markdown report** with Sharpe Ratio, Max Drawdown, top/bottom performers

Run weekly/monthly to update stock rankings.

**2. Daily Trading:**

```bash
# Dry run (no actual trades)
python agent.py --strategy two_day_momentum --dry-run

# Live trading
python agent.py --strategy two_day_momentum
```

**What it does:**
1. Downloads latest BSE-30 prices (differential update)
2. Loads pre-computed backtest summary for the strategy
3. Generates buy/sell trade calls for top performers
4. Executes trades via Upstox API

**Defaults:** ₹100k investment, top 5 stocks
**Customize:** `python agent.py --strategy two_day_momentum --investment 50000 --top-n 3`

### Schedule as Cron Job (Linux/Mac) or Task Scheduler (Windows)

**Daily Trading (after market close at 3:30 PM on weekdays):**

Linux/Mac (crontab):
```bash
30 15 * * 1-5 cd /path/to/stock-trading-agent && .venv/bin/python agent.py --strategy two_day_momentum
```

Windows (Task Scheduler):
- Run: `python agent.py --strategy two_day_momentum`
- Start in: `C:\path\to\stock-trading-agent`
- Trigger: Daily at 3:30 PM, weekdays only

**Weekly Backtest Update (optional, Sunday 10 AM):**

Linux/Mac:
```bash
0 10 * * 0 cd /path/to/stock-trading-agent && .venv/bin/python agent.py --strategy two_day_momentum --backtest
```

Windows:
- Run: `python agent.py --strategy two_day_momentum --backtest`
- Start in: `C:\path\to\stock-trading-agent`
- Trigger: Weekly on Sunday at 10:00 AM

### Command Line Options

```bash
python agent.py --help
```

**Required:**
- `--strategy STRATEGY` - Strategy name (must match file in strategies/ folder, e.g., `two_day_momentum`)

**Available flags:**
- `--backtest` - Run backtest mode (generates summary and report)
- `--dry-run` - Generate trade calls without executing (for testing)
- `--skip-download` - Skip price download, use existing data
- `--investment AMOUNT` - Total capital (default: ₹100,000)
- `--top-n N` - Number of top stocks to trade (default: 5)
- `--access-token TOKEN` - Upstox API token (or set UPSTOX_ACCESS_TOKEN env var)

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
