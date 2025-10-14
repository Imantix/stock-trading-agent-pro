# Channel Breakout Strategy - Backtest Report

**Generated:** 2025-10-14 16:34:05
**Period:** 2025-07-31 to 2025-10-13
**Universe:** BSE-30 (30 stocks)

---

## Overall Performance

| Metric | Value |
|--------|-------|
| **Total Trades** | 136 |
| **Long Trades** | 68 (33.8% win rate) |
| **Short Trades** | 68 (26.5% win rate) |
| **Net P&L** | ₹-522.41 |
| **Win Rate** | 30.15% |
| **Average Return** | -1.05% |
| **Compound Return** | -80.26% |
| **Sharpe Ratio** | -3.14 |
| **Maximum Drawdown** | -79.09% |

---

## Top Performers (by Net P&L)

| Rank | Symbol | Company | Trades | Net P&L | Avg Return | Win Rate |
|------|--------|---------|--------|---------|------------|----------|
| 1 | TSLA | Tesla Inc. | 2 | ₹79.83 | 11.91% | 50.00% |
| 2 | GOOGL | Alphabet Inc. Class A | 1 | ₹39.81 | 19.81% | 100.00% |
| 3 | META | Meta Platforms Inc. | 4 | ₹31.46 | 1.04% | 75.00% |
| 4 | HD | Home Depot Inc. | 1 | ₹30.14 | 7.89% | 100.00% |
| 5 | IBM | International Business Machines Corporation | 4 | ₹29.04 | 2.94% | 50.00% |
| 6 | ORCL | Oracle Corporation | 4 | ₹22.37 | 2.54% | 25.00% |
| 7 | AAPL | Apple Inc. | 2 | ₹16.21 | 3.67% | 50.00% |
| 8 | ACN | Accenture plc | 3 | ₹16.19 | 2.13% | 66.67% |
| 9 | JPM | JPMorgan Chase & Co. | 2 | ₹7.78 | 1.31% | 50.00% |
| 10 | MCD | McDonald's Corporation | 1 | ₹5.21 | 1.73% | 100.00% |

---

## Bottom Performers (by Net P&L)

| Rank | Symbol | Company | Trades | Net P&L | Avg Return | Win Rate |
|------|--------|---------|--------|---------|------------|----------|
| 1 | NFLX | Netflix Inc. | 5 | ₹-226.93 | -3.65% | 0.00% |
| 2 | AVGO | Broadcom Inc. | 4 | ₹-94.97 | -7.01% | 0.00% |
| 3 | TMO | Thermo Fisher Scientific Inc. | 3 | ₹-58.27 | -3.81% | 33.33% |
| 4 | BA | Boeing Company | 6 | ₹-50.15 | -3.61% | 0.00% |
| 5 | MA | Mastercard Incorporated | 5 | ₹-46.94 | -1.58% | 20.00% |
| 6 | COST | Costco Wholesale Corporation | 5 | ₹-39.16 | -0.79% | 40.00% |
| 7 | TXN | Texas Instruments Incorporated | 4 | ₹-34.12 | -4.46% | 0.00% |
| 8 | NVDA | NVIDIA Corporation | 4 | ₹-28.63 | -3.96% | 0.00% |
| 9 | CAT | Caterpillar Inc. | 2 | ₹-28.09 | -3.23% | 0.00% |
| 10 | LLY | Eli Lilly and Company | 3 | ₹-22.28 | -0.60% | 66.67% |

---

## Strategy Description

**Channel Breakout Strategy (Long/Short with Reversals)**

**Long Entry Rules:**
- Price breaks above upper channel (10-bar highest high)
- Entry at next bar's Open price
- If currently SHORT, automatically reverse to LONG

**Short Entry Rules:**
- Price breaks below lower channel (10-bar lowest low)
- Entry at next bar's Open price
- If currently LONG, automatically reverse to SHORT

**Position Sizing:**
- 1 share per trade (for backtesting purposes)
- Fixed order size: 250 units (in live trading)
- Max pyramiding: 18 concurrent positions

