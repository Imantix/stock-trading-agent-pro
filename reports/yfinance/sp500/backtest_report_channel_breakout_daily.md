# Channel Breakout Strategy - Backtest Report

**Generated:** 2025-10-15 07:42:53
**Period:** 2024-10-30 to 2025-10-14
**Universe:** BSE-30 (30 stocks)

---

## Overall Performance

| Metric | Value |
|--------|-------|
| **Total Trades** | 714 |
| **Long Trades** | 354 (37.0% win rate) |
| **Short Trades** | 360 (35.6% win rate) |
| **Net P&L** | ₹202.32 |
| **Win Rate** | 36.27% |
| **Average Return** | 0.18% |
| **Compound Return** | -70.99% |
| **Sharpe Ratio** | 0.28 |
| **Maximum Drawdown** | -98.27% |

---

## Top Performers (by Net P&L)

| Rank | Symbol | Company | Trades | Net P&L | Avg Return | Win Rate |
|------|--------|---------|--------|---------|------------|----------|
| 1 | COST | Costco Wholesale Corporation | 15 | ₹243.98 | 1.75% | 53.33% |
| 2 | IBM | International Business Machines Corporation | 12 | ₹135.95 | 4.86% | 58.33% |
| 3 | ORCL | Oracle Corporation | 13 | ₹118.65 | 6.07% | 53.85% |
| 4 | META | Meta Platforms Inc. | 15 | ₹111.58 | 1.49% | 53.33% |
| 5 | TSLA | Tesla Inc. | 13 | ₹111.24 | 3.56% | 30.77% |
| 6 | CAT | Caterpillar Inc. | 11 | ₹96.49 | 3.51% | 36.36% |
| 7 | NFLX | Netflix Inc. | 12 | ₹93.45 | 1.34% | 33.33% |
| 8 | HD | Home Depot Inc. | 14 | ₹77.96 | 1.46% | 50.00% |
| 9 | UNH | UnitedHealth Group Incorporated | 15 | ₹66.52 | 2.21% | 46.67% |
| 10 | MSFT | Microsoft Corporation | 12 | ₹51.57 | 1.16% | 50.00% |

---

## Bottom Performers (by Net P&L)

| Rank | Symbol | Company | Trades | Net P&L | Avg Return | Win Rate |
|------|--------|---------|--------|---------|------------|----------|
| 1 | LLY | Eli Lilly and Company | 17 | ₹-316.85 | -2.17% | 29.41% |
| 2 | MA | Mastercard Incorporated | 23 | ₹-166.44 | -1.27% | 21.74% |
| 3 | MCD | McDonald's Corporation | 16 | ₹-104.46 | -2.07% | 18.75% |
| 4 | PM | Philip Morris International Inc. | 17 | ₹-60.88 | -1.85% | 23.53% |
| 5 | QCOM | QUALCOMM Incorporated | 14 | ₹-51.09 | -2.08% | 50.00% |
| 6 | TMO | Thermo Fisher Scientific Inc. | 17 | ₹-50.05 | -0.62% | 23.53% |
| 7 | DHR | Danaher Corporation | 15 | ₹-42.17 | -1.28% | 33.33% |
| 8 | XOM | Exxon Mobil Corporation | 20 | ₹-40.16 | -1.75% | 25.00% |
| 9 | UPS | United Parcel Service Inc. | 18 | ₹-33.89 | -1.51% | 38.89% |
| 10 | BA | Boeing Company | 16 | ₹-33.52 | -0.58% | 37.50% |

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

