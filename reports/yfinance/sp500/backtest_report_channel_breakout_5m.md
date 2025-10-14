# Channel Breakout Strategy - Backtest Report

**Generated:** 2025-10-14 14:31:28
**Period:** 2025-08-15 to 2025-10-13
**Universe:** BSE-30 (30 stocks)

---

## Overall Performance

| Metric | Value |
|--------|-------|
| **Total Trades** | 9672 |
| **Long Trades** | 4826 (35.5% win rate) |
| **Short Trades** | 4846 (35.3% win rate) |
| **Net P&L** | ₹-222.66 |
| **Win Rate** | 35.44% |
| **Average Return** | 0.00% |
| **Compound Return** | -7.17% |
| **Sharpe Ratio** | -0.27 |
| **Maximum Drawdown** | -62.58% |

---

## Top Performers (by Net P&L)

| Rank | Symbol | Company | Trades | Net P&L | Avg Return | Win Rate |
|------|--------|---------|--------|---------|------------|----------|
| 1 | ORCL | Oracle Corporation | 192 | ₹100.05 | 0.22% | 34.90% |
| 2 | CAT | Caterpillar Inc. | 201 | ₹43.31 | 0.05% | 35.32% |
| 3 | NVDA | NVIDIA Corporation | 206 | ₹41.14 | 0.12% | 36.89% |
| 4 | DHR | Danaher Corporation | 200 | ₹24.70 | 0.07% | 34.50% |
| 5 | IBM | International Business Machines Corporation | 183 | ₹22.64 | 0.04% | 39.89% |
| 6 | PM | Philip Morris International Inc. | 195 | ₹19.07 | 0.06% | 34.87% |
| 7 | TMO | Thermo Fisher Scientific Inc. | 198 | ₹17.99 | 0.03% | 34.85% |
| 8 | ABBV | AbbVie Inc. | 195 | ₹17.93 | 0.04% | 39.49% |
| 9 | INTC | Intel Corporation | 186 | ₹12.11 | 0.22% | 37.10% |
| 10 | AAPL | Apple Inc. | 207 | ₹9.72 | 0.02% | 35.27% |

---

## Bottom Performers (by Net P&L)

| Rank | Symbol | Company | Trades | Net P&L | Avg Return | Win Rate |
|------|--------|---------|--------|---------|------------|----------|
| 1 | NFLX | Netflix Inc. | 200 | ₹-130.58 | -0.05% | 32.50% |
| 2 | AMD | Advanced Micro Devices Inc. | 199 | ₹-61.67 | -0.16% | 33.17% |
| 3 | JPM | JPMorgan Chase & Co. | 204 | ₹-57.03 | -0.09% | 29.41% |
| 4 | ADBE | Adobe Inc. | 201 | ₹-47.71 | -0.06% | 33.33% |
| 5 | TSLA | Tesla Inc. | 186 | ₹-39.64 | -0.04% | 31.18% |
| 6 | MSFT | Microsoft Corporation | 197 | ₹-31.81 | -0.03% | 37.06% |
| 7 | HON | Honeywell International Inc. | 216 | ₹-21.67 | -0.05% | 31.48% |
| 8 | MCD | McDonald's Corporation | 209 | ₹-21.10 | -0.03% | 37.80% |
| 9 | MA | Mastercard Incorporated | 200 | ₹-20.82 | -0.02% | 36.00% |
| 10 | JNJ | Johnson & Johnson | 205 | ₹-17.81 | -0.05% | 33.17% |

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

