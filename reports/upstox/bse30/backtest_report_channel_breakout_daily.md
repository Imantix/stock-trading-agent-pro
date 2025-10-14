# Channel Breakout Strategy - Backtest Report

**Generated:** 2025-10-14 13:47:59
**Period:** 2025-07-31 to 2025-10-13
**Universe:** BSE-30 (30 stocks)

---

## Overall Performance

| Metric | Value |
|--------|-------|
| **Total Trades** | 94 |
| **Long Trades** | 41 (26.8% win rate) |
| **Short Trades** | 53 (7.5% win rate) |
| **Net P&L** | ₹-304.10 |
| **Win Rate** | 15.96% |
| **Average Return** | -1.66% |
| **Compound Return** | -80.55% |
| **Sharpe Ratio** | -7.14 |
| **Maximum Drawdown** | -79.53% |

---

## Top Performers (by Net P&L)

| Rank | Symbol | Company | Trades | Net P&L | Avg Return | Win Rate |
|------|--------|---------|--------|---------|------------|----------|
| 1 | MARUTI.BO | Maruti Suzuki | 3 | ₹2,613.00 | 6.78% | 33.33% |
| 2 | ASIANPAINT.BO | Asian Paints | 1 | ₹93.60 | 3.90% | 100.00% |
| 3 | BAJFINANCE.BO | Bajaj Finance | 2 | ₹47.90 | 2.64% | 50.00% |
| 4 | ULTRACEMCO.BO | UltraTech Cement | 3 | ₹-5.00 | -0.00% | 66.67% |
| 5 | POWERGRID.BO | Power Grid | 2 | ₹-9.15 | -1.58% | 0.00% |
| 6 | BEL.BO | Bharat Electronics | 1 | ₹-10.60 | -2.72% | 0.00% |
| 7 | ICICIBANK.BO | ICICI Bank | 2 | ₹-15.30 | -0.54% | 50.00% |
| 8 | NTPC.BO | NTPC | 4 | ₹-19.75 | -1.45% | 25.00% |
| 9 | ETERNAL.BO | Eternal | 3 | ₹-25.55 | -2.54% | 33.33% |
| 10 | TATASTEEL.BO | Tata Steel | 5 | ₹-30.45 | -3.64% | 0.00% |

---

## Bottom Performers (by Net P&L)

| Rank | Symbol | Company | Trades | Net P&L | Avg Return | Win Rate |
|------|--------|---------|--------|---------|------------|----------|
| 1 | TRENT.BO | Trent | 3 | ₹-559.00 | -3.35% | 33.33% |
| 2 | M&M.BO | Mahindra & Mahindra | 4 | ₹-434.80 | -3.16% | 0.00% |
| 3 | TCS.BO | Tata Consultancy Services | 5 | ₹-318.30 | -2.00% | 20.00% |
| 4 | SUNPHARMA.BO | Sun Pharma | 6 | ₹-286.90 | -2.88% | 0.00% |
| 5 | INFY.BO | Infosys | 5 | ₹-263.70 | -3.48% | 0.00% |
| 6 | HCLTECH.BO | HCLTech | 5 | ₹-208.80 | -2.84% | 0.00% |
| 7 | RELIANCE.BO | Reliance Industries | 4 | ₹-114.30 | -2.03% | 0.00% |
| 8 | LT.BO | Larsen & Toubro | 2 | ₹-99.00 | -1.35% | 0.00% |
| 9 | KOTAKBANK.BO | Kotak Mahindra Bank | 3 | ₹-81.60 | -1.37% | 0.00% |
| 10 | TATAMOTORS.BO | Tata Motors | 3 | ₹-71.85 | -3.40% | 0.00% |

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

