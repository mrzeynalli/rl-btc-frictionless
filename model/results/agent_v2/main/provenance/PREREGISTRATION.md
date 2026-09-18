# Agent v2 — decision rules fixed before measurement (2026-09-15)

Written before the proxy audit, the feature redundancy measurement and the γ/λ search were run.
Every rule below is a DESIGN CHOICE unless tagged otherwise; none was adjusted after seeing results.

## 1. Data: BitMEX XBTUSD as the pre-2020 proxy for Binance USD-M BTCUSDT

Splice at 2020-01-01 00:00 UTC (the first monthly Binance archive, four months after launch).
Before it: BitMEX prices rescaled to the Binance level at the last common minute, BitMEX volume
rescaled to Binance USDT notional by the ratio of the two venues' volume over the 28 days before the
splice, BitMEX funding. From it: Binance prices, USDT quote volume, funding.

The proxy is accepted if, over 2020-01-01 → 2026-01-01 (both venues, development data only):
- hourly log returns of the two venues correlate at ρ ≥ 0.99; and
- a market feature computed from each venue alone correlates at Spearman ρ ≥ 0.90.
A feature that fails the second test is removed from the state. If returns fail the first test, the
proxy is rejected and training starts at 2020-04-01 on Binance data alone.

## 2. Market-state candidates and the redundancy rule

Candidates, in precedence order:
1. volatility-normalised log returns r_h / (σ_d √(h/24)) over h = 1 h, 1 d, 1 w, 4 w, 13 w, 26 w, 3 h, 3 d, 8 w
   (1 h: the reviewer; 1 d and 1 w: Liu & Tsyvinski 2021; 4 w, 8 w, 13 w: Zhang, Zohren & Roberts 2020;
   26 w: Moskowitz, Ooi & Pedersen 2012; 3 h and 3 d: Deng et al. 2017);
2. the v1 regime features: volatility_level, volatility_regime, volume_regime, funding_rate,
   drawdown_4w, trend_efficiency_1w.

On the first fold's training span (2017-01-01 → 2024-01-01), sampled hourly, walk the list in order and
keep a candidate only if its Spearman |ρ| with every feature already kept is below 0.8 — the threshold
at which v1 excluded the four-week efficiency ratio (ARCHITECTURE v1 §4.3). No candidate is selected or
removed by its relation to future returns.

## 3. γ/λ search

- Inner folds only: validate 2022 (train 2017–2021) and 2023 (train 2017–2022); outer folds untouched.
- Grid per hour: γ ∈ {0.9583, 0.9792, 0.9900} (discount horizons 24 h, 48 h, 100 h; 0.99: Andrychowicz
  et al. 2021, Tummon et al. 2020) × λ ∈ {0.90, 0.95, 0.9792} (0.90: Andrychowicz et al. 2021; 0.95:
  their default and Schulman et al. 2016; 0.9792: v1, USER). The cost break-even guard is not applied.
- One seed (11), 30 passes, a checkpoint every 5 passes, argmax evaluation, every other setting v2.
- Cell score: IQM of quarterly log return over 6 checkpoints × 8 quarters (48 scores).
- Choice by main effects: the γ level with the highest mean cell score over the three λ levels, and
  likewise for λ. A level replaces v1's value (γ = λ = 0.9792 per hour) only if its mean score exceeds
  v1's by more than one standard error of the paired difference across the 8 quarters; otherwise v1's
  value stays.

---

## Addendum — 2026-09-15, after the search, before any new measurement

USER decisions after §1–§3 had run: (a) prices, volume and funding from **BitMEX XBTUSD alone**, still settled as a
USDT linear account — the proxy audit found the two venues' funding disagreed (daily ρ 0.69; 1.84 against 3.51 bp a day);
(b) **the whole of 2026 is the test window**, so the validation folds are 2024 and 2025 only.

Consequences, fixed before rerunning anything:
- §1 no longer applies: there is no splice. Its results stay in `audit/proxy_audit.json` as the record of why.
- §2 is rerun unchanged on the BitMEX-only cache, over **all 15 candidates** in the same precedence order: the reason two of
  them were removed (disagreement between venues) no longer exists.
- §3 ran on the merged data (`search_merged_data/`) and kept v1's γ = λ = 0.9792 per hour: no level beat v1 by one standard
  error. It is not rerun by default. Between the two datasets the inputs of the search years differ only after 2020, where
  hourly returns correlate at 0.9919, plus funding and, possibly, two state features; the outcome was the null default. A rerun
  on the final design is offered to the user, not assumed.
