# Agent architecture — version 2

**Status:** designed and built 2026-09-15. Package `src/agent/`. **Main training not launched.** The γ/λ search
has run and kept v1's values (§5). Two USER decisions after it: one venue (BitMEX) rather than BitMEX spliced to
Binance, and all of 2026 as the test window (§0.2). Version 1 (BitMEX inverse
contract, BTC collateral, six regime features) is archived with its full record in
`backup/agent_inverse_regime_20260915/` — its real-data result, both looks at the 2026 test window, and its
post-mortem are summarised in §1.

**What version 2 is.** The same recurrent PPO agent, re-posed as the account the user trades: a **USDT-margined
linear perpetual account** (Binance USD-M costs and margin) on **BitMEX XBTUSD prices, volume and funding**, one
consistent venue from 2016; a market state of **volatility-normalised returns over nine horizons** plus volatility,
volume and funding features; training at **twice the real cost**; **argmax evaluation**; validation on 2024 and
2025; **the whole of 2026 locked for testing**; and γ and λ checked by a search on folds that precede every
validation quarter. Every data and state decision was fixed in
advance (`results/agent_v2/PREREGISTRATION.md`) and measured on development data only
(`results/agent_v2/audit/`).

**Update this file whenever the agent changes.** A test fails if a constant in `Config` is missing from the
provenance table in §12.

Every constant carries exactly one provenance class:

| class | meaning |
|---|---|
| **CITED** | taken from a named study, exchange document or reference implementation |
| **DERIVED** | follows from other constants or from the problem's arithmetic |
| **MEASURED** | measured on this data, this host, or this project's own training logs, with the number |
| **USER** | a decision the user made explicitly |
| **DESIGN CHOICE** | none of the above — stated openly as unverified |

---

## 0. The agent in one page

### 0.1 Summary

| component | what it is | section |
|---|---|---|
| instrument | a USDT-margined linear perpetual account, 1× entry notional, on BitMEX XBTUSD prices and funding | §3, §8 |
| time | one-minute accounting and margin checks; five-minute bars for features and reward statistics; **one decision per hour** | §3, §4.1 |
| action | target position `SHORT`, `FLAT` or `LONG`, one softmax | §4.1 |
| market state | **13 features**: volatility-normalised log returns over 1 h, 3 h, 1 d, 3 d, 1 w, 4 w, 8 w, 13 w and 26 w; the four-week volatility level and the last day against it; the last week's volume against the four before; the last funding rate | §4.2 |
| account state | 8 inputs: side one-hot, open trade's return and age, exposure, the reward's two statistics | §4.4 |
| reward | exact per-bar increment of an exponentially weighted Sharpe ratio of **USDT** equity, summed over the decision's 12 bars and scaled | §4.5 |
| ledger | linear contract: equity = wallet + BTC position × (price − entry); a 1× long cannot be liquidated, a 1× short is liquidated near a doubling | §4.6 |
| horizon | γ = λ = 0.9792 per hour (48 h discount horizon, 24.3 h credit window), kept by the pre-registered search on inner folds 2022–2023 | §5 |
| network | separate actor and critic: Linear–LayerNorm–tanh → LSTM → LayerNorm, account concatenated → two LayerNorm–tanh layers; 835,204 parameters | §6 |
| learner | recurrent PPO with the discrete-action settings of PPO's Atari experiments (CleanRL `ppo_atari_lstm`); **training cost 2× the real one** | §7 |
| budget | **60 passes** over each fold's training span, a validated checkpoint every 5 | §7.3 |
| data | BitMEX XBTUSD from 2016; expanding walk-forward from 2017: validate 2024 and 2025; episodes of 90.7 days at random starts, exploring starts | §8 |
| evaluation | **argmax** actions, each quarter from flat, real costs; IQM selection, stratified bootstrap, PBO; constant long/short/flat and two momentum rules as references; **all of 2026 locked for testing** | §9 |
| formulation | the equations and pseudo-code of everything above | §14 |

### 0.2 What changed from version 1, and why

| area | v1 | v2 | evidence |
|---|---|---|---|
| contract and account | BitMEX XBTUSD inverse; BTC wallet; reward in BTC | linear contract; USDT wallet; reward in USDT | USER (2026-09-15): the account the user trades. External review: on the inverse contract flat is long BTC in dollars and a 1× short is an exact dollar hedge — verified on v1's ledger (+10% BTC: long +20%, short 0.00% in USD) |
| data | BitMEX XBTUSD 2016– | unchanged: BitMEX XBTUSD prices, volume and funding | USER (2026-09-15), after a pre-registered audit of splicing BitMEX to Binance found the two venues' funding disagreed (daily ρ 0.69; 1.84 against 3.51 bp a day) — one venue keeps prices, volume and funding consistent (§8) |
| market state | 6 regime features, no returns | 9 volatility-normalised returns + volatility level and regime, volume regime, funding rate | USER (2026-09-15): returns and momentum; horizons from Moskowitz, Ooi & Pedersen 2012, Zhang, Zohren & Roberts 2020, Liu & Tsyvinski 2021, Deng et al. 2017; survivors of a pre-registered audit and redundancy rule (§4.2) |
| costs in training | real cost | 2× real cost; evaluation at the real cost | Deng et al. 2017 (5×); Zhang, Zohren & Roberts 2020 (cost *"as a regularising term"*); MEASURED in this project's walk-forward study: 2× ranked 1st of 16 cells and all four 1× cells ranked 8th or lower |
| evaluation actions | sampled over 4 seeds | argmax, one path | USER (2026-09-15) |
| γ, λ | 0.9792 per hour each (USER) | chosen by search on inner folds | USER request; external review: *"There is no theoretical requirement that γ and λ be equal"*; Andrychowicz et al. 2021 (γ 0.99, λ 0.9) |
| budget | 100 passes | 60 passes | MEASURED on v1: every checkpoint after pass 65 scored negative on validation, and pass-100 policies lost more on the spent window than pass-15 ones |
| folds | validate 2024, 2025 | unchanged | — |
| test | 2026-01-01 → (spent by v1) | **2026-01-01 → locked for v2** | USER (2026-09-15): the whole of 2026. January–July 2026 was v1's test window, and v1's loss there informed v2's design (§15) |
| fee and maintenance margin | BitMEX inverse: 5 bp, 0.50% | the user's USD-M account: 5 bp taker (Binance VIP 0), 0.40% (Binance BTCUSDT tier 1) | CITED |
| references | constant long, short, flat | + 1-week and 26-week momentum rules | Liu & Tsyvinski 2021; Moskowitz, Ooi & Pedersen 2012; the external review's recommendation to compare the agent with a measurable version of the trading logic and with conventional strategies |

**Unchanged:** hourly decisions on five-minute bars, the three target positions, the Sharpe reward and its
exact increments, **all eight account inputs** (side, open trade's return and age, exposure, the reward's two
statistics), the network, every PPO setting, γ and λ, exploring starts, 90.7-day episodes, five seeds, the
validation folds, IQM selection with PBO.

---

## 1. History — what version 1 established

Version 1's record is in `backup/agent_inverse_regime_20260915/ARCHITECTURE.md` (§11.5–§11.7, §18). In short:

- **Validation (2024–2025):** no edge distinguishable from zero — selected checkpoint IQM +0.019 log per
  quarter, 95% interval [−0.019, +0.057], probability of backtest overfitting 0.80.
- **v1's test window, 2026-01-01 → 2026-08-01, BTC −28.4%:** all ten runs lost in BTC terms at the selected
  checkpoint (pass 15; mean −20.2%) and at the final one (pass 100; mean −31.5%); none beat a flat account;
  constant short made +38.2%.
- **Post-mortem:** the episode sampler was not bull-biased — 42% of drawable training quarters fell — but the
  training data rewarded holding long almost everywhere: next-week return was positive in every decile of
  the weekly trend, from +63 bp in the strongest downtrend to +360 bp in the strongest uptrend, and a short in
  a downtrend did not cover the 12 bp round trip at 8 hours. Fully trained policies were trend-aligned (10 of
  10) yet still lost; selection among 20 checkpoints could not tell good from bad.
- **The same pattern appeared in an earlier study** in this project under a different learner and clock
  (walk-forward, 2026-08-18): trend learned, long lean, sealed loss; its supervised probe measured an
  information coefficient of +0.024 at 1 h against a 0.202 break-even hurdle.

**What that implies for v2.** Changing the account, the state and the horizon removes mismatches the review
identified; it does not create an edge. Whether an hourly BTC edge survives costs remains the open question
(§15), and the momentum references in §9 measure it directly.

---
## 2. The one measurement, and the one assumption

**Measured per fold, on its training span only:** the standard deviation of five-minute log returns,
σ_bar. It sets the reward's prior risk (§4.5) and, with the cost, a planning break-even.

**Assumed:** a signal worth trading has an annualised Sharpe ratio of 1 (`SIGNAL_SHARPE_ASSUMPTION`). The
cost break-even holding period is then round-trip cost / drift per bar:

| fold | training span | σ_bar | break-even (Sharpe-1 signal, 12 bp round trip) |
|---|---|---:|---:|
| search 2022 | 2017–2021 | 29.22 bp | 11.1 h |
| search 2023 | 2017–2022 | 27.95 bp | 11.6 h |
| 2024 | 2017–2023 | 26.55 bp | 12.2 h |
| 2025 | 2017–2024 | 25.54 bp | 12.7 h |

(Search folds measured on the BitMEX-to-Binance series the search ran on; 2024 and 2025 on BitMEX alone, identical
to v1's measurement.)

As the review notes, only the volatility is measured; the drift is assumed, so this is a planning
heuristic, not evidence that a trade pays after 12 hours. v1 refused any credit window below it
(`check_horizon`). v2 keeps the computation but lets the γ/λ search test windows below it
(`enforce_horizon` = False in the search, §5). The chosen 24.3 h credit window clears it in both folds.

---

## 3. Problem setup

| | |
|---|---|
| Account | USDT-margined linear perpetual, USDT collateral, the user's USD-M futures account |
| Market data | BitMEX XBTUSD one-minute bars and funding, 2016 onward, one venue (§8) |
| Accounting and margin checks | every minute |
| Bars (features, reward statistics) | five minutes |
| Decisions | **every hour** — `decision_bars` = 12 |
| Actions | target position `SHORT`, `FLAT`, `LONG` |
| Position size | a BTC quantity worth 1× the USDT wallet at entry, never rebalanced |
| Costs | 5 bp taker commission (Binance USD-M VIP 0) + 1 bp adverse slippage per leg; **training charges 2×** |
| Reward | exact differential Sharpe increment of USDT equity (USER) |
| Folds | train 2017 → Y, validate the four quarters of Y: **2024**, **2025** |
| γ/λ search folds | 2022 (4 quarters), 2023 (4 quarters), before every validation quarter |
| Test | **2026-01-01 onward, locked** (USER) |

---

## 4. The MDP

### 4.1 Decisions and actions

A categorical choice of target position (Moody & Saffell 2001; Huang 2018; Zhang, Zohren & Roberts 2020):
keeping the position costs nothing, moving from long to short pays two legs. Entering, holding, exiting and
reversing are this one choice made every hour; a holding period emerges from repeated choices to keep.

**Timing.** At decision boundary *t* the agent observes the features of the last completed five-minute bar
and its account marked at the previous minute's close. The order is live for the next 60 minutes and fills at
the first minute with a printed trade, at that minute's open plus adverse slippage. Inside the hour the ledger
runs minute by minute: funding at settlement minutes, the margin check at each minute's adverse extreme, and
a liquidation ends the episode at that minute. The reward statistics advance every five-minute bar.

**Why hourly** — kept from v1, where it was MEASURED: at five-minute decisions all three latent smoke runs
lost their entries (entropy 0.05–0.10, entries per rollout from 800–1,000 down to 5–64, one run permanently
flat); hourly runs held entropy at 0.18–0.34. Holding an action for several steps is frame skip (Mnih et al.
2015; Andrychowicz et al. 2021 *"Try frame skip if possible"*). The price is that the agent cannot react inside
an hour; the review's alternative — richer intra-hour observations at the same decision clock — is served in
part by the 1-hour and 3-hour returns (§4.2).

**Evaluation takes the most probable action** (USER, 2026-09-15). Training samples, as on-policy PPO requires,
so the policy evaluated is the argmax of a policy trained under sampling (§15).

### 4.2 State — the exogenous market

**How the state was chosen.** The user asked for returns and momentum. Candidates were taken from the
literature before any measurement, and a pre-registered rule that never looks at future returns removed the
redundant ones (`results/agent_v2/PREREGISTRATION.md` §2 and addendum; `results/agent_v2/audit/feature_redundancy.json`):

**Redundancy.** On 2017–2023, hourly, in precedence order, a candidate is kept only if its Spearman |ρ| with every
kept feature is below 0.8 (v1's threshold, at which it excluded the four-week efficiency ratio).

| candidate | source | largest \|ρ\| with a kept feature | kept |
|---|---|---|:---:|
| `return_1h` | the external review (recent hours); Deng et al. 2017 (last 45 minutes) | 0.47 (3 h) | ✓ |
| `return_1d` | Liu & Tsyvinski 2021 (Bitcoin's daily return significantly predicts 1-, 3-, 5- and 6-day-ahead returns); Deng et al. 2017 | 0.53 (3 d) | ✓ |
| `return_1w` | Liu & Tsyvinski 2021 (weekly returns: *"the top quintiles outperform the bottom quintiles at 1-4 week ahead horizons"*) | 0.64 (3 d) | ✓ |
| `return_4w` | Zhang, Zohren & Roberts 2020 (*"Returns over the past month, 2-month, 3-month and 1-year periods"*) | 0.69 (8 w) | ✓ |
| `return_13w` | Zhang, Zohren & Roberts 2020 (3 months) | 0.77 (8 w) | ✓ |
| `return_26w` | Moskowitz, Ooi & Pedersen 2012 (look-backs of 1–12 months) | 0.68 (13 w) | ✓ |
| `return_3h` | Deng et al. 2017 (momentum over *"3 h, 5 h, 1 day, 3 days, and 10 days"*) | 0.47 (1 h) | ✓ |
| `return_3d` | Deng et al. 2017 | 0.64 (1 w) | ✓ |
| `return_8w` | Zhang, Zohren & Roberts 2020 (2 months) | 0.77 (13 w) | ✓ |
| `volatility_level` | Corsi 2009; v1 | 0.26 (26 w) | ✓ |
| `volatility_regime` | Corsi 2009; v1 | 0.54 (volume regime) | ✓ |
| `volume_regime` | Barber & Odean 2008; Liu & Tsyvinski 2021; v1 | 0.54 (volatility regime) | ✓ |
| `funding_rate` | He et al. 2024; Koijen et al. 2018; v1 | 0.38 (4 w) | ✓ |
| `drawdown_4w` | Pagan & Sossounov 2003; v1 | **0.82 with `return_4w`** | ✗ |
| `trend_efficiency_1w` | Kaufman 1995; v1 | **0.985 with `return_1w`** | ✗ |

**A rule that no longer applies.** While the design spliced BitMEX (to 2019) to Binance (from 2020), a
pre-registered audit required each feature computed on either venue alone to agree at Spearman ρ ≥ 0.90; volume
(0.84) and funding (0.51) failed and left the state (`audit/proxy_audit.json`). The user then chose one venue; the
reason for removing them disappeared and they returned as candidates, and passed the redundancy rule.

Twelve-month returns — the canonical time-series momentum horizon — are not a candidate: they need a year of
history before training starts, and BitMEX data begin 2016-05-13 (DERIVED). MACD and RSI indicators (Zhang,
Zohren & Roberts 2020) normalise over a year for the same reason and were not candidates (DESIGN CHOICE).

**Definitions.** From five-minute log closes with bar return r_b, the four-week realised daily volatility
σ_d = sqrt(288 × mean of r_b² over four weeks) (Liu, Patton & Sheppard 2015; Corsi 2009), and

```
return_h          = (ln C_t − ln C_{t−h}) / (σ_d · sqrt(h / 24 h))     for h in hours
volatility_level  = ln σ_d
volatility_regime = ½ ln(one-day realised variance / four-week realised variance)
volume_regime     = ln(1 + last week's mean volume per day) − ln(1 + the previous four weeks' mean volume per day)
funding_rate      = the last funding rate settled before t
```

The normalisation by volatility scaled to the horizon is Zhang, Zohren & Roberts' (*"we normalise annual
returns as r_{t−252,t}/(σ_t √252)"*), with the five-minute realised daily volatility in place of their 60-day
exponentially weighted standard deviation (DERIVED substitution). Every feature at boundary *t* uses minutes
before *t* only (tested), is standardised with the training span's mean and standard deviation and clipped at
±10 (Andrychowicz et al.; Engstrom et al.). The cache holds all 15 candidates; `FEATURE_NAMES` selects the 13.
All six v1 features are reproduced exactly by the v2 pipeline on the same data (largest difference 6e-8, float
rounding; MEASURED 2026-09-15).

**Relation to v1's "regime, not prediction" requirement.** v1 excluded return features on the user's
instruction of 2026-09-11. The user reversed that on 2026-09-15. Four of v1's six features remain; the two
trend features v1 relied on are carried almost entirely by the 1-week and 4-week returns (ρ = 0.985 and 0.82).

**No feature was selected for predicting returns.** The review's warning applies: *"a model can identify a
trend but enter late, exit poorly, or trade too frequently"*, and v1 found that the training years' next-week
return was positive in every trend decile. Whether these inputs carry an edge after costs is measured by the
agent's validation and by the momentum references (§9), not assumed.

### 4.3 What was not added, from the review

| suggestion | decision | reason |
|---|---|---|
| explicit trade deterioration (profit given back from the trade's peak) | not added | MEASURED in this project (2026-09-04, BTCUSDT 2017–2026): next-24-hour return conditional on drawdown from a rolling 30-day peak is flat across buckets (\|t\| ≤ 0.5), and stops or trailing stops never beat the same entry rule without them. `trade_return` and `trade_age` remain |
| a sequence of intra-hour observations | not added | the 1-hour and 3-hour returns summarise the recent path; a sequence input would multiply the recurrent steps by 12 |
| an explicit abstention rule on expected net reward (Borrageiro et al. 2022) | not added | the critic estimates value but does not gate the policy; that agent learned online as a maker and took 71% of its return from funding — a different source of return |
| deterministic entry/hold hysteresis | superseded | argmax evaluation (USER) |

### 4.4 State — the endogenous account

| feature | why it is there |
|---|---|
| `short`, `flat`, `long` one-hot | the cost of each action depends on the side held |
| `trade_return` — open trade's log return / σ_d | how far the trade has run |
| `trade_age` — log(1 + bars held / credit window in bars) | how long it has run |
| `exposure` — log(1 + \|BTC position\| × price / equity) | the position's notional over equity: 1× at entry, rising as a short loses |
| `running_ratio`, `relative_risk` | the reward is a function of its two statistics (§4.5), so they make it Markov |

All clipped at ±10. Side, exposure and the reward statistics make reward and termination Markov given the
market (Subramanian, Sinha, Seraj & Mahajan, JMLR 2022). Unchanged from v1 except that exposure is notional
over equity on a linear contract.

### 4.5 Reward — exact increments of the Sharpe ratio

Unchanged from v1 except that R_b is the net log change of **USDT** equity over the bar, including
commission, slippage and funding. Moody & Saffell (IEEE TNN 2001) track A (mean) and B (second moment) with
rate η; the reward is the exact change of S = A / sqrt(B − A²) per bar divided by η, summed over the 12 bars
and scaled by sqrt((1 − γ_d²)/12). A fresh account starts at A = 0, B = σ_bar² of the fold. The downside
(Sortino) variant remains implemented. η = 1/1440 per bar, a five-day window (DESIGN CHOICE inside the
derived bounds). Formulas in §14.4.

The scale depends on γ, so it moves with the searched value: 0.0825 at γ = 0.9583 per hour, 0.0586 at 0.9792
and 0.0407 at 0.9900.

### 4.6 The ledger and its transitions — a linear USDT-margined contract

Wallet W in USDT (1 at the start), signed BTC quantity Q, entry price p̄:

```
equity(P)      = W + Q (P − p̄)
open at O:       fill p = O (1 + s ς);  Q = s · W · leverage / O;  W −= φ |Q| p;  p̄ = p
close at O:      fill p = O (1 − sign(Q) ς);  W += Q (p − p̄) − φ |Q| p
funding f:       W −= Q · C_{m−1} · f            (a long pays a positive rate)
liquidation:     equity(P_adv) ≤ |Q| P_adv (maintenance + φ)
```

with φ = 5 bp (×2 in training), ς = 1 bp (×2 in training), maintenance 0.40%, P_adv the minute's low for a long
and high for a short, and prices and funding rates from BitMEX XBTUSD. A funding rate is a fraction of position
notional on either contract type, so BitMEX's rate is charged on the USDT notional. On this contract, at 1×:

| position | BTC +10% | BTC −10% | liquidated at |
|---|---:|---:|---|
| flat | 0% | 0% | — |
| long | +10% | −10% | never (equity stays above the requirement down to a zero price; tested to 0.5% of entry) |
| short | −10% | +10% | a rise to ≈ 1.99× entry (tested: 199.04 from 100) |

(tested without costs to 1e-12). Liquidation terminates the episode with no bootstrap; episode ends are time
limits, bootstrapped with the critic (Pardo et al., ICML 2018). Adverse one-minute extremes trigger the
margin check — a conservative approximation, not an exchange's engine; funding is booked and checked before an
order fills.

---
## 5. Horizon — γ and λ from a pre-registered search

v1 fixed γ = λ = 0.99825 per bar (0.9792 per hour) by USER decision. The review pointed out that γ and λ
do different jobs — γ discounts the objective, λ trades bias for variance in the advantage estimate (Schulman
et al., GAE, ICLR 2016) — and that nothing requires them to be equal. Théate & Ernst (2021) found for trading
that *"there is an optimal value for the discount factor, which is neither too small nor too large"*, linked
to trading frequency through costs. Andrychowicz et al. (ICLR 2021) report *"the performance depends heavily
on the discount factor γ … with γ = 0.99 performing reasonably well in all environments"* and *"λ = 0.9 …
performed well"*. None of these gives a value for hourly BTC, so they bound a search.

**The search** (`scripts/run_search.sh`, `scripts/search_summary.py`; rules fixed in
`results/agent_v2/PREREGISTRATION.md` §3 before it ran):

| | |
|---|---|
| folds | **inner only**: validate 2022 (train 2017–2021) and 2023 (train 2017–2022) — before every validation quarter of §9, the nested selection Cawley & Talbot (JMLR 2010) prescribe |
| γ per hour | 0.9583, 0.9792, 0.9900 — discount horizons 1/(1 − γ) of 24 h, 48 h, 100 h; 0.99 from Andrychowicz et al. and Tummon et al. (2020) |
| λ per hour | 0.90 (Andrychowicz et al.), 0.95 (their default; common in PPO), 0.9792 (v1, USER) |
| per cell | seed 11, 30 passes, a checkpoint every 5, argmax evaluation, every other setting of v2 at the time; the break-even guard off |
| cell score | IQM of quarterly log return over 6 checkpoints × 8 quarters |
| decision | by main effects; a level replaces v1's 0.9792 only if its mean score beats v1's by more than one standard error of the paired per-quarter difference |

GAE credit windows 1/(1 − γλ) across the grid, against the break-even of 11.1–13.3 h (§2):

| γ \ λ | 0.90 | 0.95 | 0.9792 |
|---|---:|---:|---:|
| 0.9583 | 7.3 h | 11.2 h | 16.2 h |
| 0.9792 | 8.4 h | 14.3 h | 24.3 h |
| 0.9900 | 9.2 h | 16.8 h | 32.7 h |

**Result — completed 2026-09-15** (`results/agent_v2/search_merged_data/decision.json`). IQM and mean of quarterly log
return over 6 checkpoints × 8 quarters; behaviour over the same 48 evaluations:

| γ | λ | credit window | IQM | mean | positive | orders / quarter | median hold | long share | short share |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9583 | 0.90 | 7.3 h | −0.038 | −0.080 | 40% | 29 | 27 h | 0.51 | 0.12 |
| 0.9583 | 0.95 | 11.2 h | −0.111 | −0.158 | 31% | 32 | 44 h | 0.70 | 0.15 |
| 0.9583 | 0.9792 | 16.2 h | −0.047 | −0.089 | 40% | 36 | 25 h | 0.46 | 0.08 |
| 0.9792 | 0.90 | 8.4 h | −0.115 | −0.137 | 29% | 46 | 36 h | 0.56 | 0.25 |
| 0.9792 | 0.95 | 14.3 h | −0.099 | −0.123 | 23% | 35 | 46 h | 0.62 | 0.18 |
| **0.9792** | **0.9792** | **24.3 h** | **+0.0005** | **+0.012** | **52%** | 38 | 19 h | 0.40 | 0.23 |
| 0.9900 | 0.90 | 9.2 h | −0.056 | −0.118 | 38% | 19 | 55 h | 0.83 | 0.10 |
| 0.9900 | 0.95 | 16.8 h | −0.146 | −0.256 | 19% | 17 | 57 h | 0.62 | 0.19 |
| 0.9900 | 0.9792 | 32.7 h | −0.106 | −0.171 | 29% | 48 | 33 h | 0.70 | 0.26 |

| main effect | level means (mean cell IQM) | best | best − v1 | one standard error | decision |
|---|---|---|---:|---:|---|
| γ | 0.9583: −0.066; 0.9792: −0.071; 0.9900: −0.103 | 0.9583 | +0.006 | 0.058 | **keep 0.9792** |
| λ | 0.90: −0.070; 0.95: −0.119; 0.9792: −0.051 | 0.9792 | 0 | — | **keep 0.9792** |

**γ = λ = 0.9792 per hour** (0.99825 per bar) — v1's USER values, now also the only cell of nine that did not lose
on the inner folds, though its IQM of +0.0005 is indistinguishable from zero.

**Two cautions.** (1) The search ran on the BitMEX-to-Binance series, before the user chose BitMEX alone; the search
years' prices differ only after 2020 (hourly returns ρ = 0.9919), and funding and two state features differ. It was
not rerun (PREREGISTRATION.md, addendum). (2) Eight of nine cells lost on 2022–2023 after 30 passes, and long shares
of 0.40–0.83 persisted into the 2022 bear market: early evidence, from one seed, that the v1 pattern survives v2's
changes.

---

## 6. Network

```
market features (13) ─► Linear ─► LayerNorm ─► tanh ─► LSTM ─► LayerNorm ─┐
                                                                            ├─► Linear ─► LayerNorm ─► tanh ─► Linear ─► LayerNorm ─► tanh ─► Linear
account (8) ────────────────────────────────────────────────────────────────┘                                                              (3 logits | 1 value)
```

Unchanged from v1 except the input width: separate actor and critic of this shape, one recurrent step per
hourly decision, widths 128 (actor) and 256 (critic), **835,204 parameters**. LayerNorm after every hidden
layer (Lyle et al., ICML 2023; Ba, Kiros & Hinton 2016); 128 is CleanRL `ppo_atari_lstm`'s LSTM; two hidden
layers (*"Two hidden layers appear to work well for policy … and value networks … in all tested
environments"*, Andrychowicz et al.); a critic twice as wide (*"for the value function there seems to be no
downside in using wider networks"*, ibid.); orthogonal initialisation and a final policy layer 100× smaller
(ibid.; CleanRL); forget-gate bias 1 (Jozefowicz, Zaremba & Sutskever 2015); recurrence rather than stacked
observations (Hausknecht & Stone 2015). The account never enters the recurrent memory.

**Why the architecture was not changed.** Nothing measured points at capacity: v1's fully trained policies were
trend-aligned and still lost, and the review concluded that *"the literature does not establish that changing
the neural-network architecture alone will resolve these issues"*. Engstrom et al. (ICLR 2020) and
Andrychowicz et al. (ICLR 2021) find implementation and training choices dominate architecture in on-policy
RL. Holding the network fixed also keeps v1 and v2 comparable.

---

## 7. Optimisation — recurrent PPO

### 7.1 The update

Unchanged from v1 (full algorithm in §14.8). Each update collects 128 hourly decisions in each of 8
environments (1,024 samples), replays the rollout under the current parameters and stops on any mismatch above
5e-5, then for 4 epochs recomputes GAE with the current critic and takes 4 minibatch steps of 2 environments
each, rebuilding the recurrent state over a 64-decision burn-in: actor loss −min(ρÂ, clip(ρ, 0.9, 1.1)Â) −
0.01 H(π), critic loss ½(V − Ĝ)², separate Adam optimisers at 2.5e-4 decaying linearly to zero, gradient norm
clipped at 0.5.

Known PPO failure modes on this problem (v1 §7.1) — a clip that settles on a suboptimal discrete action (Hsu,
Mendler-Dünner & Hardt 2020), near-deterministic softmax policies that stop sampling (Mei et al. 2020, 2021),
overfitting a repeated history (Sato 2019; Zhang, Vinyals, Munos & Bengio 2018) — are monitored by the
diagnostics of §10.

### 7.2 Training at an inflated cost

Training environments charge `training_cost_multiplier` × (fee + slippage) = 12 bp per leg; validation, selection
and test charge the real 6 bp.

- Deng et al. (IEEE TNNLS 2017): *"a much higher c is set in (1). It is five times higher than the real TCs"*.
- Zhang, Zohren & Roberts (2020) train at 20 bp and write: *"We can treat bp as a regularising term as a large
  bp penalises turnovers and lets agents maintain current trade positions."*
- MEASURED in this project's walk-forward study (2026-08-18, hourly clock, `backup/artifacts/btc_wf_1h/
  FINAL_REPORT.md`): multipliers 1, 2, 4, 8 were searched on development data; **2× ranked first of 16 cells**
  (median development Sharpe +0.603), and the four 1× cells ranked 8th, 14th, 15th and 16th. Its real cost was
  4 bp per leg, so its best cells charged 8–16 bp; 2× here is 12 bp.
- MEASURED on v1: the pass-100 policies placed a median of about 305 orders over seven months against about 80 at
  pass 15, and lost more.

The multiplier is not searched: it is fixed by the evidence above, and the search budget went to γ and λ.

### 7.3 The training budget — 60 passes

As in v1, the budget counts passes over the fold's finite history with a validated checkpoint every 5 passes
(Moody & Saffell 2001; Deng et al. 2017: 100 epochs with validation-based stopping; Prechelt 1998: strips of
5). v1's own logs cut it to 60:

- the IQM of v1's 20 checkpoints was negative at every checkpoint after pass 65 (passes 70–100: −0.030, −0.027,
  −0.026, −0.053, −0.013, −0.007, −0.022);
- the pass-100 policies lost 31.5% on the spent window against 20.2% at pass 15.

| fold | training decisions | updates = ⌈60 × decisions / 1,024⌉ | checkpoints |
|---|---:|---:|---:|
| 2024 (2017–2023) | 61,344 | 3,595 | 12 |
| 2025 (2017–2024) | 70,128 | 4,110 | 12 |

The γ/λ search uses 30 passes (1,284 and 1,541 updates) to fit its compute; the horizon it chooses is carried to
the 60-pass run (§15).

### 7.4 Parameters

| parameter | value | provenance |
|---|---|---|
| `environments` | 8 | CITED: PPO Table 5 *"Number of actors 8"*; CleanRL `num_envs = 8` |
| `rollout` | 128 | CITED: PPO Table 5 *"Horizon (T) 128"*; CleanRL `num_steps = 128` |
| `sequence` | 128 | CITED: CleanRL trains each environment's whole rollout |
| `burn_in` | 64 | CITED: R2D2's 40-step burn-in before 80-step sequences (ratio 0.5) |
| `minibatch_sequences` | 2 | DERIVED: CleanRL `num_minibatches = 4`, 256 samples = PPO Table 5 minibatch 32 × 8 |
| `epochs` | 4 | CITED: CleanRL `update_epochs = 4` |
| `clip` | 0.1 | CITED: PPO Table 5 *"Clipping parameter ϵ 0.1×α"*; CleanRL `clip_coef = 0.1` |
| `actor_lr`, `critic_lr` | 2.5e-4, annealed | CITED: PPO Table 5; CleanRL |
| `gradient_clip` | 0.5 | CITED: CleanRL `max_grad_norm = 0.5` |
| `entropy_coefficient` | 0.01 | CITED: PPO Table 5; CleanRL |
| `recompute_advantages_per_pass` | on | CITED: Andrychowicz et al. *"recompute advantages once per data pass"* |
| value loss | MSE, no clipping | CITED: Andrychowicz et al. |
| `training_cost_multiplier` | 2 | CITED + MEASURED (§7.2) |
| `training_passes` | 60 | CITED (budget in passes) + MEASURED (the value, §7.3) |
| `validation_passes` | 5 | CITED: Prechelt 1998 |

---

## 8. Training data

### 8.1 One venue: BitMEX XBTUSD

| source | span used | contents |
|---|---|---|
| BitMEX XBTUSD (`data/bitmex_xbtusd_1m_2016_2026_08_01`) | 2016-07-02 → 2026-01-01 (cache); 2026 locked | one-minute bars from regular trades only (no forward fill), volume in USD contracts, and every funding settlement (8-hourly) |

Prices, volume and funding come from **one venue** (USER, 2026-09-15). The ledger settles them as a USDT linear
contract (§4.6). The development cache stops at 2026-01-01; nothing after it is read.

### 8.2 Why not Binance data — the audit that settled it (`results/agent_v2/audit/proxy_audit.json`)

The first v2 design used Binance USD-M BTCUSDT from 2020 (collected by `src/collector/binance_um.py`: 3,689,642
minutes from its launch on 2019-09-08, one missing; 7,682 funding settlements, all 8 h apart) with BitMEX before, and
tested that splice against pre-registered rules on 2020–2025:

| test | measured |
|---|---|
| hourly log returns, 52,589 hours | ρ = 0.9919 (rule ≥ 0.99: passed); median absolute difference 2.2 bp |
| features from each venue alone | returns 0.9945–0.9999; volatility 0.9977; volume **0.8393**; funding **0.5148** |
| daily funding | ρ = 0.69; mean **1.84 bp a day on BitMEX against 3.51 on Binance** |

Returns agree, but the two funding histories do not. Rather than train on one funding regime before 2020 and
another after, the user chose BitMEX throughout. The cost of that choice is recorded in §15: a live Binance USD-M
long paid roughly twice BitMEX's funding over 2020–2025. Binance's data stay collected for that comparison.

### 8.3 Episodes

Unchanged from v1: **2,176 hourly decisions (90.7 days)** starting at uniformly random five-minute boundaries of
the training span, first episodes staggered by a random number of rollouts (Mnih et al., A3C, 2016); a 64-decision
exploring-start prefix with real fees and funding in which a third of accounts stay flat and the rest enter a
uniformly chosen side (Sutton & Barto 2018, §5.3). The episode length is the user's 2–3-month range rounded to
whole rollouts (USER + DERIVED); the quarterly windows of Yang et al. (2020) are an analogy, not evidence for it,
as the review noted. The review's suggestion to test episode length separately is not taken up here (§13).

Uniform sampling reproduces the history's mix of regimes rather than balancing it; v1 measured 28 non-overlapping
quarters in 2017–2023 behind ~2,800 episode draws (§1, §15).

---

## 9. Evaluation and selection

- **Actions: the most probable action** (`evaluation_policy` = `argmax`, USER), one deterministic path per
  evaluation.
- **Real costs**: 5 bp + 1 bp per leg; the training multiplier never applies to evaluation.
- **Each validation quarter from a flat account**, decisions on the hour, after a 64-decision burn-in, the
  memory rebuilt every 128 decisions as in training.
- **Folds** (`FOLDS`): 2024 Q1–Q4 (train 2017–2023) and 2025 Q1–Q4 (train 2017–2024). Five seeds × 8 quarters = 40
  scores per checkpoint.
- **Selection**: the checkpoint index with the highest interquartile mean of quarterly log return, with a
  stratified bootstrap 95% interval of 50,000 replications (Agarwal et al., NeurIPS 2021); **probability of
  backtest overfitting** by CSCV over the 8 quarters and 12 checkpoints (Bailey, Borwein, López de Prado & Zhu
  2017); break-even extra slippage from turnover.
- **References through the same ledger, never used for selection**: constant long, constant short, flat (USDT
  cash), and two time-series momentum rules that hold the sign of the 1-week (Liu & Tsyvinski 2021) or 26-week
  (Moskowitz, Ooi & Pedersen 2012) volatility-normalised return, revised once a day at 00:00 UTC (DESIGN CHOICE:
  keeps their turnover far below the agent's). Example from the search smoke run: the 26-week rule lost −0.50 log
  in 2022 Q1 while BTC moved −1.5%, whipsawed by six flips — a check of the rule, not a result.
- **Test window: the whole of 2026, locked** (USER; `TEST_START` = 2026-01-01; the cache stops there;
  `development_range` refuses later dates; `TEST_LOCK.json`). BitMEX data run to 2026-08-01, seven months. When it
  is opened is a USER decision (§15).

**What the review asked for that §9 does not yet contain:** a benchmark of the user's own enter–hold–exit rule.
It needs the user's thresholds; the momentum rules stand in until then.

---

## 10. Diagnostics and logs

Everything below is written by every run, so that a poor result can be traced to its cause afterwards.

**Every update** (`runs/<year>/seed_<seed>/updates.jsonl`, one JSON line per update):

| metric | reads |
|---|---|
| `account_side_share`, `long_entry_share`, `sampled_action_share` | side concentration and the direction of entries |
| `state_action_mutual_information[_flat]` | zero exactly when the policy ignores the state |
| `policy_entropy`, `flat_long_probability`, `flat_short_probability`, `entry_decisions` | whether entries are explored |
| `exit_decisions`, `reversal_decisions`, `mean_age_at_exit_hours`, `policy_implied_holding_hours` | holding and exits |
| `approx_kl`, `clip_fraction`, `post_update_kl`, `policy_loss` | the trust region and the actor's objective |
| `actor_grad_norm`, `critic_grad_norm` | gradient norms before clipping at 0.5 — how often and how hard the clip binds |
| `value_loss`, `explained_variance_pre/post`, `target_std`, `mean_value`, `advantage_std` | critic tracking and the scale of the learning signal |
| `mean_reward`, `reward_sd`, `reward_p05`, `reward_p95` | the reward's distribution |
| `liquidations_in_rollout`, `mean_net_log_return`, `median_net_log_return`, `turnover_per_decision` | ruin, net return at the training cost, intensity |
| `replay_probability_error`, `replay_value_error`, `replay_bootstrap_error` | exactness of the recurrent replay |
| `actor_lr`, `critic_lr`, `update_seconds`, `elapsed_seconds`, `passes` | schedule and speed |

**Every checkpoint** (every 5 passes, 12 per run):

| file | contents |
|---|---|
| `policy_<update>.pt` | actor and critic weights, the scaler, the specification |
| `validation_<update>.json` | per validation quarter: log return, daily Sharpe, drawdown, orders, round trips, holding spells, long and short shares, entries, fees, funding, turnover, liquidations |
| `validation_<update>.csv.gz` | **every hourly decision** of the validation year: time, action, executed action, position, fill, log return, reward increment, equity, mark, fees, funding, liquidation, the three action probabilities, **the critic's value**, **the 13 standardised market inputs** (`z_*`) and **the 8 account inputs** (`account_*`) |
| `in_sample_<update>.json`, `.csv.gz` | the same exam on the **last training year** (2023 for fold 2024, 2024 for fold 2025): if validation is poor, a good in-sample result points at overfitting or non-stationarity, a poor one at failing to learn or a missing edge |
| `rollout_<update>.npz` | the last training rollout before the checkpoint (8 × 128 decisions): minutes, actions, sides, probabilities and values at collection, rewards, terminations, net returns, turnover, account inputs, and the advantages, returns and values after the update — what training saw and how it credited it |
| `training.json` | status, fold measurements, and the validation curve with each checkpoint's in-sample summary |

**Per run and per experiment:** `logs/<year>_<seed>.log` (the run's stdout, including a line every 25 updates),
`logs/jobs.jsonl` (starts, exits, retries), `status.json`, `queue.json` (fold measurements), `resume.pt`,
`coordinator.log`, `host.log` (load, memory and each worker's memory and CPU every 5 minutes),
`comparison/selection.json` (every checkpoint's IQM on validation **and in sample**, the selected checkpoint, its
interval, PBO, break-even slippage, and all five references per quarter), and `provenance/` — the exact `src/agent`
source, ARCHITECTURE.md, README, the pre-registration, the tests, the audits, the search decision, and
`environment.json` (Python, torch, numpy and pandas versions, `pip freeze`, git head and status, host, data and cache
manifests with their SHA-256).

---

## 11. Verification before compute

### 11.1 Unit tests — `tests/test_agent.py`, 107 tests

New or rewritten for v2:

- **Linear ledger:** a 1× long gains the price return and a 1× short loses it, flat keeps its USDT (exact to
  1e-12 without costs); exposure equals log(1 + notional/equity) and a 1× long is never liquidated down to 0.5% of
  entry; a 1× short is liquidated at the analytic price (199.04 from 100); turnover is not inflated by a
  liquidation.
- **State:** the 13 features are the pre-registered survivors in candidate order, with no clock, cycle or
  single-bar features; every candidate is causal (a change at minute k cannot move anything before the next
  boundary); returns and ratios are invariant to scaling volatility ×2 and volume ×1,000, while the volatility
  level moves by exactly log 2.
- **Folds and specification:** outer folds 2024 and 2025 whose last quarter ends where the 2026 test begins; search
  folds 2022/2023 before them; the configuration round-trips through the frozen protocol, so a run is rebuilt
  from it rather than from command-line flags.
- **Logging:** every validation decision records the critic's value and all 21 inputs, which match the scaler's
  observations exactly; every update records gradient norms, value, advantage and reward statistics and keeps the
  rollout it learned from.
- **Evaluation:** argmax is deterministic and takes the most probable action; sampled evaluation still reports a
  distribution; the momentum reference holds the signal's sign and revises only at 00:00 UTC; unknown references
  and dates from 2026-01-01 are refused.
- Unchanged v1 tests of the reward's exact increments, GAE, the PPO surrogate, the recurrent replay, exploring
  starts, time limits, IQM, the stratified bootstrap and PBO.

### 11.2 End-to-end smoke runs — 2026-09-15

- One search cell (γ 0.99, λ 0.95 per hour), both inner folds, 1 pass, on the BitMEX-to-Binance cache: training,
  argmax validation, selection, references and the frozen-protocol checks completed; replay errors 0.0.
- The main experiment on the final BitMEX-only design, seed 11, both outer folds, 2 passes, a checkpoint every pass:
  both folds completed at 1.35–1.43 s per update with replay errors 0.0; the frozen protocol recorded 13 features,
  folds 2024 and 2025 and a test start of 2026-01-01; the per-fold break-evens were 12.2 h and 12.7 h against a
  24.3 h credit window; selection computed PBO over 8 quarters; all five references were evaluated.

Returns of runs this short are not results.

### 11.3 Data and state audits — 2026-09-15

`results/agent_v2/audit/proxy_audit.json` (the splice, §8.2), `feature_redundancy.json` (BitMEX alone, §4.2) and
`feature_redundancy_merged_data.json` (the superseded splice). The v2 pipeline reproduces v1's six features and
prices exactly on the shared BitMEX data (§4.2).

### 11.4 The γ/λ search

§5.

---

## 12. Provenance of every constant

| constant | value | class | source |
|---|---|---|---|
| `decision_minutes` | 5 | USER | bar length of the features and reward statistics; accounting runs every minute |
| `decision_bars` | 12 | USER + MEASURED | hourly decisions, adopted on v1's evidence that five-minute decisions lose entry exploration (§4.1); frame skip (Mnih et al. 2015; Andrychowicz et al. 2021) |
| `fee` | 0.0005 | CITED | the user's account: Binance USD-M taker fee at VIP 0, 0.05% |
| `slippage` | 0.0001 | USER | kept from v1 |
| `entry_leverage` | 1.0 | USER | kept from v1 |
| `maintenance` | 0.004 | CITED | Binance BTCUSDT maintenance margin rate in the first notional tier, 0.40%; it cannot bind a 1× long and binds a 1× short near a doubling |
| `training_cost_multiplier` | 2.0 | CITED + MEASURED | Deng et al. 2017 (5×); Zhang, Zohren & Roberts 2020 (cost as a regulariser); this project's walk-forward study ranked 2× first of 16 cells (§7.2) |
| `hidden` | 128 | CITED | CleanRL `ppo_atari_lstm` LSTM 128 |
| `head_width` | 128 | CITED + DESIGN CHOICE | two hidden layers (Andrychowicz et al.); width equal to `hidden` is a choice |
| `critic_width_multiple` | 2 | CITED | Andrychowicz et al. on wider value networks |
| `environments` | 8 | CITED | PPO Table 5; CleanRL |
| `rollout` | 128 | CITED + DERIVED | PPO Table 5 horizon 128; CleanRL; bound rollout / 2 ≥ credit window |
| `sequence` | 128 | CITED + DERIVED | CleanRL whole-rollout sequences; bound `burn_in` + `sequence` ≥ credit window |
| `burn_in` | 64 | CITED | R2D2 burn-in ratio 0.5 (Kapturowski et al. 2019) |
| `episode_steps` | 2,176 | USER + DERIVED | the user's 2–3-month episodes; whole rollouts nearest a quarter (`quarter_episode`). Yang et al. 2020's quarterly windows are an analogy only |
| `minibatch_sequences` | 2 | DERIVED | CleanRL `num_minibatches = 4`; 256 samples = PPO Table 5 minibatch 32 × 8 |
| `epochs` | 4 | CITED | CleanRL `update_epochs` |
| `recompute_advantages_per_pass` | True | CITED | Andrychowicz et al. |
| `gamma` | 0.99825 | USER + MEASURED | 0.9792 per hour; v1's USER value, kept by the pre-registered 3 × 3 search (no level beat it by one standard error, §5) |
| `gae_lambda` | 0.99825 | USER + MEASURED | 0.9792 per hour; as `gamma` (§5) |
| `actor_lr` | 2.5e-4 | CITED | PPO Table 5; CleanRL |
| `critic_lr` | 2.5e-4 | CITED | PPO Table 5; CleanRL |
| `anneal_lr` | True | CITED | PPO Table 5; CleanRL |
| `gradient_clip` | 0.5 | CITED | CleanRL `max_grad_norm` |
| `entropy_coefficient` | 0.01 | CITED | PPO Table 5; CleanRL; Mei et al. 2020 |
| `clip` | 0.1 | CITED | PPO Table 5 (0.1 × α); CleanRL `clip_coef`; the discrete-action failure mode (Hsu et al. 2020) is monitored |
| `flat_start_fraction` | 1/3 | CITED | exploring starts uniform over the three sides (Sutton & Barto 2018 §5.3) |
| `reward` | sharpe | USER | differential Sharpe ratio; `sortino` selects the downside deviation ratio (Moody & Saffell 2001) |
| `dsr_eta` | 1/1440 | DESIGN CHOICE | per bar, a five-day window inside derived bounds |
| `dsr_initial_second_moment` | measured per fold | MEASURED | σ_bar² of the fold's training span (§2) |
| `observation_clip` | 10.0 | CITED | Engstrom et al., ICLR 2020 |
| `training_passes` | 60 | CITED + MEASURED | a budget in passes with validation stopping (Moody & Saffell 2001; Deng et al. 2017); the value from v1's validation curve and test looks (§7.3) |
| `validation_passes` | 5 | CITED | strips of 5 epochs (Prechelt 1998) |
| `updates` | derived per fold | DERIVED | ⌈`training_passes` × training decisions / 1,024⌉ |
| `threads` | 1 | MEASURED | one thread per worker; 1.4–1.9 s per update with six workers |
| `evaluation_policy` | `argmax` | USER | 2026-09-15 |
| `evaluation_seeds` | 4 | MEASURED | used only when `evaluation_policy` = `sampled` (v1: a 4-seed mean's standard error 0.12) |
| `folds` | (2024, 4), (2025, 4) | USER + CITED | expanding walk-forward (Zhang, Zohren & Roberts 2020); 2026 kept whole for testing (USER) |
| `seeds` | 11, 29, 47, 61, 83 | USER | five training seeds; the values are arbitrary |
| `enforce_horizon` | True | DERIVED | the Sharpe-1 break-even guard (§2); False in the search cells; the chosen 24.3 h window clears it |
| `bootstrap_replications` | 50,000 | CITED | rliable default (Agarwal et al. 2021) |
| `confidence_level` | 0.95 | CITED | rliable default |

Numbers outside `Config`: `TEST_START` = 2026-01-01 (USER); `SEARCH_FOLDS` 2022 and 2023 (DERIVED: the two years
before the first validation fold); the redundancy threshold 0.8 (v1's rule, fixed before measuring); the search's
grid and decision rule (CITED bounds, DESIGN CHOICE rule, fixed before measuring, §5); the return horizons (CITED,
§4.2); the momentum references' daily revision (DESIGN CHOICE); `SIGNAL_SHARPE_ASSUMPTION` = 1.0 (ASSUMPTION);
`NO_MOVEMENT` = 1e-12 and the replay tolerance 5e-5 (numerical).

---

## 13. The external review, item by item

| review recommendation | v2 | where |
|---|---|---|
| align the environment with a USDT account: ledger, sizing, funding, margin, rewards, benchmarks | **adopted** for the ledger, sizing, margin, reward and references; prices and funding stay BitMEX's for consistency (USER), after the audit showed Binance's funding differs (§8.2) | §4.6, §8 |
| define the enter–hold–exit logic as a benchmark | **partly**: two momentum rules as references; the user's own rule needs its thresholds | §9 |
| recent directional changes in the state | **adopted**: 1 h, 3 h, 1 d, 3 d returns | §4.2 |
| longer trend context | **adopted**: 4 w, 8 w, 13 w, 26 w returns | §4.2 |
| explicit deterioration of an open trade | **not adopted**: this project's measurement found no forward information in drawdown from a peak on BTC | §4.3 |
| test γ and λ independently | **adopted**: 3 × 3 search on inner folds | §5 |
| test episode length separately | **deferred**: one factor at a time; the user's range kept | §8.3 |
| evaluate the intended decision rule (deterministic vs sampled) | **adopted**: argmax (USER) | §9 |
| fresh chronological evaluation | **adopted**: the whole of 2026 locked (USER); Jan–Jul 2026 was v1's test, so only August onward is untouched by any look | §9, §15 |
| regime labels from past data only | **not applicable**: no regime label is an input; the features are causal (tested) | §4.2 |
| the 12-hour break-even is a heuristic | **adopted**: reported, not enforced in the search | §2, §5 |
| keep the Yang et al. citation as an analogy, not evidence | **adopted** | §8.3, §12 |

---
## 14. Mathematical formulation and algorithm

This section writes down exactly what the code computes, in one place. Every symbol maps to a function
or `Config` field, named in brackets; the prose sections above give the reasons and provenance. Formulas
are in LaTeX (they render on GitHub and in most Markdown viewers).

### 14.1 Time and notation

| symbol | meaning | code |
|---|---|---|
| $m$ | minute index; $O_m, H_m, L_m, C_m$ open, high, low, close ($C$ forward-filled over untraded minutes) | `market.prices` |
| $b$ | five-minute bar boundary, $b \equiv 0 \pmod 5$; the bar ending at $b$ covers minutes $b-5,\dots,b-1$ | `boundary` |
| $k$ | decision index; decision $k$ is taken at minute $t_k$ and holds for $K=12$ bars, $t_{k+1}=t_k+60$ | `decision_bars` |
| $f_m$ | funding rate settled at minute $m$ (zero at other minutes) | `market.funding` |
| $\varphi,\ \varsigma$ | commission and slippage per leg, 5 bp and 1 bp at evaluation | `fee`, `slippage` |
| $\kappa$ | training cost multiplier, 2: training charges $\kappa\varphi$ and $\kappa\varsigma$ | `training_cost_multiplier` |
| $\gamma,\ \lambda$ | discount and GAE parameter **per bar** (§5) | `gamma`, `gae_lambda` |
| $\gamma_d=\gamma^{K},\ \lambda_d=\lambda^{K}$ | per decision, i.e. per hour (§5) | `decision_gamma`, `decision_lambda` |

### 14.2 The market state $x_k$ (exogenous)

Every quantity at boundary $b$ uses minutes before $b$ only (`causal_arrays`). With the bar log return
$r_b=\ln C_{b-1}-\ln C_{b-6}$ and window lengths $D=288$ (day) and $M=8064$ (four weeks) in bars:

$$
\sigma^2_{\text{day}}(b)=\frac{1}{D}\sum_{j=0}^{D-1} r_{b-j}^2,\qquad
\sigma^2_{\text{month}}(b)=\frac{1}{M}\sum_{j=0}^{M-1} r_{b-j}^2,\qquad
\sigma_d(b)=\sqrt{D\,\sigma^2_{\text{month}}(b)}
$$

For each horizon $h\in\{1, 3, 24, 72, 168, 672, 1344, 2184, 4368\}$ hours:

$$
x^{(h)}_b=\frac{\ln C_{b-1}-\ln C_{b-1-60h}}{\sigma_d(b)\,\sqrt{h/24}}
\qquad\texttt{return\_1h, …, return\_26w}
$$

$$
x^{(\text{vl})}_b=\ln\sigma_d(b)\quad\texttt{volatility\_level},\qquad
x^{(\text{vr})}_b=\tfrac12\ln\!\big(\sigma^2_{\text{day}}(b)/\sigma^2_{\text{month}}(b)\big)\quad\texttt{volatility\_regime}
$$

together with $x^{(\text{vol})}_b$, the log of the last week's mean daily volume minus that of the four weeks before
(`volume_regime`), and $x^{(f)}_b$, the last funding rate settled before $b$ (`funding_rate`). These 13 are the state;
the cache also holds the two removed candidates (§4.2) so the audit is reproducible.
The network sees $z_k=\operatorname{clip}\big((x_{t_k}-\mu)/s,\,-10,\,10\big)$, with $\mu$ and $s$ measured on the
fold's training span only (`Scaler.fit`). Prices, volume and funding are BitMEX XBTUSD's (`load_sources`).

### 14.3 The account (endogenous) and the ledger

The account holds a USDT wallet $W$, a signed BTC quantity $Q$ and an entry price $\bar p$. Its equity at mark
$P$ is the linear-contract value

$$
E(P)=W+Q\,(P-\bar p).
$$

**Action.** $a_k\in\{\text{SHORT},\text{FLAT},\text{LONG}\}\mapsto s_k\in\{-1,0,1\}$, a *target* side. If
$s_k\neq\operatorname{sign}(Q)$, the order fills at the first minute $m$ of the hour that has a print, at its open
$O_m$ (`Ledger.execute`), with $\varphi$ and $\varsigma$ multiplied by $\kappa$ in training:

$$
\begin{aligned}
\text{close:}\quad & p=O_m\,(1-\operatorname{sign}(Q)\,\varsigma),\qquad W\leftarrow W+Q\,(p-\bar p)-\varphi\,|Q|\,p,\qquad Q\leftarrow 0\\
\text{open:}\quad & p=O_m\,(1+s_k\varsigma),\qquad Q\leftarrow s_k\,\frac{W\,\ell}{O_m},\qquad W\leftarrow W-\varphi\,|Q|\,p,\qquad \bar p\leftarrow p
\end{aligned}
$$

with entry leverage $\ell=1$ and no rebalancing while the position is held.

**Every minute** of the hour, in this order (`advance`):

1. funding: $W\leftarrow W-Q\,C_{m-1}\,f_m$ (a long pays a positive rate);
2. a margin check at $C_{m-1}$, then the pending order if this minute is eligible;
3. a margin check at the adverse extreme $P^{\text{adv}}_m=L_m$ for a long, $H_m$ for a short. **Liquidation**
   happens if
   $$E(P^{\text{adv}}_m)\le |Q|\,P^{\text{adv}}_m\,(\mu_{\text{mm}}+\varphi),\qquad \mu_{\text{mm}}=0.40\%,$$
   and the position is then closed at $P^{\text{adv}}_m$ and the episode terminates. For a 1× long,
   $E(P)\approx W P/\bar p$ against a requirement of $(W/\bar p)\,P\,(\mu_{\text{mm}}+\varphi)$, so it never binds;
   for a 1× short, $E(P)\approx W(2-P/\bar p)$ meets it at $P/\bar p\approx 2/(1+\mu_{\text{mm}}+\varphi)=1.991$;
4. the equity is marked at the close: $E_m=E(C_m)$.

**Account observation** $y_k\in\mathbb R^8$ (`Ledger.portfolio`), marked at $C_{t_k-1}$:

$$
y_k=\Big[\,\mathbb 1_{s=-1},\ \mathbb 1_{s=0},\ \mathbb 1_{s=+1},\ \
\frac{\ln(E/E_{\text{trade start}})}{\sigma_d},\ \
\ln\!\Big(1+\frac{\text{bars held}}{K\,w}\Big),\ \
\ln\!\Big(1+\frac{|Q|\,P}{E}\Big),\ \
S(A,B),\ \ \tfrac12\ln\frac{B}{B_0}\,\Big]
$$

with the credit window $w=1/(1-\gamma_d\lambda_d)$ decisions, the reward statistics $A$ and $B$ of §14.4, and the
continuous entries clipped to $\pm10$. The state is $s_k=(z_{\le k},\,y_k)$.

### 14.4 Reward

For each five-minute bar $b$ inside the decision, with $R_b=\ln(E_b/E_{b-1})$ the log change of USDT equity net of
commission, slippage and funding — at $\kappa$ times the cost in training (`differential_ratio`):

$$
\begin{aligned}
A_b &= A_{b-1}+\eta\,(R_b-A_{b-1}), &
B_b &= B_{b-1}+\eta\,(g(R_b)-B_{b-1}),\\
S(A,B) &= \frac{A}{\sqrt{B-A^2}}\ \ (\text{sharpe},\ g(R)=R^2), &
S(A,B) &= \frac{A}{\sqrt B}\ \ (\text{sortino},\ g(R)=\min(R,0)^2),
\end{aligned}
$$

$$
D_b=\frac{S(A_b,B_b)-S(A_{b-1},B_{b-1})}{\eta},\qquad
r_k=c\sum_{b\in k} D_b,\qquad
c=\sqrt{\frac{1-\gamma_d^2}{K}}\ \Big(\times\tfrac{1}{\sqrt2}\text{ for sortino}\Big),
$$

with $\eta=1/1440$. A fresh account starts at $A_0=0$ and $B_0=\hat\sigma^2_{\text{bar}}$, the bar-return
variance measured on the training span ($\hat\sigma^2_{\text{bar}}/2$ for sortino). The sum telescopes:
$\sum_{b\in k} D_b=\big(S_{\text{end}}-S_{\text{start}}\big)/\eta$.

### 14.5 The decision problem

Each episode starts at a uniformly random boundary $\tau$ of the training span, with room for the prefix
and the episode. A prefix of $L=64$ decisions builds the starting position: with probability $1/3$ the
account stays flat; otherwise it enters side $\pm1$, each with probability $1/2$, at a step
$j\sim\mathcal U\{0,\dots,63\}$ (`TrainingEnvironment.reset`). The episode then runs for at most
$T_{\text{ep}}=2176$ decisions and terminates early on liquidation. The objective is

$$
J(\theta)=\mathbb E_{\tau,\ \pi_\theta}\Big[\sum_{k=0}^{T_{\text{ep}}-1}\gamma_d^{\,k}\,r_k\Big],
$$

where a liquidation ends the sum and the time limit does not: truncated episodes are bootstrapped with the
critic.

### 14.6 Policy and value networks

The actor ($\theta$) and the critic ($\omega$) are separate branches with the same shape (`Branch`).
Widths are $n=128$ for the actor and $n=256$ for the critic.

$$
\begin{aligned}
e_k &= \tanh\big(\mathrm{LN}(W_e z_k+b_e)\big)\\
(h_k,c_k) &= \mathrm{LSTM}(e_k,\ h_{k-1},c_{k-1}),\qquad m_k=\mathrm{LN}(h_k)\\
u_k &= \tanh\big(\mathrm{LN}(W_1[m_k,\,y_k]+b_1)\big),\qquad v_k=\tanh\big(\mathrm{LN}(W_2u_k+b_2)\big)\\
\pi_\theta(\cdot\mid s_k) &= \operatorname{softmax}(W_\pi v_k+b_\pi),\qquad V_\omega(s_k)=w_V^{\top}v_k+b_V
\end{aligned}
$$

The recurrent state $(h,c)$ is never carried across rollouts. It is rebuilt from zero over the $L=64$
decisions that precede each rollout, both in training and in evaluation (`Trainer.collect`,
`market_cache`).

### 14.7 Advantage estimation and the PPO loss

With $d_k=1$ at liquidation (time limits do not set it) and $\text{valid}_k=0$ after termination
(`advantages`):

$$
\delta_k=r_k+\gamma_d(1-d_k)\,V_\omega(s_{k+1})-V_\omega(s_k),\qquad
\hat A_k=\text{valid}_k\big(\delta_k+\gamma_d\lambda_d(1-d_k)\hat A_{k+1}\big),\qquad
\hat G_k=\hat A_k+V_\omega(s_k).
$$

For each minibatch $\mathcal B$, with advantages normalised over its valid steps,
$\tilde A_k=(\hat A_k-\bar A_{\mathcal B})/(\operatorname{sd}_{\mathcal B}(\hat A)+10^{-8})$, and the probability
ratio $\rho_k(\theta)=\pi_\theta(a_k\mid s_k)/\pi_{\theta_{\text{old}}}(a_k\mid s_k)$
(`clipped_surrogate`):

$$
\mathcal L_{\text{actor}}(\theta)=-\frac{1}{|\mathcal B|}\sum_{k\in\mathcal B}
\min\!\Big(\rho_k\tilde A_k,\ \operatorname{clip}(\rho_k,1-\epsilon,1+\epsilon)\,\tilde A_k\Big)
-\beta\,\frac{1}{|\mathcal B|}\sum_{k\in\mathcal B}\mathcal H\big(\pi_\theta(\cdot\mid s_k)\big),
$$

$$
\mathcal L_{\text{critic}}(\omega)=\frac{1}{|\mathcal B|}\sum_{k\in\mathcal B}\tfrac12\big(V_\omega(s_k)-\hat G_k\big)^2,
$$

with $\epsilon=0.1$ and $\beta=0.01$. Each loss gets its own Adam optimiser, a global gradient-norm clip at
0.5 and the step size $\alpha_u=\alpha_0\,(1-u/U)$, where $\alpha_0=2.5\times10^{-4}$, $u$ is the update
number and $U$ the fold's update count.

**Budget.** $U=\big\lceil P\cdot N_{\text{train}}/(E\cdot T)\big\rceil$ for $P=60$ passes, $N_{\text{train}}$ hourly
decisions in the training span, $E=8$ environments and $T=128$ decisions per rollout. That gives 3,595 and 4,110 updates
for folds 2024 and 2025, with a checkpoint every $U/12$ updates (`fold_config`,
`checkpoint_updates`).

### 14.8 Algorithm — training one fold and one seed

```text
Algorithm 1  TRAIN(fold year Y, seed σ)
  span        ← [2017-01-01, Y-01-01)
  μ, s, σ̂_bar ← statistics of the 13 features and the 5-minute returns on span        # Scaler.fit
  B0          ← σ̂_bar²;  U ← ⌈60 · N_train / 1024⌉;  checkpoints ← {round(i·U/12) : i = 1..12}
  costs       ← κ · (φ, ς) for the training environments;  (φ, ς) for EVALUATE
  θ, ω        ← orthogonal initialisation (policy head gain 0.01, forget bias 1)
  for e = 1..8:  RESET(e);  shorten env e's first episode by 128·𝒰{0..16} decisions
  for u = 0..U-1:
      α ← α0 · (1 − u/U)
      D, bootstrap ← COLLECT()                              # Algorithm 2
      PPO-UPDATE(D, bootstrap, α)                           # Algorithm 3
      if u+1 ∈ checkpoints:
          save θ, ω;  EVALUATE(θ, fold Y)                    # Algorithm 4, stored for selection

Procedure RESET(e)
  τ ~ 𝒰{5-minute boundaries of span, leaving 64 + 2176 decisions before its end}
  new account: W = 1, Q = 0, A = 0, B = B0
  with prob 1/3: side ← 0   else: side ~ 𝒰{−1,+1},  j ~ 𝒰{0..63}
  for i = 0..63:  ADVANCE(account, i < j or side = 0 ? FLAT : side)     # training costs and funding
  cursor_e ← τ + 64 decisions;  prefix_e ← the 64 decision minutes just traversed
  if liquidated in the prefix: start a fresh flat account at cursor_e
```

```text
Algorithm 2  COLLECT()                                      # 8 environments × 128 decisions
  for each env e whose episode reached 2176 decisions or was liquidated:  RESET(e)
  (h, c) ← LSTM state after encoding z over prefix_e, from zero, under the current θ and ω
  for k = 0..127:
      z_k ← standardised features at cursor;   y_k ← account observation at close(cursor − 1)
      π_k, V_k, (h, c) ← networks(z_k, y_k, (h, c))
      a_k ~ Categorical(π_k)                                # sampled, never argmax
      r_k, d_k ← ADVANCE(account, a_k)                     # 12 bars × 5 minutes of ledger, §14.3–§14.4
      store (z_k, y_k, a_k, log π_k(a_k), π_k, V_k, r_k, d_k, valid_k)
      cursor ← cursor + 60 minutes
  bootstrap ← 0 if liquidated else V(s_128)
  prefix_e ← the last 64 decision minutes
  return D, bootstrap
```

```text
Algorithm 3  PPO-UPDATE(D, bootstrap, α)
  REPLAY: recompute π, V, bootstrap over D under θ, ω
          if any |replayed − collected| > 5e-5:  stop training (misaligned state)
  for epoch = 1..4:
      V ← collected values if epoch = 1 else REPLAY values  # advantages recomputed every pass
      Â, Ĝ ← GAE(r, V, bootstrap, d, valid; γ_d, λ_d)       # §14.7
      for each minibatch ℬ in a random partition of the 8 envs into 4 pairs:
          (h, c) ← encode the 64-decision prefix of ℬ without gradient
          Ã ← normalise Â over the valid steps of ℬ
          θ ← Adam_α( ∇θ[ −mean min(ρÃ, clip(ρ, 0.9, 1.1)Ã) − 0.01·mean 𝓗(π_θ) ] ),  ‖∇‖ ≤ 0.5
          ω ← Adam_α( ∇ω[ mean ½(V_ω − Ĝ)² ] ),                                       ‖∇‖ ≤ 0.5
```

```text
Algorithm 4  EVALUATE(θ, fold Y)                            # one checkpoint of one run; real costs
  for quarter q = 1..4 of Y:
      memory ← LSTM over z, rebuilt from zero every 128 decisions over the preceding 64
      flat account at the quarter's first hour;  for each hour k of q:
          a_k ← argmax_a π_θ(a | memory_k, y_k);  ADVANCE(account, a_k)       # evaluation_policy = argmax
      score[q] ← total USDT log return of the path
  (evaluation_policy = sampled: 4 sampled paths per quarter; score = their mean)

Algorithm 5  SELECT()                                       # after all folds and all 5 seeds
  for checkpoint index j = 1..12:
      S_j ← {score[q] of seed i, fold Y at checkpoint j}    # 5 seeds × 2 folds × 4 quarters = 40
      IQM_j ← mean of the middle half of S_j
  j* ← argmax_j IQM_j  (ties go to the earlier checkpoint)
  report: stratified bootstrap 95% interval of IQM_{j*}; PBO by CSCV over 8 quarters × 12 checkpoints;
          constant long, short, flat and the 1-week and 26-week momentum rules on every quarter

Algorithm 6  SEARCH()                                       # §5, before Algorithm 1 runs on the outer folds
  for γ_h in {0.9583, 0.9792, 0.9900}, λ_h in {0.90, 0.95, 0.9792}:
      γ, λ ← γ_h^(1/12), λ_h^(1/12);  P ← 30 passes;  a checkpoint every 5
      for fold Y in {2022, 2023}:  TRAIN(Y, seed 11)          # the break-even guard off
      score(γ_h, λ_h) ← IQM of the 6 checkpoints × 8 quarters of log returns
  for each of γ and λ: the level with the highest mean score over the other's levels,
      kept only if it beats 0.9792 by more than one standard error of the paired per-quarter difference
  result (2026-09-15): γ = λ = 0.9792 kept
```

### 14.9 What the formulation still does not do

- **Nothing in $J(\theta)$ separates regimes.** The expectation is over $\tau\sim\mathcal U(\text{span})$, so the policy
  maximises return averaged over the history's own mixture of regimes; v1 showed that mixture rewards holding long
  almost everywhere (§1). v2 changes the state and the account, not the objective.
- **The state now reaches back 26 weeks** through $x^{(4368)}$, where v1's reached four; the recurrent memory still
  spans at most 192 hours.
- **Training and evaluation differ twice, deliberately**: in cost ($\kappa=2$, §7.2) and in the action rule (sampled
  in training, argmax in evaluation, USER; §15).

---

## 15. Known limitations and open risks

1. **Whether an hourly BTC edge survives costs is unknown, and the evidence so far is against it.** v1 found none on
   validation and lost on 2026-01 → 2026-08; an earlier study measured an information coefficient of +0.024 at 1 h
   against a 0.202 hurdle; and eight of nine γ/λ search cells lost on 2022–2023 (§5). v2 removes mismatches; it does
   not create signal.
2. **BitMEX funding, not the user's venue's.** Over 2020–2025 Binance USD-M funding averaged 3.51 bp a day against
   BitMEX's 1.84 (§8.2): a long held continuously on Binance paid about 6% a year more than the agent is charged. The
   user chose consistency over venue fidelity; results that depend on holding long for weeks are the most exposed.
   Prices agree closely (hourly ρ = 0.9919).
3. **Argmax of a policy trained under sampling.** PPO optimises the sampled, entropy-regularised policy; its argmax is
   a related strategy never trained as such. v1 measured that early checkpoints can evaluate as almost always flat
   (entries in 0–1.1% of flat hours at pass 15).
4. **The horizon was searched on the superseded data**, at 30 passes and one seed, then used on BitMEX alone at 60 passes
   and five seeds. It kept the prior values; a rerun on the final design is available (`scripts/run_search.sh`, ~3.5 h).
5. **The feature set is broad in horizons.** Nine returns are nine chances to fit history; the redundancy rule removed
   near-duplicates but not correlated horizons below 0.8 (0.77 between 8 and 13 weeks). `volatility_level` is a level,
   and BTC's volatility has fallen across the years, so its standardised value drifts out of sample. `funding_rate`
   sits at BitMEX's 0.01% baseline most of the time; its information is in the excursions.
6. **2026 is not wholly untouched.** January–July 2026 was v1's test window, looked at twice; v1's losses there shaped
   v2's design. Only August 2026 onward is free of any look, and BitMEX data currently end 2026-08-01.
7. **Episode sampling reproduces the history's regime mix** (v1 §11.7); regime-balanced sampling and mirrored paths
   were proposed in v1 §18 and are not in v2.
8. **Episode length, the training cost multiplier and η were not searched.**
9. **The agent cannot react inside an hour**, and the margin model is a conservative approximation.
10. **Five seeds and two folds** give 40 scores per checkpoint; intervals will be wide.
11. **The user's own enter–hold–exit rule is not yet a reference** (§9).

---

## 16. Lineage

| generation | outcome |
|---|---|
| v4 – v6, gate, side-lock, reference PPO, all-action MPO | see `backup/agent_inverse_regime_20260915/ARCHITECTURE.md` §16 |
| recurrent PPO, regime state, hourly decisions (v1) | validation: no edge, PBO 0.80; spent 2026 window: 0 of 10 profitable at pass 15 or pass 100. Archived `backup/agent_inverse_regime_20260915` |
| v2, merged data (superseded the same day) | BitMEX spliced to Binance at 2020; audit passed on returns, failed on funding and volume; search kept γ = λ = 0.9792. Archived `backup/agent_v2_merged_data_20260915`, `results/agent_v2/search_merged_data` |
| **v2 (current)** | USDT linear account on BitMEX prices and funding; 9 return horizons + volatility, volume and funding features; 2× training cost; argmax evaluation; folds 2024–2025; all of 2026 locked |

---

## References

**Reinforcement learning — algorithms, implementation, generalisation**
- Schulman, Wolfe, Dhariwal, Radford & Klimov. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017 — Table 5.
- Schulman, Moritz, Levine, Jordan & Abbeel. High-Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR 2016.
- Andrychowicz et al. What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study. ICLR 2021, arXiv:2006.05990.
- Engstrom, Ilyas, Santurkar, Tsipras, Janoos, Rudolph & Madry. Implementation Matters in Deep Policy Gradients. ICLR 2020.
- Hsu, Mendler-Dünner & Hardt. Revisiting Design Choices in Proximal Policy Optimization. arXiv:2009.10897, 2020.
- Huang, Dossa, Ye, Braga, Chakraborty, Mehta & Araújo. CleanRL. JMLR 2022 — `ppo_atari_lstm.py`.
- Kapturowski, Ostrovski, Quan, Munos & Dabney. Recurrent Experience Replay in Distributed Reinforcement Learning. ICLR 2019.
- Hausknecht & Stone. Deep Recurrent Q-Learning for Partially Observable MDPs. AAAI Fall Symposium 2015.
- Mnih et al. Human-Level Control through Deep Reinforcement Learning. Nature 2015.
- Mnih et al. Asynchronous Methods for Deep Reinforcement Learning. ICML 2016.
- Pardo, Tavakoli, Levdik & Kormushev. Time Limits in Reinforcement Learning. ICML 2018.
- Sutton & Barto. Reinforcement Learning: An Introduction, 2nd ed. 2018 — §5.3.
- Jozefowicz, Zaremba & Sutskever. An Empirical Exploration of Recurrent Network Architectures. ICML 2015.
- Lyle, Zheng, Nikishin, Avila Pires, Pascanu & Dabney. Understanding Plasticity in Neural Networks. ICML 2023.
- Ba, Kiros & Hinton. Layer Normalization. arXiv:1607.06450, 2016.
- Mei, Xiao, Szepesvári & Schuurmans. On the Global Convergence Rates of Softmax Policy Gradient Methods. ICML 2020.
- Mei, Dai, Xiao, Szepesvári & Schuurmans. Understanding the Effect of Stochasticity in Policy Optimization. NeurIPS 2021.
- Zhang, Vinyals, Munos & Bengio. A Study on Overfitting in Deep Reinforcement Learning. arXiv:1804.06893, 2018.
- Prechelt. Early Stopping — But When? In Neural Networks: Tricks of the Trade, LNCS 1524, 1998.
- Subramanian, Sinha, Seraj & Mahajan. Approximate Information State for Approximate Planning and RL in Partially Observed Systems. JMLR 2022.
- Dietterich, Trimponias & Chen. Discovering and Removing Exogenous State Variables and Rewards for RL. ICML 2018.
- Cawley & Talbot. On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. JMLR 11, 2010.

**Trading reinforcement learning (local library `articles/problems/optimal trading/`)**
- Moody & Saffell. Learning to Trade via Direct Reinforcement. IEEE TNN 2001.
- Deng, Bao, Kong, Ren & Dai. Deep Direct Reinforcement Learning for Financial Signal Representation and Trading. IEEE TNNLS 2017.
- Zhang, Zohren & Roberts. Deep Reinforcement Learning for Trading. J. Financial Data Science 2020.
- Théate & Ernst. An Application of Deep Reinforcement Learning to Algorithmic Trading. Expert Systems with Applications 2021.
- Tummon, Raja & Ryan. Trading Cryptocurrency with Deep Deterministic Policy Gradients. IDEAL 2020, LNCS 12489.
- Yang, Liu, Zhong & Walid. Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy. ICAIF 2020.
- Borrageiro, Firoozye & Barucca. The Recurrent Reinforcement Learning Crypto Agent. IEEE Access 2022.
- Huang. Financial Trading as a Game: A Deep Reinforcement Learning Approach. 2018.
- Sato. Model-Free Reinforcement Learning for Financial Portfolios: A Brief Survey. arXiv:1904.04973, 2019.
- Pippas, Ludvig & Turkay. The Evolution of Reinforcement Learning in Quantitative Finance: A Survey. ACM Computing Surveys 2025.

**Finance and econometrics**
- Moskowitz, Ooi & Pedersen. Time Series Momentum. JFE 104(2), 2012.
- Liu & Tsyvinski. Risks and Returns of Cryptocurrency. RFS 34(6), 2021.
- Corsi. A Simple Approximate Long-Memory Model of Realized Volatility. J. Financial Econometrics 2009.
- Liu, Patton & Sheppard. Does Anything Beat 5-Minute RV? J. Econometrics 2015.
- Kaufman. Smarter Trading. McGraw-Hill 1995.
- Pagan & Sossounov. A Simple Framework for Analysing Bull and Bear Markets. J. Applied Econometrics 2003.
- Barber & Odean. All That Glitters. RFS 2008.
- He, Manela, Ross & von Wachter. Fundamentals of Perpetual Futures. 2024.
- Koijen, Moskowitz, Pedersen & Vrugt. Carry. JFE 2018.
- Sortino & van der Meer. Downside Risk. J. Portfolio Management 1991.
- Binance. USDⓈ-M futures trading fees (VIP 0 taker 0.05%); BTCUSDT leverage and margin tiers (maintenance margin 0.40% in the first tier). Binance public data (data.binance.vision) and the USD-M REST API.
- BitMEX. XBTUSD one-minute trade archive and funding history (the dataset's manifest records sources and audit).

**Evaluation**
- Agarwal, Schwarzer, Castro, Courville & Bellemare. Deep RL at the Edge of the Statistical Precipice. NeurIPS 2021.
- Bailey, Borwein, López de Prado & Zhu. The Probability of Backtest Overfitting. J. Computational Finance 2017.
