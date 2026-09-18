# Agent architecture

**Status:** current as of 2026-09-15. Package `src/agent/`. **Evaluation now takes the most probable action
(argmax; USER, 2026-09-15)**; training still samples. Every result in §11 up to 2026-09-13 was produced with
sampled evaluation and is not directly comparable (§9, §15). **The 2026 test window is spent**: all ten runs
lost at both the selected and the fully trained checkpoint (§11.6). The mathematical formulation and
pseudo-code are in §17; why the agent does not generalise, and what to change, in §18.
**Hourly decisions on five-minute bars** (USER, 2026-09-11). **The configuration was revised the same
day for hourly decisions and a finite history** (§0.2): the update shape of the discrete-action PPO
reference, a training budget counted in passes over each fold's training span, and the volatility that
sets the reward prior and the cost break-even measured per fold instead of taken from a constant.
**Verification:** unit tests 86 of 86; real-data pipeline checks passed (§11.2); pre-registered
verification of the revised configuration: **1 of 4 PASS**, all four regime-conditioned with positive
test returns at their selected checkpoints, two missing the bar by 0.03–0.05 (§11.4). **The real-data training completed on 2026-09-12 at 03:02 UTC**
(`results/agent/sharpe_bars12`). **Result: no edge distinguishable from zero** — selected checkpoint IQM
+0.019 log per quarter, 95% interval [−0.019, +0.057], probability of backtest overfitting 0.80 (§11.5).
USER decisions for the run: the Sharpe reward for training and
validation, five training seeds, 100 passes.
**Update this file whenever the agent changes.** A test fails if a constant in `Config` is missing from
the provenance table in §12.

Every constant is tagged with exactly one provenance class:

| class | meaning |
|---|---|
| **CITED** | taken from a named study or reference implementation, with the passage or table |
| **DERIVED** | follows from other constants or from the problem's arithmetic |
| **MEASURED** | measured on this data or this host, with the number — per fold where it depends on the data |
| **USER** | a decision the user made explicitly |
| **DESIGN CHOICE** | none of the above — stated openly as unverified |

---

## 0. The agent in one page

### 0.1 Summary

| component | what it is | section |
|---|---|---|
| instrument | one BitMEX XBTUSD inverse perpetual, BTC collateral, 1× entry notional | §3 |
| time | one-minute accounting and margin checks; five-minute bars for features and reward statistics; **one decision per hour** (12 bars) | §3, §4.1 |
| action | target position `SHORT`, `FLAT` or `LONG`, one softmax | §4.1 |
| market state | 6 regime features: `trend_efficiency_1w`, `drawdown_4w`, `volatility_level`, `volatility_regime`, `volume_regime`, `funding_rate` — chosen by measured relevance, redundancy and stability on 2017–2023 | §4.2, §4.3 |
| account state | 8 inputs: side one-hot, open trade's return and age, exposure, the reward's two statistics | §4.4 |
| reward | exact per-bar increment of an exponentially weighted Sharpe ratio (default) or downside deviation ratio, summed over the decision's 12 bars and scaled; its prior risk measured on each fold's training span | §4.5 |
| termination | liquidation terminates; episode ends are time limits | §4.6 |
| horizon | γ = λ = 0.99825 per bar → 24.3 h GAE credit window, 48.1 h discount horizon; cost break-even 12.2 h / 12.7 h measured per fold | §2, §5 |
| network | separate actor and critic: Linear–LayerNorm–tanh → LSTM → LayerNorm, then the account concatenated → two LayerNorm–tanh layers; 832,516 parameters | §6 |
| learner | recurrent PPO with the discrete-action settings of PPO's Atari experiments as CleanRL `ppo_atari_lstm` runs them: 8 environments × 128 decisions, 4 minibatches, 4 epochs, clip 0.1, Adam 2.5e-4 annealed, entropy 0.01; burn-in 64 | §7 |
| budget | **100 passes over each fold's training span** — 5,991 updates (fold 2024), 6,849 (fold 2025) — with a validated checkpoint every 5 passes | §7.3 |
| data | train 2017–2023 → validate 2024; train 2017–2024 → validate 2025; episodes of 2,176 decisions (90.7 days) at random start points, exploring starts | §8 |
| evaluation | the most probable action (argmax, USER 2026-09-15), one deterministic path; sampled over 4 seeds for every result up to 2026-09-13; each calendar quarter from flat; checkpoint selection by IQM, stratified bootstrap, PBO; break-even slippage; the 2026 test window is spent (§11.6) | §9 |
| formulation | the MDP, ledger, reward, networks, GAE/PPO losses and training, evaluation and selection as equations and pseudo-code | §17 |
| open problem | no generalisation to 2026; remedies from the literature, ranked | §18 |

### 0.2 What changed

**2026-09-15 — evaluation by argmax** (USER, after an external review): `evaluation_policy` = `argmax`. The
policy is still trained on sampled actions, as on-policy PPO requires; only validation, selection and test
evaluations take the most probable action. `evaluation_policy` = `sampled` reproduces the previous protocol.
MEASURED beforehand on the saved validation paths (indicative, since an argmax path visits other states):
from a flat account the most probable action is an entry in **0.0–1.1%** of hours at the selected checkpoints
(pass 15) and **3.5–14.2%** at the final ones (pass 100); while a position is held it is an exit or
reversal in 0.6–5.2% of hours. An early checkpoint can therefore evaluate as almost always flat.

**Earlier on 2026-09-11 — from the all-action MPO agent** (`backup/agent_all_action_mpo_20260911`):

| area | before | after | evidence |
|---|---|---|---|
| learner | exact counterfactual branches of every action; MPO E-step on a learned action-value head | standard recurrent PPO on sampled actions | USER decision; §7 |
| market state | 13 features incl. four single-bar values, 1-day and 1-week momentum, funding-cycle and hour encodings | 6 measured regime features | USER request; §4.2–§4.3 |
| account state | 7 inputs | 8: + trade age | §4.4 |
| reward | exact differential Sharpe increment | the same, or the exact downside-deviation (Sortino) increment, equal scale | USER request; §4.5 |
| decision interval | five minutes | hourly on five-minute bars | USER decision on smoke evidence (§4.1, §11.3) |

**Later on 2026-09-11 — revised so every figure fits hourly decisions on a finite history** (USER request;
§11.3 motivated each change):

| figure | before | after | why |
|---|---|---|---|
| training budget | 5,000 updates, matching CleanRL's Atari gradient steps | **100 passes over the fold's training span**: 5,991 / 6,849 updates | a simulator's step budget assumes unlimited fresh data; the trading studies budget epochs over a fixed history and stop by validation (Moody & Saffell 2001; Deng et al. 2017), and the hourly smoke runs overfit after a few dozen passes (§7.3, §11.3) |
| validation interval | every 500 updates | **every 5 passes**, 20 checkpoints aligned across folds | Prechelt (1998) validates at the end of strips of 5 epochs |
| rollout / sequence / burn-in | 1,024 / 256 / 128 | **128 / 128 / 64** | 1,024 existed only because the five-minute credit window was 286 steps; the hourly window (24.3) admits the discrete-action reference values (PPO Table 5; CleanRL) |
| minibatches | 4 environments | **2 environments** (4 minibatches of 256 samples) | CleanRL `num_minibatches` = 4; PPO Table 5 minibatch 32 × 8 |
| episode | 2,048 decisions | **2,176** (90.7 days) | a quarter to the nearest whole rollout |
| learning rate | 3e-4 | **2.5e-4** | PPO Table 5 (Atari) and CleanRL; 3e-4 is Andrychowicz et al.'s continuous-control default |
| clip | 0.25 | **0.1** | PPO Table 5 and CleanRL for discrete actions; 0.25 is Andrychowicz et al.'s continuous-control starting point |
| reward prior, cost break-even | a 14.8 bp constant → 22 h break-even | **measured per fold on its training span**: 26.55 bp / 25.54 bp → 12.2 h / 12.7 h | the constant came from 2023–2025, which overlaps both validation years |
| slippage stress | 1, 3 and 5 bp, reevaluated | **break-even slippage** derived from turnover | the grid had no source; the break-even needs no chosen levels |
| verification rule | mean of the last three evaluations | **select on a validation window, judge on a later test window** | mirrors the real protocol; changed after the last-three rule failed runs that had passed early and then overfit (§11.3) |

---

## 1. History — why earlier designs failed

### 1.1 Problem 1 — the entry side locked

The side-lock agent (stopped 2026-09-10) committed to one side in four of five runs (share short / flat
/ long in training rollouts):

| run | updates 1–200 | 401–600 | 801–1000 | validation @1000 | long share of real entries @500 → @1000 |
|---|---|---|---|---|---|
| 2023 s11 | .12/.71/.17 | .91/.06/.03 | **.96**/.03/.01 | −0.750 in a rising year | 0.42 → **0.00** |
| 2023 s29 | .08/.28/.64 | .03/.14/.84 | .01/.04/**.95** | +0.480 | 0.91 → 0.93 |
| 2024 s11 | .15/.21/.64 | .01/.03/.96 | .00/.03/**.96** | +0.362 | 0.85 → **1.00** |
| 2024 s29 | .09/.41/.50 | .01/.14/.85 | .00/.01/**.99** | +0.325 | 0.97 → **1.00** |
| 2025 s11 | .07/.12/.82 | .01/.05/.94 | .01/.04/**.95** | −0.503 in a falling year | 0.48 → **1.00** |

SAC-style temperature duals constrain *"average entropy … while the entropy at different states can
vary"* (Haarnoja et al. 2018), so rare entry states went deterministic. Learned termination functions
degenerate (Harb et al. 2018; Harutyunyan et al. 2019; Liu & Jing 2026, preprint). Committal
convergence is intrinsic to stochastic softmax policy gradient (Mei et al., NeurIPS 2021). Trading
agents that *"simply stop actively trading and adopt a passive approach"* are reported by Théate & Ernst
(2021).

### 1.2 Problem 2 — liquidations were not learned as catastrophic

74 training liquidations by update ~1,200, 73 of them in the four long-locked runs. The ±10 reward clip
made a wipeout cost the same as a slide; a Sharpe ratio is invariant to wealth, so only termination can
price ruin; and a 1× inverse long is liquidated by a 49.7% fall while a 1× short needs a 1,657-fold
rally.

### 1.3 The common root

That design carried 17 constants with no source and no test, and one wrong citation (the PPO paper's
Atari table uses 3 epochs, not 4). §13 lists every one.

### 1.4 The verification record of 2026-09-10 and early 09-11

Three trainers were verified on planted markets (a hidden regime with a 48 h mean dwell written into one
feature column; oracle Sharpe 3; real costs; five-minute decisions; 1,000 updates):

| trainer | outcome | measured mechanism |
|---|---|---|
| reference PPO, 13-feature state | **0 of 4**; side locks | at 0.009 sd per bar the edge was invisible in GAE advantages summing a 286-bar window of market noise; costs made the policy commit first |
| all-action PPO | **0 of 4**; frozen flat | the clipped-ratio gradient on a logit carries that action's own probability (Mei et al. 2020) |
| all-action MPO | learned a ten-times edge in 100 updates; the planted edge not in 100 | DERIVED from the signal-to-noise ratio: about 3,000 updates |

The user then chose standard PPO and a regime state. The Sharpe-3 planted test of §1.4 has not been
re-run; nothing here claims PPO would now pass it.

---

## 2. The one measurement, and the one assumption

The measurement is made **per fold, on that fold's training span only** (`Scaler.fit`): the standard
deviation of five-minute log returns is **26.55 bp over 2017–2023** (fold 2024) and **25.54 bp over
2017–2024** (fold 2025). Commission is 5 bp and adverse slippage 1 bp per leg (USER), so a round trip
costs 12 bp, 0.45–0.47 of one bar's typical move.

The assumption: a signal worth trading has an annualised Sharpe ratio of **1.0** (ASSUMPTION,
`SIGNAL_SHARPE_ASSUMPTION`). Its expected drift per bar is σ_bar / √105,120, so paying for one entry and
one exit takes 149 bars (**12.2 h**) in fold 2024 and 154 bars (**12.7 h**) in fold 2025. The credit
window must exceed it (checked per fold by `check_horizon`).

**Correction.** The previous version used a 14.8 bp constant, giving a 22 h break-even. That figure was
measured on 2023–2025 closes, which include both validation years; on the training years volatility was
80% higher. It has been removed.

---

## 3. Problem setup

| | |
|---|---|
| Instrument | one BitMEX XBTUSD inverse perpetual, BTC collateral |
| Accounting and margin checks | every minute |
| Bars (features, reward statistics) | five minutes |
| Decisions | **every hour** — `decision_bars` = 12 bars |
| Actions | target position `SHORT`, `FLAT`, `LONG` |
| Position size | 1× account equity at entry, never rebalanced |
| Costs | 5 bp commission + 1 bp slippage per leg, fixed, no curriculum |
| Reward | exact differential Sharpe increment (USER: used for training and validation); the downside-deviation variant is implemented and verified |
| Folds | train 2017–2023 → validate **2024**; train 2017–2024 → validate **2025** |
| Test | 2026-01-01 → 2026-08-01, **spent**: opened by `scripts/test_window.py` on 2026-09-12 (selected checkpoints) and 2026-09-13 (final checkpoints) at the user's instruction (§11.6) |

Expanding walk-forward folds follow Zhang, Zohren & Roberts (2020): *"We retrain our model at every 5
years, using all data available up to that point."*

---

## 4. The MDP

### 4.1 Decisions and actions

A categorical choice of target position, the representation of Moody & Saffell (2001), Huang (2018) and
Zhang, Zohren & Roberts (2020): keeping the current position costs nothing; moving from long to short
pays two legs. Whether to trade, which side, how long to hold and when to exit are this one choice made
every hour: enter from flat, keep, exit to flat, or reverse.

**Timing of one decision.** At the decision boundary *t* the agent observes the features of the last
completed five-minute bar and its account marked at the previous minute's close. The order is live for
the next 60 minutes and fills at the first minute with a printed trade, at that minute's open plus
adverse slippage; with no printed trade in the hour the position is unchanged. Inside the hour the ledger
runs minute by minute: funding is booked at settlement minutes, the margin check uses each minute's
adverse extreme, and a liquidation ends the episode at that minute. The reward statistics advance once
per five-minute bar, twelve times per decision (§4.5). In evaluation decisions fall on the hour; in
training, episodes start at random five-minute boundaries, so decision times within the hour vary.

**Why hourly.** Holding an action for several steps is frame skip, used by DQN (Mnih et al., Nature
2015); Andrychowicz et al. (ICLR 2021) recommend *"Try frame skip if possible"*. γ and λ are specified
per bar and raised to the 12th power per decision, so the horizon in time is unchanged (tested within
3%).

1. **DERIVED — holding under sampled actions.** Training samples actions (and evaluation did until
   2026-09-15), so a mean holding period of H decisions needs a per-decision probability of keeping the
   position of 1 − 1/H. The argument still governs what the policy can practise in training; an argmax
   evaluation holds for as long as keeping stays the most probable action.

   | hold | five-minute decisions: keep probability, policy entropy | hourly decisions |
   |---|---|---|
   | 12.4 h (the break-even) | 0.9933, **0.045 nats** | 0.9194, **0.336 nats** |
   | 24 h | 0.9965, 0.026 nats | 0.9583, 0.202 nats |
   | 48 h | 0.9983, 0.014 nats | 0.9792, 0.116 nats |

   At five-minute decisions the policy must be seven to eight times lower in entropy, against the entropy
   bonus, to express the same hold, and at that entropy entries from flat are rarely sampled.
2. **MEASURED — five-minute decisions lost their entries** (§11.3): in all three five-minute latent runs
   entropy fell to 0.05–0.10 and entries per rollout from 800–1,000 to 5–64; one run ended permanently
   flat. The hourly runs held entropy at 0.18–0.34 with 250–570 entries per rollout.
3. **The price.** A coarser policy cannot react inside its interval (Kalyanakrishnan et al. 2021). Margin
   is still checked every minute, and a 1× long needs a 49.7% fall to be liquidated. The five-minute data
   are kept for what they give — trend and volatility measured on fine bars.

### 4.2 State — the exogenous market

The market is **exogenous**: a one-contract trader does not move the XBTUSD price. Dietterich, Trimponias
& Chen (ICML 2018) show exogenous components *"can slow down reinforcement learning by injecting
uncontrolled variation into the reward signal"*. Market features go through the recurrent memory; the
account does not (§6). The memory takes one step per decision, reading the features of the latest
five-minute boundary.

**Regime, not prediction.** Next-bar and next-day returns are close to unpredictable, and what
predictability exists is non-stationary. Markets do show persistent regimes — Hamilton's (1989)
Markov-switching model, and for Bitcoin, regime changes in volatility dynamics (Ardia, Bluteau & Rüede
2019). The state therefore describes the regime. Deng et al. (2017) was the reference: they feed raw
price changes over the last 45 minutes and momentum over 3 and 5 hours and 1, 3 and 10 days through fuzzy
memberships for *"increasing, decreasing, and the no trend groups"*; here each regime property is carried
by one variable, and the network's first layer forms such groups itself.

**How the six were chosen** (`scripts/regime_state_analytics.py`, results in
`results/state_analytics/regime_state_analytics.json`). Thirteen candidates were computed on the
five-minute grid from bars before each boundary. On the training years 2017–2023 only, with daily samples
at 00:00 UTC (2,556 days), each was measured against regime outcomes of the following day or week. 2024
and 2025 were read only for the scale of each feature by year.

| feature | construction at boundary *t* | regime property | source |
|---|---|---|---|
| `trend_efficiency_1w` | (log close_t − log close_{t−1 week}) / Σ\|five-minute log returns\| over that week; in [−1, 1] by construction | direction and orderliness of the prevailing trend | Kaufman (1995), efficiency ratio |
| `drawdown_4w` | (log close_t − max log close over 4 weeks) / four-week daily σ | depth below the recent peak | Pagan & Sossounov (2003); Lunde & Timmermann (2004) |
| `volatility_level` | log of the four-week daily σ | the volatility regime, and a day's move against a fixed 12 bp cost | Corsi (2009), monthly HAR component; Liu, Patton & Sheppard (2015) |
| `volatility_regime` | ½ log(mean squared five-minute return over the last day / over four weeks) | whether volatility is rising or falling now | Corsi (2009) |
| `volume_regime` | log(1 + daily volume last week) − log(1 + daily volume over the four weeks before) | attention and participation | Barber & Odean (2008); Liu & Tsyvinski (2021) |
| `funding_rate` | last funding rate settled strictly before *t* | the carry an open position pays, and the crowding of the side that pays it | He, Manela, Ross & von Wachter (2024); Koijen et al. (2018) |

Each feature is standardised by the mean and standard deviation of its five-minute samples over the fold's
training span and clipped at ±10.

**Relevance** — Spearman rank correlation with the outcome, 2017–2023:

| feature | next-day log volatility | next-day \|efficiency\| | next-week variance ratio | next-day return z | next-week return z | next-day trend payoff |
|---|---:|---:|---:|---:|---:|---:|
| `volatility_level` | **0.625** | −0.031 | −0.067 | −0.005 | −0.013 | −0.028 |
| `volatility_regime` | **0.376** | **−0.151** | −0.155 | 0.030 | 0.035 | 0.003 |
| `volume_regime` | **0.362** | −0.092 | **−0.211** | 0.034 | 0.069 | 0.014 |
| `drawdown_4w` | 0.008 | 0.002 | −0.186 | 0.029 | **0.084** | **0.073** |
| `trend_efficiency_1w` | −0.027 | −0.005 | −0.119 | 0.030 | 0.045 | **0.071** |
| `funding_rate` | 0.006 | 0.023 | −0.099 | 0.010 | −0.044 | 0.047 |

The volatility and volume regimes forecast the next day's volatility in every one of the seven years.
Nothing forecasts the next day's return (all |ρ| ≤ 0.05).

**Each feature adds information the others do not** — drop-one loss of R² in a linear model on
standardised ranks:

| outcome (R² with all six) | largest unique contributions |
|---|---|
| next-day log volatility (0.583) | `volatility_level` 0.404, `volatility_regime` 0.079, `volume_regime` 0.014 |
| next-week variance ratio (0.077) | `drawdown_4w` 0.012, `volume_regime` 0.011, `volatility_regime` 0.006, `volatility_level` 0.005 |
| next-week return z (0.018) | `drawdown_4w` 0.007, `funding_rate` 0.007, `volume_regime` 0.002 |
| next-day \|efficiency\| (0.026) | `volatility_regime` 0.016, `volatility_level` 0.002 |
| next-day trend payoff (0.007) | `drawdown_4w` 0.0008, `trend_efficiency_1w` 0.0007, `volatility_level` 0.0006 |

**The trend state is non-linear.** Annualised Sharpe ratio of the next day's payoff by quintile
(2017–2023; *trend* = hold the sign of the weekly trend, *long* = hold long):

| conditioning | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---:|---:|---:|---:|---:|
| \|`trend_efficiency_1w`\| → trend | −1.13 | 0.66 | 0.15 | 0.91 | **1.95** |
| `trend_efficiency_1w` → long | 0.03 | −0.08 | 0.33 | 0.54 | **2.90** |
| `drawdown_4w` (Q5 = at the high) → trend | −0.31 | −0.69 | 0.74 | 0.69 | **2.73** |
| `volatility_level` → trend | **1.45** | 1.10 | 0.96 | −0.04 | 0.07 |
| `volume_regime` → trend | −0.24 | 0.54 | 1.19 | 1.37 | −0.16 |
| `funding_rate` → long | **1.76** | −0.56 | −0.18 | 0.90 | 1.21 |

These are in-sample regime descriptions on the training years, not evidence of a tradable edge after
costs. Rank correlations among the six are at most 0.68. Every feature except `volatility_level` is
invariant to the level of volatility and volume (tested); `volatility_level` is meant to move with it.
Boundaries lie on UTC multiples of five minutes, a minute between boundaries carries the last boundary's
observation, and a funding settlement at minute *m* becomes visible at the first boundary after *m* (all
tested).

### 4.3 What was tested and excluded

| candidate | measured reason for exclusion |
|---|---|
| single-bar return, range, volume, coverage | patterns inside a bar are not the target (USER); no regime content |
| `hour_sin/cos`, `funding_phase_sin/cos` | clock and settlement-cycle position; no regime content (USER) |
| `momentum_1w` | rank correlation **0.99** with `trend_efficiency_1w` |
| one-day trend efficiency | its only increment is next-day reversal (+0.004 R², ρ = −0.046): a return forecast, not a regime |
| four-week trend efficiency, `rally_4w` | rank correlations **0.81–0.84** with `drawdown_4w` |
| weekly variance ratio (Lo & MacKinlay 1988) | pooled it adds 0.033 R² to next week's variance ratio, but its persistence **changes sign across years** (0.44, 0.09, −0.06, 0.25, 0.15, −0.14, 0.01) — a between-year difference, not a persistent regime property (Song et al., ICLR 2020) |
| one-day variance ratio, log daily volatility | ≤ 0.005 R²; the latter is the sum of level and regime |

### 4.4 State — the endogenous account

| feature | why it is there |
|---|---|
| `short`, `flat`, `long` one-hot | the cost of each action depends on the side held |
| `trade_return` — open trade's log return / four-week daily σ | how far the trade has run |
| `trade_age` — log(1 + bars held / credit window in bars) | how long it has run: zero at entry, log 2 after one credit window (291.5 bars = 24.3 h) |
| `exposure` — log(1 + \|contracts\| / price / equity) | **liquidation happens exactly when this exceeds log(1 + 1/(maintenance + fee)) = 5.21** (tested) |
| `running_ratio`, `relative_risk` | the reward is a function of its two statistics (§4.5) |

All eight are clipped at ±10. Side, exposure and the reward statistics make reward and termination Markov
given the market (Subramanian, Sinha, Seraj & Mahajan, JMLR 2022). `trade_return` and `trade_age` are
there because the regime is not observed: bull and bear phases show duration dependence (Lunde &
Timmermann 2004), and the recurrent memory — rebuilt over a 64-decision burn-in (2.7 days) before
128-decision sequences (5.3 days) — cannot count a hold longer than about 8 days (DERIVED).

### 4.5 Reward — exact increments of the Sharpe or downside deviation ratio

Moody & Saffell (IEEE TNN 2001) track exponential estimates of the mean return `A` and a risk second
moment `B`:

```
A_t = A_{t-1} + η (R_t − A_{t-1})
sharpe:   B_t = B_{t-1} + η (R_t² − B_{t-1})            S = A / sqrt(B − A²)
sortino:  B_t = B_{t-1} + η (min(R_t, 0)² − B_{t-1})    S = A / sqrt(B)
```

and obtain the differential Sharpe and differential downside deviation ratios by *"expanding to first
order in the adaptation rate"*; the downside deviation ratio has *"no penalty for large positive
returns"* (Sortino & van der Meer 1991). The reward uses the **exact** change:

```
per bar b:       D_b = (S(A_b, B_b) − S(A_{b-1}, B_{b-1})) / η
per decision:    reward = scale × Σ_{b in the decision's 12 bars} D_b
```

`R_b` is the net log BTC equity change over the bar, including commission, slippage and funding;
statistics of a liquidated account freeze. The sum telescopes to the change in the ratio, and a 12-bar
decision reproduces 12 one-bar decisions exactly (both tested). A fresh account starts with A = 0 and B =
the fold's measured bar variance (half of it for the downside ratio).

| one-bar return (Sharpe, prior σ = 26.55 bp) | exact increment | first-order term |
|---|---:|---:|
| +1 sd | +1.00 | +1.0 |
| −3 sd | −2.99 | −3.0 |
| −20 sd (−5.1%) | **−17.70** | −20.0 |
| −50% | **−37.58** | −264.6 |
| wipeout | **−37.96** | −10,546 |

The Sharpe increment is bounded by 1/sqrt(η(1−η)) = 37.96 and the downside increment from a flat start
below by −1/sqrt(η) = 37.95 (tested). A gain is unbounded under the downside ratio by definition and
limited in practice by the largest bar return. Termination carries ruin (§4.6), because both ratios are
invariant to wealth.

| constant | value | provenance |
|---|---|---|
| `reward` | `sharpe` | USER; `sortino` is the user's comparison |
| `dsr_eta` | 1/1440 per bar, a five-day window | DESIGN CHOICE inside DERIVED bounds: slower than the 24.3 h credit window, faster than a 90.7-day episode |
| `dsr_initial_second_moment` | (26.55 bp)² in fold 2024, (25.54 bp)² in fold 2025 | MEASURED per fold on the training span; the downside second moment starts at half (DERIVED for symmetric returns) |
| scale | sqrt((1 − γ^24) / 12) = 0.0586; × 1/√2 = 0.0414 for `sortino` | DERIVED: the discounted sum of a unit-variance sequence has sd 1/sqrt(1 − γ_d²) with γ_d = γ^12, and a decision sums 12 increments; the downside increment is √2 times the Sharpe increment for symmetric returns. The scaled spreads of the two rewards agree within 5% (tested) |

With `R_b = 0` the increment has the sign opposite to `A`, so an agent that has been losing is paid to
stay flat — correct ratio accounting. A ratio does not reward leverage resets (tested for both).

### 4.6 Transitions, termination and time limits

- **Liquidation terminates** the episode at the minute it happens: no reward and no bootstrap afterwards.
  BitMEX XBTUSD base maintenance margin was 0.50% until 17 April 2020 and 0.45% afterwards (BitMEX, 13
  April 2020); the higher value is used throughout. The account stays in the rollout as invalid steps and
  is reset at the next rollout.
- **Episode ends are time limits**, bootstrapped with the critic (Pardo et al., ICML 2018); GAE masks the
  bootstrap only on liquidation (tested).
- Adverse one-minute extremes trigger the margin check — a conservative approximation, not BitMEX's
  engine. Funding alone can breach margin before an order fills, and is checked first.

---

## 5. Horizon

| | decisions | time | provenance |
|---|---:|---|---|
| cost break-even | 12.2 / 12.7 | 12.2 h (fold 2024) / 12.7 h (fold 2025) | DERIVED per fold from §2 |
| GAE credit window `1/(1−γ^12 λ^12)` | 24.3 | 24.3 h | USER: γ = λ = 0.99825 per bar; about twice the break-even |
| discount horizon `1/(1−γ^12)` | 48.1 | 48.1 h | DERIVED |
| `burn_in` | 64 | 2.7 days | CITED: R2D2 ratio 0.5 (Kapturowski et al. 2019) |
| `sequence` | 128 | 5.3 days | CITED: CleanRL trains each environment's whole rollout as one sequence; DERIVED bound `burn_in + sequence ≥ credit window`, cleared 7.9 times |
| `rollout` | 128 | 5.3 days | CITED: PPO Table 5 *"Horizon (T) 128"*; CleanRL `num_steps = 128`; DERIVED bound `rollout / 2 ≥ credit window`, cleared 2.6 times |
| episode | 2,176 | 90.7 days | USER 2–3 months; a quarter to the nearest whole rollout |

The user fixed γ = λ = 0.99825 per bar when the break-even was believed to be 22 h (§2, correction). On
the training years it is 12.2–12.7 h, so the credit window is about twice the break-even and still clears
it in both folds.

Andrychowicz et al. recommend *"GAE with λ = 0.9"* for MuJoCo; that horizon is far below the break-even,
which is the failure that killed v5 (entropy fell in 78 of 78 runs at a 2.9 h window).

---

## 6. Network

```
market features (6) ─► Linear ─► LayerNorm ─► tanh ─► LSTM ─► LayerNorm ─┐
                                                                           ├─► Linear ─► LayerNorm ─► tanh ─► Linear ─► LayerNorm ─► tanh ─► Linear
account (8) ───────────────────────────────────────────────────────────────┘                                                              (3 logits | 1 value)
```

Separate actor and critic of this shape, one recurrent step per hourly decision. Widths 128 (actor) and
256 (critic). **832,516 parameters** (actor 168,451; critic 664,065).

- **Layer normalisation after every hidden layer.** Lyle et al. (ICML 2023, `articles/deep
  RL/lyle2023.pdf`) find normalisation layers give *"the greatest improvements to plasticity"*. Gain and
  bias act *"after the normalization but before the non-linearity"* (Ba, Kiros & Hinton 2016). The LSTM
  is normalised at its output; the in-cell LN-LSTM MEASURED 13.3× slower than the fused kernel. Juliani &
  Ash (NeurIPS 2024) find LayerNorm resolves on-policy plasticity loss in training but is inconsistent for
  generalisation.
- **Widths.** 128 is CleanRL `ppo_atari_lstm`'s LSTM. Two hidden layers follow it: *"Two hidden layers
  appear to work well for policy (Fig. 22) and value networks (Fig. 20) in all tested environments"*
  (Andrychowicz et al.); their width equal to the LSTM's is a DESIGN CHOICE — the same study finds *"the
  optimal width of the policy MLP depends on the complexity of the environment"*. The critic is twice as
  wide: *"for the value function there seems to be no downside in using wider networks"* (ibid.).
- **The account does not enter the recurrent memory** (tested), so one market pass serves every sampling
  seed in evaluation.
- Recurrence rather than stacked observations follows Hausknecht & Stone (2015).
- Separate policy and value networks, tanh, orthogonal initialisation (gain √2 on hidden layers, 1 on the
  LSTM and value output) and a final policy layer 100× smaller follow Andrychowicz et al. (*"Initialize
  the last policy layer with 100× smaller weights"*) and CleanRL; the initial policy is uniform within
  0.02 (tested). The forget-gate bias of 1 follows Jozefowicz, Zaremba & Sutskever (2015).

---

## 7. Optimisation — recurrent PPO

### 7.1 Why PPO, and what is known to go wrong with it

The user chose PPO. Its known failure modes on this problem, and what the design does about each:

| failure mode | evidence | here |
|---|---|---|
| the clip converges to a suboptimal discrete action | Hsu, Mendler-Dünner & Hardt (2020): *"On discrete action spaces with sparse high rewards, standard PPO often gets stuck at suboptimal actions"*; KL-regularised PPO *"enjoys convergence guarantees when the parameterized policy class is closed under mixture"*, which *"are not preserved when using the standard clipping heuristic"* | three actions; `state_action_mutual_information` and `long_entry_share` show a lock every update. A reverse-KL penalty surrogate is the named alternative; not adopted, because its coefficient has no source for this problem |
| a near-deterministic softmax stops sampling the other actions | Mei et al. (ICML 2020; NeurIPS 2020); MEASURED in §1.4 and at five-minute decisions in §11.3 | hourly decisions (§4.1); entropy bonus 0.01; exploring starts |
| committal convergence under sampling | Mei et al. (NeurIPS 2021) | the same, plus side-share and entry diagnostics |
| overfitting a repeated history | Sato (2019): *"repeated training using the same historical dataset would end up capturing random patterns that exist only in that particular set of samples"*; Zhang, Vinyals, Munos & Bengio (2018): overfitting happens robustly and stochasticity does not prevent it; MEASURED in §11.3 | a budget in passes with checkpoints selected on validation (§7.3); PBO (§9) |
| a weak edge below the advantage noise | MEASURED in §1.4 at 0.009 sd per bar | a persistent regime and trade state; an hourly decision accumulates twelve bars of any edge. Not solved in general |
| stale or misaligned recurrent state | — | states rebuilt over the burn-in prefix; the rollout is replayed before every update and must match (tested; MEASURED error 0.0) |

### 7.2 The update

For each update:

1. **Collect** 128 hourly decisions in each of 8 environments (1,024 samples), actions sampled, starting
   from recurrent states burnt in over the stored 64-decision prefix. An environment whose episode reached
   2,176 decisions, or whose account was liquidated, is reset first.
2. **Replay** the rollout under the current parameters; stop if probabilities, values or the bootstrap
   differ from collection by more than 5e-5.
3. For each of 4 epochs, **recompute GAE** advantages and λ-returns with the current critic (γ^12, λ^12
   per decision; liquidation masks the bootstrap, time limits do not).
4. The 8 environments, in random order, form 4 minibatches of 2 (256 samples). For each: rebuild the
   recurrent state over the 64-decision prefix without gradient, then
   - normalise advantages over the minibatch's valid steps;
   - actor: minimise −min(ρÂ, clip(ρ, 0.9, 1.1)Â) − 0.01 H(π), clip the gradient norm at 0.5, Adam step;
   - critic: minimise ½(V − return)², clip the gradient norm at 0.5, Adam step.
5. Learning rates decay linearly to zero over the fold's updates; the prefix rolls forward.

16 actor and 16 critic gradient steps per update: 95,856 of each in fold 2024 and 109,584 in fold 2025.

### 7.3 The training budget — passes over a finite history

The previous 5,000 updates matched CleanRL's Atari gradient-step budget (10M frames). A simulator produces
fresh experience for every update; a trading history does not. The trading RL studies instead budget
epochs over the fixed history and decide when to stop on held-out data:

- **Moody & Saffell (2001)**: RRL systems *"use the first ten years for stochastic optimization of system
  parameters, and the subsequent ten years for validating early stopping of training"*; on simulated data
  they *"are first optimized on the training data set for 100 epochs"*.
- **Deng et al. (2017)**: *"we adopt two convenient strategies … The first strategy is the widely used
  early stopping method for DNN training. The system is only trained for 100 epochs"*, with a validation
  set of the last 3,000 of 15,000 training points for model selection.
- **Théate & Ernst (2021)**: *"a tendency to overfit was observed"*, answered with *"Dropout, L2
  regularisation and Early Stopping"*.
- **Prechelt (1998)**, *Early Stopping — But When?*: validation is measured *"only at the end of each
  strip"*, with *"strips of length 5"* epochs; slower stopping criteria buy about 4% better generalisation
  for about four times the training.

So the budget is **`training_passes` = 100** passes over the fold's training span, where a pass collects
as many decisions as the span holds, and a checkpoint is validated every **`validation_passes` = 5**
passes. Choosing the best checkpoint on the validation quarters is the validation stopping of Moody &
Saffell and Deng, made across all 20 checkpoints rather than by halting training; its cost is measured by
the probability of backtest overfitting (§9).

| fold | training decisions | updates = ⌈100 × decisions / 1,024⌉ | checkpoints | gradient steps |
|---|---:|---:|---:|---:|
| 2024 (2017–2023) | 61,344 | 5,991 | 20, every ~300 updates | 95,856 |
| 2025 (2017–2024) | 70,128 | 6,849 | 20, every ~342 updates | 109,584 |

Checkpoints lie at evenly spaced fractions of each fold's updates, so the k-th checkpoints of both folds
sit after the same number of passes and can be scored together.

**What the passes do not settle.** A PPO pass trains each collected sample for 4 epochs, so 100 passes are
not identical to 100 supervised epochs; and where held-out return peaks depends on the signal. Both are
measured in §11.4, which verifies the whole schedule on a synthetic market with a known optimum, including
where the validation and test curves peak.

### 7.4 Parameters

| parameter | value | provenance |
|---|---|---|
| `environments` | 8 | CITED: PPO Table 5 *"Number of actors 8"*; CleanRL `num_envs = 8` |
| `rollout` | 128 | CITED: PPO Table 5 *"Horizon (T) 128"*; CleanRL `num_steps = 128` |
| `sequence` | 128 | CITED: CleanRL trains each environment's whole rollout |
| `burn_in` | 64 | CITED: R2D2's 40-step burn-in before 80-step sequences |
| `minibatch_sequences` | 2 | DERIVED: CleanRL `num_minibatches = 4` over 8 environments, 256 samples = PPO Table 5 *"Minibatch size 32×8"* |
| `epochs` | 4 | CITED: CleanRL `update_epochs = 4` (PPO Table 5: 3) |
| `clip` | 0.1 | CITED: PPO Table 5 *"Clipping parameter ϵ 0.1×α"*; CleanRL `clip_coef = 0.1`. Andrychowicz et al.'s *"Start with the clipping threshold set to 0.25"* comes from continuous control, where they also found *"lower (ϵ = 0.1) … values give better performance on some of the environments"* |
| `actor_lr`, `critic_lr` | 2.5e-4 | CITED: PPO Table 5 *"Adam stepsize 2.5×10−4×α"*; CleanRL `learning_rate = 2.5e-4` |
| `anneal_lr` | linear to zero | CITED: PPO Table 5 *"α is linearly annealed from 1 to 0"*; CleanRL `anneal_lr = True` |
| `gradient_clip` | 0.5 | CITED: CleanRL `max_grad_norm = 0.5`; Andrychowicz et al.: *"Gradient clipping might slightly help but is of secondary importance"* |
| `entropy_coefficient` | 0.01 | CITED: PPO Table 5 *"Entropy coeff. 0.01"*; CleanRL `ent_coef = 0.01` |
| `recompute_advantages_per_pass` | on | CITED: Andrychowicz et al.: *"recompute advantages once per data pass"* |
| advantage normalisation | per minibatch | CITED: CleanRL `norm_adv = True` |
| value loss | MSE, no clipping | CITED: Andrychowicz et al.: *"Use GAE with λ = 0.9 but neither Huber loss nor PPO-style value loss clipping"* (CleanRL's default clips; the study is followed) |
| KL early stop | none | CITED: CleanRL `target_kl = None` |
| `training_passes` | 100 | CITED: Moody & Saffell 2001; Deng et al. 2017 (§7.3) |
| `validation_passes` | 5 | CITED: Prechelt 1998, strips of length 5 |

Andrychowicz et al. also advise *"Tune the number of transitions in each iteration (C2) if possible"*; it
has not been tuned here, and the reference value is used instead.

---

## 8. Training data

**Episodes are 2,176 hourly decisions, 90.7 days** — the whole number of rollouts nearest the quarterly
window of FinRL's ensemble strategy (Yang et al. 2020; `quarter_episode`), inside the user's 2–3 month
range — starting at uniformly random five-minute boundaries of the training span. The first episode of
each environment is shortened by a random whole number of rollouts (0 to 16), so the eight environments do
not reset together (Mnih et al., A3C, ICML 2016).

**Exploring starts** (Sutton & Barto 2018, §5.3): a 64-decision prefix (2.7 days) with real fees, funding
and entry prices builds the starting position and doubles as the recurrent burn-in; one third of accounts
stay flat and the rest enter a uniformly chosen side at a uniformly chosen hour of the prefix. A prefix
liquidation starts a fresh flat account. Sides are balanced at initialisation (tested).

**Data reuse** is set by the budget: 100 passes over each fold's span (§7.3). What addresses year-to-year
magnitude is the scale-free state (§4.2), not the window length.

---

## 9. Evaluation and selection

- **Actions are the most probable action** (`evaluation_policy` = `argmax`, USER 2026-09-15): one
  deterministic path per evaluation. **Every result up to 2026-09-13** — §11.5–§11.7, the charts, and both
  test looks — used **sampled actions** over `evaluation_seeds` = 4. MEASURED at hourly decisions under that
  protocol: one evaluation's standard deviation across sampling seeds was 0.23 log over 51 weeks, so a
  4-seed mean has a standard error of 0.12, below the 0.26–0.63 differences between training seeds.
  An argmax evaluation removes that sampling noise, but the policy it evaluates is not the one PPO
  optimised: training maximised return under sampled actions with an entropy bonus.
- **Each calendar quarter of the validation year is scored from a flat account**, decisions on the hour
  (2,160–2,208 per quarter), after a 64-decision burn-in; the memory is rebuilt every 128 decisions exactly
  as in training. Four quarters × two folds × five training seeds = 40 scores per checkpoint.
- **Checkpoint selection**: the checkpoint index with the highest **interquartile mean** of quarterly log
  return, with a **stratified bootstrap 95% interval**, 50,000 replications (Agarwal et al., NeurIPS 2021).
- **Probability of backtest overfitting** by CSCV over the eight quarters and 20 checkpoints (Bailey,
  Borwein, López de Prado & Zhu 2017); the same test Gort et al. (2022) use to reject overfitted DRL
  cryptocurrency agents.
- **Break-even slippage**: for the selected checkpoint, the extra slippage per leg at which each evaluated
  quarter's return would reach zero, return / turnover (first order; the identity is tested). It replaces a
  grid of stress levels that had no source.
- **Passive references** (constant long, short, flat) per quarter, never used for selection.
- **The Sharpe reward only** (USER): training, validation and selection use the differential Sharpe ratio, in
  `results/agent/sharpe_bars12`. The downside-deviation variant remains implemented and verified (§11.4).

---

## 10. Diagnostics

Logged every update:

| metric | reads |
|---|---|
| `account_side_share`, `long_entry_share`, `sampled_action_share` | side concentration and the realised direction of entries from flat |
| `state_action_mutual_information[_flat]` | zero exactly when the policy ignores the state — a lock |
| `policy_entropy`, `flat_long_probability`, `flat_short_probability`, `entry_decisions` | whether entries are still being explored |
| `exit_decisions`, `reversal_decisions`, `mean_age_at_exit_hours`, `policy_implied_holding_hours` | how long it holds and when it exits |
| `approx_kl`, `clip_fraction`, `post_update_kl` | the trust region |
| `explained_variance_pre/post`, `target_std`, `value_rmse_{short,flat,long}` | critic tracking, per side |
| `liquidations_in_rollout`, `mean_net_log_return`, `turnover_per_decision` | ruin, net return, trading intensity |
| `replay_probability_error`, `replay_value_error`, `replay_bootstrap_error` | exactness of the recurrent replay |
| `passes` | how many passes over the training span the run has made |

---

## 11. Verification before compute

### 11.1 Unit tests

`tests/test_agent.py`, **86 tests, all passing**. Mechanics are tested at one bar per decision; the
production configuration has its own tests. Among them:

- **specification**: hourly decisions with the horizon preserved; the credit window clears the break-even
  measured at training-year volatility; the predecessor horizon is refused; the update shape equals PPO
  Table 5 / CleanRL; the budget is 100 passes of each fold's span with 20 aligned checkpoints; the reward
  prior is never a constant; `Scaler.fit` measures volatility on the training span only; lower bounds on
  rollout, recurrent span and reward window; reward scale; the provenance check against §12;
- **features**: causality at the five-minute boundary and for funding settlements; scale invariance;
  bounded trend efficiency; no clock, cycle or single-bar features;
- **reward**, both ratios: reduction to Moody & Saffell's first-order terms, bounds on catastrophic losses,
  telescoping, scale invariance, no gain from leverage, frozen statistics after termination; the downside
  ratio's indifference to upside dispersion; equal scaled spread;
- **ledger**: a 12-bar decision reproduces 12 one-bar decisions exactly; extra slippage costs turnover ×
  slippage; trade age; the liquidation threshold of the exposure input; exploring starts;
- **network and PPO**: LayerNorm placement; near-uniform initial policy; account-free recurrence; the
  clipped surrogate stops outside the trust region; PPO learns the side of a strong drift; a locked policy
  shows no state–action information; replay and resume exactness; GAE terminations versus time limits;
- **evaluation, scoring and synthetic markets**: sampled multi-seed evaluation; break-even slippage; IQM,
  bootstrap and PBO on known cases; hidden latent regimes leave a trace in the trend state.

### 11.2 Pipeline checks on real data — MEASURED 2026-09-11

| check | result |
|---|---|
| cache rebuild (4,785,118 minutes, 2016-11-26 to 2026-01-01 exclusive) | 4.9 s |
| cached features against an independent pandas implementation, 3,256 daily boundaries | max \|difference\| ≤ 9.5e-7, correlation 1.000000 |
| **revised configuration**: `python -m agent.train`, fold 2024, 1 pass (60 updates), checkpoint and quarterly validation | completed in 296 s; **1.68 s per update** with five verification runs on the host; replay errors 0.0; fold record: 61,344 training decisions, σ_bar 26.55 bp, break-even 12.21 h, credit window 24.29 h |
| first configuration (rollout 1,024): five-minute, Sortino and hourly command lines | completed; 5.3–12.9 s per update; replay errors 0.0 |

A full run takes about 6,000–6,850 updates at 1.7 s plus 20 validations of about 3 minutes: about four
hours per worker, so one reward's six runs in parallel take roughly four to five hours.

### 11.3 Smoke tests of the first hourly configuration — the record behind the revision

Configuration: rollout 1,024, sequence 256, burn-in 128, clip 0.25, learning rate 3e-4, 1,000 updates,
the break-even and reward prior from the old 14.8 bp constant. Markets (`src/agent/synthetic.py`):

| market | construction |
|---|---|
| observable | a 48 h hidden regime written into `trend_efficiency_1w`, oracle Sharpe 30 |
| latent | Markov-switching bear / range / bull regimes (Hamilton 1989), mean dwell 10 days, drift ±0.5 daily σ in trends, daily σ 4.5% / 2.5% / 3.0%, volume 1.3× / 0.8× / 1.0×, zero-mean funding noise; four years of minute bars passed through the real feature pipeline; the regime never shown |

The latent calibration before training (vectorised, held-out year, 6 bp per leg): oracle Sharpe 10.06, the
sign of `trend_efficiency_1w` 5.57, constant long −2.08. The pre-registered rule: the mean of the last
three evaluations (every 100 updates) beats the best constant, with regime conditioning.

| run | pass bar | evaluations @100 … @1000 | last three | verdict |
|---|---:|---|---:|---|
| observable, five-minute | +0.871 | −2.78, −1.13, +0.31 (300 updates) | −1.20 | FAIL, conditioned |
| observable, hourly | +0.872 | +3.31, +3.32, +3.36 (300 updates) | **+3.33** | **PASS**, 99% of the oracle |
| latent Sharpe, five-minute, s11 | +1.467 | −5.52, −0.18, +0.38, −0.05 (flat), −0.16, −0.36, +0.99, +1.22, +1.47, +1.09 | +1.26 | FAIL |
| latent Sharpe, five-minute, s29 | +1.467 | −4.64, −0.71, −0.15, +1.03, +0.87, −0.26, −0.21, +0.16, +0.33, +0.40 | +0.30 | FAIL |
| latent Sortino, five-minute, s11 | +1.467 | −6.26, −3.20, −2.50, −2.91, +0.78, +0.21, +0.55, +0.20, +1.37, +0.68 | +0.75 | FAIL |
| latent Sharpe, hourly, s11 | +1.467 | **+2.39**, +1.33, +1.14, +0.52, +1.03, +1.14, +1.26, +0.66, +0.41, +0.67 | +0.58 | FAIL |
| latent Sharpe, hourly, s29 | +1.467 | **+2.13**, +0.70, +1.21, +1.14, +0.80, +1.18, +0.60, +0.08, +0.18, +0.30 | +0.19 | FAIL |
| latent Sortino, hourly, s11 | +1.467 | +1.78, **+1.92**, +0.65, +0.15, −0.16, +0.63, −0.40, −0.72, … | −0.72 | FAIL |

**Overfitting at hourly decisions — MEASURED.** Training return per year rose while held-out return fell
(passes over the three-year training span):

| run | updates | passes | training net log return per year | held-out return |
|---|---|---:|---:|---:|
| latent Sharpe, hourly, s11 | 51–100 | 16–31 | +3.90 | +2.39 @100 |
| | 151–200 | 47–62 | +5.94 | +1.33 @200 |
| | 251–300 | 78–94 | +7.85 | +1.14 @300 |
| | 301–321 | 94–100 | **+8.82** | |
| latent Sharpe, hourly, s29 | 51–100 | 16–31 | +2.92 | +2.13 @100 |
| | 151–200 | 47–62 | +4.73 | +0.70 @200 |
| observable, hourly | 151–200 | — | +12.78, flat since update 50 | +3.32, stable |

By update 300 the hourly policy earned more per year on its training span than the oracle earns out of
sample (+5.45): only a policy fitted to that span's path can. **Exploration of entries** — the mechanism
behind the decision interval (training rollouts):

| run | update | policy entropy | flat share | entries per rollout | P(long ∣ flat) | P(short ∣ flat) |
|---|---:|---:|---:|---:|---:|---:|
| latent Sharpe, five-minute, s11 | 50 / 200 / 300 / 407 | 0.493 / 0.074 / 0.053 / **0.001** | 0.19 / 0.22 / 0.83 / **1.00** | 1,007 / 71 / 33 / **2** | 0.44 → 0.0002 | 0.25 → 0.0000 |
| latent Sharpe, five-minute, s29 | 50 / 400 | 0.514 / 0.081 | 0.16 / 0.48 | 855 / 52 | 0.33 / 0.010 | 0.33 / 0.008 |
| latent Sortino, five-minute, s11 | 50 / 400 | 0.452 / 0.098 | 0.47 / 0.52 | 814 / 64 | 0.14 / 0.008 | 0.06 / 0.006 |
| latent Sharpe, hourly, s11 | 50 / 250 | 0.340 / **0.241** | 0.12 / 0.22 | 570 / **342** | 0.26 / 0.13 | 0.30 / 0.10 |
| latent Sharpe, hourly, s29 | 50 / 128 | 0.320 / **0.230** | 0.10 / 0.41 | 451 / **312** | 0.24 / 0.04 | 0.33 / 0.07 |

**What the record established.** Five-minute decisions lost their entries; hourly decisions recognised the
regimes and held positions for hours to days, cleared the bar at their early checkpoints, and then overfit
under a budget of many passes — so a last-checkpoint rule failed them. That is what the revision addresses:
a budget in passes with selection on validation, and a verification that judges the selected checkpoint on
data it was not selected on.

### 11.4 Verification of the revised configuration — pre-registered

`scripts/planted_edge_test.py`, `scripts/run_verification.sh`, `scripts/verification_summary.py`. The
production configuration and schedule, run as the real experiment runs:

- **latent market** (as §11.3, market seed 7): training on the first three years (25,439 hourly decisions;
  σ_bar measured on it: 19.91 bp), **validation** on the next 26 weeks (2015-01-07 to 2015-07-08),
  **test** on the last 25 weeks (2015-07-08 to 2015-12-31);
- **100 passes** (2,485 updates) with a checkpoint every **5 passes** (20 checkpoints), each scored on
  both windows with two sampling seeds;
- the checkpoint with the best **validation** return is selected and judged on the **test** window.

References through the full ledger (log return):

| window | oracle | trend rule | constant long | constant short |
|---|---:|---:|---:|---:|
| validation | +3.326 | +2.135 | −2.875 | +0.664 |
| test | +2.121 | +1.394 | −4.841 | +0.802 |

**Pass rule, set before any result:** the selected checkpoint's test return beats the best of flat,
constant long and constant short on the test window (**+0.802**), **and** on the test window it is long
more often in bull than in bear regimes and short more often in bear than in bull regimes.

| run | purpose |
|---|---|
| Sharpe, seeds 11 and 29 | the production configuration |
| Sortino, seeds 11 and 29 | the reward comparison |
| Sharpe, seed 47, a checkpoint every 2 passes (50) | measurement only: where validation and test return peak, at finer resolution; not part of the verdict |

**Results — 2026-09-11** (log returns on the 25-week test window; +0.802 is +123%):

| run | selected checkpoint (best validation) | its test return | best test at any checkpoint | checkpoints above the bar | median test after pass 30 | corr(validation, test) | verdict |
|---|---|---:|---|---:|---:|---:|---|
| Sharpe, seed 11 | pass 5 (validation +1.63) | +0.75 | +1.23 at pass 40 | 6 / 20 | +0.76 | +0.59 | FAIL, conditioned |
| Sharpe, seed 29 | pass 60 (validation +2.03) | +0.77 | +1.18 at pass 40 | 8 / 20 | +0.59 | +0.38 | FAIL, conditioned |
| Sortino, seed 11 | pass 5 (validation +1.45) | +0.44 | +0.89 at pass 75 | 1 / 20 | +0.39 | +0.19 | FAIL, conditioned |
| Sortino, seed 29 | pass 95 (validation +1.36) | **+1.33** | +1.48 at pass 70 | 15 / 20 | +0.99 | +0.81 | **PASS**, 95% of the trend rule |

Across the 80 checkpoints the validation and test returns correlate +0.43.

**Findings.**

1. **The agent learns the hidden regimes under the production schedule.** At its selected checkpoint every
   run was long more often in bull than bear regimes and short more often in bear than bull on the test
   window, held positions for 6–8 hours, and earned a positive test return (+0.44 to +1.33, 31–95% of the
   causal trend rule). One of four cleared the pre-registered bar; two missed it by 0.03 and 0.05.
2. **The learning curve.** Fast learning to about pass 5; a noisy dip between passes 10 and 25, in which
   conditioning was lost and regained; recovery and a plateau from about pass 30; flat to mildly declining
   afterwards. No run ended unconditioned after 100 passes, and the monotone overfitting of the first
   configuration (§11.3) did not recur. The probe (every 2 passes, measurement only) peaked at +1.47 at pass
   14 and +1.32 at pass 40, with its unconditioned dip at passes 16–26.
3. **Selection on one half-year window is the weak link.** With two sampling seeds on 26 weeks, validation
   predicts test only weakly (+0.43), and two runs selected their pass-5 checkpoint although later
   checkpoints did far better on test. The real experiment selects on 8 quarters × 5 training seeds × 4
   sampling seeds per checkpoint, which reduces this noise without removing it; PBO (§9) measures what
   remains.
4. **Sharpe against Sortino:** no difference larger than the difference between seeds — the best and the
   weakest runs were both Sortino.

---

### 11.5 Real-data result — 2026-09-12

Ten runs (5 seeds × 2 folds), Sharpe reward, 100 passes, 20 checkpoints each; 5,991 updates (fold 2024)
and 6,849 (fold 2025) at 1.5–2.1 s per update; all ten completed with **zero liquidations**.

**Selection** (`results/agent/sharpe_bars12/comparison/selection.json`), over 5 seeds × 8 quarters at each
of the 20 common checkpoints:

| | value |
|---|---|
| selected checkpoint | 3 of 20, pass 15 |
| IQM of quarterly log return | **+0.0190** |
| 95% stratified bootstrap interval | **[−0.0186, +0.0573]** — contains zero |
| positive quarters | 57% (23 of 40) |
| **probability of backtest overfitting** | **0.80** |
| break-even extra slippage per leg | median 3.7 bp, IQM 5.3 bp; 57% of paths above zero |

**Per fold at the selected checkpoint** (validation-year log return, BTC):

| fold | mean over 5 seeds | best | worst | constant long | constant short | flat |
|---|---:|---:|---:|---:|---:|---:|
| 2024 | +0.216 | +0.387 | −0.109 | **+0.401** | −0.671 | 0 |
| 2025 | −0.032 | +0.213 | −0.380 | −0.305 | **+0.121** | 0 |

**The checkpoint curve never settles.** The IQM oscillates between −0.08 and +0.02 across the 20
checkpoints with no stable positive region, and the last six checkpoints are negative. The PBO of 0.80
says the same thing formally: a checkpoint chosen on these quarters ranks below median out of sample four
times in five, so the +0.019 is best read as selection noise, not skill.

**What did work.** The agent's behaviour is sane and matches the design: 19–137 orders per quarter (against
~1,700 untrained), positions held for hours to days, both sides used in most runs, no liquidation in
40 validation years of trading, and the replay checks exact throughout. The machinery learns what it is
given — it learned the synthetic regimes (§11.4) — but on BTC 2017–2025 after 12 bp round-trip costs there
is no edge it could find that survives its own selection.

**Consequence.** The 2026 test window stays locked. Spending it on a result whose interval contains zero
and whose selection is 80% likely to be noise would waste the only untouched data.

**Reading the runs** — `scripts/financial_charts.py` writes `results/agent/sharpe_bars12/charts/index.html`,
a self-contained interactive report of the selected checkpoint for both validation folds and all five
training seeds: BTC price coloured by the position held with every entry and exit marked, equity in BTC
against constant long, constant short and the five-seed portfolio (banded over the four sampling seeds),
value in USD against buy-and-hold, drawdown, cumulative commission and funding, the policy's action
probabilities, quarter-by-quarter returns, holding-period and per-trade outcome distributions, a summary
table and every completed position. Two alignment details matter for reading it honestly: the saved frames
label each row with the decision instant but record price, position and equity as of the *end* of that
hour, so the page shifts price and equity onto the decision grid (one point longer, ending at the final
close) rather than drawing every marker an hour early; and the charts show one sampled path, since the
policy samples its actions and never takes the argmax (§9), with the four-sampling-seed band and mean
alongside. Checked against a coin-margined 1x model, the 1,207 charted positions have a median residual of
+4.7 bp per trade — fills at the next bar's open, slippage and funding over the hold.

### 11.6 Test window — SPENT 2026-09-12

The seal held through development and was opened once, on the user's instruction, by
`scripts/test_window.py` — the entrypoint §11.5 said did not exist. One pass over
**2026-01-01 to 2026-08-01** (212 days, the extent of the collected data), each of the ten runs at
the checkpoint already selected on validation, four sampling seeds each, from a flat account, with
the scaler saved inside each checkpoint (fitted on that fold's training span alone; nothing was
refitted on the test window). The market arrays were rebuilt from the raw bars with the full
2016-2025 history flowing in causally, and verified identical to the development cache over every
overlapping minute. No per-decision frame, per-checkpoint score or model state was written, so
nothing can be fitted to this window afterwards.

BTC fell from 87,657 to 62,767 over the window, **−28.4%**. Mean over the four sampling seeds:

| policy | return BTC | Sharpe BTC | return USD | Sharpe USD | max DD | orders | median hold |
|---|---:|---:|---:|---:|---:|---:|---:|
| trained 2017–2023, seed 11 | −5.4% | +0.11 | −32.1% | −0.38 | −22.5% | 54 | 10 h |
| trained 2017–2023, seed 29 | −22.3% | −0.71 | −44.2% | −0.79 | −32.5% | 114 | 8 h |
| trained 2017–2023, seed 47 | **−0.1%** | +0.11 | −28.3% | −1.26 | −17.9% | 86 | 15 h |
| trained 2017–2023, seed 61 | −27.3% | −0.54 | −47.8% | −0.54 | −39.9% | 50 | 7 h |
| trained 2017–2023, seed 83 | −11.8% | −0.31 | −36.7% | −0.53 | −32.2% | 274 | 6 h |
| trained 2017–2024, seed 11 | −27.3% | −0.73 | −47.8% | −0.71 | −37.3% | 74 | 13 h |
| trained 2017–2024, seed 29 | −13.6% | −0.28 | −38.0% | −0.69 | −32.7% | 100 | 8 h |
| trained 2017–2024, seed 47 | −38.3% | −1.24 | −55.7% | −0.98 | −46.1% | 34 | 24 h |
| trained 2017–2024, seed 61 | −22.0% | −0.71 | −44.0% | −0.91 | −38.6% | 256 | 7 h |
| trained 2017–2024, seed 83 | −34.1% | −1.27 | −52.7% | −1.01 | −44.7% | 72 | 13 h |
| **always short** | **+38.2%** | **+1.40** | −0.8% | −3.06 | −24.6% | 1 | — |
| **always long** | −38.3% | −0.58 | −55.8% | −0.44 | −54.2% | 1 | — |
| **flat (hold BTC)** | 0.0% | 0.00 | −28.2% | −0.99 | 0.0% | 0 | — |

**Every one of the ten lost money in BTC and none beat doing nothing.** Mean −13.4% for the models
trained through 2023 and −27.1% for those trained through 2024; mean Sharpe −0.27 and −0.85. The
best, at −0.1%, merely matched a flat account. All ten beat a constant long, none came within 38
points of the constant short that this window rewarded, and in dollars none beat simply holding the
collateral. Zero liquidations, and order counts (34–274 over seven months) stayed in the designed
range, so the behaviour was sane — it was the direction that was wrong.

**The failure is directional, not mechanical.** In a market that fell 28% the policies were long
15–57% of the time and short 0–44%. The single run that leaned net short (trained 2017–2023, seed
47: 0% long, 44% short) is the single run that did not lose. A state built from trend efficiency,
drawdown, volatility and volume did not stop the others from leaning long into a sustained decline
— consistent with a 2017–2024 training history in which BTC mostly rose, and with §11.5's finding
that the +0.019 validation IQM was selection noise (PBO 0.80).

**Consequence.** The out-of-sample result agrees with the validation reading: there is no edge
here. The window is spent and cannot be re-used; any further work needs new held-out data, not a
new selection over this one.

**Sharpe caveat.** The USD Sharpe of the constant short (−3.06 on a −0.82% return) is an artefact
of a nearly constant dollar value: a tiny negative drift divided by a very small standard deviation.
The BTC column is the account's native numeraire and the one to read.

**Second look — the fully trained checkpoints, 2026-09-13.** On the user's instruction, the same window,
the same protocol and the same guarantees (nothing written but `TEST_LOCK.json`, which now lists both
accesses), with every run at its **final** checkpoint (pass 100; update 5,991 / 6,849). The checkpoint was
fixed before the look, and the result cannot select anything: the window was already spent.

| policy (final checkpoint) | return BTC | Sharpe BTC | return USD | Sharpe USD | max DD | orders | median hold | long / short |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trained 2017–2023, seed 11 | −6.6% | −0.02 | −33.0% | −0.47 | −25.0% | 324 | 10 h | 49% / 39% |
| trained 2017–2023, seed 29 | −6.6% | +0.00 | −33.0% | −0.45 | −29.9% | 396 | 10 h | 49% / 37% |
| trained 2017–2023, seed 47 | −66.4% | −2.54 | −75.9% | −2.02 | −67.7% | 179 | 15 h | 37% / 47% |
| trained 2017–2023, seed 61 | −11.9% | −0.43 | −36.8% | −0.87 | −34.8% | 336 | 11 h | 51% / 36% |
| trained 2017–2023, seed 83 | −56.9% | −2.79 | −69.1% | −2.09 | −60.0% | 308 | 12 h | 54% / 32% |
| trained 2017–2024, seed 11 | −18.7% | −0.23 | −41.6% | −0.47 | −40.4% | 236 | 14 h | 58% / 22% |
| trained 2017–2024, seed 29 | **−4.5%** | +0.12 | −31.5% | −0.37 | −34.3% | 342 | 10 h | 41% / 39% |
| trained 2017–2024, seed 47 | −41.8% | −1.09 | −58.2% | −0.95 | −48.2% | 270 | 13 h | 60% / 28% |
| trained 2017–2024, seed 61 | −49.9% | −1.63 | −64.0% | −1.28 | −52.3% | 302 | 11 h | 57% / 27% |
| trained 2017–2024, seed 83 | −51.7% | −0.50 | −65.3% | −0.32 | −66.0% | 23 | 12 h | 97% / 0% |
| always short / always long / flat | +38.2% / −38.3% / 0.0% | +1.40 / −0.58 / 0.00 | | | | | | |

**Training longer made it worse.** Mean BTC return −29.7% for the models trained through 2023 and −33.3%
for those trained through 2024 (mean Sharpe −1.16 and −0.67), against −13.4% and −27.1% at the selected
checkpoint. Six of ten runs lost more than at pass 15, and **five of ten lost more than a constant long**.
The median run placed about 305 orders, against about 80 at pass 15. The final policies are trend-aligned
on validation (§11.7), yet they still spent 37–97% of a falling market long. The larger losses are
consistent with a trend follower whipsawed by rallies inside a decline, plus four times the cost. That
mechanism is **not measured**: measuring it would mean mining the spent window.

**Consequence.** Neither the selected nor the fully trained policy generalises to 2026. The failure is
therefore not only a selection accident (§11.7); §18 sets out the remedies.

### 11.7 Why the deployed policy leaned long — post-mortem, 2026-09-12

Asked why an agent trained on random quarter-long episodes should carry a bull-market bias,
`scripts/episode_distribution.py` and the saved validation frames answer it, and the answer is not
the one the question assumes.

**The sampler was uniform, and bear regimes were not rare.** Each of the 8 environments draws a
start uniformly from the fold's training span (710,016 / 815,424 distinct legal 5-minute-aligned
starts) and runs one contiguous 90.7-day episode, preceded by the 2.7-day exploring-start prefix
whose side is a 50/50 coin flip. Measured over the drawable starts: **42.5%** of quarters fell
(fold 2024; 40.1% for fold 2025), and **22.8%** of 212-day stretches were as bad as the test
window's −28.4%. Uniform sampling is unbiased for the base rate — it reproduces the history's mix
rather than balancing it — but that mix was not lopsided in *count*.

**It is lopsided in magnitude, and thin in independent scenarios.** Mean quarter log return +0.131
against a median of +0.089, p95 +0.929 against p5 −0.572: up-quarters are larger, so expected
reward favours long exposure even at 4:3 counts. More binding, the span holds only **28** (fold
2024) or **32** (fold 2025) non-overlapping quarters. The 2,819 and 3,223 episodes actually drawn
each overlap about 100 others, so the effective sample is ~28 macro-scenarios, not ~2,800.

**The data rewards momentum, and the trained policy learnt momentum.** Forward return by decile of
`trend_efficiency_1w` over 2017–2024 rises monotonically from +63 bp (strongest downtrend) to
+360 bp (strongest uptrend) per week; by `drawdown_4w`, from −7 bp at the deepest to +273 bp at the
peak. At the **final** checkpoint (pass 100) **10 of 10** runs trade with the trend — buy strength,
short weakness — mean alignment +0.087.

**What was deployed was not that policy.** At the **selected** checkpoint (pass 15) only **3 of 10**
are trend-aligned; the mean is *negative*, −0.035. Those policies buy weakness and short strength,
and they are the most lopsided of the run: checkpoint 3 ranks 4th of 20 by long-entry share (72.4%
of entries long, 1.94x long/short exposure, against 58.4% and 1.34x at pass 100). A dip-buyer is
long into every leg of a sustained decline, which is exactly the §11.6 test result — and the one run
that did not lose there is the one that leaned net short.

**The defect is in selection, not in episode sampling.** The validation IQM curve oscillates between
−0.081 and +0.019 with no stable region; pass 15 is simply its maximum, which is what PBO 0.80 says
formally. **But the alternative was no better**: pass 100 scores −0.0219 on validation. Trend
alignment did not convert into profit, because the momentum edge (~21 bp per 8 hours in the best
decile) sits barely above the 12 bp round trip — the same cost wall as every earlier study.

**Caveat on one measurement.** A constant 1x inverse long is *liquidated* in 2018 and 2020, so
year-long constant-long returns over the training span are wipeouts, not a measure of what being
long paid. The quarter-return distribution above is that measure.

---

## 12. Provenance of every constant

| constant | value | class | source |
|---|---|---|---|
| `decision_minutes` | 5 | USER | bar length of the features and reward statistics; accounting runs every minute |
| `decision_bars` | 12 | USER | hourly decisions on five-minute bars, adopted on the evidence of §4.1 and §11.3; frame skip (Mnih et al. 2015; Andrychowicz et al. 2021) |
| `fee` | 0.0005 | USER | realistic taker commission, no curriculum |
| `slippage` | 0.0001 | USER | |
| `entry_leverage` | 1.0 | USER | |
| `maintenance` | 0.005 | CITED | BitMEX XBTUSD base maintenance margin 0.50% until 17 Apr 2020, 0.45% after (BitMEX, 13 Apr 2020); higher value used throughout |
| `hidden` | 128 | CITED | CleanRL `ppo_atari_lstm` LSTM 128 |
| `head_width` | 128 | CITED + DESIGN CHOICE | two hidden layers (Andrychowicz et al.); width equal to `hidden` is a choice |
| `critic_width_multiple` | 2 | CITED | *"for the value function there seems to be no downside in using wider networks"* (Andrychowicz et al.) |
| `environments` | 8 | CITED | PPO Table 5 *"Number of actors 8"*; CleanRL |
| `rollout` | 128 | CITED + DERIVED | PPO Table 5 *"Horizon (T) 128"*; CleanRL `num_steps`; bound rollout / 2 ≥ credit window |
| `sequence` | 128 | CITED + DERIVED | CleanRL trains whole rollouts; bound `burn_in` + `sequence` ≥ credit window |
| `burn_in` | 64 | CITED | R2D2 ratio 0.5 (Kapturowski et al. 2019) |
| `episode_steps` | 2,176 | USER + CITED + DERIVED | 2–3 month windows (user); quarterly window of Yang et al. 2020; whole rollouts nearest a quarter (`quarter_episode`) |
| `minibatch_sequences` | 2 | DERIVED | CleanRL `num_minibatches = 4`; 256 samples = PPO Table 5 minibatch 32 × 8 |
| `epochs` | 4 | CITED | CleanRL `update_epochs` |
| `recompute_advantages_per_pass` | True | CITED | Andrychowicz et al. |
| `gamma` | 0.99825 | USER | per bar, raised to the 12th power per decision; §5 |
| `gae_lambda` | 0.99825 | USER | per bar, raised to the 12th power per decision; §5 |
| `actor_lr` | 2.5e-4 | CITED | PPO Table 5; CleanRL |
| `critic_lr` | 2.5e-4 | CITED | PPO Table 5; CleanRL |
| `anneal_lr` | True | CITED | PPO Table 5 (α annealed); CleanRL |
| `gradient_clip` | 0.5 | CITED | CleanRL `max_grad_norm` |
| `entropy_coefficient` | 0.01 | CITED | PPO Table 5; CleanRL; Mei et al. 2020 |
| `clip` | 0.1 | CITED | PPO Table 5 (0.1 × α); CleanRL `clip_coef`; the clip's discrete-action failure mode (Hsu et al. 2020) is monitored |
| `flat_start_fraction` | 1/3 | CITED | exploring starts, uniform over the three sides (Sutton & Barto 2018 §5.3) |
| `reward` | sharpe | USER | differential Sharpe ratio; `sortino` selects the downside deviation ratio (Moody & Saffell 2001) |
| `dsr_eta` | 1/1440 | DESIGN CHOICE | per bar, a five-day window, inside the derived bounds of §4.5 |
| `dsr_initial_second_moment` | measured per fold | MEASURED | five-minute return variance of the fold's training span: (26.55 bp)², (25.54 bp)² |
| `observation_clip` | 10.0 | CITED | *"Observation Clipping … usually [−10, 10]"* (Engstrom et al., ICLR 2020) |
| `training_passes` | 100 | CITED | 100 epochs with validation-based stopping (Moody & Saffell 2001; Deng et al. 2017) |
| `validation_passes` | 5 | CITED | strips of 5 epochs (Prechelt 1998) |
| `updates` | derived per fold | DERIVED | ⌈`training_passes` × training decisions / (`environments` × `rollout`)⌉: 5,991 and 6,849 |
| `threads` | 1 | MEASURED | one thread per worker; 1.68 s per update and ~1 GB per worker |
| `evaluation_policy` | `argmax` | USER | 2026-09-15, after an external review; `sampled` was the protocol of every result up to 2026-09-13 (§9) |
| `evaluation_seeds` | 4 | MEASURED | used only when `evaluation_policy` = `sampled`: a 4-seed mean's standard error 0.12 against 0.26–0.63 between training seeds (§9) |
| `bootstrap_replications` | 50,000 | CITED | rliable default (Agarwal et al. 2021) |
| `confidence_level` | 0.95 | CITED | rliable default |

Numbers outside `Config`: `SEEDS` = 11, 29, 47, 61, 83 (USER: five training seeds; the values are arbitrary); `SIGNAL_SHARPE_ASSUMPTION` = 1.0 (ASSUMPTION); the per-fold σ_bar and break-even
(MEASURED, DERIVED); `QUARTER_BARS` = a quarter of a Julian year in five-minute bars (DERIVED; the quarterly
window CITED to Yang et al. 2020); the reward scale (DERIVED); the feature windows (DERIVED from their
sources); the trade-age unit (DERIVED) and its log compression (DESIGN CHOICE); `NO_MOVEMENT` = 1e-12, a
numerical floor; the replay tolerance 5e-5 (float32 round-off; MEASURED error 0.0); PBO partitions = 8
quarters (DERIVED); `BTC_TRAINING_FIVE_MINUTE_SD` = 26.55 bp, the scale of the observable synthetic market
(MEASURED on 2017–2023); the latent market's parameters (§11.3), calibrated before training.

---

## 13. Audit — constants of earlier designs

| constant (earlier) | value | status then | what became of it |
|---|---|---|---|
| gate and dual constants | various | unsourced or fitted after the fact | **removed** (§1.1) |
| `advantage_scale_decay`, `critic_epochs` | 0.05, 2 | untested | **removed** |
| `epochs` | 4 | cited as the PPO paper's Atari setting — **wrong**, Table 5 uses 3 | **kept, re-cited** to CleanRL |
| counterfactual constants, `dsr_clip`, variance floor | — | untested or arbitrary | **removed**; exact reward increments |
| 18 portfolio inputs, raw momenta, one-minute encoder | — | unverified | **replaced** (2026-09-10) |
| planted-edge "PASS" of the side-lock agent | +0.144 vs constant long | compared with one constant only | **over-claimed**; rules now require beating every constant with regime conditioning |
| `policy_kl_step`, action-value head, counterfactual branches | — | cited | **removed** with the all-action trainer |
| 13-feature state | — | cited individually, never tested jointly | **replaced** by six measured features |
| five-minute decisions | — | USER | **replaced** by hourly decisions (USER) |
| `FIVE_MINUTE_RETURN_SD` = 14.8 bp, `COST_BREAK_EVEN_HOURS` = 22 | constants | "MEASURED" — on 2023–2025, **overlapping both validation years** | **replaced** by per-fold measurement on training spans |
| `updates` = 5,000 | | CITED to a simulator's step budget | **replaced** by 100 passes (§7.3) |
| `validation_every` = 500 updates | | DESIGN CHOICE | **replaced** by a checkpoint every 5 passes (Prechelt 1998) |
| `rollout` 1,024, `sequence` 256, `burn_in` 128, 4 environments per minibatch, episode 2,048 | | DERIVED for five-minute decisions | **replaced** by the discrete-action reference shape |
| `clip` 0.25, learning rate 3e-4 | | CITED to continuous control | **replaced** by the discrete-action references |
| `stress_slippage` 1, 3, 5 bp | | DESIGN CHOICE | **replaced** by break-even slippage |
| last-three verification rule | | pre-registered | **replaced** by select-on-validation, judge-on-test — after it failed runs that had passed early and then overfit (§11.3) |

---

## 14. Fixes and their status

| proposal | evidence | decision |
|---|---|---|
| exploration where the entry decision is made | Haarnoja et al. 2018; Mei et al. 2020, 2021 | **adopted**: one softmax with the entropy bonus, exploring starts, hourly decisions |
| remove the reward clip | van Hasselt et al. 2016 | **adopted**: exact increments |
| a budget in passes with validation stopping | Moody & Saffell 2001; Deng et al. 2017; Théate & Ernst 2021; Prechelt 1998; Sato 2019 | **adopted** (§7.3) |
| measure data-dependent quantities per fold | leakage found in §2 | **adopted** |
| liquidation as a constraint, catastrophe predictors | Chow et al. 2017; Lipton et al.; Sootla et al. 2022 | **not adopted**: each needs a weight no study pins |
| plasticity interventions | Lyle et al. 2023; Juliani & Ash 2024 | **LayerNorm adopted**; shrink-and-perturb in reserve |
| KL-regularised surrogate | Hsu et al. 2020 | **in reserve** if the clip locks |
| IQM, bootstrap, PBO | Agarwal et al. 2021; Bailey et al. 2017; Gort et al. 2022 | **adopted** |

---

## 15. Known limitations and open risks

1. **PPO's signal-to-noise problem at weak edges is not solved** (§1.4). The synthetic verification uses an
   edge strong enough to test the machinery; the real market's edge, if any, is weaker.
2. **Where held-out return peaks on real data is unknown.** The budget and checkpoint spacing come from the
   trading literature and are verified on a synthetic market (§11.4); selection among 20 checkpoints on
   the validation years has a cost that PBO measures but does not remove.
3. **The reference hyperparameters were tuned for Atari**, not for this problem; Andrychowicz et al. advise
   tuning the transitions per iteration. None has been tuned here.
4. **The horizon rests on an assumption and a USER value**: the break-even assumes a Sharpe 1.0 signal, and
   γ = λ = 0.99825 per bar was fixed when the break-even was believed to be 22 h; it is 12.2–12.7 h.
5. **The features were selected on 2017–2023**; choosing six of thirteen candidates is itself a degree of
   freedom, and their relevance is to regime properties in-sample, not to profit after costs.
6. **The agent cannot react inside an hour**; an adverse hour is taken in full.
7. **The Sortino evidence is thin**: at five-minute decisions it began anti-conditioned; its first hourly run
   peaked at +1.92 and fell to −0.72 on the last-three rule. The downside ratio is unbounded for gains.
8. **`dsr_eta` and the head width are design choices.** No study pins either; tuning η on the synthetic
   market would tune it to our own generator.
9. **`funding_rate` sits at the 0.01% baseline most of the time**; its information is in the excursions.
10. **Two folds and five seeds** give 40 scores per checkpoint; intervals will still be wide.
11. **The agent does not generalise out of sample.** All ten runs lost on 2026 at both the selected and the
    final checkpoint (§11.6). The causes and remedies are in §18.
12. **Argmax evaluation of a policy trained under sampling.** PPO optimised the sampled, entropy-regularised
    policy; its argmax is a different strategy that was never trained as such, and early checkpoints can
    evaluate as nearly always flat (§0.2). The saved checkpoints were selected under sampled evaluation, so
    any argmax result needs its own selection on validation, not the old one.

---

## 16. Lineage

| generation | outcome |
|---|---|
| v4 | converged to permanent flat |
| v5 | 18 architectures × 78 runs; entropy fell monotonically in 78/78 |
| v6 stage A | horizon fix alone: collapse rate unchanged |
| gate agent | both runs collapsed via the gate |
| side-lock agent | entry side locked in 4 of 5 runs; stopped 2026-09-10 |
| reference PPO, redesigned MDP | planted verification 0 of 4 |
| all-action PPO, then MPO | PPO step froze; MPO learned a ten-times edge; not verified at the planted edge |
| recurrent PPO, regime state, hourly decisions | smoke tests: regimes recognised at hourly decisions, then overfitting under many passes |
| **current** | the same agent with the discrete-action reference shape, a budget in passes with validation selection, and per-fold measurement of the reward prior and break-even; synthetic verification 1 of 4; **real data: no edge distinguishable from zero, PBO 0.80 (§11.5); 2026 test: 0 of 10 profitable at pass 15 or at pass 100 (§11.6)** |

Archives: `backup/perpetual_v4_20260909`, `backup/perpetual_v4_v5_before_v6_20260908`,
`backup/perpetual_v6_stage_a_20260909`, `backup/agent_gate_collapse_20260909`,
`backup/agent_alpha_windup_20260910`, `backup/agent_side_lock_20260910`,
`backup/agent_all_action_mpo_20260911`. Smoke-test results of the first hourly configuration are kept in
`results/smoke/`; the previous feature cache in `results/agent/cache_v2_13_features`.

---

## 17. Mathematical formulation and algorithm

This section writes down exactly what the code computes, in one place. Every symbol maps to a function
or `Config` field, named in brackets; the prose sections above give the reasons and provenance. Formulas
are in LaTeX (they render on GitHub and in most Markdown viewers).

### 17.1 Time and notation

| symbol | meaning | code |
|---|---|---|
| $m$ | minute index; $O_m, H_m, L_m, C_m$ open, high, low, close ($C$ forward-filled over untraded minutes) | `market.prices` |
| $b$ | five-minute bar boundary, $b \equiv 0 \pmod 5$; the bar ending at $b$ covers minutes $b-5,\dots,b-1$ | `boundary` |
| $k$ | decision index; decision $k$ is taken at minute $t_k$ and holds for $K=12$ bars, $t_{k+1}=t_k+60$ | `decision_bars` |
| $f_m$ | funding rate settled at minute $m$ (zero at other minutes) | `market.funding` |
| $\varphi,\ \varsigma$ | commission and slippage per leg, 5 bp and 1 bp | `fee`, `slippage` |
| $\gamma,\ \lambda$ | discount and GAE parameter **per bar**, 0.99825 each | `gamma`, `gae_lambda` |
| $\gamma_d=\gamma^{K},\ \lambda_d=\lambda^{K}$ | per decision, 0.97920 each | `decision_gamma`, `decision_lambda` |

### 17.2 The market state $x_k$ (exogenous)

Every quantity at boundary $b$ uses minutes before $b$ only (`causal_arrays`). With the bar log return
$r_b=\ln C_{b-1}-\ln C_{b-6}$ and window lengths $D=288$ (day), $W=2016$ (week) and $M=8064$ (four weeks)
in bars:

$$
\sigma^2_{\text{day}}(b)=\frac{1}{D}\sum_{j=0}^{D-1} r_{b-j}^2,\qquad
\sigma^2_{\text{month}}(b)=\frac{1}{M}\sum_{j=0}^{M-1} r_{b-j}^2,\qquad
\sigma_d(b)=\sqrt{D\,\sigma^2_{\text{month}}(b)}
$$

$$
\begin{aligned}
x^{(1)}_b &= \operatorname{clip}\!\left(\frac{\ln C_{b-1}-\ln C_{b-1-5W}}{\sum_{j=0}^{W-1}|r_{b-j}|},\,-1,\,1\right)
&&\texttt{trend\_efficiency\_1w (signed Kaufman ratio)}\\
x^{(2)}_b &= \frac{\ln C_{b-1}-\max_{0\le j<M}\ln C_{b-1-5j}}{\sigma_d(b)} \;\le 0
&&\texttt{drawdown\_4w}\\
x^{(3)}_b &= \ln \sigma_d(b) &&\texttt{volatility\_level}\\
x^{(4)}_b &= \tfrac12\ln\!\big(\sigma^2_{\text{day}}(b)/\sigma^2_{\text{month}}(b)\big) &&\texttt{volatility\_regime}\\
x^{(5)}_b &= \ln\!\big(1+D\,\bar v_{\text{week}}\big)-\ln\!\big(1+D\,\bar v_{\text{prior 4 weeks}}\big) &&\texttt{volume\_regime}\\
x^{(6)}_b &= \text{last funding rate settled before } b &&\texttt{funding\_rate}
\end{aligned}
$$

where $\bar v$ is mean volume per bar over the window. The network sees the standardised, clipped vector
$z_k=\operatorname{clip}\big((x_{t_k}-\mu)/s,\,-10,\,10\big)$, with $\mu$ and $s$ measured on the fold's
training span only (`Scaler.fit`).

### 17.3 The account (endogenous) and the ledger

The account holds a BTC wallet $W$, a USD contract quantity $Q$ (sign = side) and an entry price
$\bar p$. Its equity at mark $P$ is the inverse-contract value

$$
E(P)=W+Q\left(\frac{1}{\bar p}-\frac{1}{P}\right).
$$

**Action.** $a_k\in\{\text{SHORT},\text{FLAT},\text{LONG}\}\mapsto s_k\in\{-1,0,1\}$, a *target* side.
If $s_k\neq\operatorname{sign}(Q)$, the order fills at the first minute $m$ of the hour that has a print, at
its open $O_m$ (`Ledger.execute`):

$$
\begin{aligned}
\text{close:}\quad & p=O_m\,(1-\operatorname{sign}(Q)\,\varsigma),\qquad W\leftarrow W+Q\Big(\tfrac{1}{\bar p}-\tfrac1p\Big)-\varphi\,\tfrac{|Q|}{p},\qquad Q\leftarrow 0\\
\text{open:}\quad & p=O_m\,(1+s_k\varsigma),\qquad Q\leftarrow s_k\big\lfloor W\,O_m\,\ell\big\rfloor,\qquad W\leftarrow W-\varphi\,\tfrac{|Q|}{p},\qquad \bar p\leftarrow p
\end{aligned}
$$

with entry leverage $\ell=1$ and no rebalancing while the position is held. A round trip costs
$2(\varphi+\varsigma)=12$ bp.

**Every minute** of the hour, in this order (`advance`):

1. funding: $W\leftarrow W-\dfrac{Q}{C_{m-1}}\,f_m$ (a long pays a positive rate);
2. a margin check at $C_{m-1}$, then the pending order if this minute is eligible;
3. a margin check at the adverse extreme $P^{\text{adv}}_m=L_m$ for a long, $H_m$ for a short.
   **Liquidation** happens if
   $$E(P^{\text{adv}}_m)\le \frac{|Q|}{P^{\text{adv}}_m}\,(\mu_{\text{mm}}+\varphi),\qquad \mu_{\text{mm}}=0.5\%,$$
   in which case the position is closed at $P^{\text{adv}}_m$ and the episode terminates;
4. the equity is marked at the close: $E_m=E(C_m)$.

**Account observation** $y_k\in\mathbb R^8$ (`Ledger.portfolio`), marked at $C_{t_k-1}$:

$$
y_k=\Big[\,\mathbb 1_{s=-1},\ \mathbb 1_{s=0},\ \mathbb 1_{s=+1},\ \
\frac{\ln(E/E_{\text{trade start}})}{\sigma_d},\ \
\ln\!\Big(1+\frac{\text{bars held}}{K\,\kappa}\Big),\ \
\ln\!\Big(1+\frac{|Q|}{P\,E}\Big),\ \
S(A,B),\ \ \tfrac12\ln\frac{B}{B_0}\,\Big]
$$

with the credit window $\kappa=1/(1-\gamma_d\lambda_d)=24.3$ decisions, the reward statistics $A$ and $B$ of
§17.4, and the continuous entries clipped to $\pm10$. The state is $s_k=(z_{\le k},\,y_k)$: the market
enters through recurrent memory, and the account enters directly.

### 17.4 Reward

For each five-minute bar $b$ inside the decision, with $R_b=\ln(E_b/E_{b-1})$ net of commission, slippage
and funding (`differential_ratio`):

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
c=\sqrt{\frac{1-\gamma_d^2}{K}}=0.0586\ \Big(\times\tfrac{1}{\sqrt2}\text{ for sortino}\Big),
$$

with $\eta=1/1440$. A fresh account starts at $A_0=0$ and $B_0=\hat\sigma^2_{\text{bar}}$, the bar-return
variance measured on the training span ($\hat\sigma^2_{\text{bar}}/2$ for sortino). The sum telescopes:
$\sum_{b\in k} D_b=\big(S_{\text{end}}-S_{\text{start}}\big)/\eta$.

### 17.5 The decision problem

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

### 17.6 Policy and value networks

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

### 17.7 Advantage estimation and the PPO loss

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

**Budget.** $U=\big\lceil P\cdot N_{\text{train}}/(E\cdot T)\big\rceil$ for $P=100$ passes, $N_{\text{train}}$ hourly
decisions in the training span, $E=8$ environments and $T=128$ decisions per rollout. That gives 5,991 updates
for fold 2024 and 6,849 for fold 2025, with a checkpoint every $U/20$ updates (`fold_config`,
`checkpoint_updates`).

### 17.8 Algorithm — training one fold and one seed

```text
Algorithm 1  TRAIN(fold year Y, seed σ)
  span        ← [2017-01-01, Y-01-01)
  μ, s, σ̂_bar ← statistics of the six features and the 5-minute returns on span        # Scaler.fit
  B0          ← σ̂_bar²;  U ← ⌈100 · N_train / 1024⌉;  checkpoints ← {round(i·U/20) : i = 1..20}
  θ, ω        ← orthogonal initialisation (policy head gain 0.01, forget bias 1)
  for e = 1..8:  RESET(e);  shorten env e's first episode by 128·𝒰{0..16} decisions
  for u = 0..U-1:
      α ← α0 · (1 − u/U)
      D, bootstrap ← COLLECT()                              # Algorithm 2
      PPO-UPDATE(D, bootstrap, α)                           # Algorithm 3
      if u+1 ∈ checkpoints:
          save θ, ω;  EVALUATE(θ, validation year Y)        # Algorithm 4, stored for selection

Procedure RESET(e)
  τ ~ 𝒰{5-minute boundaries of span, leaving 64 + 2176 decisions before its end}
  new account: W = 1, Q = 0, A = 0, B = B0
  with prob 1/3: side ← 0   else: side ~ 𝒰{−1,+1},  j ~ 𝒰{0..63}
  for i = 0..63:  ADVANCE(account, i < j or side = 0 ? FLAT : side)     # real fees and funding
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
      r_k, d_k ← ADVANCE(account, a_k)                     # 12 bars × 5 minutes of ledger, §17.3–17.4
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
      Â, Ĝ ← GAE(r, V, bootstrap, d, valid; γ_d, λ_d)       # §17.7
      for each minibatch ℬ in a random partition of the 8 envs into 4 pairs:
          (h, c) ← encode the 64-decision prefix of ℬ without gradient
          Ã ← normalise Â over the valid steps of ℬ
          θ ← Adam_α( ∇θ[ −mean min(ρÃ, clip(ρ, 0.9, 1.1)Ã) − 0.01·mean 𝓗(π_θ) ] ),  ‖∇‖ ≤ 0.5
          ω ← Adam_α( ∇ω[ mean ½(V_ω − Ĝ)² ] ),                                       ‖∇‖ ≤ 0.5
```

```text
Algorithm 4  EVALUATE(θ, year Y)                            # one checkpoint of one run
  for quarter q = 1..4 of Y:
      memory ← LSTM over z, rebuilt from zero every 128 decisions over the preceding 64
      if evaluation_policy = argmax:                         # USER, 2026-09-15
          flat account at the quarter's first hour;  for each hour k of q:
              a_k ← argmax_a π_θ(a | memory_k, y_k);  ADVANCE(account, a_k)
          score[q] ← total BTC log return of the path
      else:                                                  # sampled, every result to 2026-09-13
          for sampling seed ξ = 0..3:
              flat account at the quarter's first hour;  for each hour k of q:
                  a_k ~ π_θ(· | memory_k, y_k);  ADVANCE(account, a_k)
              G[q, ξ] ← total BTC log return of the path
          score[q] ← mean over ξ of G[q, ξ]

Algorithm 5  SELECT()                                       # after both folds and all 5 seeds
  for checkpoint index j = 1..20:
      S_j ← {score[q] of seed i, fold Y at checkpoint j}    # 5 seeds × 2 folds × 4 quarters = 40
      IQM_j ← mean of the middle half of S_j
  j* ← argmax_j IQM_j  (ties go to the earlier checkpoint)
  report: stratified bootstrap 95% interval of IQM_{j*}; PBO by CSCV over the 8 quarters × 20 checkpoints
```

### 17.9 What the formulation says about the failure

Two properties of this formulation bear directly on §11.6–§11.7:

- **Nothing in $J(\theta)$ separates regimes.** The expectation is over $\tau\sim\mathcal U(\text{span})$, so
  the policy maximises return averaged over the history's own mixture of regimes. A rule that pays in the
  common regime and loses in a rarer one is optimal if the average favours it. In the training span,
  up-quarters are larger than down-quarters (§11.7).
- **The information that identifies a regime is short.** $z_k$ reaches back at most four weeks, and the
  LSTM memory is rebuilt from a 64-decision prefix every 128 decisions, so it spans at most 192 hours
  (eight days). The policy has no direct view of a trend lasting months, which is exactly what 2026 was.

---

## 18. Why it does not generalise to 2026, and what to change

Written 2026-09-13, after both looks at the spent window (§11.6). Proposals here are **not yet part of the
agent**; each is tagged with its literature, with this project's own evidence, and with the risk that it
makes things worse. The 2026-01 to 2026-07 window cannot be used to judge any of them.

### 18.1 What is established

| finding | evidence |
|---|---|
| neither checkpoint generalises | all 10 runs lost on 2026 at pass 15 (mean −20.2% BTC) and at pass 100 (mean −31.5%); none beat flat (§11.6) |
| validation could not tell good from bad | the checkpoint IQM curve is flat noise in [−0.08, +0.02]; PBO 0.80 (§11.5) |
| the history contains few independent scenarios | 28–32 non-overlapping quarters behind ~3,000 episode draws (§11.7) |
| the edge the data offers is about the size of the cost | best trend decile: ~21 bp over 8 h against a 12 bp round trip (§11.7) |
| the policies keep a long lean that the state does not remove | 37–97% of 2026 spent long at pass 100 (§11.6) |
| **earlier studies in this project already tried several textbook remedies** | project record, walk-forward study of 2026-08-18 (the `cryptorl` package; its artifacts are no longer on disk): training at 2–8× cost, a 4-hour clock, walk-forward retraining and continuous sizing all behaved as intended, development was positive out of sample, and the sealed window still lost on 10 of 10 seeds. Its supervised probe measured an information coefficient of +0.024 at 1 h against a 0.202 break-even hurdle, and no horizon cleared. The agent there also *"learned trend and went 63–73% net long through a round trip"* |

The last row matters most. The same pattern — trend learned, long lean, sealed loss — has now appeared under
two different learners, two clocks and two protocols. That points first at the **problem** (a weak edge
against the cost on one asset) and only second at this agent's design.

### 18.2 Four causes, in order of how much evidence supports them

1. **Too little predictable signal after costs at this horizon.** If this is the cause, no change to state,
   parameters or architecture can fix it; only the horizon, the cost, or the source of return can.
2. **Too few independent scenarios, hence overfitting.** Cobbe, Klimov, Hesse, Kim & Schulman (ICML 2019)
   *"find that agents overfit to surprisingly large training sets"*, and their environments had thousands of
   levels. This agent has about 28 quarters.
3. **Non-stationarity.** Regimes change and are persistent (Hamilton 1989; Ang & Timmermann 2012), and
   momentum strategies suffer *"infrequent and persistent strings of negative returns"* in panic states
   after declines (Daniel & Moskowitz, JFE 2016 — shown for cross-sectional momentum). A policy frozen for
   seven months meets regimes its training mixture weights lightly.
4. **Selection noise.** Real but secondary: the fully trained policy did no better (§11.6).

### 18.3 The remedies the literature offers, by lever

**A. Protocol and evaluation** — the one area where the failure is demonstrated, and the cheapest to fix.

| remedy | literature | addresses | risk |
|---|---|---|---|
| **A1. A new sealed test window.** Data from 2026-08-01 onward are not collected yet; collect and seal them before any new design is trained. 2024 and 2025 have been used for selection, and 2026-01 to 07 is spent. | Bailey, Borwein, López de Prado & Zhu (2017) | every later claim | waiting months for enough data |
| **A2. Select for robustness, not the peak.** Deflate the selected Sharpe for the number of checkpoints and configurations tried, and replace one validation year with combinatorially purged cross-validation over many train/test splits. | Bailey & López de Prado, *The Deflated Sharpe Ratio* (JPM 2014); López de Prado, *Advances in Financial Machine Learning* (2018), ch. 11–12 | cause 4, and exposes cause 2 honestly | it will more often say "no edge" — which is the point |
| **A3. Average checkpoints instead of choosing one.** Stochastic weight averaging finds wider optima that generalise better. | Izmailov, Podoprikhin, Garipov, Vetrov & Wilson (UAI 2018) | cause 4 | shown for supervised learning; for PPO policies it is untested here |
| **A4. Retrain walk-forward, often.** Moody & Saffell trained on about two months and traded two weeks before rolling; FinRL's ensemble retrains every quarter. A model trained through Q1 2026 would have seen the start of the decline. | Moody & Saffell (2001); Yang, Liu, Zhong & Walid (ICAIF 2020) | cause 3 | **already tried** in the 2026-08-18 walk-forward study, which still lost; costs compute on every roll |
| **A5. A feasibility gate before any RL.** Test simple, published rules through this ledger with these costs, walk-forward 2018–2025, before training an agent: volatility-scaled time-series momentum at 1-, 4-, 12- and 26-week look-backs, and funding carry. If none clears the deflated-Sharpe bar, stop. | Moskowitz, Ooi & Pedersen, *Time Series Momentum* (JFE 2012); Hurst, Ooi & Pedersen, *A Century of Evidence on Trend-Following Investing* (JPM 2017); Liu & Tsyvinski (RFS 2021) report a strong time-series momentum effect in cryptocurrencies; Koijen, Moskowitz, Pedersen & Vrugt, *Carry* (JFE 2018) | cause 1 — the only remedy that tests it | none; it is cheap. Project record: two trivial rules beat all 40 learned tracks of the removed portfolio agent, and carry was the only profitable benchmark in the walk-forward study |

**B. Training data — the episode distribution.** The RL generalisation literature ranks training diversity
first: Kirk, Zhang, Grefenstette & Rocktäschel (JAIR 2023) survey data augmentation, environment
randomisation and more varied training contexts as the main levers for generalising to unseen situations.

| remedy | literature | addresses | risk |
|---|---|---|---|
| **B1. Regime-balanced episode sampling.** Label each candidate start by the regime of its episode (bull, bear or range, dated as in Pagan & Sossounov 2003), then sample the labels with equal probability instead of in proportion to history. A stronger variant up-weights the worst regime. | Sagawa, Koh, Hashimoto & Liang, *Distributionally Robust Neural Networks* (ICLR 2020): optimising the worst group improves worst-group generalisation; Pagan & Sossounov (2003) | cause 3 and the long lean directly | it changes what is optimal: a policy that is right in bear markets can give up bull-market return |
| **B2. More independent scenarios by resampling.** A stationary block bootstrap of return paths, with blocks of weeks so that regimes survive, or generated paths. | Politis & Romano, *The Stationary Bootstrap* (JASA 1994); Wiese, Knobloch, Korn & Kretschmer, *Quant GANs* (Quantitative Finance 2020); Koshiyama, Firoozye & Treleaven (Quantitative Finance 2020) calibrate trading strategies on cGAN-generated paths | cause 2 | a bootstrap cannot invent a regime the history lacks; generators can inject artefacts a policy learns to exploit |
| **B3. Drift randomisation — mirrored markets.** With probability ½, train an episode on the mirrored path $\ln C'_m=2\ln C_\tau-\ln C_m$ (highs and lows swapped, funding sign flipped). An unconditional long tilt then earns nothing on average, and only conditional direction pays. | DESIGN PROPOSAL by analogy to domain randomisation (Tobin et al., IROS 2017) and augmentation in RL (Laskin et al., *RL with Augmented Data*, NeurIPS 2020); no trading study found that does exactly this | the long lean, directly | BTC's asymmetries are real: crashes are faster than rallies, and mirroring creates the opposite. It can teach a symmetry the market does not have |
| **B4. Train at an inflated cost.** Charge 2–5× the real cost in training and evaluate at the real cost, so turnover is penalised. | Deng et al. (2017) train at *"five times higher than the real TCs"*; Zhang, Zohren & Roberts (2020) treat cost *"as a regularising term"*; the walk-forward study selected 2× and 8× on this data | the higher turnover at pass 100 (about 305 orders against 80) | **already tried**; it reduces cost, not directional error |

**C. Training parameters.**

| remedy | literature | addresses | risk |
|---|---|---|---|
| **C1. Regularise the policy network**: L2 weight decay, or dropout on the actor head. | Liu, Li, Kang & Darrell, *Regularization Matters in Policy Optimization* (ICLR 2021): *"only regularizing the policy network is typically the best"*; Cobbe et al. (2019): L2, dropout, data augmentation and batch normalisation improve generalisation | cause 2 | a new constant that needs its own evidence |
| **C2. Fewer passes.** The validation curve has no rising region after about pass 15–30, and pass 100 did worse on 2026. | Zhang, Vinyals, Munos & Bengio (2018) on overfitting in RL; Prechelt (1998) | cause 2 | 2024 already showed the early checkpoint is noise too; this helps only together with A2 |
| **C3. Robust objectives.** Adversarial perturbation of dynamics, or a risk-constrained objective. | Pinto, Davidson, Sukthankar & Gupta, *Robust Adversarial RL* (ICML 2017); Chow, Ghavamzadeh, Janson & Pavone (JMLR 2017) | cause 3 | costly, and weakly connected to this failure |

**D. State information.**

| remedy | literature | addresses | risk |
|---|---|---|---|
| **D1. A slow trend horizon.** The state reaches back four weeks and the memory eight days (§17.9); time-series momentum is documented at 1–12 month look-backs. Add one trend measure at 12–26 weeks, chosen by the existing relevance/redundancy/stability analytics on 2017–2023. | Moskowitz, Ooi & Pedersen (2012); Hurst, Ooi & Pedersen (2017) | cause 3: a months-long decline becomes visible | the 4-week efficiency ratio was excluded at ρ = 0.81–0.84 with `drawdown_4w`; a longer horizon must clear the same test |
| **D2. An explicit regime probability.** The posterior of a persistent-state model (a statistical jump model, or a Markov-switching model) fitted on the training span only, as one feature. | Nystrup, Lindström & Madsen, *Learning hidden Markov models with persistent states by penalizing jumps* (Expert Systems with Applications 2020); Hamilton (1989) | cause 3, compactly | a regime label estimated in-sample can still be wrong at a regime never seen |
| **D3. Audit non-stationary levels.** `volatility_level` is a log level standardised with the training mean. BTC's volatility has trended down over the years, so its z-score drifts out of sample. Measure the drift on 2025 (not on 2026) and replace levels with ratios if it is large. | Andrychowicz et al. (2021) on observation normalisation; Song, Jiang, Tu, Du & Neyshabur, *Observational Overfitting in RL* (ICLR 2020) | cause 3 | small |

**E. Architecture** — the least supported lever.

| remedy | literature | addresses | risk |
|---|---|---|---|
| **E1. Smaller, not larger.** With about 28 independent scenarios, capacity to memorise is the danger. Try halving the widths, or a feed-forward policy on explicit multi-horizon features instead of an LSTM that can memorise era-specific paths. | Engstrom et al. (ICLR 2020) and Andrychowicz et al. (ICLR 2021): implementation and training choices dominate architecture in on-policy RL; Cobbe et al. (2019) found deeper networks help — with high-dimensional images and thousands of levels, the opposite of this problem | cause 2 | may underfit a real but weak signal |
| **E2. Seed ensembles with hysteresis.** Average several seeds' decisions, but enter or exit only when the majority agrees and holds. | Yang et al. (2020) ensemble strategy | cause 4 | MEASURED here on 2024: averaging probabilities *raised* churn to 326–410 orders a quarter and lost −0.436; a 3-of-5 majority vote made +0.037. Ensembles need a hysteresis rule |

### 18.4 Recommendation — which lever first

**Do not start with architecture.** Nothing in the evidence points at capacity or the network's form. The
literature on on-policy RL says architecture is not where results are won, and with about 28 independent
quarters a bigger network overfits sooner.

**In order:**

1. **Feasibility gate (A5), before any new training.** Run the simple published rules through this exact
   ledger, walk-forward 2018–2025, with a deflated-Sharpe criterion fixed in advance. Two earlier studies in
   this project point to failure here. If no rule clears the cost at an hourly-to-daily horizon, the
   productive changes are to the **problem**, not the agent: a longer holding horizon, lower execution cost
   (maker rather than taker), or carry as the source of return.
2. **Fix the protocol (A1, A2)** in the same step, because any later claim needs it: seal the post-August
   2026 data; select with purged cross-validation and a deflated Sharpe.
3. **If the gate is passed, change the training data before anything else (B1, then B3).** It is the lever
   the generalisation literature ranks first, and it targets the observed failure — the long lean — directly.
   Keep B4, which is already known to be necessary.
4. **Then the state (D1, D2)**, so that a months-long regime is observable at all. Select any new feature
   with the existing analytics on 2017–2023, never on returns.
5. **Then training parameters (C1, C2)**, together with checkpoint averaging (A3).
6. **Architecture last (E1)**, and only toward smaller.

**To answer the question directly:** training data and protocol first; state information second; training
parameters third; architecture last. The largest risk is that the gate says the edge is not there. That is
the most useful result available: it would save another full study.

---

## References

**Reinforcement learning — algorithms and implementation**
- Schulman, Wolfe, Dhariwal, Radford & Klimov. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017 — Table 5 (Atari).
- Schulman, Moritz, Levine, Jordan & Abbeel. High-Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR 2016.
- Andrychowicz et al. What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study. ICLR 2021, arXiv:2006.05990.
- Engstrom, Ilyas, Santurkar, Tsipras, Janoos, Rudolph & Madry. Implementation Matters in Deep Policy Gradients. ICLR 2020.
- Hsu, Mendler-Dünner & Hardt. Revisiting Design Choices in Proximal Policy Optimization. arXiv:2009.10897, 2020.
- Huang, Dossa, Ye, Braga, Chakraborty, Mehta & Araújo. CleanRL. JMLR 2022 — `ppo_atari_lstm.py` (defaults verified 2026-09-11).
- Kapturowski, Ostrovski, Quan, Munos & Dabney. Recurrent Experience Replay in Distributed Reinforcement Learning. ICLR 2019.
- Hausknecht & Stone. Deep Recurrent Q-Learning for Partially Observable MDPs. AAAI Fall Symposium 2015.
- Mnih et al. Human-Level Control through Deep Reinforcement Learning. Nature 2015.
- Mnih et al. Asynchronous Methods for Deep Reinforcement Learning. ICML 2016.
- Kalyanakrishnan et al. An Analysis of Frame-skipping in Reinforcement Learning. arXiv:2102.03718, 2021.
- Pardo, Tavakoli, Levdik & Kormushev. Time Limits in Reinforcement Learning. ICML 2018.
- Sutton & Barto. Reinforcement Learning: An Introduction, 2nd ed. 2018 — §5.3.
- Jozefowicz, Zaremba & Sutskever. An Empirical Exploration of Recurrent Network Architectures. ICML 2015.

**Overfitting, stopping and failure analysis**
- Prechelt. Early Stopping — But When? In Neural Networks: Tricks of the Trade, LNCS 1524, Springer 1998.
- Zhang, Vinyals, Munos & Bengio. A Study on Overfitting in Deep Reinforcement Learning. arXiv:1804.06893, 2018.
- Sato. Model-Free Reinforcement Learning for Financial Portfolios: A Brief Survey. arXiv:1904.04973, 2019.
- Gort, Liu, Zheng, Gao, Wang, Wang, Zhu & Walid. Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting. arXiv:2209.05559, 2022.
- Haarnoja et al. Soft Actor-Critic Algorithms and Applications. arXiv:1812.05905, 2018.
- Mei, Xiao, Szepesvári & Schuurmans. On the Global Convergence Rates of Softmax Policy Gradient Methods. ICML 2020.
- Mei, Xiao, Dai, Li, Szepesvári & Schuurmans. Escaping the Gravitational Pull of Softmax. NeurIPS 2020.
- Mei, Dai, Xiao, Szepesvári & Schuurmans. Understanding the Effect of Stochasticity in Policy Optimization. NeurIPS 2021.
- Liu & Jing. arXiv:2609.05508, 2026 (preprint, not peer reviewed).
- Harb, Bacon, Klissarov & Precup. When Waiting is not an Option. AAAI 2018.
- Harutyunyan, Dabney, Borsa, Heess, Munos & Precup. The Termination Critic. AISTATS 2019.
- Lyle, Zheng, Nikishin, Avila Pires, Pascanu & Dabney. Understanding Plasticity in Neural Networks. ICML 2023.
- Ba, Kiros & Hinton. Layer Normalization. arXiv:1607.06450, 2016.
- Juliani & Ash. A Study of Plasticity Loss in On-Policy Deep Reinforcement Learning. NeurIPS 2024.
- van Hasselt, Guez, Hessel, Mnih & Silver. Learning Values Across Many Orders of Magnitude. NeurIPS 2016.
- Lipton et al. Combating Reinforcement Learning's Sisyphean Curse with Intrinsic Fear. arXiv:1611.01211.
- Chow, Ghavamzadeh, Janson & Pavone. Risk-Constrained RL with Percentile Risk Criteria. JMLR 2017.
- Sootla et al. Sauté RL. ICML 2022.

**MDP and state design**
- Dietterich, Trimponias & Chen. Discovering and Removing Exogenous State Variables and Rewards for RL. ICML 2018.
- Subramanian, Sinha, Seraj & Mahajan. Approximate Information State for Approximate Planning and RL in Partially Observed Systems. JMLR 2022.
- Song, Jiang, Tu, Du & Neyshabur. Observational Overfitting in Reinforcement Learning. ICLR 2020.

**Trading reinforcement learning (local library `articles/problems/optimal trading/`)**
- Moody & Saffell. Learning to Trade via Direct Reinforcement. IEEE TNN 2001.
- Deng, Bao, Kong, Ren & Dai. Deep Direct Reinforcement Learning for Financial Signal Representation and Trading. IEEE TNNLS 2017.
- Huang. Financial Trading as a Game: A Deep Reinforcement Learning Approach. 2018.
- Zhang, Zohren & Roberts. Deep Reinforcement Learning for Trading. J. Financial Data Science 2020.
- Théate & Ernst. An Application of Deep Reinforcement Learning to Algorithmic Trading. Expert Systems with Applications 2021.
- Yang, Liu, Zhong & Walid. Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy. ICAIF 2020.

**Finance and econometrics — regimes and features**
- Hamilton. A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle. Econometrica 1989.
- Ardia, Bluteau & Rüede. Regime Changes in Bitcoin GARCH Volatility Dynamics. Finance Research Letters 2019.
- Pagan & Sossounov. A Simple Framework for Analysing Bull and Bear Markets. J. Applied Econometrics 2003.
- Lunde & Timmermann. Duration Dependence in Stock Prices: An Analysis of Bull and Bear Markets. JBES 2004.
- Kaufman. Smarter Trading: Improving Performance in Changing Markets. McGraw-Hill 1995.
- Lo & MacKinlay. Stock Market Prices Do Not Follow Random Walks. RFS 1988.
- Corsi. A Simple Approximate Long-Memory Model of Realized Volatility. J. Financial Econometrics 2009.
- Liu, Patton & Sheppard. Does Anything Beat 5-Minute RV? J. Econometrics 2015.
- Barroso & Santa-Clara. Momentum Has Its Moments. JFE 2015.
- Moreira & Muir. Volatility-Managed Portfolios. J. Finance 2017.
- Barber & Odean. All That Glitters. RFS 2008.
- Liu & Tsyvinski. Risks and Returns of Cryptocurrency. RFS 2021.
- He, Manela, Ross & von Wachter. Fundamentals of Perpetual Futures. 2024.
- Koijen, Moskowitz, Pedersen & Vrugt. Carry. JFE 2018.
- Sortino & van der Meer. Downside Risk. J. Portfolio Management 1991.
- BitMEX. Reduction in Base Maintenance Margin — XBT and ETH Series, 13 April 2020.

**Evaluation**
- Agarwal, Schwarzer, Castro, Courville & Bellemare. Deep RL at the Edge of the Statistical Precipice. NeurIPS 2021.
- Bailey, Borwein, López de Prado & Zhu. The Probability of Backtest Overfitting. J. Computational Finance 2017.

**Generalisation, robustness and resampling (§18)**
- Cobbe, Klimov, Hesse, Kim & Schulman. Quantifying Generalization in Reinforcement Learning. ICML 2019, PMLR 97:1282–1289.
- Kirk, Zhang, Grefenstette & Rocktäschel. A Survey of Zero-shot Generalisation in Deep Reinforcement Learning. JAIR 76:201–264, 2023.
- Liu, Li, Kang & Darrell. Regularization Matters in Policy Optimization — An Empirical Study on Continuous Control. ICLR 2021, arXiv:1910.09191.
- Laskin, Lee, Stooke, Pinto, Abbeel & Srinivas. Reinforcement Learning with Augmented Data. NeurIPS 2020.
- Tobin, Fong, Ray, Schneider, Zaremba & Abbeel. Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. IROS 2017.
- Pinto, Davidson, Sukthankar & Gupta. Robust Adversarial Reinforcement Learning. ICML 2017.
- Sagawa, Koh, Hashimoto & Liang. Distributionally Robust Neural Networks for Group Shifts. ICLR 2020.
- Izmailov, Podoprikhin, Garipov, Vetrov & Wilson. Averaging Weights Leads to Wider Optima and Better Generalization. UAI 2018.
- Politis & Romano. The Stationary Bootstrap. JASA 89(428), 1994.
- Wiese, Knobloch, Korn & Kretschmer. Quant GANs: Deep Generation of Financial Time Series. Quantitative Finance 20(9), 2020.
- Koshiyama, Firoozye & Treleaven. Generative Adversarial Networks for Financial Trading Strategies Fine-Tuning and Combination. Quantitative Finance 21(5), 2021 (online 2020), arXiv:1901.01751.
- Nystrup, Lindström & Madsen. Learning Hidden Markov Models with Persistent States by Penalizing Jumps. Expert Systems with Applications 150:113307, 2020.
- Ang & Timmermann. Regime Changes and Financial Markets. Annual Review of Financial Economics 4, 2012.
- Moskowitz, Ooi & Pedersen. Time Series Momentum. JFE 104(2), 2012.
- Hurst, Ooi & Pedersen. A Century of Evidence on Trend-Following Investing. J. Portfolio Management 44(1), 2017.
- Daniel & Moskowitz. Momentum Crashes. JFE 122(2):221–247, 2016.
- Bailey & López de Prado. The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. J. Portfolio Management 40(5), 2014.
- López de Prado. Advances in Financial Machine Learning. Wiley 2018 — ch. 11–12.
