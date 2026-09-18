# Agent architecture

**Status:** current as of 2026-09-10. Package `src/agent/`.
**Verification: reference PPO FAILED 0/4 (§11.3); all-action PPO FAILED 0/4 (§11.6);
the MPO step on learned action values passed its diagnostic at a ten-times edge and
has not yet been verified at the planted edge (§11.7). The real-data training has
NOT been launched.**
**Update this file whenever the agent changes.** A test fails if a constant in
`Config` is missing from the provenance table in §12.

Every constant is tagged with exactly one provenance class:

| class | meaning |
|---|---|
| **CITED** | taken from a named study or reference implementation, with the passage or table |
| **DERIVED** | follows from other constants or from the problem's arithmetic |
| **MEASURED** | measured on this data or this host, with the number |
| **USER** | a decision the user made explicitly |
| **DESIGN CHOICE** | none of the above — stated openly as unverified |

---

## 0. What changed in this generation

| area | previous agent (stopped 2026-09-10) | now | evidence |
|---|---|---|---|
| action evaluation | one sampled action per state, GAE advantage | every action on the same bar by exact counterfactual branches, at all three reachable accounts | Huang 2018; Ciosek & Whiteson 2018; Sinclair et al. 2023 (§7) |
| policy step | clipped ratio surrogate | MPO E-step under a KL bound, fitted by cross-entropy | Abdolmaleki et al. 2018; Mei et al. 2020 (§7.2) |
| action head | keep/switch gate × direction softmax, learnable continuation bias | one softmax over target positions SHORT / FLAT / LONG | trading-RL standard (§4.1); termination-degeneracy literature (§1.1) |
| exploration | two self-tuning entropy duals with targets, init, lr, floor and cap | fixed entropy bonus 0.01 | PPO paper, CleanRL; Mei et al. 2020 (§7) |
| extra objective | action-enumerated counterfactual branches (5 untested constants) | removed | our logs: did not prevent collapse (§13) |
| advantage scale | slow global EMA (decay 0.05, untested) | per-minibatch normalisation | PPO reference, CleanRL, Andrychowicz et al. |
| reward | first-order differential Sharpe, clipped at ±10 | exact increment of the same Sharpe ratio, no clip | Moody & Saffell 2001 (§4.5) |
| market state | 60 one-minute bars × 6 through their own LSTM + 13 raw slow features | 13 dimensionless 5-minute features into one LSTM | §4.2, §4.4 |
| account state | 18 inputs through their own LSTM | 7 Markov-sufficient inputs, no recurrence | Subramanian et al. 2022; Song et al. 2020 |
| normalisation | none | LayerNorm after every hidden layer | Lyle et al. 2023; Ba et al. 2016 |
| episodes | 14.2 days | one quarter (92.4 days), staggered | user request; Yang et al. 2020 (§8) |
| folds | validate 2023, 2024, 2025 | validate 2024, 2025 (train through 2023 / 2024) | user decision |
| selection | median over runs, 90-day blocks stepped 15 days, "55% positive" rule | IQM, stratified bootstrap CI, probability of backtest overfitting; no invented threshold | Agarwal et al. 2021; Bailey et al. 2017 |
| cost | 38–48 s per update, 882,251 parameters | **5.2 s per update**, 834,820 parameters | measured |

---

## 1. Why the previous agent failed

### 1.1 Problem 1 — the entry side locked

Measured over the stopped run's training rollouts (share of time short / flat / long):

| run | updates 1–200 | 401–600 | 801–1000 | validation @1000 | long share of real entries @500 → @1000 |
|---|---|---|---|---|---|
| 2023 s11 | .12/.71/.17 | .91/.06/.03 | **.96**/.03/.01 | −0.750 in a rising year | 0.42 → **0.00** |
| 2023 s29 | .08/.28/.64 | .03/.14/.84 | .01/.04/**.95** | +0.480 | 0.91 → 0.93 |
| 2024 s11 | .15/.21/.64 | .01/.03/.96 | .00/.03/**.96** | +0.362 | 0.85 → **1.00** |
| 2024 s29 | .09/.41/.50 | .01/.14/.85 | .00/.01/**.99** | +0.325 | 0.97 → **1.00** |
| 2025 s11 | .07/.12/.82 | .01/.05/.94 | .01/.04/**.95** | −0.503 in a falling year | 0.48 → **1.00** |

Four of five runs committed to one side, and the commitment strengthened with
training. The direction dual sat at its 1e-4 floor (median 0.0001–0.0004) the
whole time. The failure is documented in the literature at three levels:

- **The regulariser averaged over the wrong states.** SAC's automatic temperature,
  which both duals copied, is by construction a constraint on *average* entropy:
  *"we formulate a constrained optimization problem where the average entropy of
  the policy is constrained, while the entropy at different states can vary"*
  (Haarnoja et al. 2018, arXiv:1812.05905). Flat states were 0.1–7% of samples, so
  the average met its 0.6 target while the entry decision was near-deterministic.
- **The gate was a learned termination function, and those degenerate.** Harb et
  al. (AAAI 2018) show option termination collapses without a deliberation cost;
  Harutyunyan et al. (AISTATS 2019) that return-maximising termination is
  ill-posed; Liu & Jing (arXiv:2609.05508, preprint, not peer reviewed) name the
  mechanism: *"A state commits to whichever action was first sampled with a
  positive advantage; the score function then vanishes, and the state never
  updates again."* Their remedy is exploration at the lower level.
- **Committal behaviour is intrinsic to stochastic softmax policy gradient.** Mei
  et al. (NeurIPS 2021) prove a trade-off: an algorithm either converges to the
  global optimum with probability 1 at a rate no better than O(1/t), or converges
  faster but fails with positive probability. Entropy regularisation changes the
  exact-gradient rate from O(1/t) to linear (Mei et al., ICML 2020).

In trading specifically, Théate & Ernst (2021) observe DRL agents that *"simply
stop actively trading and adopt a passive approach (buy and hold or sell and
hold strategies)"*, and Sato (2019) that repeated training on one historical path
captures patterns that do not generalise. The training data are also net long:
56–60% of 14- and 92-day windows in 2017–2024 rose, and constant long had a
Sharpe ratio of 0.63 (§11.2 tests exactly this attractor).

### 1.2 Problem 2 — liquidations were not learned as catastrophic

74 training liquidations by update ~1,200, **73 of them in the four long-locked
runs** and one in the short-locked run; none in validation. Per 200 updates the
2024 s29 run went 0, 2, 4, 8, 8, 2 — not decreasing.

- **The clip made a wipeout cost the same as a slide.** Every liquidation step
  paid exactly −10 × 0.059. Clipping *"can result in qualitatively different
  behavior"* (van Hasselt et al., NeurIPS 2016). §4.5 removes the clip.
- **Rare catastrophes are forgotten.** DRL agents *"periodically revisit these
  states upon forgetting their existence under a new policy"* (Lipton et al.,
  Intrinsic Fear).
- **A Sharpe ratio is blind to wealth.** It is invariant to the level of equity,
  so no Sharpe-type reward can by itself price ruin; termination must (§4.6).
- **The dominant driver was problem 1.** A 1× inverse long is liquidated by a
  49.7% fall; a 1× short only by a 1,657-fold rally (§4.3). Side-locked long
  policies held through the 2018, 2020 and 2021 crashes.

### 1.3 The common root

The previous design carried **17 constants with no source and no test** —
gate and direction targets, dual initial value, cap and floor, advantage-scale
decay, five counterfactual constants, the reward clip and variance floor,
episode length, three hidden widths, the evaluation block schedule — plus one
**wrong citation** (the PPO paper's Atari table uses 3 epochs, not 4). Two of
them sat exactly where the failures happened. §13 lists every one and what
became of it.

---

## 2. The one measurement, and the one assumption

The five-minute log-return standard deviation on the cached BitMEX XBTUSD series
is **14.8 bp** (MEASURED). Commission is 5 bp and adverse slippage 1 bp per leg
(USER), so a round trip costs **12 bp = 0.81 of one bar's typical move**.

A signal worth an annualised Sharpe ratio of **1.0 — an ASSUMPTION, not a
measurement** — produces 0.046 bp of expected drift per bar and needs
`0.0012 / 0.0000046 = 264 bars = 22 hours` in position to pay for one entry and
one exit. The discount horizon (§5) is anchored to that break-even. A weaker
signal would need longer holds; a stronger one shorter.

---

## 3. Problem setup

| | |
|---|---|
| Instrument | one BitMEX XBTUSD inverse perpetual, BTC collateral |
| Decisions / accounting | every 5 minutes / every 1 minute |
| Actions | target position `SHORT`, `FLAT`, `LONG` |
| Position size | 1× account equity at entry, never rebalanced |
| Costs | 5 bp commission + 1 bp slippage per leg, fixed, no curriculum |
| Folds | train 2017–2023 → validate **2024**; train 2017–2024 → validate **2025** |
| Test | **everything from 2026-01-01 is locked; no test entrypoint exists** |

Expanding walk-forward folds follow Zhang, Zohren & Roberts (2020): *"We retrain
our model at every 5 years, using all data available up to that point."*

---

## 4. The MDP

### 4.1 Decisions and actions

A categorical choice of target position, the representation of Moody & Saffell
(2001), Huang (2018) and Zhang, Zohren & Roberts (2020), who call it *target
orders*: keeping the current position costs nothing; moving from long to short
pays two legs. The previous gate factorisation is gone together with the eight
constants it needed (§13).

An order lasts one decision interval and fills at the first regular-trade open
inside it; with no printed trade the position is unchanged.

### 4.2 State — the exogenous market

The market is **exogenous**: a one-contract trader does not move the XBTUSD
price. Dietterich, Trimponias & Chen (ICML 2018) show that exogenous state and
reward components *"can slow down reinforcement learning by injecting
uncontrolled variation into the reward signal"* and that separating them is
what makes an MDP decomposable. Here the separation is architectural: market
features go through the recurrent memory, the account does not (§6).

Every feature is **dimensionless**. BTC's annualised volatility was 119% in 2017,
96% in 2018, 44% in 2023 and 45% in 2025 (MEASURED), so a raw return means a
different thing in each year. Divided by the ex-ante volatility it does not —
the construction of Moskowitz, Ooi & Pedersen (JFE 2012), used as an RL state by
Lim, Zohren & Roberts (2019) and Zhang, Zohren & Roberts (2020).

All features at boundary *t* use minutes strictly before *t* (tested).

| feature | construction | why | source |
|---|---|---|---|
| `bar_return` | last 5-min log return / σ_bar, where σ_bar² is the mean squared 5-min return over the last day (288 bars) | the return path the memory integrates | ex-ante vol scaling: Moskowitz et al. 2012. Five-minute returns: *microstructure noise biases finer sampling*, Liu, Patton & Sheppard (JoE 2015). A full day: intraday periodicity dominates high-frequency volatility, Andersen & Bollerslev (JEF 1997) |
| `bar_range` | log(high/low) of the bar / σ_bar | intrabar volatility | Parkinson (J. Business 1980) extreme-value estimator |
| `bar_volume` | log(1+bar volume) − log(1+mean bar volume over the last day) | abnormal volume | Barber & Odean (RFS 2008): abnormal volume as an attention proxy |
| `bar_coverage` | share of the bar's minutes with a printed trade | orders fill only against printed trades | mechanical (ledger rule) |
| `momentum_1d` | 1-day log return / (σ_bar √288) | daily time-series momentum | Liu & Tsyvinski (RFS 2021): *"For Bitcoin daily returns, the current return positively and statistically significantly predicts 1-day, 3-day, 5-day, and 6-day ahead returns"* |
| `momentum_1w` | 1-week log return / (σ_bar √2016) | weekly time-series momentum | Liu & Tsyvinski: *"For Bitcoin weekly returns, the current return … predicts 1-week, 2-week, 3-week, and 4-week ahead returns"* |
| `volatility_regime` | ½ log(σ²_day / σ²_4 weeks) | volatility is persistent, and it is the Sharpe reward's denominator | Andersen & Bollerslev 1997 (volatility persistence); window mirrors the attention construction below |
| `volume_regime` | log(1+daily volume last week) − log(1+daily volume over the four weeks before) | investor attention | Liu & Tsyvinski: attention *"predicts high future returns over 1-2 week horizons for Bitcoin"*; their measure is *"minus the average of previous four weeks"* |
| `funding_rate` | last settled funding rate | perpetual carry and crowding | He, Manela, Ross & von Wachter, *Fundamentals of Perpetual Futures* (2024); Koijen et al., *Carry* (JFE 2018) |
| `funding_phase_sin/cos` | position in the 8 h settlement cycle (MEASURED: settlements at 04:00, 12:00, 20:00 UTC, interval constant at 8 h) | a settlement at the decision minute is paid before the order fills | mechanical; sinusoidal encoding as Huang (2018) and Carapuço, Neves & Horta (2018) |
| `hour_sin/cos` | time of day | intraday periodicity | Andersen & Bollerslev 1997; encoding as Huang 2018 |

The longest look-back is five weeks, so the cache starts on 2016-11-26 and
training still begins on 2017-01-01.

**Measured on the rebuilt cache**, the scaled features have nearly the same
spread in training and in validation — `bar_return` sd 1.028 vs 1.037,
`momentum_1d` 0.882 vs 0.898, `momentum_1w` 1.044 vs 1.104 for the 2024 fold —
despite validation volatility being half the training average. Fewer than 0.1%
of values reach the observation clip.

### 4.3 State — the endogenous account

The account part is the smallest set that makes reward and termination Markov
given the market — an *information state* in the sense of Subramanian, Sinha,
Seraj & Mahajan (JMLR 2022):

| feature | why it is needed |
|---|---|
| `short`, `flat`, `long` one-hot | the cost of each action depends on the side held (Huang 2018 encodes position the same way) |
| `trade_return` — open trade's log return / one-day ex-ante volatility | inverse-contract exposure drifts with the trade's P&L (Carapuço et al. 2018 include unrealised P&L) |
| `exposure` — log(1 + \|contracts\| / price / equity) | **liquidation happens exactly when this exceeds log(1 + 1/(maintenance + fee)) = 5.21** (tested). A 1× long reaches it after a 49.7% fall; a 1× short needs a 1,657-fold rally |
| `running_sharpe`, `relative_volatility` | the reward is a function of the two Sharpe statistics; without them the same action in the same market pays differently |

### 4.4 What was removed from the state, and why

State designs in the trading-RL studies of the local library:

| study | market state | account state |
|---|---|---|
| Moody & Saffell 2001 | recent returns | previous position |
| Deng et al. 2017 | raw recent price changes, features learned; they note *"a widely known drawback of technical analysis is its poor generalization ability"* | previous action |
| Huang 2018 | 8 most recent log returns of close and tick volume, running z-score over 96, clipped at 10; minute, hour, weekday encodings | position one-hot |
| Carapuço et al. 2018 | price features; sin/cos time of day | position, unrealised profit |
| Zhang, Zohren & Roberts 2020 | normalised close; returns over 1, 2, 3 months and 1 year normalised by daily volatility; MACD; RSI | previous position |
| Théate & Ernst 2021 | OHLCV of the last τ days | position |
| Jiang, Xu & Liang 2017 | prices of the last 50 periods normalised by the latest close | previous weights |
| **previous agent** | 60 one-minute bars × 6 features through a dedicated LSTM, plus 13 raw slow features | 18 inputs through a second LSTM |
| **now** | 13 dimensionless features at the decision interval | 7 inputs |

Multi-horizon inputs are standard (Zhang et al. use horizons from one month to
one year). What had no counterpart in any of these studies was a **second
recurrent encoder over one-minute bars inside every decision step**. It:

1. duplicated the market LSTM, whose job is exactly to integrate the per-step
   history (Hausknecht & Stone 2015 replace stacked frames with recurrence);
2. fed raw, unscaled one-minute returns whose meaning changes with each year's
   volatility;
3. ran 60 recurrent steps inside every decision step of both branches, against
   one step for the market memory. The redesign as a whole took an update from
   38–48 s to 5.2 s; the encoder's share of that was not measured separately.

Also removed: `hours_since_funding` and `last_known_funding_interval_hours` (the
interval is a constant 8 h in all 78,888 hourly samples, so it carried no
information), unnormalised momentum at seven hand-picked horizons, the one-hour
volume ratio, and eleven account inputs not needed for the Markov property
(position age, favourable and adverse excursion, trade and account drawdown,
maintenance headroom, trade fees and funding, previous reward, previous-action
one-hot) with their ×10 multipliers and ±5 clips. Each spurious input is one more
thing to overfit: agents *"may mistakenly correlate reward with certain spurious
features from the observations"* (Song et al., *Observational Overfitting in
Reinforcement Learning*, ICLR 2020). Sato (2019) warns in the other direction that
price-only states *"are an underrepresentation of the market environment"*; the
BTC-only constraint rules out cross-asset and external data, so the features are
chosen from what the crypto-return literature finds predictive on BTC's own data.

### 4.5 Reward — the exact differential Sharpe increment

Moody & Saffell (IEEE TNN 2001) track exponential estimates of the first two
moments of return,

```
A_t = A_{t-1} + η (R_t − A_{t-1})        B_t = B_{t-1} + η (R_t² − B_{t-1})        S = A / sqrt(B − A²)
```

and obtain the differential Sharpe ratio by *"expanding to first order in the
adaptation rate"*. The reward is now the **exact** change,

```
D_t = (S(A_t, B_t) − S(A_{t-1}, B_{t-1})) / η,        reward = D_t × sqrt(1 − γ²)
```

where `R_t` is the net log BTC equity change including commission, slippage and
funding. For ordinary returns it equals the first-order term (tested within 0.02
for |z| ≤ 3). For extreme returns it stays bounded by 1/sqrt(η(1−η)) = 37.96,
because one huge return inflates the denominator as fast as the numerator:

| one-step return | exact increment | first-order term | previous (clipped) |
|---|---:|---:|---:|
| +1 sd | +1.00 | +1.00 | +1.00 |
| −3 sd | −2.99 | −3.00 | −3.00 |
| −0.5% | −3.38 | −3.39 | −3.39 |
| **−5%** | **−25.6** | −34.7 | **−10** |
| **−50%** | **−37.8** | −468 | **−10** |
| wipeout | −37.96 | −18,670 | −10 |

The clip and the variance floor (§13) are gone. What remains is the Sharpe
ratio's own limitation: −50% and a wipeout are nearly the same reward, because
the ratio is invariant to wealth. That is why termination carries ruin (§4.6).

Why a Sharpe-type reward at all: across 78 predecessor runs trained on plain log
growth, net directional exposure alone explained R² = 0.96 / 0.82 / 0.63 of
validation return. A ratio does not reward leverage resets (exposure scales
return and volatility together) while their fees are pure loss (tested).

| constant | value | provenance |
|---|---|---|
| `dsr_eta` | 1/1440, a five-day window | DESIGN CHOICE inside DERIVED bounds: slower than the 286-step credit window (else the reward's own drift reads as signal), faster than an episode (so it settles). No study pins it; Moody & Saffell note only that η = 0 is an infinite average |
| `dsr_initial_second_moment` | 14.8 bp² | MEASURED — a fresh account starts with the asset's own variance |
| reward scale | sqrt(1 − γ²) = 0.0591 | DERIVED: the discounted sum of a unit-variance sequence has sd 1/sqrt(1 − γ²) ≈ 17 |

**Known property:** with `R_t = 0` the increment has the sign opposite to `A`, so
an agent that has been losing is paid to stay flat. That is correct Sharpe
accounting. Both `R_t` and `D_t` are logged every step.

### 4.6 Transitions, termination and time limits

- **Liquidation terminates** the episode: no reward and no bootstrap afterwards,
  so the loss of every future reward is the cost of ruin — the one channel a
  wealth-invariant reward leaves. BitMEX XBTUSD base maintenance margin was 0.50%
  until 17 April 2020 and 0.45% afterwards (BitMEX announcement, 13 April 2020);
  the higher value is used throughout, conservative by 0.05% of notional.
- **Episode ends are time limits**, bootstrapped with the critic: *time limits and
  true terminations must be treated differently* (Pardo et al., ICML 2018). GAE
  masks the bootstrap only on liquidation (tested).
- Adverse one-minute extremes (low for longs, high for shorts) trigger the margin
  check — a conservative approximation, not a reconstruction of BitMEX's engine.

---

## 5. Horizon

| | steps | hours | provenance |
|---|---:|---:|---|
| cost break-even | 264 | 22.0 | DERIVED from §2, including its assumption |
| GAE credit window `1/(1−γλ)` | 286 | 23.8 | USER: γ = λ = 0.99825 |
| γ horizon `1/(1−γ)` | 571 | 47.6 | DERIVED |
| `sequence` | 256 | 21.3 | DERIVED: `burn_in + sequence` must cover the credit window |
| `burn_in` | 128 | 10.7 | CITED: R2D2 stores 80-step sequences behind a 40-step burn-in, ratio 0.5 (Kapturowski et al. 2019) |
| `rollout` | 1024 | 85.3 | DERIVED: the average step sits half a rollout before the bootstrap and must still see a credit window |

Andrychowicz et al. recommend *"GAE with λ = 0.9"* for MuJoCo; that horizon (10
steps, 50 minutes here) is far below the cost break-even, which is the failure
that killed v5 (entropy fell in 78 of 78 runs at a 2.9 h window). γ is *"one of
the most important hyperparameters and should be tuned per environment"* — here
it is set by the break-even arithmetic and fixed by the user.

---

## 6. Network

```
market features (13) ─► Linear ─► LayerNorm ─► tanh ─► LSTM ─► LayerNorm ─┐
                                                                            ├─► Linear ─► LayerNorm ─► tanh ─► Linear ─► LayerNorm ─► tanh ─► Linear
account (7) ────────────────────────────────────────────────────────────────┘                                                              (3 logits | 1 value)
```

Separate actor and critic of this shape. Widths 128 (actor) and 256 (critic).
**834,820 parameters** (actor 169,219; critic 665,601).

- **Layer normalisation after every hidden layer.** Lyle et al. (ICML 2023,
  `articles/deep RL/lyle2023.pdf`) find normalisation layers give *"the greatest
  improvements to plasticity"*, then *"apply the best-performing intervention,
  layer normalization, to a standard DQN architecture and obtain significant
  improvements in performance on the Arcade Learning Environment"*; the only
  change was *"the incorporation of layer normalization after each hidden layer
  in the network"*. The gain and bias act *"after the normalization but before
  the non-linearity"* (Ba, Kiros & Hinton 2016), who also note LayerNorm is
  *"very effective at stabilizing the hidden state dynamics in recurrent
  networks"*. The LSTM is normalised at its output: Ba et al.'s in-cell LN-LSTM
  MEASURED 13.3× slower than the fused kernel (138.7 ms vs 10.4 ms per 384-step
  forward and backward, width 128), which would have multiplied a run's cost by
  about four.
- **Caveat on the on-policy evidence.** Juliani & Ash (NeurIPS 2024) find, for
  on-policy PPO, that LayerNorm resolves plasticity loss in training performance
  but is inconsistent for generalisation, and that it works best combined with a
  soft shrink-and-perturb. The latter is the next evidence-backed intervention if
  plasticity loss appears (§15).
- **The account does not enter the recurrent memory** (tested): it is Markov, so
  memory is spent on the market, and one market pass serves every sampling seed.
- Separate policy and value networks, tanh, orthogonal initialisation and a final
  policy layer 100× smaller follow Andrychowicz et al. (ICLR 2021): *"Initialize
  the last policy layer with 100× smaller weights"*; the resulting initial policy
  is uniform within 0.02 (tested). Hidden-layer gain √2, LSTM weight gain 1 and
  critic head gain 1 follow CleanRL `ppo_atari_lstm`; the forget-gate bias of 1
  follows Jozefowicz, Zaremba & Sutskever (ICML 2015).

---

## 7. Optimisation — all-action recurrent PPO

### 7.1 Why the sampled estimator was replaced

Reference PPO failed the planted-edge verification (§11.3), and the diagnostic in
§11.4 located the cause. GAE advantages of sampled actions sum a 286-bar credit
window of exogenous market noise, so a per-bar edge of 0.009 sd was invisible; with
costs, the policy committed to a side before the edge could register. Mei et al.
(NeurIPS 2021) prove that fast policy optimisation must commit with positive
probability *"in the absence of external oracle information, which allows an
algorithm to determine the difference between optimal and sub-optimal actions
given only on-policy samples"*. Their stochastic setting observes only the sampled
action's reward.

In this market that information is available. Under the *"zero market impact
hypothesis"* (Huang 2018) the reward of every action on a bar is computable, so the
trainer uses it.

### 7.2 What is computed

At every step, for each environment:

1. **Three sibling accounts**: the accounts reachable over the previous bar under
   SHORT, FLAT and LONG. One of them is the visited account (checked against the
   environment every step; MEASURED gap 0.0).
2. **Nine exact branches**: each sibling stepped under each action through the same
   five one-minute bars by the same `advance` accounting the environment uses,
   giving reward r(s, a), liquidation flag and next account s'_a.
3. **Action-value targets** r(s, a) + γ (1 − terminal) V(s'_a), with V the critic at
   each branch's next account on the shared next market memory. These are exact for
   that bar, and they are the regression target of the critic's action-value head
   (the critic outputs a state value and three action values).

The policy step reads the **head**, never the raw target: one bar's outcome is not
Q^π, MPO's E-step is defined on a learned critic, and Sinclair et al. distil
counterfactual values into a network for the same reason. Feeding the raw one-step
values to the step was tried first and failed (§11.7): at the planted edge strength
the policy committed to the sign of the current bar and ended anti-correlated with
the regime.

The policy step is MPO's (Abdolmaleki et al., ICLR 2018), taken at **all three
siblings** of every step:

```
Â(s, a) = Q(s, a) − Σ_b π_old(b|s) Q(s, b)
q(a|s)  ∝ π_old(a|s) · exp(Â(s, a) / η),   η set so that mean KL(q ‖ π_old) = policy_kl_step
loss    = − Σ_a q(a|s) log π_θ(a|s) − entropy_coefficient · H(π_θ(·|s))
```

Abdolmaleki et al. obtain q *"in closed form"* as *"q(a|s) ∝ π(a|s, θ) exp(Q(s,a)/η)"*
with η from a convex dual under *"E[KL(q(a|s), π(a|s, θ))] < ϵ"*. With three actions
the integral over actions is exact, so η is found by bisection on the realised KL,
which decreases in it. The M-step is their supervised fit to q — the forward KL,
which they note *"reduces the tendency of the entropy of the parametric policy to
collapse"*.

**Why not PPO's clipped surrogate here.** It was tried first, over the same
all-action values, and failed (§11.6). Its gradient on an action's logit carries
that action's own probability as a factor, so once a policy has become
deterministic it stops responding to its critic — the *gravitational pull of
softmax* (Mei et al. 2020). MPO's step moves in distribution space instead: the
gradient of the M-step loss is π_θ − q, which stays large for an action the critic
prefers even when that action's current probability is near zero. MEASURED: from a
probability of 1e-6 with a one-unit advantage, one step of ε = 0.1 lifts the
starved action above 1e-2.

- **Ciosek & Whiteson (AAAI 2018), expected policy gradients:** EPG *"integrates
  across the action when estimating the gradient, instead of relying only on the
  action in the sampled trajectory"*, and it *"reduces the variance of the gradient
  estimates"*.
- **Allen et al. (2017), Mean Actor-Critic:** it *"uses the agent's explicit
  representation of all action values to estimate the gradient"* to *"reduce
  variance in the policy gradient estimate"*.
- **Sinclair et al. (ICML 2023), MDPs with exogenous inputs:** *"Since we can
  simulate Q† for any action, we use the common random numbers insight (Ng &
  Jordan, 2000) for variance reduction and sum across all actions."* Their Q† comes
  from a hindsight planner and is biased upward. Here Q comes from the policy's own
  critic, so that hindsight bias does not arise.

The critic regresses the visited account on its GAE λ-return, which carries value
across the horizon. The two unvisited siblings regress on the one-step expected
target Σ_a π_old(a|s) Q(s, a) (expected SARSA). Training values at accounts the
policy is not holding is what stops the critic extrapolating to states it never
saw: the failure Fujimoto, Meger & Precup (ICML 2019) call extrapolation error.

### 7.3 Why the siblings guard against commitment

Updating at every reachable account is exploring starts (Sutton & Barto 2018, §5.3)
applied at every step, not only at episode start. A policy that holds short still
receives gradient at the long and flat accounts. The first policy-improvement step
("keep long when already long in an up regime") can therefore happen while the
policy is locked; only then does entering long become valuable. §11.5 measures
learning in exactly that order.

**One constant is introduced**, `policy_kl_step` = 0.1, MPO's own E-step bound. It
replaces two: the PPO clip and the per-minibatch advantage scale, because the
temperature adapts to the advantage scale by construction. The uniform weight over
the three siblings is exploring starts' uniform start distribution over the
account's side. The cost is 9.5 s per update against 5.2 s (MEASURED), for the 72
extra branch accounts stepped at every step.

### 7.4 Parameters

| parameter | value | provenance |
|---|---|---|
| `policy_kl_step` | 0.1 | CITED: MPO's E-step bounds the mean KL(q ‖ π_old) by ε; their experiments use 0.1 (Abdolmaleki et al., ICLR 2018, Table 2), as does V-MPO for DMLab single-task |
| `actor_lr`, `critic_lr` | 3e-4 | CITED: *"0.0003 is a safe default"* (Andrychowicz et al.) |
| `anneal_lr` | linear to zero | CITED: Andrychowicz et al.; CleanRL default |
| `gradient_clip` | 0.5 | CITED: CleanRL `max_grad_norm`; OpenAI baselines (Engstrom et al. 2020); Andrychowicz et al.: *"of secondary importance"* |
| `epochs` | 4 | CITED: CleanRL `ppo_atari_lstm` `update_epochs = 4`. (The PPO paper's Atari Table 5 uses **3**; the previous file wrongly cited it as 4.) |
| `recompute_advantages_per_pass` | on | CITED: recomputing each pass *"performs best among all variants"* (Andrychowicz et al.) |
| advantage normalisation | per minibatch | CITED: CleanRL `norm_adv = True`; Andrychowicz et al.: *"seems not to affect the performance too much"* |
| `entropy_coefficient` | 0.01 | CITED: PPO paper Atari Table 5 and CleanRL `ent_coef`. Entropy regularisation makes softmax policy gradient converge linearly rather than at O(1/t) (Mei et al., ICML 2020). Andrychowicz et al. found no regulariser helps *in continuous control*, where Gaussian policies explore by construction |
| `environments` | 8 | CITED: CleanRL `ppo_atari_lstm`; Andrychowicz et al. |
| `minibatch_sequences` | 4 | DERIVED: 4 × 256 = 1,024 samples per minibatch, 32 gradient steps per update |
| `updates` | 5,000 | CITED by budget parity: 5,000 × 32 = 160,000 actor gradient steps against CleanRL `ppo_atari_lstm`'s 10M / 1,024 × 4 × 4 = 156,240 |
| value loss | MSE, no value clipping | CITED: *"neither Huber loss nor PPO-style value loss clipping"* (Andrychowicz et al.) |
| KL early stop | none | CITED: CleanRL `target_kl = None` |

The recurrent state is rebuilt under current parameters from the stored prefix
for every minibatch, never taken stale from collection; replaying the rollout
must reproduce the collected probabilities and values to 5e-5 or the update
aborts (measured error: 0.0).

---

## 8. Training data

**Episodes are one quarter long** — 26 rollouts of 1,024 steps, 92.4 days — and
start at uniformly random points of the training span. The user proposed 2–3
month windows so that each year's different trend and magnitude would not be
learned as one block.

What the evidence says about that proposal:

- The previous agent already sampled random 14.2-day windows across the whole
  training span, not whole years. Longer episodes do **not** add regime
  diversity per update; eight parallel environments at eight random points do.
- Longer episodes **do** remove a train/validation mismatch: median validation
  holding spells reached 904 and 1,350 hours in one run and 165 hours in another,
  while a training episode lasted 340 hours, so the consequences of multi-week
  holds were never experienced in training. They also
  match the evaluation window exactly (§9) and the quarterly window FinRL's
  ensemble strategy retrains and validates on: *"we retrain our three agents at
  every three months"*, with *"a 3-month validation rolling window"* (Yang, Liu,
  Zhong & Walid, ICAIF 2020).
- **What actually addresses year-to-year magnitude is scale-free features**
  (§4.2): the measured feature spreads in the 44–54% volatility validation years
  match those of the 2017–2023 training span.

The first episode of each environment is shortened by a random whole number of
rollouts (DERIVED), so resets spread across updates rather than replacing all
eight market windows at once every 26 updates. Parallel actors exist to
decorrelate updates (Mnih et al., A3C, ICML 2016); synchronised resets would undo
that.

**Exploring starts** (Sutton & Barto 2018, §5.3): every episode starts flat,
short or long with equal probability (`flat_start_fraction` = 1/3), after a
128-step prefix with real fees, funding and entry prices, so exits are practised
from the first update. Measured on real data at initialisation: sides 0.33 /
0.33 / 0.34.

---

## 9. Evaluation and selection

- **Actions are sampled, never taken by argmax**, with `evaluation_seeds` = 4.
- **Each calendar quarter of the validation year is scored from a flat account** —
  the training episode length, and FinRL's validation window. Four quarters × two
  folds × three training seeds = 24 scores per checkpoint.
- **Aggregate: interquartile mean** of quarterly log return, with a **stratified
  bootstrap 95% interval** over seeds within each quarter, 50,000 replications
  (Agarwal et al., NeurIPS 2021; rliable defaults `reps = 50000`, percentile
  method). With three runs the interval will be wide, and that is the honest size
  of the uncertainty.
- **Checkpoint selection** takes the highest IQM over the 10 checkpoints. Because
  the best of ten is chosen on the data it is scored on, the **probability of
  backtest overfitting** is estimated by combinatorially symmetric
  cross-validation over the eight quarters: C(8, 4) = 70 in-sample/out-of-sample
  splits (Bailey, Borwein, López de Prado & Zhu, J. Computational Finance 2017).
- **Passive references** per quarter (constant long, short, flat) for context,
  never as a reward or a threshold.
- **Slippage sensitivity** at `stress_slippage` 1, 3 and 5 bp, reported and never
  used for selection.
- **No pass or fail threshold is invented.** The previous "median block above
  zero and 55% of blocks positive" rule had no source.

---

## 10. Diagnostics

Logged every update:

| metric | reads |
|---|---|
| `account_side_share`, `long_entry_share` | **problem 1**: side concentration, and the realised direction of actual entries from flat. The previous probability-averaged metric read 0.97 long in a run whose every real entry went short |
| `state_action_mutual_information[_flat]` | zero exactly when the policy ignores the market |
| `policy_entropy`, `switch_probability_mean`, `policy_implied_holding_hours` | exploration and holding |
| `liquidations_in_rollout` | **problem 2** |
| `mean_net_log_return`, `median_net_log_return`, `turnover_per_decision` | both reward signals; turnover is now relative to the trade's starting equity, so a liquidation can no longer book 2.6e12 |
| `explained_variance_post`, `target_std`, `value_rmse_{short,flat,long}` | critic tracking, per side |
| `approx_kl`, `clip_fraction`, `post_update_kl`, `replay_*_error` | trust region and exactness |

**Adequacy of `evaluation_seeds`**, MEASURED on the stopped run's full-year
validations: the within-run variance across sampling seeds averaged 0.032, while
the variance between training seeds' means ranged from 0.0008 (2024 at update
1000) to 0.67 (2023). Where training seeds disagree, more training seeds, not
more sampling seeds, narrow the result.

---

## 11. Verification before compute

### 11.1 Unit tests

`tests/test_agent.py`, 60 tests, all passing: feature causality (changing minute *k* cannot
change any observation before boundary *k+1*), scale invariance of every price
feature under doubled volatility and 1,000× volume, bounds by construction,
funding and clock encodings, the exact reward's reduction to Moody & Saffell's
term and its bound, telescoping, the liquidation threshold of the exposure input,
turnover at liquidation, balanced exploring starts, staggered episodes, LayerNorm
placement, near-uniform initial policy, account-free recurrence, GAE termination
versus time-limit handling, replay and resume exactness, entropy bonus effect,
quarter tiling, IQM, bootstrap and PBO on known cases, and the planted markets.
It also fails if any `Config` field is missing from §12.

### 11.2 Planted markets — pre-registered

`src/agent/synthetic.py`: a hidden regime flips sign with a ~48 h mean dwell and
adds drift while it holds; the previous minute's regime is written into the
`momentum_1w` column, with three noise columns as distractors. Held-out 90-day
window, real costs, two seeds each, 1,000 updates, the mean of the last three
evaluations.

| test | what it isolates | pre-registered pass |
|---|---|---|
| **symmetric** | can the machinery learn a planted edge at all | beat the better of constant long and constant short |
| **drift** | BTC's MEASURED in-sample Sharpe ratio (0.627, 2017–2023) added as unconditional drift, so constant long pays over the training span — the attractor that locked the previous agent | beat the better of constant long and constant short |

A policy locked onto either side cannot pass.

**Market construction, recorded before any result.** The symmetric test is the
previous generation's (600,000 minutes, market seed 7, training minutes
20,000–450,000, validation 460,000–590,000). The drift test was first built on
the same 299-day span, and MEASURED there it did not carry the attractor: constant
long lost 1.61 log (liquidated) while constant short gained 0.59, because an
unconditional Sharpe of 0.63 is invisible over 299 days of 48% volatility. The
two drift runs were stopped before their first evaluation. The drift test now
trains over **seven years** of the same process (the length of BTC's 2017–2023
span), same market seed, where **59.8% of 92-day training windows rise** against
**57.3% for BTC** (MEASURED), and validates on the following 90 days. Its
synthetic calendar starts in 2010 so that seven years end before the 2026 test
lock, which refused the first attempt; the generator does not use the calendar.

References on the held-out windows, MEASURED before training:

| market | oracle | constant long | constant short | pass if above |
|---|---:|---:|---:|---:|
| symmetric | +0.415 | −0.191 | +0.159 | **+0.159** |
| drift | +0.612 | +0.141 | −0.166 | **+0.141** |

In the drift market the in-sample attractor (long) is also the better constant
out of sample, so an always-long policy ties the criterion and fails.

**Result: FAILED, 0 of 4.** See §11.3. Per the rule set before launch, the six
real-data runs were **not** started.

### 11.3 Results — 2026-09-10

1,000 updates each, held-out 90-day window, mean of the last three evaluations
(two sampling seeds each):

| run | log return | pass above | share long when regime up / down | share short when up / down | outcome |
|---|---:|---:|---|---|---|
| symmetric s11 | **−2.04** | +0.159 | 0.37 / 0.36 | 0.29 / 0.36 | no conditioning; still trading about hourly |
| symmetric s29 | **−0.82** | +0.159 | 0.01 / 0.07 | 0.90 / 0.88 | locked short |
| drift s11 | **−0.23** | +0.141 | 0.00 / 0.00 | 0.99 / 1.00 | locked short (entropy 0.013), the in-sample *losing* side |
| drift s29 | **−0.93** | +0.141 | 0.00 / 0.27 | 0.00 / 0.63 | flat in up regimes, trading in down regimes |

Training trajectories (MEASURED, per 100 updates): policy entropy fell from 1.10
to 0.01–0.11; implied holding rose from 0.2 h to only 1–2 h (the regime lasts
48 h on average); side concentration reached 0.79–0.96 in three of four runs.

**The previous agent was no better on this test.** Re-reading its planted-edge
logs: 99–100% short in training from update ~200 in both capacities, validation
indistinguishable from constant short (§13). **Neither design has ever shown it
can learn the planted regime.** Both lock onto a side, and the new one also
fails to learn to hold within this budget.

What a pipeline diagnostic on the same market shows is recorded in §11.4.

### 11.4 Pipeline diagnostic — a bug, or the learning signal?

Symmetric planted market, 100 updates, three variants
(`scripts/diagnose_regime_signal.py`; logs in `results/planted_edge/diagnostic/`).
"Agreeing − opposing" is the mean GAE advantage of sampled actions on the regime's
side minus those against it, with a naive t-statistic that ignores
autocorrelation.

| variant | edge per bar | P(long∣up) − P(long∣down), updates 10 → 50 → 100 | P(short∣down) − P(short∣up) @100 | sides S/F/L @100 | entropy @100 | agreeing − opposing, t range |
|---|---:|---|---:|---|---:|---|
| zero cost, planted Sharpe 3 | 0.009 sd | +0.06 → +0.14 → +0.03 | −0.01 | .27 / .45 / .28 | 0.82 | −0.78 to +1.02 |
| zero cost, Sharpe 30 | 0.093 sd | +0.10 → +0.27 → +0.30 | +0.46 | .40 / .31 / .29 | 0.84 | −0.38 to +2.65 |
| real cost, Sharpe 30 | 0.093 sd | +0.03 → +0.02 → +0.01 | +0.23 | .61 / .36 / .03 | 0.27 | −2.58 to +2.97 |

**Findings (MEASURED):**

1. **Not a pipeline bug.** The regime column equals the previous minute's regime
   in 100% of probes, and with zero cost and a strong edge the policy's side
   becomes steadily regime-dependent while net return per bar rises from +0.11 to
   +0.53 bp.
2. **At the planted strength the learning signal is below the estimator's noise.**
   Without any cost, agreeing and opposing advantages are indistinguishable for
   100 updates. Arithmetic (DERIVED): the edge is a one-bar quantity of 0.009 sd,
   while each advantage sums a 286-bar credit window of exogenous market noise,
   about √286 ≈ 17 times the one-bar noise.
3. **Costs make the policy commit before it learns the edge.** With real costs and
   a ten-times stronger edge, entropy fell to 0.27 in 100 updates and the long
   action died in both regimes (3% long) while short kept partial conditioning.
   This is the committal convergence of Mei et al. (NeurIPS 2021) and the necrosis
   of Liu & Jing (2026), now reproduced without any gate.

**What follows — evidence, not yet implemented:**

- **The binding constraint is the learning signal, not the architecture.** The
  gate agent and the flat softmax both lock onto a side; LayerNorm and the new
  features do not change the signal-to-noise ratio of the advantage. Financial
  data have a *"very low signal-to-noise ratio"* (Sato 2019; also Zhang, Zohren &
  Roberts 2020).
- **Compare actions on the same exogenous path.** Because the market is
  exogenous, the reward of every action on a bar is computable: action
  augmentation (Huang 2018), paired opposite-action experiences (Théate & Ernst
  2021), hindsight learning in MDPs with exogenous inputs (Sinclair et al., ICML
  2023). A same-bar comparison of long against short removes the other 285 bars
  of noise from the comparison. The removed counterfactual module was a rollout
  version of this idea with five untested constants.
- **Guard against commitment.** Mei et al. (2021) prove that faster stochastic
  policy optimisation fails with positive probability and propose ensembles;
  Juliani & Ash (2024) find soft shrink-and-perturb with LayerNorm the most
  reliable on-policy plasticity intervention; Liu & Jing (2026) restore
  lower-level exploration.
- **Temporal resolution.** Andrychowicz et al.: *"Try frame skip if possible"*. A
  longer decision interval raises the edge per decision relative to its noise,
  but conflicts with the user's five-minute decision.

### 11.5 All-action values — the diagnostic, repeated

The same 100-update diagnostic after every action was given its exact reward on
the same bar (§7.2), still with PPO's clipped surrogate as the policy step:

| variant | sampled-advantage trainer @100 | all-action trainer @100 |
|---|---|---|
| real cost, edge Sharpe 30 | long action dead (3% long), 61% short, conditioning +0.01 | **P(long∣up) − P(long∣down) = +0.975**, P(short∣down) − P(short∣up) = +0.75, sides .43/.13/.44, **+0.99 bp per bar** |
| real cost, planted Sharpe 3 | — | flat first, as costs require; Q(long) − Q(short) by regime grew 0.00 → +0.057; long share 0.19 by update 100 |
| zero cost, planted Sharpe 3 | no learning (+0.02 bp per bar) | **+1.56 bp per bar**; short conditioning +0.20 |

At the strong edge the order of learning was the one policy improvement predicts:
hold first (mostly flat), then long in up regimes, then short in down regimes.

### 11.6 All-action values with PPO's step — verification, and why it failed

The pre-registered verification (§11.2) was then re-run unchanged. **0 of 4 again**,
but with a different failure: no side lock anywhere.

| run | last-3 mean | pass above | how it ended |
|---|---:|---:|---|
| symmetric s11 | +0.012 | +0.159 | 98% flat, ~8 entries per rollout |
| symmetric s29 | −0.006 | +0.159 | 69% flat |
| drift s11 | **0.000** | +0.141 | 100% flat, zero entries from update ~150, zero orders in validation |
| drift s29 | −0.068 | +0.141 | wanders between sides, no conditioning |

MEASURED over training:

- **The critic kept improving.** Its spread across the three action values grew from
  0.18 to 0.47 (symmetric s11) and 0.58 to 0.88 (drift s11); explained variance rose
  from 0.11 to 0.65.
- **The policy stopped following it.** Entropy fell from 0.106 to 0.001–0.013 within
  100–200 updates, entries per rollout fell from 120–200 to under 15, and in drift
  s11 to exactly zero, frozen for 850 updates.

**The mechanism.** In the clipped surrogate the gradient on an action's logit is
π_old(a) · Â(a): it carries the action's own probability as a factor. Once one
action has probability ≈ 1, the others receive no gradient whatever the critic
says, and the policy sits at a fixed point the values cannot move. Mei, Xiao, Dai,
Li, Szepesvári & Schuurmans (NeurIPS 2020) analyse exactly this as softmax's
*gravitational pull*; Mei et al. (2020, ICML) show entropy regularisation does not
repair it in general, and here the fixed 0.01 bonus did not, because advantages are
normalised to unit scale, which makes near-determinism the optimum of the
regularised objective.

That is why the policy step is now MPO's (§7.2): it moves in distribution space, so
a starved action recovers at a rate set by the KL bound rather than by its own
probability.

### 11.7 The MPO step — two diagnostics

The same 100-update diagnostic at real cost, first with the step reading the raw
one-step values, then reading the critic's learned action-value head. "Value gap"
is Q(long) − Q(short) on the values the step reads, regime up minus regime down.

| step reads | edge | long conditioning @100 | short conditioning @100 | value gap @100 | net per bar | sides S/F/L |
|---|---|---:|---:|---:|---:|---|
| raw one-step values | planted Sharpe 3 | **−0.86** | +0.09 | −0.090 | −0.52 bp | .05 / .46 / .49 |
| raw one-step values | Sharpe 30 | +0.95 | −0.02 | +0.161 | +0.27 bp | .02 / .55 / .43 |
| learned action values | planted Sharpe 3 | −0.005 | +0.012 | −0.005 | −0.13 bp | .13 / .06 / .82 |
| learned action values | Sharpe 30 | **+0.49** | **+0.42** | +0.114 | **+0.51 bp** | .56 / .04 / .40 |

**Findings (MEASURED):**

1. **On raw one-step values the step followed single bars.** Each target contains
   one bar's realised return; a KL-bounded exponential tilt of that sample moved
   the policy by 0.1 nats per update toward whatever the recent bars said. At the
   planted edge it ended anti-correlated with the regime. This is why the step now
   reads the learned head (§7.2).
2. **On the learned head there is no lock in either direction.** At the ten-times
   edge the policy learned both sides — long in up regimes, short in down regimes —
   and was profitable from update 50 on. At the planted edge the value gap stayed
   within ±0.02 of zero for 100 updates, and so did the conditioning.
3. **Why the planted edge had not appeared by update 100 (DERIVED).** The
   t-statistic of a drift estimate grows with edge × √samples, so an edge ten times
   smaller needs about a hundred times the samples for the same significance. The
   ten-times edge's value gap cleared its noise at about update 30, which puts the
   planted edge on the order of 3,000 updates — beyond the verification's 1,000,
   inside the real-data budget of 5,000.

---

## 12. Provenance of every constant

| constant | value | class | source |
|---|---|---|---|
| `decision_minutes` | 5 | USER | |
| `fee` | 0.0005 | USER | realistic taker commission, no curriculum |
| `slippage` | 0.0001 | USER | |
| `entry_leverage` | 1.0 | USER | |
| `maintenance` | 0.005 | CITED | BitMEX XBTUSD base maintenance margin 0.50% until 17 Apr 2020, 0.45% after (BitMEX, 13 Apr 2020); higher value used throughout |
| `hidden` | 128 | CITED + test | CleanRL `ppo_atari_lstm` LSTM 128; planted-edge test §11 |
| `head_width` | 128 | CITED | two hidden layers after the memory, Andrychowicz et al.; width equal to `hidden` |
| `critic_width_multiple` | 2 | CITED | *"no downside in using wider networks"* for the value function (Andrychowicz et al.) |
| `environments` | 8 | CITED | CleanRL `ppo_atari_lstm`; Andrychowicz et al. |
| `rollout` | 1024 | DERIVED | rollout / 2 ≥ credit window |
| `sequence` | 256 | DERIVED | `burn_in` + `sequence` ≥ credit window |
| `burn_in` | 128 | CITED | R2D2 ratio 0.5 (Kapturowski et al. 2019) |
| `episode_steps` | 26,624 | USER + CITED | 2–3 month windows (user); quarterly window of Yang et al. 2020; must be a whole number of rollouts |
| `minibatch_sequences` | 4 | DERIVED | 1,024 samples per minibatch, 32 gradient steps per update |
| `epochs` | 4 | CITED | CleanRL `update_epochs` |
| `recompute_advantages_per_pass` | True | CITED | Andrychowicz et al. |
| `gamma` | 0.99825 | USER | derivation §5 |
| `gae_lambda` | 0.99825 | USER | derivation §5 |
| `actor_lr` | 3e-4 | CITED | Andrychowicz et al. |
| `critic_lr` | 3e-4 | CITED | Andrychowicz et al. |
| `policy_kl_step` | 0.1 | CITED | MPO's E-step KL bound (Abdolmaleki et al., ICLR 2018, Table 2); V-MPO's DMLab single-task value |
| `anneal_lr` | True | CITED | Andrychowicz et al.; CleanRL |
| `gradient_clip` | 0.5 | CITED | CleanRL; OpenAI baselines via Engstrom et al. 2020 |
| `entropy_coefficient` | 0.01 | CITED | PPO paper Atari Table 5; CleanRL; Mei et al. 2020 |
| `flat_start_fraction` | 1/3 | CITED | exploring starts, uniform over the three sides (Sutton & Barto 2018 §5.3) |
| `dsr_eta` | 1/1440 | DESIGN CHOICE | inside derived bounds, §4.5 |
| `dsr_initial_second_moment` | 2.19e-6 | MEASURED | 14.8 bp five-minute sd |
| `observation_clip` | 10.0 | CITED | *"Observation Clipping … usually [−10, 10]"* (Engstrom et al., ICLR 2020) |
| `updates` | 5000 | CITED | gradient-step parity with CleanRL `ppo_atari_lstm` |
| `validation_every` | 500 | DESIGN CHOICE | ten checkpoints; the cost of choosing among them is measured by PBO (§9) |
| `threads` | 1 | MEASURED | one thread per worker, six workers on eight cores |
| `evaluation_seeds` | 4 | DESIGN CHOICE | inherited; adequacy measured in §10 |
| `bootstrap_replications` | 50,000 | CITED | rliable default (Agarwal et al. 2021) |
| `confidence_level` | 0.95 | CITED | rliable default |
| `stress_slippage` | 1, 3, 5 bp | DESIGN CHOICE | sensitivity only, never selection |

Derived and measured numbers outside `Config`: the 14.8 bp sd (MEASURED), the
22 h break-even (DERIVED under the Sharpe-1 ASSUMPTION), reward scale (DERIVED),
five-week look-back (DERIVED from the feature definitions), PBO partitions = 8
quarters (DERIVED), 6 workers (MEASURED: 1.06 GB peak and 5.2 s per update per
worker), replay tolerance 5e-5 (float32 round-off bound; measured error 0.0),
LayerNorm ε = 1e-5 (PyTorch default).

---

## 13. Audit — the previous design's constants

| constant (previous) | value | status then | what became of it |
|---|---|---|---|
| `initial_persistence`, `initial_flat_persistence` | 0.9965 | derived from the Sharpe-1 break-even | **removed** with the gate |
| gate entropy target | entropy at 2× break-even | "2×" unsourced | **removed** |
| `entropy_target` (direction) | 0.6 nats | unsourced | **removed**; the dual averaged over states (§1.1) |
| `entropy_alpha_init` | 0.005 | inherited | **removed** |
| `entropy_alpha_lr` | 1e-3 | CleanRL SAC — a temperature for SAC, not for a gate | **removed** |
| `entropy_alpha_max` | 0.05 | fitted after the fact to seven peaks from our own two runs | **removed** |
| `entropy_alpha_min` | 1e-4 | unsourced; the direction dual sat on it all run | **removed** |
| `advantage_scale_decay` | 0.05 | untested | **replaced** by per-minibatch normalisation (CITED) |
| `critic_epochs` | 2 | unsourced | **removed**; standard PPO shares epochs |
| `epochs` | 4 | cited as the PPO paper's Atari setting — **wrong**, Table 5 uses 3 | **kept, re-cited** to CleanRL |
| `counterfactual_stride / horizon / actor_weight / critic_weight / warmup` | 128 / 48 / 0.5 / 0.5 / 10 | untested, inherited from v5 | **removed**: its actor loss fell to 1e-7–1e-8 exactly when a side collapsed, and 4 of 5 runs locked anyway |
| `dsr_clip`, variance floor | 10, 1e-3 | arbitrary guards; the clip bound on every liquidation | **replaced** by the exact increment (§4.5) and a float64-epsilon guard |
| `kl_stop_scope`, `target_kl` | off, 0.01 | off | **removed** |
| `fast_minutes`, `local_width`, `position_hidden`, 32-unit head layer | 60, 64, 32, 32 | unverified | **removed** with the redesigned state |
| slow features | 7 raw momenta 15 min–10 days, 2 vols, 1 h volume ratio, 3 funding fields | inherited, unnormalised; one field constant | **replaced** by §4.2 |
| portfolio features | 18, with ×10 multipliers and ±5 clips | inherited | **replaced** by §4.3 |
| `episode_steps` | 4096 (14.2 days) | untested | **replaced** by one quarter |
| `flat_start_fraction` | 1/3 | derived from the gate's jump chain | **kept, re-justified** as exploring starts |
| observation clip | ±8 | unsourced | **replaced** by ±10 (CITED) |
| `block_days`, `block_step_days`, `block_evaluation_seeds`, 55% rule | 90, 15, 2, 0.55 | unsourced | **replaced** by calendar quarters, IQM, bootstrap CI, PBO |
| `maintenance` | 0.005 | listed as "measured" without a source | **kept, sourced** (§12) |
| validation guards | rollout ≥ 3× window, reward window ≥ 2× window, sequence ≥ window/2 | invented multipliers | **replaced** by derived conditions (§5) |
| experiment guard | refuse credit window < 20 h | arbitrary | **replaced** by the 22 h break-even |
| LSTM forget-gate bias | 1 | uncited | **kept, cited** (Jozefowicz et al. 2015) |
| `YEARS` | 2023, 2024, 2025 | — | **changed** to 2024, 2025 (USER) |
| planted-edge "PASS" | last-3 mean +0.144 against constant long −0.191 | compared with constant long only | **over-claimed**: constant short earned +0.159 on the same window, and the previous agent's own training logs show it **99–100% short from update ~200** in both capacities, on a training span where constant short gained +0.83. It had locked onto the sample's direction, not learned the regime. The pass rule now requires beating both constants (§11.2) |

---

## 14. The proposed fixes for problems 1 and 2

The previous report ranked six remedies. What the evidence supported, and what
was done:

| proposal | evidence | decision |
|---|---|---|
| Restore exploration where the entry decision is made | Liu & Jing 2026 (preprint): exploration at the lower level; SAC's average-entropy constraint (verbatim, §1.1); Mei et al. 2020, 2021 | **Adopted in its evidence-supported form**: the factorisation is gone, so every decision — including every entry — is one softmax regularised by the standard entropy bonus at every state. Liu & Jing's ε = 0.1 mixing was not adopted: a constant from a tabular study, and it would have to be carried through PPO's ratios. A per-state direction constraint has no study behind it |
| PID update for both dual coefficients | Stooke, Achiam & Abbeel, ICML 2020 | **Moot**: the duals are removed |
| Liquidation as a constraint with a learned multiplier | Chow et al. 2017; Ray et al. 2019 | **Not adopted**: needs a cost limit and multiplier gains no study pins for this problem. Instead the clip is removed (§4.5), the exact liquidation distance is in the state (§4.3), termination carries ruin (§4.6), and liquidations are logged every update. 73 of 74 previous liquidations came from long-locked runs, so problem 1's fix is also problem 2's main fix |
| Densify the rare cost (catastrophe predictor, Sauté state augmentation) | Lipton et al.; Sootla et al., ICML 2022 | **Not adopted**: each adds a model or a penalty scale. The state now carries the exact Markov quantity that decides liquidation |
| Pop-Art instead of the reward clip | van Hasselt et al. 2016 | **Superseded**: the exact increment needs no clip; value targets stay near unit scale by the derived reward scale |
| Partial network resets against primacy bias | Nikishin et al. 2022; Juliani & Ash 2024 | **Not adopted as resets**: Juliani & Ash find many interventions fail on-policy and that LayerNorm with a soft shrink-and-perturb works best. LayerNorm is adopted (user request, Lyle et al. 2023); shrink-and-perturb is the next step if plasticity loss appears |
| Mirror (sign-flip) augmentation | none found for trading; the inverse contract is not mirror-symmetric | **Rejected** |
| Mutual-information regularisation | Leibfried & Grau-Moya 2019 | **Rejected**: a 97%-long learned marginal would make always-long free |
| A larger fixed liquidation penalty | none | **Rejected**: a guessed weight |
| IQM, bootstrap intervals, probability of backtest overfitting | Agarwal et al. 2021; Bailey et al. 2017 | **Adopted** (§9) |
| **A test of the side-lock attractor itself** | — | **Added**: the drift planted market (§11.2) |

---

## 15. Known limitations and open risks

1. **The horizon rests on an assumption.** The 22 h break-even assumes a Sharpe
   1.0 signal. The user fixed γ = λ = 0.99825 on that basis.
2. **`dsr_eta`, `validation_every`, `evaluation_seeds` and `stress_slippage` are
   design choices**, bounded or measured but not pinned by any study.
3. **The features are chosen from what the literature finds predictive, not
   shown to be tradable at five minutes after 12 bp.** Liu & Tsyvinski's sample
   ends in 2018 and they report a *"somewhat smaller"* momentum effect since 2013.
   Daily and weekly momentum implies multi-day holds, consistent with the 22 h
   break-even, but whether any edge survives costs remains empirical.
4. **A fixed entropy bonus does not guarantee against committal convergence**
   (Mei et al. 2021). `long_entry_share`, `account_side_share` and
   `state_action_mutual_information_flat` are logged every update to show it early.
5. **A wealth-invariant reward cannot price ruin by itself**; termination is the
   only channel. Liquidations are logged every update.
6. **LayerNorm is applied at the LSTM output, not inside the cell** (§6).
7. **The untrained policy trades almost every bar** (MEASURED: −11.2 log return in
   one quarter with 19,540 orders), so learning to hold comes first.
8. **Two folds and three seeds** give eight quarters and 24 scores per
   checkpoint; intervals will be wide and PBO has limited resolution.

---

## 16. Lineage

| generation | outcome |
|---|---|
| v4 | converged to permanent flat; zero orders in every development year |
| v5 | 18 architectures × 78 runs; entropy fell monotonically in 78/78; variant ranking did not reproduce across seeds (rank correlation −0.18) |
| v6 stage A | horizon fix alone: entropy recovered in 9/16 runs; collapse rate unchanged |
| gate agent | + differential Sharpe; both runs collapsed via the gate, one to 100% long and one to 100% flat |
| side-lock agent | + gate entropy dual, capped; 3 seeds × 3 folds; entry side locked in 4 of 5 runs by update 600–800; 74 training liquidations; stopped at ~1,200 updates |
| **current** | reference PPO on a redesigned MDP: target-position softmax, dimensionless market state, Markov account, exact Sharpe increment, LayerNorm, quarterly episodes, folds 2024 and 2025 |

Archives: `backup/perpetual_v4_20260909`, `backup/perpetual_v4_v5_before_v6_20260908`,
`backup/perpetual_v6_stage_a_20260909`, `backup/agent_gate_collapse_20260909`,
`backup/agent_alpha_windup_20260910`, `backup/agent_side_lock_20260910` (code,
logs, checkpoints, the previous planted-edge results and this file's
predecessor).

---

## References

**Reinforcement learning — algorithms and implementation**
- Schulman, Wolfe, Dhariwal, Radford & Klimov. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017 — Table 5.
- Andrychowicz et al. What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study. ICLR 2021, arXiv:2006.05990.
- Engstrom, Ilyas, Santurkar, Tsipras, Janoos, Rudolph & Madry. Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO. ICLR 2020, arXiv:2005.12729.
- Huang, Dossa, Ye, Braga, Chakraborty, Mehta & Araújo. CleanRL. JMLR 2022 — `ppo_atari_lstm.py`.
- Kapturowski, Ostrovski, Quan, Munos & Dabney. Recurrent Experience Replay in Distributed Reinforcement Learning. ICLR 2019.
- Hausknecht & Stone. Deep Recurrent Q-Learning for Partially Observable MDPs. AAAI Fall Symposium 2015.
- Mnih et al. Asynchronous Methods for Deep Reinforcement Learning. ICML 2016.
- Pardo, Tavakoli, Levdik & Kormushev. Time Limits in Reinforcement Learning. ICML 2018.
- Sutton & Barto. Reinforcement Learning: An Introduction, 2nd ed. 2018 — §5.3 exploring starts.
- Jozefowicz, Zaremba & Sutskever. An Empirical Exploration of Recurrent Network Architectures. ICML 2015.

**Failure analysis — exploration, termination, plasticity**
- Haarnoja et al. Soft Actor-Critic Algorithms and Applications. arXiv:1812.05905, 2018.
- Mei, Xiao, Szepesvári & Schuurmans. On the Global Convergence Rates of Softmax Policy Gradient Methods. ICML 2020, arXiv:2005.06392.
- Mei, Dai, Xiao, Szepesvári & Schuurmans. Understanding the Effect of Stochasticity in Policy Optimization. NeurIPS 2021, arXiv:2110.15572.
- Liu & Jing. arXiv:2609.05508, 2026 (preprint, not peer reviewed) — policy necrosis in learned termination.
- Harb, Bacon, Klissarov & Precup. When Waiting is not an Option: Learning Options with a Deliberation Cost. AAAI 2018.
- Harutyunyan, Dabney, Borsa, Heess, Munos & Precup. The Termination Critic. AISTATS 2019.
- Lyle, Zheng, Nikishin, Avila Pires, Pascanu & Dabney. Understanding Plasticity in Neural Networks. ICML 2023. Local: `articles/deep RL/lyle2023.pdf`.
- Ba, Kiros & Hinton. Layer Normalization. arXiv:1607.06450, 2016.
- Juliani & Ash. A Study of Plasticity Loss in On-Policy Deep Reinforcement Learning. NeurIPS 2024, arXiv:2405.19153.
- Moalla, Miele, Pyatko, Pascanu & Gulcehre. No Representation, No Trust: Connecting Representation, Collapse, and Trust Issues in PPO. NeurIPS 2024.
- Nauman et al. Overestimation, Overfitting, and Plasticity in Actor-Critic: the Bitter Lesson of Reinforcement Learning. ICML 2024.
- Nikishin, Schwarzer, D'Oro, Bacon & Courville. The Primacy Bias in Deep Reinforcement Learning. ICML 2022.
- van Hasselt, Guez, Hessel, Mnih & Silver. Learning Values Across Many Orders of Magnitude. NeurIPS 2016.
- Lipton et al. Combating Reinforcement Learning's Sisyphean Curse with Intrinsic Fear. arXiv:1611.01211.
- Stooke, Achiam & Abbeel. Responsive Safety in RL by PID Lagrangian Methods. ICML 2020.
- Chow, Ghavamzadeh, Janson & Pavone. Risk-Constrained RL with Percentile Risk Criteria. JMLR 2017.
- Sootla et al. Sauté RL. ICML 2022.
- Leibfried & Grau-Moya. Mutual-Information Regularization in MDPs and Actor-Critic Learning. CoRL 2019.

**MDP and state design**
- Dietterich, Trimponias & Chen. Discovering and Removing Exogenous State Variables and Rewards for RL. ICML 2018.
- Subramanian, Sinha, Seraj & Mahajan. Approximate Information State for Approximate Planning and RL in Partially Observed Systems. JMLR 2022.
- Song, Jiang, Tu, Du & Neyshabur. Observational Overfitting in Reinforcement Learning. ICLR 2020.
- Sinclair et al. Hindsight Learning for MDPs with Exogenous Inputs. ICML 2023.

**Trading reinforcement learning (local library `articles/problems/optimal trading/`)**
- Moody & Saffell. Learning to Trade via Direct Reinforcement. IEEE TNN 2001.
- Deng, Bao, Kong, Ren & Dai. Deep Direct Reinforcement Learning for Financial Signal Representation and Trading. IEEE TNNLS 2017.
- Huang. Financial Trading as a Game: A Deep Reinforcement Learning Approach. 2018.
- Carapuço, Neves & Horta. Reinforcement Learning Applied to Forex Trading. Applied Soft Computing 2018.
- Zhang, Zohren & Roberts. Deep Reinforcement Learning for Trading. J. Financial Data Science 2020.
- Théate & Ernst. An Application of Deep Reinforcement Learning to Algorithmic Trading. Expert Systems with Applications 2021.
- Yang, Liu, Zhong & Walid. Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy. ICAIF 2020.
- Jiang, Xu & Liang. A Deep RL Framework for the Financial Portfolio Management Problem. arXiv:1706.10059, 2017.
- Tummon, Raja & Ryan. Trading Cryptocurrency with Deep Deterministic Policy Gradients. 2020.
- Sato. Model-Free Reinforcement Learning for Financial Portfolios: A Brief Survey. 2019.
- Hambly, Xu & Yang. Recent Advances in Reinforcement Learning in Finance. Mathematical Finance 2023.

**Finance and econometrics**
- Moskowitz, Ooi & Pedersen. Time Series Momentum. JFE 2012.
- Lim, Zohren & Roberts. Enhancing Time-Series Momentum Strategies Using Deep Neural Networks. J. Financial Data Science 2019, arXiv:1904.04912.
- Liu & Tsyvinski. Risks and Returns of Cryptocurrency. RFS 2021 (NBER w24877).
- Andersen & Bollerslev. Intraday Periodicity and Volatility Persistence in Financial Markets. J. Empirical Finance 1997.
- Liu, Patton & Sheppard. Does Anything Beat 5-Minute RV? J. Econometrics 2015.
- Parkinson. The Extreme Value Method for Estimating the Variance of the Rate of Return. J. Business 1980.
- Barber & Odean. All That Glitters: The Effect of Attention and News on the Buying Behavior of Individual and Institutional Investors. RFS 2008.
- He, Manela, Ross & von Wachter. Fundamentals of Perpetual Futures. 2024. Local: `articles/market simulation/manela2022.pdf`.
- Koijen, Moskowitz, Pedersen & Vrugt. Carry. JFE 2018.
- BitMEX. Reduction in Base Maintenance Margin — XBT and ETH Series, 13 April 2020.

**Evaluation**
- Agarwal, Schwarzer, Castro, Courville & Bellemare. Deep RL at the Edge of the Statistical Precipice. NeurIPS 2021 — rliable.
- Bailey, Borwein, López de Prado & Zhu. The Probability of Backtest Overfitting. J. Computational Finance 2017.
