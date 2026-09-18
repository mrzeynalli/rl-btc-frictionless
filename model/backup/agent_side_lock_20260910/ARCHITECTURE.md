# Agent architecture

**Status:** current as of 2026-09-09. Package `src/agent/`.
**Update this file whenever the agent changes.** Every constant below either
cites a source, is derived from a measurement, or is explicitly marked as
unverified. Nothing here is a guess carried forward from an earlier generation.

---

## 1. The one measurement everything hangs on

The five-minute log-return standard deviation on the cached BitMEX XBTUSD series
is **14.8 bp**. Commission is 5 bp and adverse slippage 1 bp per leg, so a round
trip costs **12 bp = 0.81 of one bar's typical move**.

A signal worth an annualised Sharpe of 1.0 produces 0.046 bp of expected drift
per bar, so it needs

```
0.0012 / 0.0000046  =  264 bars  =  22 hours
```

in position before it has paid for one entry and one exit. **Twenty-two hours is
the cost break-even**, and the discount factors, the holding priors, the entropy
floor and the BPTT window are all derived from it.

---

## 2. Problem setup

| | |
|---|---|
| Instrument | one BitMEX XBTUSD inverse perpetual, BTC collateral |
| Decisions / accounting | every 5 minutes / every 1 minute |
| Actions | `SHORT`, `FLAT`, `LONG` |
| Position size | 1× account equity at entry, never rebalanced |
| Costs | 5 bp commission + 1 bp slippage per leg, **fixed, no curriculum** |
| Training / validation | 2017–2022 / 23 / 24 → validate on 2023 / 2024 / 2025 |
| Test | **everything from 2026-01-01 is locked; no test entrypoint exists** |

Flat retains BTC collateral and earns exactly zero, so the reward has a natural
zero point.

---

## 3. Network

Separate actor and critic branches, no shared trunk — Andrychowicz et al. find
separate policy and value networks better on 4 of 5 environments.

Each branch stacks three recurrent encoders and a fusion head:

```
fast (60 x 6 minute features) ─► LSTM(6 → local_width) ──┐
                                                         ├─► Linear+Tanh ─► LSTM(hidden → hidden) ─┐
slow (13 causal market features) ────────────────────────┘                                          │
                                                                                                    ├─► Linear(→ head_width)
account (18 portfolio features) ─► LSTM(18 → position_hidden) ──────────────────────────────────────┤   ► Tanh
                                                                                                    │   ► Linear(→32) ► Tanh
slow, account (direct skip connections) ────────────────────────────────────────────────────────────┘   ► Linear(→4)
```

| width | actor | critic | why |
|---|---:|---:|---|
| `hidden` | **128** | **256** | CleanRL `ppo_atari_lstm` uses 128 for its recurrent baseline. Andrychowicz: *"no downside in using wider networks"* for the value function and a benefit to making it wider than the policy (their HalfCheetah optimum was 16–32 policy / 256 value). Tummon et al. reached **256** by Optuna search on a BitMEX crypto task, so the previous 64 was very likely undersized. |
| `head_width` | **128** | **256** | scales with `hidden`; two hidden layers in the head, which Andrychowicz find works well for both networks |
| `local_width` | 64 | 64 | **not widened for the critic** — the citation covers the value *MLP*, not shared feature encoders. Widening everything reached 989k parameters against 630k training decisions. |
| `position_hidden` | 32 | 32 | same reasoning |

**882,251 parameters** (actor 204k, critic 678k). Activations are `tanh`
throughout — Andrychowicz: *"tanh activations perform best and relu worst"*.

**The actor head is initialised with `orthogonal_(gain=0.01)`**, i.e. 100×
smaller than standard. Andrychowicz call this their *most surprising finding*:
*"Initialize the last policy layer with 100× smaller weights."*

### 3.1 The action head is factorised, and this matters

The actor emits 4 numbers, not 3. One is a **gate** logit and three are
**direction** logits:

```
P(keep current side)      = sigmoid(keep_logit + continuation_bias[current])
P(a | switch)             = softmax(alternatives, current side masked to -inf)
P(a)                      = P(switch) · P(a | switch)      for a ≠ current
                          = P(keep)                        for a = current
```

`continuation_bias` is a **learnable** parameter, one entry per side. It is a
starting belief, not a constraint.

This factorisation is why exploration must be regulated on **both** factors
separately — see §6.

---

## 4. Reward: differential Sharpe ratio

Moody & Saffell (IEEE TNN 2001). Per-step marginal contribution to an
exponentially weighted Sharpe ratio:

```
A_t = A_{t-1} + η(R_t − A_{t-1})
B_t = B_{t-1} + η(R_t² − B_{t-1})

         B_{t-1}(R_t − A_{t-1}) − ½ A_{t-1}(R_t² − B_{t-1})
D_t  =  ─────────────────────────────────────────────────────
                    (B_{t-1} − A_{t-1}²)^{3/2}
```

`R_t` is the net log BTC equity change including commission, slippage and
funding. Reward = `D_t × reward_scale`.

**Why not plain log equity growth:** across 78 predecessor runs, net directional
exposure alone explained **R² = 0.96 / 0.82 / 0.63** of validation return. That
objective pays for exposure, not skill — in a year BTC rose, its argmax is to go
long in January and stop thinking. A Sharpe-based reward prefers activity when
activity adds skill, names no direction, and references no benchmark policy. It
also prices out the leverage-reset flicker for free: raising exposure scales
return and volatility together while the fees are pure loss.

| parameter | value | source |
|---|---|---|
| `dsr_eta` | **1/1440** (5-day window) | derived: longer than the 23.8 h credit window so the baseline is stable across it, shorter than the 14.2-day episode so it converges in-episode, wide enough to hold several trades |
| `dsr_initial_second_moment` | **2.19e-6** | measured: the 5-minute return variance. A fresh account starts with the asset's own variance so the denominator is sane from bar one |
| `reward_scale` | **√(1−γ²) = 0.0591** | derived, not tuned. `D_t` is roughly a z-score, so the discounted return would have scale ~17; the critic head starts near zero and Adam moves a weight by ~its learning rate per step, so 160,000 steps could not travel that far. This puts the value target back at unit scale (measured `target_std` ≈ 0.6) |
| `dsr_clip`, variance floor | 10.0, 1e-3 | arbitrary guards. **Correction (2026-09-10):** previously recorded as never load-bearing. The clip is rare (fraction ~1e-3) but binds on exactly the most important events: every liquidation step is clipped to −10, so a 46% slide and a 100% wipeout both pay the same −0.591. See §14 |

**Known property:** with `R_t = 0`, `D_flat = −½AB/(B−A²)^{3/2}`, whose sign is
opposite to `A`. An agent that has been losing is therefore *paid* to stay flat.
That is correct Sharpe accounting and a hazard for learning; it is why the gate
entropy floor exists. Both `R_t` and `D_t` are logged every step so the plain
log-equity objective can be recomputed offline without retraining.

---

## 5. Horizon — the fix that defined this generation

The predecessor discounted with `γλ = 0.9716`, a **2.9-hour** advantage window
that collected 1.6 bp against 12 bp of cost. Entering was negative in expectation
everywhere the gradient could see, and **entropy fell monotonically in 78 of 78
runs**.

| | steps | hours | |
|---|---:|---:|---|
| cost break-even | 264 | 22.0 | measured |
| **GAE credit window** `1/(1−γλ)` | **286** | **23.8** | γ = λ = 0.99825 |
| `γ` horizon `1/(1−γ)` | 571 | 47.6 | |
| **BPTT `sequence`** | **256** | 21.3 | must span the credit window it carries |
| `burn_in` | 128 | 10.7 | R2D2 stores 80-step sequences behind a 40-step burn-in: **ratio 0.5** |
| `rollout` | 1024 | 85.3 | GAE cannot see past a rollout's end; ≥ 3× the window |
| counterfactual horizon | 48 | 4.0 | **inconsistent** — 4 h explicit plus a value bootstrap. Verdicts are not flat-biased (flat share 0.19–0.27 vs 0.333 uniform) but it leans on the critic beyond 4 h |

---

## 6. Exploration — two duals, one per factor

**This is where every collapse observed so far actually happened.** v4 converged
to permanent flat; v5 to one side held for months; both runs of the previous
agent collapsed too — one to 100% long, one to 100% flat. In *every* case
`direction entropy stayed at 0.64–0.68` (its maximum is ln 2 = 0.693) while the
**gate** collapsed to `P(keep) → 1`. A single entropy target placed on the
direction watched the one variable that never moved.

Each factor now gets its own coefficient, tuned by gradient the way SAC tunes its
temperature (Haarnoja et al. 2018, arXiv:1812.05905 §5):

| dual | target | derivation |
|---|---:|---|
| **gate** `H(keep/switch)` | **0.0138 nats** | the entropy of a Bernoulli whose mean dwell time is **twice the cost break-even (44 h)**. A 24 h policy sits at 0.023, a 44 h policy at 0.014, the collapsed runs reached 0.003 |
| **direction** `H(side \| switch)` | **0.600 nats** | of a possible 0.693. A guard that has never yet bound |

| coefficient | value | source |
|---|---:|---|
| `entropy_alpha_lr` | **1e-3** | CleanRL SAC exactly: `a_optimizer = Adam([log_alpha], lr=q_lr)` with `q_lr = 1e-3`. Optimising `log_alpha` rather than `alpha` is also theirs |
| `entropy_alpha_max` | **0.05** | **empirical, see 6.2** |
| `entropy_alpha_min` | 1e-4 | effectively zero; reached only when entropy is above target and no pressure is wanted. Verified harmless |

### 6.2 Why the coefficient must be bounded here but not in SAC

SAC leaves its temperature unbounded and can afford to: for a Gaussian policy
`dH/d(log σ) = 1` **exactly**, so the dual can always move the entropy and never
has to wind up. A Bernoulli gate saturates instead. With `z = logit(P(keep))`:

```
dH/dz = −z·p·(1−p)

P(keep)     mean hold    H(gate)    |dH/dz|    relative to target
0.9999        833 h      0.0010     0.00092         0.08x
0.998106       44 h      0.0138     0.01185         1.00x   <- target
0.9965         24 h      0.0233     0.01971         1.66x
0.78            0.4 h    0.5269     0.21719        18.3x
```

The restoring force is **13× too weak to escape a collapse and 18× too strong
once escaped** — a spring whose stiffness rises with displacement. The
coefficient integrates upward while the policy is stuck, then that same
coefficient slams it past the target.

**Observed, first real run:** 2023 s11 sat at gate entropy 0.0004–0.0007 with
1370–2505 h holds from update 100–175 while α climbed to **0.190** (27× its
start); by update 200 gate entropy was **0.3808** (27× *above* target), holding
time 0.7 h and turnover 44× healthy. Three of five runs ended below the 22 h
break-even, i.e. unable to cover their own costs. The floor added to stop
under-trading was causing over-trading.

**The bound is empirical, from our own two experiments:**

| peak α | outcome |
|---|---|
| 0.0089, 0.0090, 0.0122, 0.0346, 0.0474 | mechanism worked, gate overshot ≤ 11× target |
| 0.0926, 0.1903 | gate overshot 19–36× target, holds below break-even |

**0.05** sits above every value in the first set and below every value in the
second. `entropy_alpha_saturated` is logged every update so a constantly binding cap —
which would leave the dual no headroom to break a collapse — stays visible.

**Caveat:** 2023 s29 overshot 25× at a peak α of only 0.022, so the cap reduces
but may not eliminate the overshoot. The gate-entropy and saturation diagnostics
are what will show whether it is sufficient.

An entropy *target* only applies pressure from below — above it the coefficient
decays to its floor — so this is an exploration floor, **not a forced trading
rate**. Maximum entropy on the gate would mean switching every other bar, which
under these costs is the worst policy available.

Both duals are applied **uniformly across states and never weighted by
`P(switch)`** — that weighting would let the agent escape the penalty by simply
never switching.

**On the contrary citation:** Andrychowicz find entropy regularisation
unnecessary, but say explicitly it is because *"careful policy initialization is
enough to guarantee good exploration"* — true for Gaussian policies in continuous
control, which explore by construction. In a discrete action space CleanRL's
Atari configurations carry a fixed `ent_coef = 0.01`, and our own logs show the
collapse directly. The floor stands on the discrete-domain precedent and on
measurement, not on Andrychowicz.

### 6.1 Holding priors

| | value | mean dwell |
|---|---:|---:|
| `initial_persistence` | 0.9965 | 23.8 h |
| `initial_flat_persistence` | 0.9965 | 23.8 h |

Equal, and that is a derivation. In the jump chain over {short, flat, long} every
state is entered equally often, so time in a state is proportional to its dwell
time — **equal dwell times put one third of the time flat and two thirds in the
market**. The predecessor used 0.9 against 0.5, a 5× bias toward being in the
market, and replacing it with 24 h against 17 h silently removed that protection;
one run then collapsed to permanent flat. Protection now comes from the gate
floor, which resists collapse toward flat and toward a held side alike.

Both remain **learnable**.

---

## 7. PPO

| parameter | value | source |
|---|---|---|
| `epochs` / `critic_epochs` | 4 / 2 | PPO paper Atari Table 5 and CleanRL both use 4; Andrychowicz: *"going over the experience multiple times appears to be crucial"* |
| `clip` | **0.25** | Andrychowicz: *"Start with the clipping threshold set to 0.25"* |
| `gradient_clip` | 0.5 | Andrychowicz: *"might slightly help but is of secondary importance"*, threshold makes little difference |
| `actor_lr` / `critic_lr` | 3e-4 / 3e-4 | Andrychowicz: *"Adam ... 0.0003 is a safe default"* |
| **`anneal_lr`** | **linear → 0** | Andrychowicz: *"Linearly decaying the learning rate may slightly improve performance"*; CleanRL anneals by default |
| **`recompute_advantages_per_pass`** | **True** | Andrychowicz propose and measure it: stale advantages *"can in fact hurt"*, per-pass recomputation *"performs best among all variants"*. Measured cost: **+5%** |
| **`kl_stop_scope`** | **`off`** | CleanRL `ppo_atari_lstm` sets `target_kl = None`; Andrychowicz: the PPO clip *"already enforces the trust region which makes KL penalties or constraints redundant"*. As a sticky flag it cancelled every remaining actor minibatch on 8–16% of updates while median KL sat **14× below** target — deleting gradient steps on outliers. `'pass'` and `'update'` remain available; KL is logged either way |
| advantage scale | slow global EMA (decay 0.05) | per-minibatch standardisation forces unit-scale gradients even from a batch with no signal, which at a per-sample SNR near 0.05 means confidently learning noise. **Decay rate is arbitrary and untested** |
| `environments` | 8 | Andrychowicz: performance *decreases sharply* as parallel environments rise, because it shortens the recurrent experience chunks. CleanRL `ppo_atari_lstm` also uses 8 |

### 7.1 Update budget

```
CleanRL ppo_atari_lstm : 10,000,000 steps / 1024 batch × 4 epochs × 4 minibatches
                       = 156,240 actor gradient steps

this agent             : 5,000 updates × 4 epochs × 4 chunks × 2 groups
                       = 160,000 actor gradient steps
```

The inherited **400 updates gave 12,800 — twelve times short** — and had been
carried from v4 through three generations without ever being justified.
`minibatch_sequences` moved 8 → 4 alongside `sequence` 128 → 256, which keeps the
minibatch at 1024 samples and the budget intact.

Measured: **33 s/update, 1.7 GB per worker** → ~46 h per 5,000-update run.

### 7.2 Counterfactual objective

At a stride of 128 steps, copied ledgers try all three actions and then follow the
sampled policy for 48 steps before a value bootstrap. Actor and critic weights
0.5, warm-up 10 updates. All inherited from v5 and **untested**.

**Known limitation:** it is *not* the safety net it was assumed to be. The
gradient on a logit carries the softmax Jacobian, `∂L/∂logit_b = −π_b(Adv_b −
mean)`, so it fades in proportion to the very probability it is trying to raise.
Measured: `cf_actor_loss` runs ~1e-4 while trading and drops to **1e-7/1e-8**
whenever flat share reaches 1.0. It slows collapse; it cannot reverse it.

---

## 8. Observation

18 portfolio features (`PORTFOLIO_WIDTH`), 6 per-minute features over a 60-minute
window, 13 causal slow features.

The last two portfolio entries are the **running Sharpe statistics** `A` and `B`.
Without them the reward depends on history the agent cannot see: the same action
in the same market state would pay differently according to recent performance,
which is not a Markov decision process and which the critic cannot fit.

Training episodes begin from varied states: one third flat, one third short, one
third long, with position ages spanning the whole prefix (measured on real data:
**34% / 31% / 35%**). The side is chosen explicitly rather than left to a random
walk — driving the prefix at the 24 h holding prior left **88% of episodes flat
and produced no short positions at all**.

---

## 9. Evaluation

**Actions are sampled, never taken by argmax.** With a persistence gate, leaving
flat under argmax requires `P(switch) > 0.5` — a ten-minute mean dwell — while
the design targets a day. The predecessor evaluated by argmax and placed **zero
orders in 420,000 decisions** across every completed run; the same checkpoints,
sampled, placed 101 to 1,272. Argmax does not summarise a persistence-gated
policy, it replaces it with a degenerate one.

- Per checkpoint: full validation year, **4 sampling seeds**, reported as a
  distribution.
- At selection: overlapping **90-day blocks stepped 15 days**, each starting
  flat, **2 seeds**, scored **per seed rather than as an ensemble** — after a
  predecessor whose outcome was decided by which direction a seed picked, the
  spread across seeds *is* the result.
- Stress: re-scored at 1, 3 and 5 bp slippage. Never trained below live cost.

Block counts, seed counts and window lengths are **arbitrary choices of mine**.

---

## 10. Diagnostics

Logged every update. `state_action_mutual_information_flat` is the one that
matters: it is **exactly zero when the entry rule ignores the market**, which is
what made every predecessor run a bet on the year's direction rather than a
trading rule.

| metric | reads |
|---|---|
| `gate_entropy`, `direction_entropy` | which factor is collapsing |
| `entropy_alpha`, `gate_entropy_alpha` | whether either floor is engaging |
| `state_action_mutual_information[_flat]` | is the policy a function of the market at all |
| `switch_probability_mean/sd`, `policy_implied_holding_hours` | holding behaviour |
| `mean_net_log_return`, `reward_sd`, `reward_clip_fraction` | both reward signals, and whether guards bind |
| `advantage_scale`, `target_std`, `explained_variance_post` | is the critic tracking |
| `single_bar_round_trips`, `flicker_share` | leverage resets counted apart from trades |
| `replay_ratio_error`, `counterfactual_replay_error` | exactness (must stay < 5e-5) |

---

## 11. Honest status of every constant

**Derived from measurement:** γ, λ, `rollout`, `sequence`, `reward_scale`,
`dsr_eta`, `dsr_initial_second_moment`, both holding priors, `gate_entropy_target`,
`flat_start_fraction`.

**Cited:** `clip`, `gradient_clip`, `actor_lr`, `critic_lr`, `anneal_lr`,
`epochs`, `environments`, `burn_in`, `updates`, `kl_stop_scope`,
`recompute_advantages_per_pass`, network widths, tanh, head initialisation,
separate-and-wider value network.

**Measured on this host:** LSTM over GRU (1.75× faster, no measured difference),
`threads = 1`, `fee`, `maintenance`.

**Still unverified — no source, not yet tested:**
`local_width`, `episode_steps`, `advantage_scale_decay`, `entropy_alpha_lr` and
its bounds, `dsr_clip`, the variance floor, all five counterfactual parameters,
`fast_minutes`, and every evaluation block/seed count. **Network capacity is
cited but not verified for this problem** — Andrychowicz are explicit that
optimal policy width is environment-dependent and that both too-small and
too-large hurt.

---

## 12. Verification before compute

A synthetic market with a **planted, known-exploitable pattern**
(`src/agent/synthetic.py`): a hidden regime flips sign with a ~48 h mean dwell and
adds drift while it holds; the previous minute's regime is written into one
observation column, so it is causally available and predicts the drift ahead.

Measured: the edge pays **6.6× its own round-trip cost**, and the oracle policy
returns **+0.465 log** against constant-long's **+0.172** over the same window
after real costs.

If the agent cannot find a signal we planted ourselves and know pays several
times its costs, the fault is in the machinery and no amount of real data will
help. Solving a problem known to be solvable before trusting a result on one that
may not be is standard practice for debugging RL.

### 12.1 Result, 2026-09-09 — PASSED

600 updates per capacity, validated on a held-out 90-day slice that is **bearish**
(constant-long −0.191), so the agent could not succeed by defaulting to long.

| validation log return | 100 | 200 | 300 | 400 | 500 | 600 | last-3 mean | seed spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| actor 64 / critic 128 (231k) | +0.071 | +0.113 | +0.131 | +0.112 | +0.125 | +0.116 | **+0.118** | 0.024 |
| actor 128 / critic 256 (882k) | +0.012 | +0.121 | +0.143 | +0.146 | +0.144 | +0.143 | **+0.144** | **0.013** |
| oracle +0.415, constant long −0.191 | | | | | | | | |

Both learned the planted edge, capturing 28% and 34% of the oracle. Zero
liquidations; replay exactness held at 2.4e-7 throughout, far inside the 5e-5
tolerance.

**The gate floor works, and it works continuously.** Gate entropy sat below its
0.0138 target on **58% (small) and 68% (large)** of all updates, with the
coefficient climbing 7-10x above its initial value to push back. A single rescue
is visible in the small run: at update 480 gate entropy fell to 0.0037 - the exact
level at which every previous generation died - with holding time running out to
107 h; alpha tripled and by update 540 entropy was back to 0.017 and holds to
17 h. The pull toward gate collapse is constant, and the dual holds the line.

**Capacity chosen: actor 128 / critic 256.** Higher return (+23% relative), half
the sampling spread, and a clean saturating learning curve where the small
configuration was non-monotonic. Cost is 38.6 s/update against 16.6.

**Caveat this test cannot settle:** the planted market has a strong clean signal,
so it measures learning capability, not resistance to overfitting noise. At 882k
parameters against 630k real training decisions - 1.4 parameters per decision over
65 passes - the larger network carries real overfitting risk on live data that the
synthetic task cannot expose. Validation every 500 updates with common-budget
selection is the guard.

---

## 13. Lineage

| generation | outcome |
|---|---|
| v4 | converged to permanent flat; zero orders in every development year |
| v5 | 18 architectures × 78 runs. Entropy fell monotonically in **78/78**. Variant ranking did not reproduce across seeds (rank correlation **−0.18**) |
| v6 stage A | horizon fix alone: entropy recovered in **9/16** runs, entropy/turnover death 26.9% → 12.5%. Collapse rate unchanged |
| previous agent | + differential Sharpe. Both runs collapsed via the **gate**, one to 100% long and one to 100% flat |
| current | + gate entropy floor, 5,000 updates, wider network, longer BPTT, annealing, per-pass advantages |

Archives with manifests: `backup/perpetual_v4_20260909`,
`backup/perpetual_v4_v5_before_v6_20260908`,
`backup/perpetual_v6_stage_a_20260909`.

---

## 14. Known issues — open, found 2026-09-10 in the main run

Verified by measurement; **none fixed yet**, because any change to `src/agent/`
aborts the frozen run.

**1. Direction entropy is averaged over the wrong states.** The direction target
is met on average while the entry decision from flat has concentrated on one side.
Evaluating the update-500 checkpoints on training-period windows:

| run | flat share | dirH when flat | dirH when holding | average | P(long \| switch) when flat |
|---|---:|---:|---:|---:|---:|
| 2023/s11 | 0.028 | 0.437 | 0.689 | 0.682 | 0.181 |
| 2023/s29 | 0.016 | 0.290 | 0.665 | 0.659 | 0.915 |
| 2024/s11 | 0.001 | 0.374 | 0.646 | 0.645 | 0.872 |
| 2024/s29 | 0.035 | 0.149 | 0.667 | 0.649 | 0.965 |
| 2025/s11 | 0.070 | 0.251 | 0.672 | 0.642 | 0.351 |

Flat states are 0.1–7% of samples, so the average sits above the 0.6 target and
the dual stays idle while the entry choice is nearly constant. This is the gate
bug again, one level down: the regularised quantity is averaged over a population
dominated by states where the decision in question does not arise. Training long
share is 0.95–0.97 in four runs; out of sample 85–97% of entries are long in
three. Timing of entries is market-dependent (`MIflat` > 0); side mostly is not.

**2. Liquidations occur in training and the penalty is capped.** ~58 across five
runs by update 930, all longs: a 1× inverse long breaches maintenance at a 49.7%
fall, while for a 1× short value/requirement = 1/(maintenance+fee) = 182 always,
so it cannot breach. The zero-equity line sits 0.55% below the breach line, so an
ordinary one-minute low often crosses both. In 2017–2024, a random 1× long held 30
days is liquidated from 1.97% of entry minutes (11 episodes: Dec 2017–Jan 2018,
Nov 2018, Feb–Mar 2020, Apr–May 2021). Data gaps are ruled out (largest move
across a gap is 3.8%). The mostly-short run liquidated once; long runs 11–22 times,
and the run with the longest holds is rising (0, 2, 4, 8, 8 per 200 updates) — the
agent is not learning to avoid them, consistent with the −0.591 cap. Validation at
update 500: zero liquidations.

**3. Turnover diagnostic divides by near-zero equity.** In `Ledger._close`,
turnover is `|contracts| / price / max(value, EPS)`; at liquidation `value` is
≈0 or floored at 1e-12, booking 270 (gradual) to 2.6e12 (wipeout) per event.
Diagnostic only — `turnover` and `net_log_return` feed the summary dict and never
the loss, and post-liquidation steps are masked invalid — but `turnover_per_decision`
and `mean_net_log_return` are corrupted on ~1.3% of updates. Use medians.

---

## References

- Andrychowicz et al. **What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study.** ICLR 2021. arXiv:2006.05990 — 250,000 agents; clip, gradient clip, learning rate, annealing, advantage recomputation, separate/wider value network, tanh, 100× smaller policy head, parallel-environment count.
- Schulman et al. **Proximal Policy Optimization Algorithms.** 2017. arXiv:1707.06347 — epochs, clip.
- Haarnoja et al. **Soft Actor-Critic Algorithms and Applications.** 2018. arXiv:1812.05905 §5 — automatically tuned entropy coefficient.
- Kapturowski et al. **Recurrent Experience Replay in Distributed Reinforcement Learning.** ICLR 2019 — burn-in to sequence ratio.
- Moody & Saffell. **Learning to Trade via Direct Reinforcement.** IEEE TNN 2001 — differential Sharpe ratio.
- CleanRL, `ppo_atari_lstm.py` — recurrent PPO reference: 8 environments, LSTM 128, `ent_coef` 0.01, `target_kl` None, `anneal_lr` True, update budget.
- Tummon, Raja & Ryan. **Trading Cryptocurrency with Deep Deterministic Policy Gradients.** 2020. Local: `articles/problems/optimal trading/tummon2020.pdf` — Optuna-tuned hidden 256, batch 128 on a BitMEX crypto task. DDPG and feed-forward, so most of its table does not transfer; it uses ReLU, which Andrychowicz find worst for on-policy RL.
