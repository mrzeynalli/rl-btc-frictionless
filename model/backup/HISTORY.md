# BTCUSDT Recurrent PPO

Binance USD-M `BTCUSDT` perpetual futures only. No cross-asset data is used.

**The current architecture is `cryptorl.hourly_eq`** — see [Study 5](#study-5--hourly-gru-ppo-with-a-log-equity-reward-current)
below. Everything before it is kept because each failure is the reason the next
one exists, not because any of it is still the model.

The earlier generations share a fixed study window of 1 June 2025 inclusive
through 1 June 2026 exclusive; Study 5 uses five years, 1 June 2021 to
1 June 2026.

The repository holds two generations of the same experiment. The second exists
because the first failed in a way that was worth diagnosing precisely.

| | Study 1 — five-minute | Study 2 — hourly |
|---|---|---|
| Package | `cryptorl` | `cryptorl.hourly` |
| Artifacts | `artifacts/single-asset/btc5m_gru_ppo/` | `artifacts/single-asset/btc1h_gru_ppo/` |
| Decision interval | 5 minutes | **1 hour** (features still 5-minute) |
| Actions | `{−1, 0, +1}` | `{−1, −0.5, 0, +0.5, +1}` |
| State | 46 dims, raw z-scored levels | 42 dims, stationary by construction |
| Protocol | one 70/15/15 split, 5 seeds | 5-fold walk-forward, 10 seeds |
| Round-trip cost | 8 bp | 12 bp (honest taker) |
| Outcome | −21.7% to −81.7% on test | see `artifacts/single-asset/btc1h_gru_ppo/FINAL_REPORT.md` |

## Why there is a second study

Study 1's five seeds all lost heavily. The diagnosis:

- **The policy never left initialisation.** Test-time maximum action
  probability was 0.342–0.353 against a uniform 0.3333; PPO's clip fraction was
  exactly 0.0 on every update of every seed.
- **Every trade was the argmax of three near-equal numbers.** A ~3 pp learned
  short tilt became 89% SHORT under argmax evaluation.
- **The losses were entirely execution costs.** Replaying the recorded actions
  at zero cost gave −7.7% to +6.3%, with exposure t-statistics below 0.9.
- **The economics could not close at five minutes.** Break-even information
  coefficient at the realised holds was 0.17–0.30; the state delivered 0.007.

The redesign that followed — the MDP, the reward, the feature set, the PPO
details, and which finding each change addressed — lived in
`HOURLY_ARCHITECTURE.md`, which has been removed along with the study
artifacts. `cryptorl.hourly` remains the record of what was built.

## Study 2 pipeline

- `cryptorl.hourly.data` — native 5m contract and mark bars plus funding, with
  the microstructure columns (quote volume, trades, taker-buy) the first
  collector discarded. Writes an audit manifest.
- `cryptorl.hourly.features` — 38 stationary market features + 4 self-state
  features; walk-forward folds; per-fold robust `arcsinh` normalisation fit on
  training data only.
- `cryptorl.hourly.env` — hourly decisions, graded exposure, a 10%-of-equity
  no-trade band, reward-only turnover penalty, finite episodes with equity
  reset, and a vectorised wrapper for 16 randomly offset environments.
- `cryptorl.hourly.model` — input encoder + independent GRU actor and critic.
- `cryptorl.hourly.ppo` — vectorised recurrent PPO with reset-aware BPTT,
  adaptive entropy control, and an exact rollout-replay invariant.
- `cryptorl.hourly.train` — one (fold, seed) run. No early stopping.
- `cryptorl.hourly.signal_check` — supervised information coefficient versus
  the break-even cost hurdle, run *before* any RL.
- `cryptorl.hourly.evaluate` / `report` — sealed-test opener, matched-turnover
  null, cost sweep, stitched walk-forward curve, SVG charts.

## Running it

```bash
.venv/bin/python -m cryptorl.hourly.data          # collect and audit
.venv/bin/python -m cryptorl.hourly.signal_check  # cost hurdle, before any RL
.venv/bin/python -m pytest tests/test_hourly.py   # 20 invariants

systemd-run --unit=cryptorl-hourly --collect \
  --property=WorkingDirectory=/root/CryptoRL \
  --property=StandardOutput=append:/root/CryptoRL/artifacts/single-asset/btc1h_gru_ppo/experiment.log \
  /root/CryptoRL/.venv/bin/python -m cryptorl.hourly.experiment \
    --output-dir /root/CryptoRL/artifacts/single-asset/btc1h_gru_ppo \
    --updates 1000 --workers 7 --cpu-threads-per-run 1
```

Progress is always valid JSON at `artifacts/single-asset/btc1h_gru_ppo/experiment.json` and
`artifacts/single-asset/btc1h_gru_ppo/<fold>/seed_<seed>/training.json`. The evaluator
refuses to open the test months until all 50 runs report `completed` with
`test_performance_accessed == false`.

## Data

`data/BTCUSDT_5M_HOURLY/` — 114,048 native 5m bars (8,928 warm-up + 105,120
study), 1,188 funding settlements with unrounded `calc_time`, and a manifest
recording the three genuine zero-volume bars from the 2025-08-29 Binance halt.

Study 1's data stays in `data/BTCUSDT_5M_2025_2026/`.

## BTC online volatility-event study

`cryptorl.online` implements the next BTC-only experiment without changing the
completed hourly studies:

- Volatility events replace fixed hourly decisions.  The variance threshold is
  calibrated only on 2025 and capped at six elapsed hours per event.
- 2025 is split in two.  January--February learns an initialisation; its
  terminal state — weights, optimiser, causal normaliser, utility moments,
  conviction anchor, reward scale — seeds the main run through December.
  January--April 2026 is a single causal online validation pass; May--June is
  opened only if a configuration-level robustness gate passes.
- The four predeclared candidates are continuous-position GRU and sampled PPO,
  each with the **differential Sharpe ratio** and **quadratic mean--variance
  utility**.
- Validation and test continue updating parameters.  Only terminal training
  states are eligible, PPO always uses a recorded local sampling stream, and no
  deterministic action argmax is used.
- Calendar features are removed.  The funding-settlement countdown remains
  because it directly determines position economics.  True bid/ask/depth data
  is not claimed; taker imbalance and trade activity are transaction-flow
  proxies.

### What the first pass of this study got wrong

The first pass stopped at the validation gate with two candidates flat and two
losing.  None of that was informative, because four defects each independently
guaranteed it.  Each now has a named regression test in `tests/test_online.py`.

- **A ratio objective cannot choose a position size.**  `U(k·r) == U(r)`
  exactly, transaction costs included, so nothing pushed the direct policy's
  output above the 10% no-trade band and it held zero for all 3,102 validation
  events.  The policy head is now a *conviction*; size comes from the
  volatility target and a causal `ConvictionAnchor`, and only for the
  scale-free objective.  Quadratic utility prices scale itself and still sizes
  to zero when that is right.
- **The trained object was not the executed object.**  Training optimised an
  unbanded, ungated, unheld path, and the online update propagated its own
  recomputed exposure instead of the position the ledger held.  Band, hold and
  gate now live in the differentiable rollout — exact forward, straight-through
  backward for band and gate, exact zero gradient for the hold — and the online
  update replays the realised trajectory.
- **Sampled execution churned.**  The action stream was indistinguishable from
  i.i.d. draws and burned 29.7% of equity in a quarter.  A predeclared
  `minimum_hold_events` makes that structurally impossible; the policy-gradient
  term is masked to decision events so no-op actions are not credited with what
  followed.
- **The gate was not a hurdle.**  It required beating carry's return *realised
  in the validation window*, which was 12.5% in a quarter BTC fell 22%.  The
  hurdle is now measured over an equally long window ending at `TRAIN_END`, and
  a candidate that declines to trade is reported as `abstained` rather than as
  a failed Sharpe of exactly zero.

Reward scale is also normalised before GAE — the differential Sharpe arm's
value loss was 30.1 against the quadratic arm's 0.003, which silently made the
shared recurrent trunk learn at two different rates in a comparison meant to
isolate the objective.

### Running it

Collect the native one-minute mark-price series used for basis features, verify
the invariants, check the cost hurdle, then launch the study:

```bash
.venv/bin/python -m cryptorl.online.data
.venv/bin/python -m pytest tests/test_online.py
.venv/bin/python -m cryptorl.online.signal_check   # cost hurdle, before any RL

.venv/bin/python -m cryptorl.online.experiment \
  --output-dir artifacts/single-asset/btc_vol_event_online \
  --offline-updates 1000 --workers 7
```

`signal_check` fits a ridge probe on the initialisation window and evaluates it
on the main training window — the same handoff the RL protocol uses, so the
answer is out of sample without reading a single 2026 event.  The experiment
runs it first and **refuses to start** if no horizon shows signal
distinguishable from zero; `--ignore-signal-check` overrides that deliberately.

The runner then calibrates minimum hold, learning rate and weight decay across
the ten fixed seeds on three chronological inner folds wholly contained in
2025, retrains all four candidates for those seeds, and writes progress to
`experiment.json`.  A profile that never trades can no longer win calibration
merely because zero beats a loss.  If every candidate fails the predeclared
gate, it exits without computing May--June performance.
`--allow-missing-basis` exists only for smoke tests and records the missing
basis in the data manifest.

## Study 5 — hourly GRU-PPO with a log-equity reward (current)

`cryptorl.hourly_eq`. Five years of history, one hourly decision, three target
positions, and a reward that is nothing but the realised change in equity.
Every earlier study answered "what should the reward be?" with more machinery;
this one answers it with the account statement.

### The model

| | |
|---|---|
| Data | 1 Jun 2021 – 1 Jun 2026, 535,104 five-minute bars, 43,824 hourly decisions |
| Decision | once per hour, on the twelve five-minute bars that just closed |
| Execution | the next five-minute open, i.e. the top of the following hour |
| State | 12 bars × {log return, log high-low range, log quote-volume change, taker-buy imbalance} + {3h, 5h, 1d, 3d, 10d momentum, 24h realised volatility, last settled funding rate} + position one-hot |
| Network | one shared GRU(32) over the five-minute stream, actor and critic heads on the hidden state read at each hour, 6,532 parameters |
| Actions | `SHORT`, `FLAT`, `LONG` at a fixed 1× notional. `FLAT` *is* the close action |
| Cost | 3 bp fee + 1 bp slippage, charged as `c·|q_t − q_{t−1}|` on executed turnover |
| Funding | applied at the exact Binance settlement timestamp to the position open immediately before it |
| Reward | `log(E_{t+1}/E_t)` after price P&L, funding and cost, times one constant |

No differential Sharpe, no volatility division, no reward EWMA, no turnover
penalty on top of the real cost, no edge gate, no conviction anchor, no shadow
portfolio, no regime label, no calendar feature, no funding countdown.

### The protocol

1. Four expanding walk-forward folds inside 2021-06 – 2025-05, each validated on
   the six months that follow its training window.
2. A predeclared 12-point grid (learning rate × entropy coefficient × discount)
   × 4 folds × 5 seeds, economic costs. One configuration **and one training
   budget** are chosen from the median validation total log return — the budget
   is part of the selection because validation performance is not monotone in it.
3. The selected configuration is re-run on the same folds with ten seeds in both
   cost arms, and the zero-cost diagnostic gate is evaluated there — before the
   test year is touched.
4. One retrain on 2021-06 – 2025-05 at the selected budget, ten seeds, both arms.
   These runs evaluate nothing.
5. The sealed 2025-06 – 2026-05 year is opened exactly once, deterministic argmax,
   frozen weights and frozen normaliser, hidden state warmed causally over the
   preceding 240 hours.

Normalisation is fitted on each fold's own training window and frozen. Every
seed is reported with the median, the interquartile range and a bootstrap
interval; no seed is ever selected.

### Three places the specification left a choice, and what was chosen

- **Equity accounting.** The specification writes the interval return in logs
  and then defines the reward as `log(E_{t+1}/E_t)`. Equity compounds
  arithmetically here — `E_{t+1} = E_t(1 + q r + F − C)` with `r` the simple
  price return — because that is what a 1× position realises. It agrees with the
  log form exactly for `LONG` and to second order for `SHORT`.
- **One configuration for both arms.** "Two versions of exactly the same learned
  policy" is read as one configuration, selected on the economic arm, used
  unchanged in both. Letting the zero-cost arm tune itself separately would
  inflate the very diagnostic it exists to provide.
- **The gate does not withhold the test.** A failed zero-cost gate is reported
  in bold and the test year is still opened once, because a single frozen
  evaluation selects nothing. `--stop-if-gate-fails` implements the strict
  reading instead.

### Running it

```bash
.venv/bin/python -m cryptorl.hourly_eq.data           # build and audit the 5-year dataset
.venv/bin/python -m cryptorl.hourly_eq.signal_check   # cost hurdle, before any RL
.venv/bin/python -m pytest tests/test_hourly_eq.py    # 41 invariants

systemd-run --unit=cryptorl-hourly-eq --collect \
  --property=WorkingDirectory=/root/CryptoRL \
  --property=StandardOutput=append:/root/CryptoRL/artifacts/single-asset/btc1h_gru_ppo_equity/experiment.log \
  --property=StandardError=append:/root/CryptoRL/artifacts/single-asset/btc1h_gru_ppo_equity/experiment.log \
  /root/CryptoRL/.venv/bin/python -m cryptorl.hourly_eq.experiment \
    --output-dir /root/CryptoRL/artifacts/single-asset/btc1h_gru_ppo_equity \
    --updates 600 --workers 8
```

The run is resumable: every stage skips runs whose `training.json` already says
`completed`, and incomplete runs get one automatic retry before anything reads
their results. Progress is always valid JSON in `experiment.json`.

### What the supervised probe says before any of this

A ridge probe on exactly the agent's state, fitted per fold on training and
scored on validation, against the information coefficient needed to pay 8 bp a
round trip at that holding period:

| horizon | median IC | break-even IC | folds clearing |
|---|---|---|---|
| 1h | +0.017 | 0.183 | 0/4 |
| 3h | +0.031 | 0.107 | 0/4 |
| 6h | +0.043 | 0.077 | 0/4 |
| 12h | +0.039 | 0.054 | 1/4 |
| 24h | +0.039 | 0.039 | 2/4 |
| 72h | −0.001 | 0.023 | 2/4 |

The hourly decision is a factor of eleven short of its own hurdle. That is the
number the zero-cost arm exists to test against.
