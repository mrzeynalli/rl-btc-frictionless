# Hourly GRU-PPO on BTCUSDT, five-minute state, log-equity reward

Study window 2021-06-01 inclusive to 2026-06-01 exclusive.  43824 hourly decisions built from 535104 five-minute bars with 31.0 days of feature warm-up, and 5934 funding settlements applied at their exact timestamps.  BTCUSDT USD-M perpetual futures only.

## Protocol

| item | value |
|---|---|
| decision interval hours | 1 |
| observation | 12 five-minute bars x 4 variables + 7 slow variables + position one-hot |
| actions | ['SHORT', 'FLAT', 'LONG'] |
| exposure | fixed 1x notional |
| reward | log change in realised portfolio equity, constant multiplier only |
| funding | applied at exact Binance settlement timestamps to the position open before them |
| selection rule | median over folds and seeds of the validation total_log_return |
| maximum updates | 600 |
| evaluation interval | 25 |
| cost arms | {'zero_cost': 0.0, 'economic': 0.00039999999999999996} |
| selection seeds | [11, 29, 47, 71, 101] |
| final seeds | [11, 29, 47, 71, 101, 137, 179, 211, 257, 293] |

## Supervised signal check, before any RL

Ridge probe on exactly the agent's state, fitted on each fold's training window and scored on that fold's validation window.  The hurdle is the information coefficient at which a sign-following position pays the 8 bp round trip at that holding period.

| horizon (h) | median IC | break-even IC | folds clearing |
|---|---|---|---|
| 1 | +0.0169 | 0.1828 | 0/4 |
| 3 | +0.0305 | 0.1071 | 0/4 |
| 6 | +0.0426 | 0.0766 | 0/4 |
| 12 | +0.0391 | 0.0543 | 1/4 |
| 24 | +0.0385 | 0.0391 | 2/4 |
| 72 | -0.0005 | 0.0233 | 2/4 |

## Configuration and budget, selected on validation folds only

Rule declared before the search: **median over folds and seeds of the validation total_log_return**.  288 (configuration, budget) cells were scored.

Selected **c04** at **300 updates**, median validation log return +0.1732.

```json
{
  "environments": 32,
  "rollout_steps": 128,
  "minibatches": 2,
  "epochs": 4,
  "gamma": 0.99,
  "gae_lambda": 0.95,
  "clip_range": 0.2,
  "value_clip_range": 0.2,
  "value_coefficient": 0.5,
  "entropy_coefficient": 0.1,
  "learning_rate": 0.0003,
  "max_gradient_norm": 0.5,
  "normalise_advantages": true
}
```

| rank | config | budget | median | IQR |
|---|---|---|---|---|
| 1 | c04 | 300 | +0.1732 | [-0.0154, +0.2811] |
| 2 | c04 | 550 | +0.1726 | [-0.0514, +0.2753] |
| 3 | c04 | 525 | +0.1575 | [-0.1592, +0.3538] |
| 4 | c04 | 450 | +0.1461 | [-0.0523, +0.3118] |
| 5 | c04 | 475 | +0.1437 | [-0.0764, +0.2789] |
| 6 | c05 | 475 | +0.1395 | [+0.0067, +0.2378] |
| 7 | c05 | 450 | +0.1374 | [-0.0592, +0.2614] |
| 8 | c04 | 425 | +0.1358 | [-0.1486, +0.3120] |
| 9 | c05 | 575 | +0.1342 | [-0.0379, +0.2119] |
| 10 | c04 | 125 | +0.1321 | [-0.0531, +0.1948] |

The momentum benchmark's lookback was chosen the same way and on the same folds: **72 hours**.

## Zero-cost diagnostic gate

Criteria: median zero-cost validation log return exceeds zero, exceeds the best simple benchmark, and more than half the runs sit above the 95th percentile of their own matched-turnover circular-shift null.

| quantity | value |
|---|---|
| runs | 40 |
| policy median validation log return | +0.2392 |
| policy IQR | [+0.1619, +0.4461] |
| best benchmark median | +0.3070 |
| runs beating the best benchmark | 52.5% |
| runs above the 95th percentile of their matched-turnover null | 22.5% |
| **gate passed** | **False** |

> The zero-cost policy does not consistently beat matched-turnover noise and the simple rules.  Under the predeclared reading of the protocol that is a **signal** result, not a cost result: the test numbers below should not be read as evidence that fees and funding are what stands in the way.

## Sealed test year, 1 June 2025 to 1 June 2026

Opened once, after every final run reported complete.  Deterministic argmax, frozen weights and frozen normaliser, hidden state warmed causally over the preceding 240 hours.  Median across seeds with the interquartile range in brackets.

| strategy | total return | annualised | Sharpe | Sortino | max drawdown | turnover/h |
|---|---|---|---|---|---|---|
| always_flat (zero_cost) | +0.00% | +0.00% | +0.000 | +0.000 | +0.00% | 0.000 |
| always_long (zero_cost) | -32.00% | -32.00% | -0.936 | -0.911 | -50.84% | 0.000 |
| momentum_72h (zero_cost) | -34.83% | -34.83% | -1.041 | -1.068 | -41.15% | 0.102 |
| always_flat (economic) | +0.00% | +0.00% | +0.000 | +0.000 | +0.00% | 0.000 |
| always_long (economic) | -32.03% | -32.03% | -0.937 | -0.912 | -50.84% | 0.000 |
| momentum_72h (economic) | -54.37% | -54.37% | -1.907 | -1.956 | -55.61% | 0.102 |
| **PPO zero_cost** | +19.86% [-1.74%, +32.05%] | +19.86% [-1.74%, +32.05%] | +0.4649 [-0.0704, +0.7614] | +0.4210 [-0.0499, +0.6321] | -27.67% [-33.71%, -20.30%] | +0.6216 [+0.5508, +0.6763] |
| **PPO economic** | -11.18% [-21.14%, -1.95%] | -11.18% [-21.14%, -1.95%] | -0.4052 [-0.7969, -0.0813] | -0.2469 [-0.4928, -0.0376] | -30.42% [-39.07%, -20.95%] | +0.0072 [+0.0030, +0.0111] |

### Every seed

| arm | seed | total return | Sharpe | max drawdown | turnover/h | long/flat/short | price / funding / cost | null percentile |
|---|---|---|---|---|---|---|---|---|
| zero_cost | 11 | +34.74% | +0.800 | -20.10% | 0.691 | 0.45/0.33/0.22 | +0.3632 / +0.0045 / -0.0000 | 90.7 |
| zero_cost | 29 | -6.66% | -0.203 | -33.43% | 0.562 | 0.54/0.37/0.09 | -0.0042 / -0.0074 / -0.0000 | 68.2 |
| zero_cost | 47 | +70.99% | +1.425 | -18.98% | 0.701 | 0.49/0.18/0.33 | +0.6004 / +0.0070 / -0.0000 | 95.9 |
| zero_cost | 71 | +13.01% | +0.320 | -20.90% | 0.547 | 0.40/0.21/0.39 | +0.1773 / +0.0180 / -0.0000 | 69.2 |
| zero_cost | 101 | +70.39% | +1.342 | -18.38% | 0.424 | 0.44/0.13/0.43 | +0.5926 / +0.0191 / -0.0000 | 94.5 |
| zero_cost | 137 | +20.96% | +0.484 | -33.80% | 0.750 | 0.54/0.15/0.31 | +0.2636 / +0.0038 / -0.0000 | 83.7 |
| zero_cost | 179 | -7.46% | -0.201 | -47.15% | 0.619 | 0.50/0.11/0.39 | -0.0117 / +0.0088 / -0.0000 | 57.0 |
| zero_cost | 211 | +23.98% | +0.647 | -22.65% | 0.434 | 0.55/0.29/0.16 | +0.2709 / -0.0008 / -0.0000 | 90.7 |
| zero_cost | 257 | -8.28% | -0.224 | -45.08% | 0.632 | 0.46/0.08/0.46 | -0.0292 / +0.0170 / -0.0000 | 50.9 |
| zero_cost | 293 | +18.76% | +0.446 | -32.69% | 0.625 | 0.61/0.11/0.28 | +0.2463 / +0.0000 / -0.0000 | 83.1 |
| economic | 11 | -21.84% | -0.814 | -35.05% | 0.003 | 0.41/0.59/0.00 | -0.1877 / -0.0032 / -0.0096 | 36.5 |
| economic | 29 | -1.85% | -0.093 | -19.37% | 0.013 | 0.10/0.90/0.00 | +0.0477 / +0.0003 / -0.0464 | 71.5 |
| economic | 47 | -30.95% | -0.900 | -48.67% | 0.001 | 0.97/0.00/0.02 | -0.2481 / -0.0348 / -0.0028 | 53.3 |
| economic | 71 | +0.54% | +0.015 | -33.56% | 0.007 | 0.54/0.34/0.12 | +0.0950 / +0.0008 / -0.0236 | 77.3 |
| economic | 101 | -8.57% | -0.379 | -20.64% | 0.002 | 0.20/0.80/0.00 | -0.0573 / +0.0020 / -0.0064 | 45.9 |
| economic | 137 | +7.51% | +0.235 | -20.11% | 0.011 | 0.17/0.47/0.35 | +0.1366 / +0.0228 / -0.0396 | 65.6 |
| economic | 179 | -19.85% | -0.969 | -27.28% | 0.014 | 0.13/0.81/0.06 | -0.1483 / +0.0027 / -0.0496 | 21.5 |
| economic | 211 | -21.57% | -0.747 | -43.23% | 0.004 | 0.64/0.36/0.00 | -0.1627 / -0.0138 / -0.0136 | 58.4 |
| economic | 257 | -2.27% | -0.078 | -21.88% | 0.011 | 0.18/0.41/0.41 | +0.0298 / +0.0283 / -0.0372 | 49.0 |
| economic | 293 | -13.78% | -0.431 | -40.41% | 0.008 | 0.45/0.43/0.12 | -0.0675 / +0.0051 / -0.0268 | 50.7 |

### Matched-turnover null

The null rotates each policy's own action sequence against history, which preserves every holding length and the exact turnover while destroying alignment with the returns those actions were chosen for.  A percentile above 95 means the policy beat its own turnover-matched noise.

| arm | median percentile | seeds above 95 |
|---|---|---|
| zero_cost | 83.4 | 1 of 10 |
| economic | 52.0 | 0 of 10 |

### Post-hoc regimes

Regimes are assigned after the fact from the trailing 30-day return and are never in the state; the GRU is left to infer conditions from returns, momentum, volatility and order flow.

| arm | regime | hours | median total log return | long/flat/short |
|---|---|---|---|---|
| zero_cost | bull | 986 | -0.0353 | 0.55/0.17/0.23 |
| zero_cost | bear | 1688 | -0.0279 | 0.61/0.13/0.26 |
| zero_cost | chop | 6086 | +0.2484 | 0.47/0.18/0.33 |
| economic | bull | 986 | -0.0011 | 0.43/0.46/0.00 |
| economic | bear | 1688 | -0.0822 | 0.55/0.41/0.05 |
| economic | chop | 6086 | -0.0318 | 0.23/0.49/0.03 |

## Charts

![test_equity.svg](test_equity.svg)

![validation_budget_curve.svg](validation_budget_curve.svg)

## Files

| file | what it holds |
|---|---|
| `experiment.json` | protocol, dataset audit, stage progress, selection, gate |
| `selection.json` | the configuration and budget chosen on the folds |
| `zero_cost_gate.json` | the diagnostic gate, run before the test was opened |
| `test_results.json` | every seed, benchmark, null and breakdown on the test year |
| `test_seed_table.csv` | the per-seed table above, machine readable |
| `test_trajectories/` | hourly position, P&L decomposition and equity per seed |
| `<split>/<arm>/<config>/seed_<n>/updates.jsonl` | one line per PPO update |
| `<split>/<arm>/<config>/seed_<n>/training.json` | the validation curve for that run |
