# BTCUSDT Recurrent PPO

Binance USD-M `BTCUSDT` perpetual futures only. No cross-asset data is used.

The current model is the flat set of modules in `src/`. Everything superseded —
five earlier studies, the direct-reinforcement line, the walk-forward runs and
their artifacts — is under `backup/`, with `backup/HISTORY.md` carrying the
narrative of why each one exists.

## Layout

```
src/            the current model, flat modules — run with PYTHONPATH=src
  data.py         builds the 5-minute study dataset from data/1MIN + data/FUNDING
  features.py     market construction, walk-forward splits, normalisers
  env.py          vectorised hourly environment, funding and cost accounting
  model.py        GruActorCritic
  ppo.py          PPO trainer
  train.py        one (split, arm, config, seed) run
  experiment.py   the whole study: grid search, selection, gates, final test
  evaluation.py   causal policy rollout, metrics, benchmarks, nulls
  evaluate.py     opens the sealed test window
  signal_check.py pre-declared information-coefficient hurdle
  probe.py        short probe used to fix the configuration
  report.py       writes FINAL_REPORT.md
  collector/      Binance downloader (klines + funding), a package
artifacts/      the current study's results
tests/          invariants for the current model
backup/         superseded code, tests and artifacts (see backup/README.md)
data/           market data, shared by both trees
```

## The model

`GruActorCritic`, **6,532 parameters**:

```
12 five-minute bars x 4 vars ──► GRU(4 → 32) ──► h₁₂
                                                  │
7 slow vars + 3 funding-schedule vars ─────────┐  │
position one-hot [3] ──────────────────────────┤  │
                                    concat(42) ◄──┘
                                  ╱                ╲
                   Linear(42→32) tanh       Linear(42→32) tanh
                   Linear(32→3)             Linear(32→1)
                   action logits             state value
```

- The GRU hidden state **carries across hourly decisions** rather than being
  rebuilt each hour; hour *h* feeds twelve consecutive bars and the heads read
  the state after the twelfth.
- Slow variables and the position one-hot enter **at the head**, not the
  recurrent stream: the position is the agent's own state and must be readable
  at the decision instant without rewriting twelve bars of history.
- Action `{SHORT, FLAT, LONG}` at fixed 1x notional. γ = 0.5, λ = 0.5, matched
  to the two-step causal reach.
- Reward is the log change in realised equity after price P&L, exact funding
  settlements and executed turnover cost. Nothing else.

Study window 2021-06-01 to 2026-06-01, hourly decisions, 4-fold walk-forward.
Configuration chosen by median validation log return over folds and seeds
(5 seeds), then run on the held-out test with 10 seeds. Selected `c08` at 225
updates. Cost arms: `zero_cost` (0 bp) and `economic` (4 bp one-way).

## Results

Held-out test, 10 seeds, in `artifacts/`:

| arm | median total return | median Sharpe | seeds positive | median turnover |
|---|---:|---:|---:|---:|
| `zero_cost` | +18.7% | +0.44 | 9/10 | 0.74 /h |
| `economic` (4 bp) | −12.7% | −0.35 | 4/10 | 0.022 /h |

The gap is entirely cost. Best economic seeds were 211 (+20.0%, Sharpe +0.46)
and 29 (+12.9%, +0.32); worst were 137 (−55.2%) and 47 (−40.4%). Profitable
seeds held long for 55–144 h; badly losing seeds held short for ~31 h.

**The study's own pre-declared gates rejected it.** The zero-cost gate failed
(policy beat the best simple benchmark in 35% of runs, and only 10% cleared the
95th percentile of their matched-turnover null; the bar was half). The signal
check failed in advance: median IC 0.018 at 1 h against a break-even IC of
0.183, with 0 of 4 folds clearing at any horizon. No test seed cleared the null.

## Running it

```bash
PYTHONPATH=src .venv/bin/python -m data              # build the study dataset
PYTHONPATH=src .venv/bin/python -m signal_check      # IC hurdle, before training
PYTHONPATH=src .venv/bin/python -m experiment --output-dir artifacts --workers 8
PYTHONPATH=src .venv/bin/python -m report --output-dir artifacts
PYTHONPATH=src .venv/bin/python -m pytest tests
```

Collecting fresh market data:

```bash
PYTHONPATH=src .venv/bin/python -m collector.binance
PYTHONPATH=src .venv/bin/python -m collector.five_minute
```

## Data

`data/1MIN/` and `data/FUNDING/` hold the USD-M perpetual archive.
`data/UNTOUCHED_2026_07/` is a sealed holdout and is never read by any code
here. Other coins exist in the archive and are deliberately unused — this study
is BTCUSDT only, including as a cross-asset feature.
