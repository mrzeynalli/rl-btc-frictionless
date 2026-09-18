**Single-asset recurrent trading agent — version 2**

A **USDT-margined linear perpetual account** — the user's USD-M futures account — on **BitMEX XBTUSD
prices, volume and funding** (one consistent venue from 2016). **Hourly decisions on five-minute bars**: the agent chooses a target
position — SHORT, FLAT (USDT cash) or LONG at 1× entry notional — once an hour. Recurrent PPO with
separate LSTM actor and critic and layer normalisation after every hidden layer. The market state is
volatility-normalised log returns over nine horizons (1 h, 3 h, 1 d, 3 d, 1 w, 4 w, 8 w, 13 w, 26 w),
the volatility level and regime, the volume regime and the funding rate; the account state is the side, the open trade's return and age,
exposure, and the reward's own statistics. The reward is the exact per-bar change in an
exponentially weighted Sharpe ratio of USDT equity. Training charges twice the real cost; evaluation
takes the most probable action at the real cost.

**Status (2026-09-15): built, tested and smoke-run end to end; the pre-registered γ/λ search kept γ = λ = 0.9792
per hour (ARCHITECTURE.md §5). The main run has not been launched. The whole of 2026 is locked for testing.** Version 1
(BitMEX inverse contract, BTC collateral, six regime features) and its full record are archived in
`backup/agent_inverse_regime_20260915/`.

**The design, the evidence behind every constant, the data audit and the formulation with pseudo-code
are in [ARCHITECTURE.md](ARCHITECTURE.md).** Update it with every change to `src/agent/`; a test fails
if a configuration constant is missing from its provenance table. Decision rules fixed before
measurement are in `results/agent_v2/PREREGISTRATION.md`.

## Protocol

| | |
|---|---|
| Decisions | hourly (`decision_bars` = 12 five-minute bars) |
| Folds | train 2017 → Y, validate the four quarters of Y: 2024, 2025 |
| γ/λ search | inner folds 2022 and 2023, seed 11, 3 × 3 grid, 30 passes — kept 0.9792 / 0.9792 per hour |
| Seeds | 11, 29, 47, 61, 83 (USER) |
| Reward | differential Sharpe of USDT equity (USER) |
| Training | 60 passes over each fold's span — 3,595 / 4,110 updates of 8 × 128 hourly decisions; episodes of 2,176 decisions (90.7 days) starting flat, short or long; 2× the real cost |
| Costs at evaluation | 5 bp taker commission (Binance USD-M VIP 0) + 1 bp slippage per leg |
| Validation | a checkpoint every 5 passes (12 per run), each quarter from a flat account, argmax actions |
| Selection | interquartile mean of quarterly log return, stratified bootstrap 95% interval, probability of backtest overfitting, break-even slippage; constant long/short/flat and 1-week and 26-week momentum rules as references |
| Test | **2026-01-01 onward, locked** (USER) |

## Running it

```bash
# Engineering checks; no locked market data are evaluated.
PYTHONPATH=src .venv/bin/python -m pytest -q

# Pre-registered state redundancy rule on 2017-2023 (development data only).
PYTHONPATH=src .venv/bin/python scripts/feature_redundancy.py

# Binance USD-M data, kept for comparing funding with the user's venue; not used by the agent.
PYTHONPATH=src .venv/bin/python -m collector.binance_um --output data/binance_um_btcusdt_1m --end YYYY-MM-DD

# The gamma/lambda search (about 3.5 hours on this host), then its pre-registered decision.
bash scripts/run_search.sh
PYTHONPATH=src .venv/bin/python scripts/search_summary.py

# The main run, after Config carries the searched gamma and lambda: [workers].
bash scripts/run_agent.sh 6

# Progress and the learning diagnostics of every experiment.
.venv/bin/python scripts/agent_status.py
```

Measured at **1.4–2.0 s per update with six workers and ~1 GB per worker**. The main run is 38,525 updates over ten
runs — roughly 19 core-hours plus validation, about 4 hours with 6 workers.

## Diagnostics to watch

`account_side_share`, `long_entry_share` and `state_action_mutual_information` show whether the policy
locks onto one side regardless of the market. `policy_entropy` and the flat-to-long and flat-to-short
probabilities show whether entries are still explored. `exit_decisions`, `mean_age_at_exit_hours` and
`policy_implied_holding_hours` show how long it holds. `liquidations_in_rollout` tracks ruin.
`replay_*_error` must stay at zero.

## Layout

Outputs are under `results/agent_v2/`: `cache/` (BitMEX market arrays to 2026-01-01, all 15 candidate features),
`audit/` (feature redundancy; the superseded splice audit), `search_merged_data/<cell>/` and its `decision.json`,
and `main/` for the main run, holding `protocol.json` (the frozen specification every run is rebuilt
from), `queue.json`, `TEST_LOCK.json`, `status.json`, `runs/<year>/seed_<seed>/` and
`comparison/selection.json`. The agent's data are in `data/bitmex_xbtusd_1m_2016_2026_08_01/`. Version 1's results remain in `results/agent/`; its scripts
(test window, charts, synthetic verification, state analytics) are in
`backup/agent_inverse_regime_20260915/`.

Profitability and generalisation remain empirical questions. These changes align the agent with the
account the user trades and give it the information the user asked for; they do not create signal,
and a null result is a result.
