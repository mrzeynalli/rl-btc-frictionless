**Single-asset recurrent trading agent**

One BitMEX XBTUSD inverse perpetual with BTC collateral. **Hourly decisions on
five-minute bars**: features, reward statistics and margin checks run on five-minute
bars and one-minute accounting, and the agent chooses a target position — SHORT, FLAT or
LONG at 1× entry notional — once an hour. Recurrent PPO with separate LSTM actor and
critic and layer normalisation after every hidden layer. The state is six measured
market-regime features — weekly trend efficiency, four-week drawdown, volatility level,
volatility regime, volume regime and funding rate — and the account: side, the open
trade's return and age, exposure, and the reward's own statistics. The reward is the
exact per-bar change in an exponentially weighted Sharpe ratio or, as a comparison, in
the downside deviation (Sortino) ratio, summed over the decision's twelve bars.

**Status (2026-09-11): the configuration was revised so every figure fits hourly decisions on
a finite history (ARCHITECTURE.md §0.2); unit tests and the real-data pipeline checks pass; the
pre-registered verification of the revised configuration passed 1 of 4 runs, with all four
regime-conditioned and positive on held-out data (§11.4). The real-data training completed on 2026-09-12:
**no edge distinguishable from zero** — IQM +0.019 log per quarter, 95% interval [−0.019, +0.057],
probability of backtest overfitting 0.80 (§11.5). The 2026 test window remains locked.**

**The full design, the evidence behind every constant and every feature, the smoke-test
record and the audit of earlier generations are in [ARCHITECTURE.md](ARCHITECTURE.md).**
Update it with every change to `src/agent/`; a test fails if a configuration constant is
missing from its provenance table.

The package is self-contained. It builds its own market cache from
`data/bitmex_xbtusd_1m_2016_2026_08_01` and imports nothing from an earlier
generation. Everything before it is in `backup/`.

## Protocol

| | |
|---|---|
| Decisions | hourly (`decision_bars` = 12 five-minute bars) |
| Folds | train 2017–2023 → validate 2024; train 2017–2024 → validate 2025 |
| Seeds | 11, 29, 47, 61, 83 — ten runs (USER) |
| Reward | differential Sharpe (USER); the downside-deviation variant is implemented and verified |
| Training | 100 passes over each fold's training span — 5,991 / 6,849 updates of 8 × 128 hourly decisions (PPO's Atari settings as CleanRL runs them); episodes of 2,176 decisions (90.7 days) starting flat, short or long |
| Costs | 5 bp commission + 1 bp slippage per leg, fixed, no curriculum |
| Validation | a checkpoint every 5 passes (20 per run), each calendar quarter from a flat account, four sampled seeds, never argmax |
| Selection | interquartile mean of quarterly log return, stratified bootstrap 95% interval, probability of backtest overfitting across checkpoints, break-even slippage |
| Measured per fold | five-minute volatility of the training span (reward prior, cost break-even), training length (update budget) |
| Test | **everything from 2026-01-01 is locked; no test entrypoint exists** |

## Running it

```bash
# Engineering checks; no locked market data are evaluated.
PYTHONPATH=src .venv/bin/python -m pytest -q

# Evidence for the state: relevance, redundancy and stability of the candidate features, 2017-2023.
PYTHONPATH=src .venv/bin/python scripts/regime_state_analytics.py

# Pre-registered verification on synthetic markets: <passes> <passes per checkpoint> "<mode>:<reward>:<bars>:<seed> ...".
bash scripts/run_verification.sh 100 5 "latent:sharpe:12:11 latent:sortino:12:11"
.venv/bin/python scripts/verification_summary.py

# Run or resume one reward's queue: [sharpe|sortino] [workers] [decision_bars, default 12].
bash scripts/run_agent.sh sharpe 6

# Progress and the learning diagnostics of every experiment.
.venv/bin/python scripts/agent_status.py

# The test window, SPENT: selected checkpoints 2026-09-12, final checkpoints 2026-09-13; printed, nothing saved.
# PYTHONPATH=src .venv/bin/python scripts/test_window.py --spend-the-test-window [--checkpoint final]

# Interactive financial charts for a finished experiment, then serve them on 127.0.0.1:8787.
PYTHONPATH=src .venv/bin/python scripts/financial_charts.py
systemd-run --unit=charts-server --collect \
  --working-directory="$PWD/results/agent/sharpe_bars12/charts" \
  "$PWD/.venv/bin/python" -m http.server 8787 --bind 127.0.0.1
# From a laptop: ssh -N -L 8787:localhost:8787 netcup, then open http://localhost:8787
```

Measured at **1.7 s per update with the host loaded and ~1 GB per worker**: a fold-2024 run (5,991
updates, 20 validations) takes about 3.7 hours and a fold-2025 run (6,849 updates) about 4.1 hours,
about 39 core-hours for the ten runs — roughly 8 hours with 6 workers (the host's memory ceiling) or
16–20 hours with 2.

## Diagnostics to watch

`account_side_share`, `long_entry_share` and `state_action_mutual_information` show
whether the policy is locking onto one side regardless of the market. `policy_entropy`,
`entry_decisions` and the flat-to-long and flat-to-short probabilities show whether it is
still exploring entries — their collapse is how five-minute decisions failed.
`exit_decisions`, `mean_age_at_exit_hours` and `policy_implied_holding_hours` show how
long it holds. `liquidations_in_rollout` tracks ruin. `replay_*_error` must stay at zero.

## Layout

Outputs are under `results/agent/`: `cache/` for the market arrays (shared), and one
directory per experiment, `<reward>_bars<decision_bars>/`, holding `protocol.json`,
`queue.json` for the frozen specification, `TEST_LOCK.json` recording that the 2026 window was
spent on 2026-09-12 (ARCHITECTURE.md §11.6), `status.json` for
progress, `runs/<year>/seed_<seed>/` for update logs, resume states and validation
checkpoints, and `comparison/selection.json` for the selected checkpoint, its interval,
the probability of backtest overfitting, passive references and slippage sensitivity, and
`charts/` for the interactive financial report (price with the decisions marked on it, equity in
BTC and USD against constant long and short, drawdown, costs, conviction, quarters, holding
periods, trade outcomes and every completed position; the Plotly bundle is written alongside it,
so the page needs no internet).
Verification writes to `results/verification/`; the first configuration's smoke tests are kept
in `results/smoke/`; the state analytics are in `results/state_analytics/`.

Profitability and generalisation remain empirical questions. These changes remove
obstacles to learning; they do not create signal, and a null result is a result.
