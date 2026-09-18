**Single-asset recurrent trading agent**

One BitMEX XBTUSD inverse perpetual with BTC collateral, five-minute decisions
and one-minute accounting. Actions are target positions SHORT, FLAT or LONG at
1× entry notional. Recurrent PPO with separate LSTM actor and critic, layer
normalisation after every hidden layer, and a reward equal to the exact per-step
change in an exponentially weighted Sharpe ratio of net BTC equity.

**Status (2026-09-10): the design failed its pre-registered planted-edge
verification (0 of 4 runs, ARCHITECTURE.md §11.3), so the real-data training has
not been launched.**

**The full design, the evidence behind every constant, and the audit of the
previous generation are in [ARCHITECTURE.md](ARCHITECTURE.md).** Update it with
every change to `src/agent/`; a test fails if a configuration constant is missing
from its provenance table.

The package is self-contained. It builds its own market cache from
`data/bitmex_xbtusd_1m_2016_2026_08_01` and imports nothing from an earlier
generation. Everything before it is in `backup/`.

## Protocol

| | |
|---|---|
| Folds | train 2017–2023 → validate 2024; train 2017–2024 → validate 2025 |
| Seeds | 11, 29, 47 — six runs |
| Training | 5,000 updates, episodes one quarter long, starting flat, short or long |
| Costs | 5 bp commission + 1 bp slippage per leg, fixed, no curriculum |
| Validation | every 500 updates, each calendar quarter from a flat account, four sampled seeds, never argmax |
| Selection | interquartile mean of quarterly log return, stratified bootstrap 95% interval, probability of backtest overfitting across checkpoints |
| Test | **everything from 2026-01-01 is locked; no test entrypoint exists** |

## Running it

```bash
# Engineering checks; no locked market data are evaluated.
PYTHONPATH=src .venv/bin/python -m pytest -q

# Verification before compute: planted edge, symmetric and with BTC's own drift.
bash scripts/run_planted_edge.sh 1000

# Run or resume the queue.
bash scripts/run_agent.sh

# Progress and the learning diagnostics.
.venv/bin/python scripts/agent_status.py
```

Measured at **~5.2 s per update and ~1.1 GB per worker**, so the six runs train in
parallel in roughly eight hours plus validation (about four minutes per checkpoint).

## Diagnostics to watch

`account_side_share` and `long_entry_share` show whether the policy is locking onto
one side, which is how the previous agent failed. `state_action_mutual_information_flat`
is zero when the entry rule ignores the market. `liquidations_in_rollout` tracks the
second failure. Both reward signals — the Sharpe increment and the raw net log
return — are logged every update.

## Layout

Outputs are under `results/agent/`: `protocol.json`, `queue.json` and
`TEST_LOCK.json` for the frozen specification; `status.json` for progress;
`cache/` for the market arrays; `runs/<year>/seed_<seed>/` for update logs,
resume states and validation checkpoints; `comparison/selection.json` for the
selected checkpoint, its interval, the probability of backtest overfitting,
passive references and slippage sensitivity. Planted-edge verification writes to
`results/planted_edge/`.

Profitability and generalisation remain empirical questions. These changes remove
obstacles to learning; they do not create signal, and a null result is a result.
