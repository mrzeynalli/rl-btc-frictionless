# CryptoRL model

This is the complete executable project. The workspace root intentionally
contains only `articles/`, `paper/`, and `model/` apart from hidden metadata.

## Data migration (September 5, 2026)

The public BitMEX XBTUSD bulk import replaces the two old Binance Spot datasets.
It is a **data-only change**: the agent described below is still the last Spot
implementation; futures conversion and retraining have not been performed.
After retirement, its old Spot collector can re-download Spot data, but training
must not be run against the new inverse-perpetual data without redesign.

New dataset: `data/bitmex_xbtusd_1m_2016_2026_08_01/` (see its README and manifest).
Import progress is in `status.json`; logs are in `results/bitmex_data_import/`.
The service `cryptorl-bitmex-ohlcv-download.service` is independent of SSH and
resumes completed days after interruption. One-minute **OHLCV Parquet** is
retained with `open_time`, `open`, `high`, `low`, `close`, `volume` (USD contracts),
alongside a compact Parquet of realized XBTUSD funding settlements. Raw trades,
reference indexes, API caches and duplicate daily shards are permanently
removed; the two retired Spot directories were removed after the
final OHLCV audit. Models, configurations and past results are preserved.
The former extended importer was stopped after an auxiliary reference-data
failure; it is no longer the active download workflow.

## Current Spot architecture

At each hourly decision, the model reads the sixty one-minute bars that have
already closed. Each bar has return, range, volume-change, and taker-imbalance
features. A 32-unit GRU or LSTM compresses this fast sequence, and its output is
joined to ten slower hourly market features to form a 42-value market state.

```text
60 completed 1-minute bars × 4 ──► GRU(32) ───────┐
                                                  ├─► 42 market values
10 hourly market features ────────────────────────┘

42 market values ─► Linear(42,32) ─► tanh ─► BUY / SELL / HOLD scores
                                                       │
                              average five seed scores │
                                                       ▼
                    valid-inventory mask: CASH or BTC ─► executed order

critic: 42 market values + 7 portfolio values ─► Linear ─► tanh ─► value
```

The actor is position-independent: changing only the account inventory cannot
change its raw market scores. The execution layer then makes order semantics
valid: CASH allows BUY/HOLD; BTC allows SELL/HOLD. Spot is long-only, so SELL
means exit BTC into cash, not open a short. There are no fractional exposures,
leverage, futures, or funding rates.

The critic additionally receives current inventory, cumulative open-trade
return, accumulated open loss, open-trade drawdown, holding time, recent
downside deviation, and recent turnover. These variables help it estimate the
future reward of the current situation without leaking portfolio state into the
directional actor.

## Reward and exploration

The reward is the net log return after the configured 4 bp one-way trading
cost, including every BUY or SELL position change. There is no accumulated-loss
or downside reward penalty. Entropy uses an adaptive controller targeting 60%
of the executable two-action maximum, bounded between coefficients 0.005 and
0.05.

Training uses randomly sampled contiguous one-to-five-month blocks. Sampling is
balanced across falling, sideways, and rising blocks and across lower/higher
variation. Starts are balanced between inherited CASH and BTC inventory, no
artificial first-step entry charge is applied, and every recurrent state gets a
causal 240-hour burn-in.

## Current training protocol

The fixed model uses `gamma=0.95`, `lambda=0.95`, actor learning rate `0.0003`,
critic learning rate `0.0001`, and a detached critic representation. Five seeds
train only on data before 2026. Their position-independent market logits are
averaged before one common inventory-valid execution layer evaluates January
1–July 21, 2026 at 4, 6, 8, and 10 bp one-way costs. Exact settings are in
[spot_v3_1m.json](configs/spot_v3_1m.json).

At every random-episode reset the rollout stores the causal post-burn-in GRU
state. PPO replay injects that same state, and training aborts if the pre-update
probability ratio or value does not reproduce the collected rollout within
numeric tolerance.

## Layout

```text
src/        current Spot implementation; Binance and public BitMEX collectors
tests/      development checks; never imported by training or inference
configs/    tuning and evaluation specifications
data/       BitMEX import; retired Spot datasets removed after its final audit
results/    current logs, tables, checkpoints, and charts
backup/     superseded source and prior results
.venv/      local Python environment
```

## Commands

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m collector.bitmex_ohlcv --workers 4 --prune-extras
```

The previous `collector.one_minute` and `experiment` entrypoints remain for
reproducing Spot work; they do not train a futures trader. Do not restart the
Spot experiment after data retirement unless deliberately restoring its dataset.

The experiment is resumable: completed jobs are detected from their
`training.json` record. Per-update metrics include pre/post-update critic
explained variance and critic prediction dispersion. Aggregate progress is in
`results/spot_v3_1m/experiment_status.json`; subprocess outcomes are appended to
`results/spot_v3_1m/logs/jobs.jsonl`; final financial charts are placed in
`results/spot_v3_1m/charts/`.

The detached background run is managed by `cryptorl-spot-v3-1m-training.service`
so it survives SSH and chat disconnection.
