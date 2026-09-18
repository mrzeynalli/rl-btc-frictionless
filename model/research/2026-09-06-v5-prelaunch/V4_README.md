# research perpetual v4

A portfolio-aware recurrent PPO research agent for **BitMEX XBTUSD inverse
perpetual contracts**, using five-minute decisions and one-minute accounting.
The previous source, configurations, tests and Spot results are preserved in
[the archive](backup/spot_before_perpetual_v4_20260905/README_ARCHIVE.md), with
SHA-256 hashes for all 2,783 archived files. Earlier backups remain in place.
The investigation is in [REVIEW.md](research/2026-09-05-review/REVIEW.md).

## Architecture and trading objective

Each actor and critic has its own market encoder and decision memory:

```text
60 completed minute bars × 6 features → local CNN(32) ─┐
13 causal market/funding features ────────────────────┼→ Linear(64) → GRU(64) → head
16 actual account/position features ──────────────────┘
actor head: SHORT / FLAT / LONG probabilities
critic head: expected discounted future reward
```

Actor and critic share neither weights nor optimizer/clipping budgets. Account
state includes contract exposure, unrealized P&L, holding age, net open-trade
return, favorable/adverse excursion, drawdown, margin headroom, fees, funding,
and previous action/reward. Repeating the current target holds the same fixed
contract quantity; FLAT closes a position; changing direction closes and opens.
There are no mandatory holding times, technical trading signals, profit targets,
or stop-loss trading rules. The liquidation approximation is a simulator constraint.

The economic reward is `100 × log(equity_BTC_after / equity_BTC_before)`, net of
execution costs and realized funding, with a fixed scale. Entry contract
notional is at most 1× BTC equity converted to USD, rounded down to whole
contracts. Quantity remains fixed until a side change. Costs are fixed at
5 bp commission plus 1 bp adverse slippage per execution leg; a reversal pays
for both legs. These are research assumptions, not historical fee-tier estimates.

**The collateral and primary objective are BTC-denominated.** Flat contracts
still hold BTC collateral. A 1× short contract position initially hedges the
collateral's USD exposure; a 1× long initially adds exposure to it. Reports show
both BTC and USD returns and the USD return of simply holding BTC collateral.

Orders fill at the first regular-trade open in the next five-minute interval.
Funding at a minute boundary applies to the existing position before that
minute's order. Missing-trade minutes carry stale marks and explicit missingness
features, have zero volume and no fills, and preserve open contracts/funding.
Features use only completed minutes and already-settled funding. The fast CNN
summarizes the latest hour; slow features span 15 minutes through ten days.

The available archive contains regular-trade OHLCV and funding, not historical
exchange mark/index prices or order-book executions. Therefore marking uses
trade prices and margin checks use adverse intrabar extrema. Funding notional,
slippage, liquidation, and final-closure marks are approximations. Final-closure
sensitivity is reported separately from marked terminal wealth. This simulator
is suitable for the present research comparison, not exact exchange replication.
See the [inverse-contract guide](https://www.bitmex.com/app/inversePerpetualsGuide)
and [contract specification](https://www.bitmex.com/app/contract/XBTUSD).

## Frozen development protocol

| Fold | Training interval (UTC) | Validation interval (UTC) |
|---|---|---|
| 2023 | 2017-01-01 through 2022-12-31 | Calendar 2023 |
| 2024 | 2017-01-01 through 2023-12-31 | Calendar 2024 |
| 2025 | 2017-01-01 through 2024-12-31 | Calendar 2025 |
| Final retraining | 2017-01-01 through 2025-12-31 | None |

Data from **2026-01-01 onward are locked**. The loader uses pre-2026 Parquet
predicates; split and evaluator guards reject test dates. The generated cache
ends immediately before the cutoff. The coordinator contains no test stage and
no unlock switch. This applies to the new run; historical results already
examined during the earlier investigation remain in backup.

Each development fold trains seeds 11, 29, 47, 71 and 101 for 400 updates, with
calendar-year validation every 50 updates. Each fold fits its own observation
scaler on that fold's training interval only. One common update budget is selected
by mean net validation log BTC growth across all 15 seed/year pairs, keeping all
seeds and preferring an earlier budget on ties. Thus these validation returns are
development/selection results, not untouched out-of-sample performance estimates.

After selection, each fold's five members average action probabilities on one
actual shared account, maintaining their own recurrent states. Ensemble validation
reports costs of 6, 8 and 10 bp per leg and flat, constant-long and constant-short
baselines. The coordinator then retrains the five seeds on all pre-2026 data at
the selected budget and stops. It never evaluates these final models on test data.

The initial PPO configuration is 16 environments, 512-step rollouts,
128-step sequences, 128-step recurrent burn-in, 4,096-decision episodes,
four policy/value epochs and two additional critic epochs. Actor learning rate
is 0.0001; critic rate is 0.0003. Gamma has a 24-hour half-life and gamma×lambda
a two-hour half-life. The entropy coefficient decreases from 0.005 to zero over
the first 200 updates, with the same schedule in final retraining.

Random contiguous training episodes start flat. Economic liquidation terminates
and masks bootstrap; artificial episode boundaries bootstrap normally. Current
weights rebuild the recorded recurrent prefix at each PPO minibatch. Pre-update
replay must reproduce collected probabilities and values within 0.00005 or abort.
Validation refreshes the same finite prefix every 512 decisions. There is no
rolling reward standardization or value-scale change.

The canonical settings are in [config.py](src/perpetual/config.py); the launch
snapshot is [perpetual_v4.json](configs/perpetual_v4.json). The output protocol
records the source hash and rejects mixed source versions on resume/new jobs.

## Running and observing

From `/root/research/model`:

```bash
# Software checks and a deliberately easy synthetic learning diagnostic.
.venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python scripts/verify_perpetual.py

# Train/validate only; resume safely if interrupted.
bash scripts/run_perpetual_v4.sh

# Read progress; no market data or model evaluation is loaded.
.venv/bin/python scripts/perpetual_status.py
systemctl status cryptorl-perpetual-v4.service --no-pager
```

The background service `cryptorl-perpetual-v4.service` runs two workers, each
with two CPU threads, at lower scheduling priority. Its process group has a
400% CPU quota and a 6 GiB memory ceiling. It survives SSH/chat disconnection.
The transient systemd service does not automatically persist across a host reboot;
the saved checkpoints can resume through the same command afterward.

Progress and all new outputs are under `results/perpetual_v4/`:

- `status.json`, `protocol.json`, `TEST_LOCK.json`: progress, fixed specification and lock.
- `runs/<fold>/seed_<seed>/training.json` and `updates.jsonl`: training diagnostics and validation curves.
- `resume.pt`: full optimizer, random generator, account and recurrent-history state, saved every 10 updates.
- `policy_*.pt` and `validation_*.json/csv`: scheduled policies and development evaluations.
- `selection.json`, `ensemble_validation_*.json`, `development_report.json`: generated after development training.
- `runs/final/`: final pre-2026 training; no test evaluation files are produced.
- `logs/` and `service.log`: job and coordinator logs; `failure.json` records a coordinator failure.
- `verification/`: engineering, synthetic learning and pre-2023 integration checks.

The synthetic diagnostic establishes that the implementation can learn an easy
observable trading pattern. It provides no evidence of a real-market edge.
Early negative sampled reward is expected under near-uniform costly exploration;
validation trading behavior, cost sensitivity and consistency across folds/seeds
are the evidence that matters for the research decision.

The retained BitMEX collection modules are in `src/collector/`; data and import
provenance stay in their existing locations. No market data were moved or deleted
as part of the agent migration.
