**Single-asset recurrent trading research: v5**

The current experiment trades one BitMEX XBTUSD inverse perpetual with BTC
collateral, five-minute decisions and one-minute accounting. Actor and critic
use separate GRU or LSTM networks for minute patterns, market memory and
position history, with direct current-account inputs. There is no CNN.

V5 compares dropout, LayerNorm, calibrated BatchNorm and four-head attention
individually and in combinations. Sixteen architecture configurations plus
two PPO-only learning controls each use seven seeds and three development
folds: **378 runs**. Each run has 400 updates and full-year validation every
100 updates. Seeds are 11, 29, 47, 71, 101, 137 and 179.

The algorithm combines recurrent PPO with a separate counterfactual
policy-gradient objective and action-value regression. Copied accounts try
all three actions and follow the sampled policy for four hours. The actual
PPO buffer contains only sampled behavior-policy actions. Dropout masks are
recorded and replayed exactly. BatchNorm uses initial training-only calibration
with frozen running statistics and learned affine parameters.

The continuation gate is trainable, with initial probabilities 0.5 when flat
and 0.9 when holding. Training starts include untouched flat accounts and valid
simulated position histories. Validation starts flat. There are no forced
holding periods, price-based entry rules, trading bonuses or inactivity
penalties. Profitability and generalization remain empirical questions.

The reward is 100 times net log BTC equity change. Entry notional is 1× account
equity; repeating a side holds the same contracts. Fees are 5 bp and adverse
slippage 1 bp per execution leg. A flat futures position retains BTC collateral.
This is single-asset position control, not allocation across assets.

Training folds use 2017–2022, 2017–2023 and 2017–2024, with validation on
2023, 2024 and 2025 respectively. **All data from 2026-01-01 onward remain
locked.** V5 uses the existing pre-2026 cache in read-only mode and adds
training-boundary checks to counterfactual continuations. There is no test or
automatic final-retraining stage.

The full rationale, architecture, limitations and settings are in
[the v5 design](research/2026-09-06-v5-prelaunch/DESIGN.md). The investigation
that motivated it is in [the v4 review](research/2026-09-06-v4-review/REVIEW.md)
and [the seed 11 analysis](research/2026-09-06-seed11/REVIEW.md).

From /root/research/model:

```bash
# Engineering checks; no locked market test data are evaluated.
.venv/bin/python -m pytest -q

# Run or resume the development queue.
bash scripts/run_perpetual_v5.sh

# Read progress without loading market data or evaluating policies.
.venv/bin/python scripts/perpetual_v5_status.py
systemctl status cryptorl-perpetual-v5.service --no-pager
```

The background service runs three workers with one PyTorch thread each.
The transient systemd service survives chat/SSH disconnection; a server reboot
requires restarting the command, which resumes saved checkpoints.

New outputs are under results/perpetual_v5/:

- protocol.json, queue.json, TEST_LOCK.json: fixed specification, queue and lock.
- status.json: queue progress and recent diagnostics.
- runs/&lt;variant&gt;/&lt;year&gt;/seed_&lt;seed&gt;/: update logs, resume states and validation checkpoints.
- validation_*.json/csv.gz: metrics and complete development trajectories.
- comparison/: common-budget selection and actual seven-member ensemble evaluations.
- logs/, the systemd journal and failure.json: process output and failures.

Training tracks position occupancy, entry/exit/reversal decisions, holding age,
costs, conditional critic errors, counterfactual comparisons and PPO replay/KL
checks. Architecture selection considers all seeds and years, and evaluates
the resulting single-account ensemble. A flat policy cannot pass the
development-candidate screen. That screen is not a claim of proven optimality
or deployment readiness.

V4 source and results remain unchanged in src/perpetual and
results/perpetual_v4. Its original README is preserved
[here](research/2026-09-06-v5-prelaunch/V4_README.md). Earlier source/results are
in [the existing archive](backup/spot_before_perpetual_v4_20260905/README_ARCHIVE.md).
The inverse-contract ledger, causal features and data provenance remain shared
with v4, including the limitations of trade-price marks and assumed execution
costs in place of historical order-book reconstruction.
