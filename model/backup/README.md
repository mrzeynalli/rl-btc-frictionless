# Backup — superseded studies

Archived 2026-09-02. Nothing here is on the active path; the current model is
the flat module set in `src/` with its artifacts in `artifacts/`.

## What is here

| path | what it is | artifacts | outcome |
|---|---|---|---|
| `src/cryptorl/hourly_ddr/` | direct reinforcement, no critic, continuous position | `artifacts/btc1h_gru_ddr/` | failed |
| `src/cryptorl/wf/` | walk-forward direct reinforcement, 1h and 4h clocks | `artifacts/btc_wf_1h/`, `artifacts/btc_wf_4h/` | 0/10 seeds positive on both clocks |
| `src/cryptorl/hourly_eq/` | GRU-PPO, log-equity reward (the earlier PPO study) | `artifacts/btc1h_gru_ppo_equity/` | 2/10 seeds positive, zero-cost gate failed |
| `src/cryptorl/hourly_dsr/` | differential Sharpe objective | — | superseded |
| `src/cryptorl/hourly/` | first hourly rework | — | superseded |
| `src/cryptorl/online/` | online / live-decision module | — | not launched |
| `src/cryptorl/*.py` | the original v1 single-asset core (env, features, model, ppo, train, …) | — | superseded |
| `tests/` | the five test modules covering the archived code | — | passed at archive time (128 tests) |

The `v2_probe` artifacts are the short probe that fixed the current study's
configuration before it ran.

## Running anything in here

These modules use `cryptorl.*` package imports, so they need the package root
on the path:

```bash
PYTHONPATH=backup/src .venv/bin/python -m cryptorl.wf.experiment --help
PYTHONPATH=backup/src .venv/bin/python -m pytest backup/tests
```

These modules resolve the repository root by walking up from their own file,
which from `backup/src/cryptorl/...` lands on `backup/` rather than the real
root. `backup/data` is therefore a symlink to `../data`, so their data paths
still resolve and the archived code stays byte-identical to what produced the
results above.
