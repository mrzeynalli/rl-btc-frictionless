#!/usr/bin/env bash
# Regime-signal diagnostic at real costs, planted edge and a ten-times edge, in parallel.
set -euo pipefail
cd /root/research/model
export PYTHONPATH=/root/research/model/src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
OUT=${1:-results/planted_edge/diagnostic_mpo_qhead}
UPDATES=${2:-100}
mkdir -p "$OUT"
pids=()
for SHARPE in 3 30; do
  .venv/bin/python -u scripts/diagnose_regime_signal.py --cost real --oracle-sharpe "$SHARPE" --updates "$UPDATES" \
    > "$OUT/real_s${SHARPE}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
