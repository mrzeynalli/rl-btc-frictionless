#!/usr/bin/env bash
# Verification before compute: the symmetric planted edge (can it learn at all?)
# and the drift variant (does it learn the market or the sample's direction?),
# two seeds each, in parallel. Exits non-zero if any run fails, so the service
# result reflects the runs rather than the shell.
#
#   bash scripts/run_planted_edge.sh [updates] [modes]
#   bash scripts/run_planted_edge.sh 1000 "symmetric drift"
set -euo pipefail
cd /root/research/model
export PYTHONPATH=/root/research/model/src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
UPDATES=${1:-1000}
MODES=${2:-"symmetric drift"}
mkdir -p results/planted_edge
pids=()
for MODE in $MODES; do
  for SEED in 11 29; do
    .venv/bin/python -u scripts/planted_edge_test.py --mode "$MODE" --seed "$SEED" --updates "$UPDATES" \
      > "results/planted_edge/${MODE}_seed_${SEED}.log" 2>&1 &
    pids+=("$!")
  done
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
