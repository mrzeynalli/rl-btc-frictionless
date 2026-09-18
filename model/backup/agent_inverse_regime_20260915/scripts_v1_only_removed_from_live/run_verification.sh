#!/usr/bin/env bash
# Pre-registered verification on synthetic markets (scripts/planted_edge_test.py), one
# process per job, in parallel. Exits non-zero if any run fails, so a service result
# reflects the runs.
#
#   bash scripts/run_verification.sh <passes> <evaluate_passes> "<mode>:<reward>:<decision_bars>:<seed> ..."
#   bash scripts/run_verification.sh 100 2 "latent:sharpe:12:11 latent:sortino:12:11"
set -euo pipefail
cd /root/research/model
export PYTHONPATH=/root/research/model/src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
PASSES=$1
EVERY=$2
JOBS=$3
shift 3
OUT=results/verification
mkdir -p "$OUT"
pids=()
for JOB in $JOBS; do
  IFS=: read -r MODE REWARD BARS SEED <<< "$JOB"
  .venv/bin/python -u scripts/planted_edge_test.py --mode "$MODE" --reward "$REWARD" --decision-bars "$BARS" \
    --seed "$SEED" --passes "$PASSES" --evaluate-passes "$EVERY" --output "$OUT" "$@" \
    > "$OUT/${MODE}_${REWARD}_bars${BARS}_seed${SEED}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
