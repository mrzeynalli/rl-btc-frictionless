#!/usr/bin/env bash
# Verification before compute: can the learner find a planted, known-exploitable
# edge? Sweeps network capacity at the same time, which is the one architectural
# choice no citation can settle.
set -euo pipefail
cd /root/research/model
export PYTHONPATH=/root/research/model/src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
UPDATES=${1:-600}
mkdir -p results/planted_edge
for CAP in small large; do
  .venv/bin/python -u scripts/planted_edge_test.py --capacity "$CAP" --updates "$UPDATES" \
    > "results/planted_edge/${CAP}.log" 2>&1 &
done
wait
