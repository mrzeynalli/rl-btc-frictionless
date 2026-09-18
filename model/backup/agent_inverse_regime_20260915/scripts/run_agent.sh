#!/usr/bin/env bash
# Three seeds over two walk-forward folds (2024, 2025), 5,000 updates each, for one reward.
# Hourly decisions by default. Measured on real data at 9.8-12.9 s per update (hourly)
# and ~1.0 GB per worker; each validation checkpoint adds a few minutes. Resumes from
# checkpoints.
#
#   bash scripts/run_agent.sh [sharpe|sortino] [workers] [decision_bars]
set -euo pipefail
REWARD=${1:-sharpe}
WORKERS=${2:-6}
BARS=${3:-12}
cd /root/research/model
export PYTHONPATH=/root/research/model/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec /root/research/model/.venv/bin/python -u -m agent.experiment --reward "$REWARD" --decision-bars "$BARS" \
  --workers "$WORKERS" --threads 1 --passes 100 --validation-passes 5 --output "results/agent/${REWARD}_bars${BARS}"
