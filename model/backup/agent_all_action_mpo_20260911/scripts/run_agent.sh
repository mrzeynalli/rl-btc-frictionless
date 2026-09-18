#!/usr/bin/env bash
# Three seeds over two walk-forward folds (2024, 2025), 5,000 updates each.
# Measured at ~5.2 s per update and ~1.1 GB per worker, so all six jobs run at
# once on this host. Resumes from checkpoints.
set -euo pipefail
cd /root/research/model
export PYTHONPATH=/root/research/model/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec /root/research/model/.venv/bin/python -u -m agent.experiment --workers 6 --threads 1 --updates 5000
