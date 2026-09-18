#!/usr/bin/env bash
# Three seeds over three development folds, 5,000 updates. Resumes from checkpoints.
set -euo pipefail
cd /root/research/model
export PYTHONPATH=/root/research/model/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec /root/research/model/.venv/bin/python -u -m agent.experiment --workers 5 --threads 1 --updates 5000
