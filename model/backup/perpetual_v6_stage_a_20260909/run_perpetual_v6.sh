#!/usr/bin/env bash
# Stage A: credit-assignment horizon and holding-time prior, one architecture,
# seven seeds, three development folds. Resumes from saved checkpoints.
set -euo pipefail
cd /root/research/model
export PYTHONPATH=/root/research/model/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec /root/research/model/.venv/bin/python -u -m perpetual_v6.experiment --workers 2 --threads 1 --updates 400
