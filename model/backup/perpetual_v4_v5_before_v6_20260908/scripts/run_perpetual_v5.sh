#!/usr/bin/env bash
set -euo pipefail
cd /root/research/model
export PYTHONPATH=/root/research/model/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec /root/research/model/.venv/bin/python -u -m perpetual_v5.experiment --workers 3 --threads 1 --updates 400
