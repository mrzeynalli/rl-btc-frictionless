#!/usr/bin/env bash
set -euo pipefail
cd /root/research/model
export PYTHONPATH=/root/research/model/src
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
exec /root/research/model/.venv/bin/python -u -m perpetual.experiment --workers 2 --threads 2 --updates 400
