#!/usr/bin/env bash
# Pre-registered gamma/lambda search (results/agent_v2/PREREGISTRATION.md §3): 9 cells on the inner
# folds 2022 and 2023, seed 11, 30 passes, a checkpoint every 5. Three cells at a time, two workers
# each: six training processes, the host's measured ceiling. Resumable.
#
#   bash scripts/run_search.sh
set -euo pipefail
cd /root/research/model
export PYTHONPATH=/root/research/model/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
cells=()
for g in 0.9583 0.9792 0.9900; do for l in 0.90 0.95 0.9792; do cells+=("$g $l"); done; done
printf '%s\n' "${cells[@]}" | xargs -P 3 -L 1 bash -c '
  out="results/agent_v2/search/gamma_$0_lambda_$1"
  mkdir -p "$out"
  /root/research/model/.venv/bin/python -u -m agent.experiment --search --seed 11 --workers 2 --threads 1 \
    --passes 30 --validation-passes 5 --gamma-hour "$0" --lambda-hour "$1" --output "$out" >> "$out/coordinator.log" 2>&1
  echo "cell gamma=$0 lambda=$1 finished with $?"'
PYTHONPATH=/root/research/model/src /root/research/model/.venv/bin/python scripts/search_summary.py
