#!/usr/bin/env bash
# Agent v2 main run: outer folds 2024 and 2025 (four quarters each) x five seeds, 60 passes, a
# checkpoint every 5 passes, then selection. Measured at 1.4-2.0 s per update with six workers and
# ~1 GB per worker; about 4-5 hours with six workers. Resumable. 2026 stays locked.
#
# Logs: <output>/coordinator.log (this script), <output>/host.log (load and memory every 5 minutes),
# <output>/logs/<year>_<seed>.log and runs/<year>/seed_<seed>/ (per run), <output>/provenance/.
#
#   bash scripts/run_agent.sh [workers]
set -euo pipefail
WORKERS=${1:-6}
OUT=results/agent_v2/main
cd /root/research/model
mkdir -p "$OUT"
export PYTHONPATH=/root/research/model/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
( set +e  # a monitor must never stop because one probe finds nothing
  while true; do
    { date -u +%FT%TZ; uptime; free -m | sed -n 2p; ps -eo pid,rss,pcpu,etime,args --sort=-rss | grep '[a]gent.train' | head -8; echo; } >> "$OUT/host.log" 2>&1
    sleep 300
  done ) &
MONITOR=$!
trap 'kill $MONITOR 2>/dev/null || true' EXIT
/root/research/model/.venv/bin/python -u -m agent.experiment --workers "$WORKERS" --threads 1 \
  --passes 60 --validation-passes 5 --output "$OUT" 2>&1 | tee -a "$OUT/coordinator.log"
