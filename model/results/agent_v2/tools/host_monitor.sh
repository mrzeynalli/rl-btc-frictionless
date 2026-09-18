#!/usr/bin/env bash
# Host monitor for the running agent-v2-main unit; exits when that unit stops.
OUT=/root/research/model/results/agent_v2/main
echo "# monitor restarted $(date -u +%FT%TZ): the first monitor stopped after its first sample (set -e in its subshell)" >> "$OUT/host.log"
while systemctl is-active --quiet agent-v2-main; do
  { date -u +%FT%TZ; uptime; free -m | sed -n 2p; ps -eo pid,rss,pcpu,etime,args --sort=-rss | grep '[a]gent.train' | head -8; echo; } >> "$OUT/host.log" 2>&1
  sleep 300
done
