#!/usr/bin/env bash
# One-glance status of the agentx d64 experiment chain: servers, per-arm progress, GPU.
# Env override: RESULT_ROOT (default /home/zx/results)
RESULT_ROOT="${RESULT_ROOT:-/home/zx/results}"
echo "=== $(date '+%m-%d %H:%M:%S') ==="
echo "-- servers --"
pgrep -af "vllm.entrypoints.cli.main serve|vllm serve" | grep -oE "scheduler-cls [a-zA-Z_.]+|--agentinfer" | sort | uniq -c || echo "no server"
echo "-- arm progress (requests.jsonl) --"
for d in "$RESULT_ROOT"/*-d64-*/; do
  f="$d/requests.jsonl"
  [[ -f "$f" ]] || continue
  n=$(wc -l < "$f")
  s=$(python3 -c "
import json,collections
c=collections.Counter()
for l in open('$f'): c[json.loads(l).get('status')]+=1
print(dict(c))")
  done=$([[ -f "$d/summary.json" ]] && echo DONE || echo running)
  echo "$(basename "$d"): $n reqs $s $done"
done
echo "-- gpu --"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' '; echo
