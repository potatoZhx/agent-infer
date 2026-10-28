#!/usr/bin/env bash
# Cold-start one arm, run AgentX trace replay on the duration64 trace, collect artifacts, stop server.
# All three arms share identical HBM/CPU capacity settings via COMMON_ARGS (unified offload).
#
# Usage: run-d64-tp2.sh <vanilla|progress-ttl|mori> <task_num> <concurrency> <task_timeout_s> <request_timeout_s> [util]
# Env overrides: AGENTINFER_ROOT, VENV, MODEL_PATH, MODEL_NAME, TRACE_PATH, RESULT_ROOT,
#                REPLAY_CONFIG, PORT, MORI_MODE (full|placement_only)
#
# Smoke:      ./run-d64-tp2.sh mori 1 1 43200 7200 0.75
# Group A:    ./run-d64-tp2.sh <arm> 16 8 43200 7200 0.75   (serial config, max_inflight_requests=1)
# Group B:    REPLAY_CONFIG=<suite>/config/replay_agentX_inflight8.yaml ./run-d64-tp2.sh <arm> 32 16 43200 7200 0.75
set -euo pipefail

ARM="$1"; TASK_NUM="$2"; CONC="$3"; TASK_TIMEOUT="$4"; REQ_TIMEOUT="$5"; UTIL="${6:-0.75}"
MORI_MODE="${MORI_MODE:-full}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="${AGENTINFER_ROOT:-/home/zx/agent-infer-mori-formal}"
VENV="${VENV:-/home/zx/agent-infer-mori/.venv-vllm-0.31.0-cu129}"
MODEL="${MODEL_PATH:-/home/models/Qwen3-Coder-30B-A3B-Instruct-FP8}"
SERVED="${MODEL_NAME:-qwen3-coder-30b}"
TRACE="${TRACE_PATH:-$SCRIPT_DIR/data/traces-duration64-ctx120k-gap10min.jsonl}"
REPLAY_CONFIG="${REPLAY_CONFIG:-$SCRIPT_DIR/config/replay_agentX_serial.yaml}"
RESULT_ROOT="${RESULT_ROOT:-/home/zx/results}"
PORT="${PORT:-8300}"
LOG_DIR="${LOG_DIR:-$RESULT_ROOT/logs}"

ARM_TAG="$ARM-d64-${TASK_NUM}x${CONC}"
RESULT_DIR="$RESULT_ROOT/${ARM_TAG}"
SERVER_LOG="$LOG_DIR/server-${ARM_TAG}.log"
REPLAY_LOG="$LOG_DIR/replay-${ARM_TAG}.log"

export PYTHONPATH="$WORKTREE"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_LOGGING_LEVEL=INFO
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p "$LOG_DIR"
[[ -e "$RESULT_DIR" ]] && { echo "result dir exists: $RESULT_DIR" >&2; exit 2; }
[[ -f "$TRACE" ]] || { echo "trace missing (gunzip data/*.jsonl.gz first?): $TRACE" >&2; exit 2; }
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "port $PORT already in use" >&2; ss -ltnp 2>/dev/null | grep ":${PORT} " >&2; exit 2
fi
if pgrep -af "vllm" | grep -vq "$$"; then
  echo "WARNING: pre-existing vllm processes:" >&2; pgrep -af vllm >&2 || true
fi

# Unified capacity settings shared by ALL arms (identical HBM KV and CPU tier).
COMMON_ARGS=(
  --served-model-name "$SERVED"
  --host 127.0.0.1 --port "$PORT"
  --tensor-parallel-size 2
  --gpu-memory-utilization "$UTIL"
  --max-model-len 118784
  --max-num-seqs 64
  --async-scheduling
  --block-size 16
  --enable-prefix-caching
  --prefix-caching-hash-algo sha256
  --enable-prompt-tokens-details
  --shutdown-timeout 60
  --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":17179869184,"kv_offload_backend":"cpu","lazy_offload":false}}'
)

start_server() {
  cd "$WORKTREE"
  case "$ARM" in
    vanilla)
      "$VENV/bin/python" -m vllm.entrypoints.cli.main serve "$MODEL" \
        "${COMMON_ARGS[@]}" \
        --scheduler-cls vllm.v1.core.sched.async_scheduler.AsyncScheduler \
        > "$SERVER_LOG" 2>&1 &
      ;;
    progress-ttl)
      rm -f "/tmp/agentinfer-ttl-$PORT.sock"
      AGENTCACHE_VLLM_LIFECYCLE_SOCKET="/tmp/agentinfer-ttl-$PORT.sock" \
      "$VENV/bin/vllm" serve "$MODEL" --agentinfer "${COMMON_ARGS[@]}" \
        > "$SERVER_LOG" 2>&1 &
      ;;
    mori)
      rm -f "/tmp/agentinfer-mori-$PORT.sock"
      local MORI_ARGS=(
        --additional-config '{"agentcache":{"strategy":"mori","schedule_interval_seconds":5.0,"mori":{"mode":"'"$MORI_MODE"'","window_size":5}}}'
      )
      AGENTCACHE_VLLM_LIFECYCLE_SOCKET="/tmp/agentinfer-mori-$PORT.sock" \
      "$VENV/bin/vllm" serve "$MODEL" --agentinfer "${COMMON_ARGS[@]}" "${MORI_ARGS[@]}" \
        > "$SERVER_LOG" 2>&1 &
      ;;
    *) echo "unknown arm $ARM" >&2; exit 2;;
  esac
  SERVER_PID=$!
  echo "[arm $ARM/$MORI_MODE] server pid $SERVER_PID, log $SERVER_LOG"
}

wait_healthy() {
  local deadline=$((SECONDS + 1800))
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[arm $ARM] server died; tail of log:" >&2; tail -60 "$SERVER_LOG" >&2; exit 3
    fi
    if [ $SECONDS -gt $deadline ]; then echo "[arm $ARM] health timeout" >&2; exit 4; fi
    sleep 5
  done
  echo "[arm $ARM] healthy after ${SECONDS}s"
  grep -E "GPU KV cache size|Available KV cache|Model loading took|per_rank=" "$SERVER_LOG" | tail -6 || true
}

run_replay() {
  cd "$WORKTREE"
  "$VENV/bin/vllm" bench serve --agentinfer replay \
    --config "$REPLAY_CONFIG" \
    --trace-type agentX \
    --trace-path "$TRACE" \
    --base-url "http://127.0.0.1:$PORT" \
    --metrics-url "http://127.0.0.1:$PORT/metrics" \
    --endpoint /v1/completions \
    --model "$SERVED" \
    --interval-mode trace \
    --task-num "$TASK_NUM" \
    --max-concurrency "$CONC" \
    --task-timeout-seconds "$TASK_TIMEOUT" \
    --request-timeout-seconds "$REQ_TIMEOUT" \
    --result-dir "$RESULT_DIR" \
    2>&1 | tee "$REPLAY_LOG"
}

stop_server() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 90); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 2; done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
  fi
  pkill -f "vllm.entrypoints.cli.main serve" 2>/dev/null || true
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 5
}

trap stop_server EXIT
start_server
wait_healthy
run_replay
stop_server
trap - EXIT
echo "[arm $ARM/$MORI_MODE] done: $RESULT_DIR"
