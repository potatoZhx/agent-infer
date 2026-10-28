#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

set -Eeuo pipefail

SUITE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SUITE_DIR" rev-parse --show-toplevel)
CONFIG=$SUITE_DIR/config/replay16-8.yaml
TRACE=$SUITE_DIR/traces/cc-2.1.152-ptkvc-64g-terminal-32_16.requests.jsonl
TRACE_SHA256=d38b0afe97ea0dd0689d32bff6a888831637f3eb963660055d58a8b4d30c7b4a

MODEL_PATH=${MODEL_PATH:-}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8}
BASELINE_PYTHON=${BASELINE_PYTHON:-python3}
MORI_PYTHON=${MORI_PYTHON:-}
MORI_PYTHONPATH=${MORI_PYTHONPATH:-}
CLIENT_PYTHON=${CLIENT_PYTHON:-$BASELINE_PYTHON}
EXPECTED_VLLM_SERIES=${EXPECTED_VLLM_SERIES:-0.31}

PORT=${PORT:-19147}
BASE_URL=${BASE_URL:-http://127.0.0.1:$PORT}
RESULT_ROOT=${RESULT_ROOT:-$REPO_ROOT/results/mori-formal/swe-cc-local-trace-$(date -u +%Y%m%dT%H%M%SZ)}
LIFECYCLE_SOCKET=${LIFECYCLE_SOCKET:-/tmp/agentinfer-mori-formal-swe-cc-$PORT.sock}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.31}
CPU_BYTES=${CPU_BYTES:-17179869184}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-107632}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1024}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
BLOCK_SIZE=${BLOCK_SIZE:-16}
TASK_NUM=${TASK_NUM:-16}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-8}
SERVER_START_TIMEOUT_SECONDS=${SERVER_START_TIMEOUT_SECONDS:-2400}
SERVER_STOP_TIMEOUT_SECONDS=${SERVER_STOP_TIMEOUT_SECONDS:-180}
COOLDOWN_SECONDS=${COOLDOWN_SECONDS:-30}

CLI_CODE='import sys; from agentinfer.agentcache.entrypoints.cli.main import main; raise SystemExit(main(["vllm", *sys.argv[1:]]))'
KV_CONFIG='{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":'"$CPU_BYTES"',"kv_offload_backend":"cpu","lazy_offload":false}}'
PT_CONFIG='{"agentcache":{"strategy":"progress_ttl","schedule_interval_seconds":5.0,"progress_ttl":{"ttl_prefill_model_intercept_seconds":0.04072930711406778,"ttl_prefill_model_linear_seconds_per_1k_tokens":0.013789636679399002,"ttl_prefill_model_quadratic_seconds_per_1k_tokens_squared":0.0006700705771443464,"decode_step_fixed_seconds":0.06026559022348978,"decode_step_seconds_per_context_token":0.0,"decode_step_seconds_per_request":0.00004228412518563339}}}'
MORI_CONFIG='{"agentcache":{"strategy":"mori","schedule_interval_seconds":5.0,"mori":{"mode":"full","window_size":5}}}'

SERVER_PID=
MONITOR_PID=
CURRENT_RUNTIME=
RUN_INITIALIZED=false

usage() {
  cat <<'EOF'
Usage:
  run_matrix.sh validate
  run_matrix.sh plan
  run_matrix.sh preflight
  run_matrix.sh run {vanilla|pt|mori}
  run_matrix.sh matrix
  run_matrix.sh compare

Required for preflight/run/matrix:
  MODEL_PATH         local model directory or model ID
  BASELINE_PYTHON    Python with native vLLM 0.31.x (default: python3)
  MORI_PYTHON        Python with the same vLLM version plus both MORI patches

Optional:
  MORI_PYTHONPATH    patched vLLM overlay prepended only for the MORI arm
  RESULT_ROOT        new output directory; existing paths are rejected for runs
  MODEL_NAME, PORT, CUDA_VISIBLE_DEVICES and the common matrix variables documented in README.md
EOF
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

python_path_for_arm() {
  local arm=$1
  if [[ $arm == mori && -n $MORI_PYTHONPATH ]]; then
    printf '%s:%s\n' "$MORI_PYTHONPATH" "$REPO_ROOT"
  else
    printf '%s\n' "$REPO_ROOT"
  fi
}

python_for_arm() {
  local arm=$1
  if [[ $arm == mori ]]; then
    printf '%s\n' "$MORI_PYTHON"
  else
    printf '%s\n' "$BASELINE_PYTHON"
  fi
}

agentinfer_cli() {
  PYTHONPATH=$REPO_ROOT "$CLIENT_PYTHON" -c "$CLI_CODE" "$@"
}

validate_suite() {
  require_command sha256sum
  local actual_sha
  actual_sha=$(sha256sum "$TRACE" | awk '{print $1}')
  [[ $actual_sha == "$TRACE_SHA256" ]] || fail "trace SHA-256 mismatch: $actual_sha"
  command -v "$CLIENT_PYTHON" >/dev/null 2>&1 || fail "CLIENT_PYTHON is not executable: $CLIENT_PYTHON"
  PYTHONPATH=$REPO_ROOT SUITE_CONFIG=$CONFIG EXPECTED_TRACE=$TRACE \
    "$CLIENT_PYTHON" - <<'PY'
import os
from pathlib import Path

from agentinfer.agentbench.replay.config import load_replay_config

config = load_replay_config(Path(os.environ["SUITE_CONFIG"]))
expected_trace = Path(os.environ["EXPECTED_TRACE"]).resolve()
assert config.experiment.task_num == 16
assert config.experiment.max_concurrency == 8
assert config.replay.trace_type == "agentinfer"
assert config.replay.trace_path == expected_trace
assert config.backend.endpoint == "/v1/messages"
assert config.replay.interval_mode == "trace"
assert config.replay.sample_seed == 0
print(f"validated config={os.environ['SUITE_CONFIG']}")
print(f"validated trace={expected_trace}")
PY
  printf 'trace_sha256=%s\n' "$actual_sha"
}

environment_facts() {
  local arm=$1
  local python
  local pythonpath
  python=$(python_for_arm "$arm")
  pythonpath=$(python_path_for_arm "$arm")
  [[ -n $python ]] || fail "MORI_PYTHON is required"
  command -v "$python" >/dev/null 2>&1 || fail "$arm Python is not executable: $python"
  PYTHONPATH=$pythonpath EXPECTED_ARM=$arm EXPECTED_VLLM_SERIES=$EXPECTED_VLLM_SERIES REPO_ROOT=$REPO_ROOT \
    "$python" - <<'PY'
import importlib.metadata as metadata
import inspect
import json
import os
from pathlib import Path

import agentinfer
import vllm
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler

arm = os.environ["EXPECTED_ARM"]
version = metadata.version("vllm")
series = os.environ["EXPECTED_VLLM_SERIES"]
if not (version == series or version.startswith(f"{series}.")):
    raise SystemExit(f"{arm}: expected vLLM {series}.x, got {version}")

agentinfer_path = Path(inspect.getfile(agentinfer)).resolve()
repo_root = Path(os.environ["REPO_ROOT"]).resolve()
if repo_root not in agentinfer_path.parents:
    raise SystemExit(f"{arm}: AgentInfer is not imported from checkout: {agentinfer_path}")

hooks = {
    "hbm_hook": hasattr(KVCacheManager, "register_mori_block_selection_provider"),
    "blockpool_hook": hasattr(BlockPool, "register_mori_block_selection_provider"),
    "simplecpu_hook": hasattr(SimpleCPUOffloadScheduler, "register_mori_offload_tier_provider"),
    "simplecpu_snapshot": hasattr(SimpleCPUOffloadScheduler, "get_mori_offload_tier_snapshot"),
}
if arm == "mori" and not all(hooks.values()):
    raise SystemExit(f"mori: required provider hooks are missing: {hooks}")
if arm != "mori" and any(hooks.values()):
    raise SystemExit(f"{arm}: baseline vLLM is contaminated by MORI hooks: {hooks}")

print(json.dumps({
    "arm": arm,
    "python": os.sys.executable,
    "agentinfer": str(agentinfer_path),
    "vllm": inspect.getfile(vllm),
    "vllm_version": version,
    "hooks": hooks,
}, sort_keys=True))
PY
}

vllm_base_version() {
  local arm=$1
  local python
  local pythonpath
  python=$(python_for_arm "$arm")
  pythonpath=$(python_path_for_arm "$arm")
  PYTHONPATH=$pythonpath "$python" - <<'PY'
from importlib.metadata import version
print(version("vllm").split("+", 1)[0])
PY
}

preflight() {
  validate_suite
  require_command curl
  require_command setsid
  [[ -n $MODEL_PATH ]] || fail "MODEL_PATH is required"
  [[ ${#LIFECYCLE_SOCKET} -le 99 ]] || fail "lifecycle socket path is too long: $LIFECYCLE_SOCKET"
  [[ ! -e $LIFECYCLE_SOCKET && ! -e ${LIFECYCLE_SOCKET}.dp0 ]] || \
    fail "lifecycle socket already exists: $LIFECYCLE_SOCKET[.dp0]"
  if curl --noproxy '*' --silent --fail --max-time 1 "$BASE_URL/health" >/dev/null 2>&1; then
    fail "an existing service is already healthy at $BASE_URL"
  fi
  environment_facts vanilla
  environment_facts mori
  local baseline_version
  local mori_version
  baseline_version=$(vllm_base_version vanilla)
  mori_version=$(vllm_base_version mori)
  [[ $baseline_version == "$mori_version" ]] || \
    fail "baseline and MORI vLLM versions differ: $baseline_version vs $mori_version"
  printf 'preflight_ok vllm=%s base_url=%s\n' "$baseline_version" "$BASE_URL"
}

stop_monitor() {
  if [[ -n ${MONITOR_PID:-} ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill -TERM "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  MONITOR_PID=
}

stop_server() {
  stop_monitor
  if [[ -n ${SERVER_PID:-} ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -INT -- "-$SERVER_PID" 2>/dev/null || kill -INT "$SERVER_PID" 2>/dev/null || true
    local deadline=$((SECONDS + SERVER_STOP_TIMEOUT_SECONDS))
    while (( SECONDS < deadline )) && kill -0 "$SERVER_PID" 2>/dev/null; do
      sleep 2
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
      sleep 5
    fi
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -KILL -- "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=
  if [[ -n ${CURRENT_RUNTIME:-} && -d $CURRENT_RUNTIME ]] && command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw \
      --format=csv,noheader > "$CURRENT_RUNTIME/gpu-after.csv" || true
  fi
  [[ ! -e ${LIFECYCLE_SOCKET}.dp0 ]] || unlink "${LIFECYCLE_SOCKET}.dp0"
  [[ ! -e $LIFECYCLE_SOCKET ]] || unlink "$LIFECYCLE_SOCKET"
}

on_exit() {
  local rc=$?
  stop_server
  if [[ $RUN_INITIALIZED == true ]]; then
    printf 'exit_code=%s\nended_at=%s\n' "$rc" "$(date --iso-8601=seconds)" > "$RESULT_ROOT/TERMINAL.txt"
    if (( rc != 0 )); then
      printf 'invalid\n' > "$RESULT_ROOT/INVALID"
    fi
  fi
}

initialize_output() {
  [[ ! -e $RESULT_ROOT ]] || fail "RESULT_ROOT already exists: $RESULT_ROOT"
  mkdir -p "$RESULT_ROOT"/{runs,runtime,comparisons,provenance}
  RUN_INITIALIZED=true
  trap on_exit EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
}

record_provenance() {
  sha256sum "$TRACE" "$CONFIG" > "$RESULT_ROOT/provenance/input-sha256.txt"
  git -C "$REPO_ROOT" rev-parse HEAD > "$RESULT_ROOT/provenance/agentinfer-head.txt"
  git -C "$REPO_ROOT" status --short > "$RESULT_ROOT/provenance/agentinfer-status.txt"
  {
    printf 'MODEL_PATH=%q\n' "$MODEL_PATH"
    printf 'MODEL_NAME=%q\n' "$MODEL_NAME"
    printf 'BASELINE_PYTHON=%q\n' "$BASELINE_PYTHON"
    printf 'MORI_PYTHON=%q\n' "$MORI_PYTHON"
    printf 'MORI_PYTHONPATH=%q\n' "$MORI_PYTHONPATH"
    printf 'PORT=%q\n' "$PORT"
    printf 'GPU_MEMORY_UTILIZATION=%q\n' "$GPU_MEMORY_UTILIZATION"
    printf 'CPU_BYTES=%q\n' "$CPU_BYTES"
    printf 'TENSOR_PARALLEL_SIZE=%q\n' "$TENSOR_PARALLEL_SIZE"
    printf 'MAX_MODEL_LEN=%q\n' "$MAX_MODEL_LEN"
    printf 'MAX_NUM_SEQS=%q\n' "$MAX_NUM_SEQS"
    printf 'MAX_NUM_BATCHED_TOKENS=%q\n' "$MAX_NUM_BATCHED_TOKENS"
    printf 'BLOCK_SIZE=%q\n' "$BLOCK_SIZE"
    printf 'TASK_NUM=%q\n' "$TASK_NUM"
    printf 'MAX_CONCURRENCY=%q\n' "$MAX_CONCURRENCY"
    printf 'CUDA_VISIBLE_DEVICES=%q\n' "${CUDA_VISIBLE_DEVICES:-}"
  } > "$RESULT_ROOT/provenance/settings.env"
  environment_facts vanilla > "$RESULT_ROOT/provenance/baseline-environment.json"
  environment_facts mori > "$RESULT_ROOT/provenance/mori-environment.json"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -q > "$RESULT_ROOT/provenance/nvidia-smi-q.txt"
  fi
}

common_server_args() {
  COMMON_ARGS=(
    serve "$MODEL_PATH"
    --served-model-name "$MODEL_NAME"
    --host 127.0.0.1
    --port "$PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --async-scheduling
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --block-size "$BLOCK_SIZE"
    --shutdown-timeout 60
    --enable-prefix-caching
    --prefix-caching-hash-algo sha256
    --enable-prompt-tokens-details
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --kv-transfer-config "$KV_CONFIG"
  )
}

server_args_for_arm() {
  local arm=$1
  common_server_args
  SERVER_ARGS=("${COMMON_ARGS[@]}")
  case "$arm" in
    vanilla)
      SERVER_ARGS+=(--middleware agentinfer.agentcache.core.api_adapter.AgentCacheIdentityMiddleware)
      ;;
    pt)
      SERVER_ARGS+=(--agentinfer --additional-config "$PT_CONFIG")
      ;;
    mori)
      SERVER_ARGS+=(--agentinfer --additional-config "$MORI_CONFIG")
      ;;
    *)
      fail "unknown arm: $arm"
      ;;
  esac
}

replay_args_for_arm() {
  local arm=$1
  REPLAY_ARGS=(
    bench serve --agentinfer replay
    --config "$CONFIG"
    --base-url "$BASE_URL"
    --model "$MODEL_NAME"
  )
  if [[ $TASK_NUM != 16 ]]; then
    REPLAY_ARGS+=(--task-num "$TASK_NUM")
  fi
  if [[ $MAX_CONCURRENCY != 8 ]]; then
    REPLAY_ARGS+=(--max-concurrency "$MAX_CONCURRENCY")
  fi
  REPLAY_ARGS+=(--result-dir "$RESULT_ROOT/runs/$arm")
}

print_plan() {
  [[ -n $MODEL_PATH ]] || fail "MODEL_PATH is required to print the concrete plan"
  local arm
  local python
  local pythonpath
  for arm in vanilla pt mori; do
    python=$(python_for_arm "$arm")
    [[ -n $python ]] || python='<MORI_PYTHON>'
    pythonpath=$(python_path_for_arm "$arm")
    server_args_for_arm "$arm"
    printf 'pythonpath[%s]=%s\n' "$arm" "$pythonpath"
    printf 'server[%s]=' "$arm"
    printf '%q ' "$python" -c "$CLI_CODE" "${SERVER_ARGS[@]}"
    printf '\n'
    replay_args_for_arm "$arm"
    printf 'client[%s]=' "$arm"
    printf '%q ' "$CLIENT_PYTHON" -c "$CLI_CODE" "${REPLAY_ARGS[@]}"
    printf '\n'
  done
}

start_server() {
  local arm=$1
  local runtime=$2
  local python
  local pythonpath
  python=$(python_for_arm "$arm")
  pythonpath=$(python_path_for_arm "$arm")
  server_args_for_arm "$arm"

  [[ ! -e $LIFECYCLE_SOCKET && ! -e ${LIFECYCLE_SOCKET}.dp0 ]] || \
    fail "lifecycle socket already exists before $arm: $LIFECYCLE_SOCKET[.dp0]"
  mkdir -p "$runtime"
  CURRENT_RUNTIME=$runtime
  printf '%q ' "$python" -c "$CLI_CODE" "${SERVER_ARGS[@]}" > "$runtime/server-command.txt"
  printf '\n' >> "$runtime/server-command.txt"
  printf '%s\n' "$pythonpath" > "$runtime/server-pythonpath.txt"
  environment_facts "$arm" > "$runtime/import-paths.json"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw \
      --format=csv,noheader > "$runtime/gpu-before.csv"
  fi

  (
    cd "$REPO_ROOT"
    exec setsid env \
      PYTHONPATH="$pythonpath" \
      AGENTCACHE_VLLM_LIFECYCLE_SOCKET="$LIFECYCLE_SOCKET" \
      TOKENIZERS_PARALLELISM=false \
      "$python" -c "$CLI_CODE" "${SERVER_ARGS[@]}"
  ) > "$runtime/server.log" 2>&1 &
  SERVER_PID=$!
  printf '%s\n' "$SERVER_PID" > "$runtime/server.pid"

  if command -v nvidia-smi >/dev/null 2>&1; then
    (
      while true; do
        nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,temperature.gpu,power.draw \
          --format=csv,noheader
        sleep 5
      done
    ) > "$runtime/gpu-monitor.csv" 2> "$runtime/gpu-monitor.err" &
    MONITOR_PID=$!
  fi

  local deadline=$((SECONDS + SERVER_START_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -200 "$runtime/server.log" >&2 || true
      fail "$arm server exited before becoming healthy"
    fi
    if curl --noproxy '*' --silent --show-error --fail "$BASE_URL/health" > "$runtime/health.txt" 2>/dev/null; then
      curl --noproxy '*' --silent --show-error --fail "$BASE_URL/v1/models" > "$runtime/models.json"
      curl --noproxy '*' --silent --show-error --fail "$BASE_URL/metrics" > "$runtime/metrics-cold.prom"
      return
    fi
    sleep 10
  done
  tail -200 "$runtime/server.log" >&2 || true
  fail "$arm server health timeout after ${SERVER_START_TIMEOUT_SECONDS}s"
}

assert_completed() {
  local run_dir=$1
  RUN_MANIFEST=$run_dir/manifest.json "$CLIENT_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

manifest = Path(os.environ["RUN_MANIFEST"])
payload = json.loads(manifest.read_text(encoding="utf-8"))
if payload.get("status") != "completed":
    raise SystemExit(f"run is not completed: {manifest}: {payload.get('status')}")
summary = manifest.parent / "summary.json"
if not summary.is_file():
    raise SystemExit(f"completed run has no summary: {summary}")
PY
}

run_arm() {
  local arm=$1
  local run_dir=$RESULT_ROOT/runs/$arm
  local runtime=$RESULT_ROOT/runtime/$arm
  [[ ! -e $run_dir && ! -e $runtime ]] || fail "arm output already exists: $arm"
  start_server "$arm" "$runtime"
  replay_args_for_arm "$arm"
  (
    cd "$REPO_ROOT"
    printf '%q ' "$CLIENT_PYTHON" -c "$CLI_CODE" "${REPLAY_ARGS[@]}" > "$runtime/client-command.txt"
    printf '\n' >> "$runtime/client-command.txt"
    agentinfer_cli "${REPLAY_ARGS[@]}"
  ) > "$runtime/client.log" 2>&1
  curl --noproxy '*' --silent --show-error --fail "$BASE_URL/metrics" > "$runtime/metrics-post-client.prom"
  assert_completed "$run_dir"
  stop_server
  CURRENT_RUNTIME=
  if (( COOLDOWN_SECONDS > 0 )); then
    sleep "$COOLDOWN_SECONDS"
  fi
}

official_reports() {
  local vanilla=$RESULT_ROOT/runs/vanilla
  local pt=$RESULT_ROOT/runs/pt
  local mori=$RESULT_ROOT/runs/mori
  for run_dir in "$vanilla" "$pt" "$mori"; do
    assert_completed "$run_dir"
  done
  agentinfer_cli bench serve --agentinfer summarize "$vanilla" "$pt" "$mori" \
    --output-csv "$RESULT_ROOT/comparisons/combined-summary.csv"
  agentinfer_cli bench serve --agentinfer compare --json --baseline "$vanilla" --candidate "$pt" \
    > "$RESULT_ROOT/comparisons/vanilla-vs-pt.json"
  agentinfer_cli bench serve --agentinfer compare --json --baseline "$vanilla" --candidate "$mori" \
    > "$RESULT_ROOT/comparisons/vanilla-vs-mori.json"
  agentinfer_cli bench serve --agentinfer compare --json --baseline "$pt" --candidate "$mori" \
    > "$RESULT_ROOT/comparisons/pt-vs-mori.json"
}

command_name=${1:-}
case "$command_name" in
  validate)
    [[ $# -eq 1 ]] || fail "validate takes no arguments"
    validate_suite
    ;;
  plan)
    [[ $# -eq 1 ]] || fail "plan takes no arguments"
    print_plan
    ;;
  preflight)
    [[ $# -eq 1 ]] || fail "preflight takes no arguments"
    preflight
    ;;
  run)
    [[ $# -eq 2 ]] || fail "run requires exactly one arm"
    arm=$2
    [[ $arm == vanilla || $arm == pt || $arm == mori ]] || fail "unknown arm: $arm"
    preflight
    initialize_output
    record_provenance
    run_arm "$arm"
    ;;
  matrix)
    [[ $# -eq 1 ]] || fail "matrix takes no arguments"
    preflight
    initialize_output
    record_provenance
    run_arm vanilla
    run_arm pt
    run_arm mori
    official_reports
    ;;
  compare)
    [[ $# -eq 1 ]] || fail "compare takes no arguments"
    [[ -d $RESULT_ROOT ]] || fail "RESULT_ROOT does not exist: $RESULT_ROOT"
    mkdir -p "$RESULT_ROOT/comparisons"
    official_reports
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
