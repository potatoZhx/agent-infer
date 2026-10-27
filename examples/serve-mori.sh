#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
MORI_CPU_BYTES=${MORI_CPU_BYTES:-17179869184}
export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=${AGENTCACHE_VLLM_LIFECYCLE_SOCKET:-/tmp/agentinfer-mori-lifecycle.sock}

AGENTCACHE_CONFIG='{"agentcache":{"strategy":"mori","schedule_interval_seconds":5.0,"mori":{"mode":"full","window_size":5}}}'
KV_TRANSFER_CONFIG='{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":'"$MORI_CPU_BYTES"',"kv_offload_backend":"cpu","lazy_offload":false}}'

# Full mode requires both vLLM 0.31 MORI provider patches documented in
# agentinfer/agentcache/patches/README.md.
exec vllm serve "$MODEL" \
  --agentinfer \
  --enable-prefix-caching \
  --additional-config "$AGENTCACHE_CONFIG" \
  --kv-transfer-config "$KV_TRANSFER_CONFIG"
