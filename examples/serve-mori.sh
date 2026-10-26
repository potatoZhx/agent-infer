#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
MORI_CPU_BYTES=${MORI_CPU_BYTES:-17179869184}
export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=${AGENTCACHE_VLLM_LIFECYCLE_SOCKET:-/tmp/agentinfer-mori-lifecycle.sock}

AGENTCACHE_CONFIG='{"agentcache":{"strategy":"mori","schedule_interval_seconds":5.0,"mori":{"mode":"full","window_size":5}}}'
KV_TRANSFER_CONFIG='{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"CPUOffloadingSpec","cpu_bytes_to_use":'"$MORI_CPU_BYTES"',"eviction_policy":"MoriCPUCachePolicy","cache_policy_module_path":"agentinfer.agentcache.mori_cpu_policy","store_threshold":0}}'

# Full mode also requires the vLLM 0.31 HBM provider patch documented in
# agentinfer/agentcache/patches/README.md.
exec vllm serve "$MODEL" \
  --agentinfer \
  --additional-config "$AGENTCACHE_CONFIG" \
  --kv-transfer-config "$KV_TRANSFER_CONFIG"
