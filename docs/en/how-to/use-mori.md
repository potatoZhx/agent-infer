# Use MORI with SimpleCPUOffloadConnector

This guide enables full MORI on one embedded vLLM rank. The preferred combination is vLLM 0.31,
`SimpleCPUOffloadConnector`, the CPU backend, and eager offload. This path uses SimpleCPU's native CPU tensors and
asynchronous D2H/H2D transfers; it does not use the `OffloadingConnector` shared-memory allocator.

## Prerequisites

- AgentInfer and the vLLM 0.31 source tree are available;
- one embedded backend rank;
- prefix caching is enabled;
- `kv_offload_backend=cpu`;
- `lazy_offload=false`; and
- the server middleware can resolve a Program identity for every request.

Full mode fails startup if the HBM hook, SimpleCPU hook, or DRAM capacity snapshot is missing. At runtime, missing or
malformed identity and overlapping requests for one Program receive a terminal request error. Invalid victim advice
falls back to the native free-queue order with a warning and a fallback counter. To evaluate placement alone,
explicitly set `"mode":"placement_only"`.

## Apply both narrow vLLM patches

Run these commands from the vLLM 0.31 source root, in order:

```bash
git apply --check \
  /path/to/agent-infer/agentinfer/agentcache/patches/vllm-0.31-mori-hbm-provider.patch
git apply \
  /path/to/agent-infer/agentinfer/agentcache/patches/vllm-0.31-mori-hbm-provider.patch

git apply --check \
  /path/to/agent-infer/agentinfer/agentcache/patches/vllm-0.31-mori-simple-cpu-provider.patch
git apply \
  /path/to/agent-infer/agentinfer/agentcache/patches/vllm-0.31-mori-simple-cpu-provider.patch
```

Rebuild or reinstall vLLM using that source tree's normal procedure. The SimpleCPU patch depends on the instance-local
`BlockPool` provider contract from the first patch, so their order is significant. Without a registered provider, both
native HBM and CPU BlockPools retain their original `popleft_n()` path and do not maintain a MORI typed index.

## Start the server

The repository includes a complete example:

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct \
MORI_CPU_BYTES=17179869184 \
bash examples/serve-mori.sh
```

Its essential configuration is equivalent to:

```bash
export AGENTCACHE_VLLM_LIFECYCLE_SOCKET=/tmp/agentinfer-mori-lifecycle.sock

AGENTCACHE_CONFIG='{"agentcache":{"strategy":"mori","schedule_interval_seconds":5.0,"mori":{"mode":"full","window_size":5}}}'
KV_TRANSFER_CONFIG='{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":17179869184,"kv_offload_backend":"cpu","lazy_offload":false}}'

vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --agentinfer \
  --enable-prefix-caching \
  --additional-config "$AGENTCACHE_CONFIG" \
  --kv-transfer-config "$KV_TRANSFER_CONFIG"
```

`cpu_bytes_to_use` is server-wide; SimpleCPU derives rank-local capacity from the vLLM world size. Use vLLM's
`cpu_bytes_to_use_per_rank` when a deployment needs an explicit per-rank value. MORI consumes the connector's observed
block count and scheduler block size instead of deriving capacity from the command-line byte value again.

## Send a request with Program identity

The server middleware resolves the Program identity below from `X-Claude-Code-Session-Id`. Reuse the same header for
later rounds. The client does not send a tier, idleness, future return time, or replay plan.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Claude-Code-Session-Id: mori-demo-session' \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Summarize the MORI cache policy."}],
    "max_tokens": 128
  }'
```

A subagent must have a separate agent identity. For example, send both
`X-Claude-Code-Session-Id: mori-demo-session` and `X-Claude-Code-Agent-Id: reviewer-1`. The server then creates a
separate Program; clients should not represent concurrent subagents by reusing the lead Program ID.

## Runtime behavior

The SimpleCPU adapter closes the following loop at startup and runtime:

1. Read total, resident, and pending blocks from `SimpleCPUOffloadScheduler` and normalize them to tokens.
2. Register a CPU BlockPool typed-victim provider ordered `INACTIVE -> BUSY -> IDLE`.
3. Keep a D2H destination non-evictable during transfer; native `free_blocks()` publishes readiness and LRU sequence
   after commit.
4. Add server-resolved Program owners on load hits, already-cached stores, and shared content.
5. Let vLLM revalidate victim count, uniqueness, queue membership, refcount, and block ID before commit.
6. Keep HBM ordered `INACTIVE -> IDLE -> BUSY`; vLLM still owns hash eviction, allocation, and D2H/H2D execution.

MORI does not actively prefetch. CPU tier is a protection priority; a later inference still triggers SimpleCPU's native
on-demand H2D load.

Queued requests older than `mori.starvation_timeout_seconds` (default `60.0`, finite and positive) take precedence
across all promotion groups, oldest first. If the oldest feasible request cannot fit, MORI reserves its next turn by
demoting idle GPU residents or marking reasoning residents for demotion after completion. New arrivals join the
pending queue. This serving fairness extension can change sticky placement after the timeout; it is not a bound on
response time. It still requires a request to fit the full GPU quota and in-flight reasoning requests to complete.
Diagnostics include `max_wait_seconds`; normal MORI priority and idleness ordering apply before the timeout.

## Startup checks and troubleshooting

| Symptom | Cause and action |
| --- | --- |
| `requires the MORI BlockPool provider patch` | The first HBM/BlockPool patch is missing or the patches were reversed |
| `requires lazy_offload=false` | Set `lazy_offload` to `false` in the SimpleCPU extra configuration |
| `requires kv_offload_backend=cpu` | Full MORI currently rejects the SimpleCPU disk backend |
| `requires server-resolved Program identity` | The request has no supported body metadata or identity header |
| `requires ... register_mori_block_selection_provider` | The running vLLM installation lacks the HBM hook |
| Missing DRAM quota | The SimpleCPU snapshot hook is not visible, usually because an unpatched vLLM is imported |

Inspect the imported implementation and hooks with:

```bash
python - <<'PY'
import inspect
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler

print(inspect.getfile(SimpleCPUOffloadScheduler))
print(hasattr(SimpleCPUOffloadScheduler, "register_mori_offload_tier_provider"))
print(hasattr(SimpleCPUOffloadScheduler, "get_mori_offload_tier_snapshot"))
PY
```

Source entry points are `examples/serve-mori.sh`, `agentinfer/agentcache/mori/simple_cpu_policy.py`,
`agentinfer/agentcache/mori/vllm_adapter.py`, and
`agentinfer/agentcache/patches/vllm-0.31-mori-simple-cpu-provider.patch`. See the
[MORI design](../../../design/module/scheduling/mori-scheduling.md) for architecture and invariants.
