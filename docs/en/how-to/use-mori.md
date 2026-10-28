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
malformed identity receives a terminal request error. Overlapping requests use the branch rules below. Invalid victim advice
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

MORI keeps a placement while it fits and there is no eligible replacement. Events wake a shared planning cycle;
arrivals use existing slack without scanning for victims. Normal queue order is CPU returns, other returns, then new
Programs by context size and relative idleness. A replacement must have a complete feasible victim plan. Acting
victims require stable residence and either an idleness advantage or an otherwise idle GPU. An aged waiter may request
lazy relief from Reasoning victims. The beneficiary owns the promised space until its admission or cancellation.
Reasoning victims retain their GPU reservation until completion or confirmed native cancellation. Queued serial
followers cannot prolong that drain.

The engineering defaults in `agentcache.mori` are `min_residency_seconds=1.0`, `idleness_margin=0.05`, and
`max_wait_seconds=30.0`. These extensions are not paper parameters. `max_wait_seconds` triggers fairness selection;
it is not a maximum latency guarantee because running inference must finish. The old
`mori.starvation_timeout_seconds` setting remains unsupported. Diagnostics distinguish observed `max_wait_seconds`
from `configured_max_wait_seconds`.

MORI requires one native KV cache group. Effective HBM capacity excludes the permanent null block. A request whose
prompt plus maximum output exceeds that capacity is rejected before admission. The normal logical claim rounds
context plus 100 decode tokens to HBM blocks; speculative headroom is capped by total capacity. CPU claims round
context to CPU chunks. Shared capacity is discounted only with current-prompt native block identity evidence;
unknown or invalidated evidence is charged conservatively. CPU placement does not certify that a complete CPU copy
is ready. Native refcounts, transfer fences, allocation and recomputation remain authoritative.

### Concurrent branches and lifecycle

A normal serial client may keep its existing Program identity. If another request arrives before the previous one
finishes, MORI creates a temporary independent owner. Completing that owner releases its protection, while native
cache contents remain available for opportunistic reuse. It does not infer which result the application will adopt.
The next real prompt determines the new context, including truncation or context switches.

For a branch that should retain protection between rounds, send a stable `lineage_id` in canonical identity metadata:

```json
{"vllm_xargs":{"agentic_context":"{\"program_id\":\"session:lead\",\"lineage_id\":\"research\"}"}}
```

Requests within the same explicit lineage serialize. Different lineages have independent context, timing, tier and
lifecycle. These internal owners do not create application parent-blocking relationships. Continue to use separate
agent identities for actual subagents. The API middleware generates an internal `lifecycle_token` per attempt;
clients should not set it. A delayed terminal receipt can release only that attempt's latest idle lineage.

See the [2026-10-28 redesign and acceptance contract](../../../design/module/scheduling/mori-redesign-20261028.zh-CN.md)
for the paper comparison, defaults, evidence and unrun GPU validation.

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
