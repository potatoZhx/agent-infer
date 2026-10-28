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

GPU membership is the admission decision: the current request goes directly to the engine, while serial followers
wait for its completion. Growth beyond the aggregate GPU budget triggers Acting-first, highest-idleness demotion.
Reasoning victims retain their GPU commitment until completion or confirmed native cancellation; serial followers
cannot prolong that drain. Pending inference is ordered by CPU returns, other returns, then new Programs by context
size and idleness. Candidates that cannot fit do not block later candidates. Waiting inference may reclaim Acting
commitments in idleness order. Without pressure, placement stays unchanged.

There is no beneficiary-owned future capacity, wait timeout, minimum residence or extra idleness margin.
`agentcache.mori` accepts only `mode` and `window_size`; remove the former `min_residency_seconds`, `idleness_margin`,
and `max_wait_seconds` settings. The old `starvation_timeout_seconds` remains unsupported. The diagnostic
`max_wait_seconds` is an observed wait, not a configuration or latency guarantee.

SimpleCPU now requires hook API 2: rebuild with both current native patches. GPU/BUSY backups can use empty,
INACTIVE or BUSY CPU blocks, but cannot evict IDLE working sets. Rejected stores never fall back to ordinary LRU.
Evidence: `agentinfer/scheduling/mori/strategy.py:141-284`, `agentinfer/agentcache/mori/simple_cpu_policy.py:82-94`.

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

There is no `lineage_id` protocol or client-provided branch declaration. A legacy key is ignored like other unknown
metadata and is no longer serialized. Overlapping requests, including overlaps while the first request awaits
admission, get separate temporary owners. There is no named-branch queue of serial followers.

These internal owners do not create application parent-blocking relationships. Use separate agent identities for
actual subagents. The API middleware generates an internal `lifecycle_token` per attempt; clients need not set it.
A delayed terminal receipt can release only the latest idle attempt's owner. Completing or cancelling a temporary
branch releases its logical protection; native cache contents may remain available for opportunistic reuse.
Evidence: `agentinfer/scheduling/request_branches.py:14-58`, `agentinfer/scheduling/runtime.py:527-561`.

See the [state and execution refactor design](../../../design/module/scheduling/mori-state-refactor-20261028.zh-CN.md)
for the protocol removal, evidence and the remaining proposed changes.

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
