# MORI native vLLM hooks

Full MORI targets the locally validated vLLM 0.31.x build. Apply both patches,
in this order, from the root of that vLLM source tree before building/installing
it:

```bash
git apply --check /path/to/agent-infer/agentinfer/agentcache/patches/vllm-0.31-mori-hbm-provider.patch
git apply /path/to/agent-infer/agentinfer/agentcache/patches/vllm-0.31-mori-hbm-provider.patch
git apply --check /path/to/agent-infer/agentinfer/agentcache/patches/vllm-0.31-mori-simple-cpu-provider.patch
git apply /path/to/agent-infer/agentinfer/agentcache/patches/vllm-0.31-mori-simple-cpu-provider.patch
```

The hook is deliberately narrow. Registration bootstraps the native free queue
and allocates the MORI sequence index once; when no provider is registered the
native path neither maintains that index nor changes its `popleft_n()` path.
AgentInfer proposes an exact ordered set of
free-queue block IDs; native vLLM validates every ID, queue membership, refcount,
and uniqueness before it mutates the queue. Hash eviction and allocation remain
native. After bootstrap, readiness, owner, and LRU-sequence deltas maintain the
AgentInfer typed index.

If runtime advice is incomplete, invalid, duplicated, non-evictable, or raises an exception, the pool uses native LRU
after rejecting the entire advice batch. Both paths publish the same removal callbacks. The per-pool
`_mori_fallback_count` records every fallback; warnings are emitted at powers of two to bound repeated logging.
Startup hook capability checks remain strict.

The second patch exposes that same instance-local contract through
`SimpleCPUOffloadScheduler`. It also reports rank-local capacity, resident
blocks, and pending D2H/H2D blocks. Request ownership is observed while an
eager store destination is pinned; abandoned stores remove that observation
before the empty block returns to the free queue. Store completion and the
native `free_blocks()` transition publish the final ready/LRU state. The
supported configuration is `kv_offload_backend=cpu` with
`lazy_offload=false`; unsupported modes fail during registration.

`SimpleCPUOffloadConnector` is the preferred MORI CPU tier. Its native worker
allocates ordinary CPU tensors and optionally pins them; it does not use the
`OffloadingConnector` shared-memory allocator. See
[`docs/en/how-to/use-mori.md`](../../../docs/en/how-to/use-mori.md) or
[`docs/zh/how-to/use-mori.md`](../../../docs/zh/how-to/use-mori.md) for the
complete launch and request example.

The patches are required only for `mori.mode=full`. The explicit
`mori.mode=placement_only` ablation does not install native cache policies.
