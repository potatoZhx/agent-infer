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

For ordinary HBM allocation, incomplete, invalid, duplicated, non-evictable or failing advice falls back to native LRU
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

## CPU store admission (API 2)

The SimpleCPU provider now requires API 2 and the BlockPool patch's
`store_admission_api_version=1` capability. Rebuild with both current patches;
an older installed patch is rejected at startup instead of silently using the
old eviction behavior.

An incoming CPU store may replace only equal-or-lower protection classes:
BUSY GPU backups can replace empty/INACTIVE or BUSY blocks, but cannot evict
IDLE CPU working sets. IDLE stores can use all three classes. Shared blocks
retain their strongest CPU protection. The eligible count is maintained by the
typed index; it is not computed by scanning the CPU pool per store.

All three eager paths (positional, boundary handoff, and finish-time flush) use
this admission rule. CPU advice rejection leaves the entire batch unchanged
and **does not fall back to native LRU**. Rejected positional stores do not
advance the cursor or pin GPU sources. Boundary handoffs cannot be retried once
the source is gone; a skipped final backup likewise does not certify CPU
residency. Tier labels remain logical protection, not proof of a READY copy.

Evidence: `agentinfer/agentcache/mori/simple_cpu_policy.py:82-94`,
`tests/agentcache/test_mori_native_patch.py:155-294`.
