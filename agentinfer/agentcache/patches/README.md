# MORI native vLLM hook

Full MORI targets the locally validated vLLM 0.31.x build. Apply the patch from
the root of that vLLM source tree before building/installing it:

```bash
git apply --check /path/to/agent-infer/agentinfer/agentcache/patches/vllm-0.31-mori-hbm-provider.patch
git apply /path/to/agent-infer/agentinfer/agentcache/patches/vllm-0.31-mori-hbm-provider.patch
```

The hook is deliberately narrow. Registration bootstraps the native free queue
and allocates the MORI sequence index once; when no provider is registered the
native path neither maintains that index nor changes its `popleft_n()` path.
AgentInfer proposes an exact ordered set of
free-queue block IDs; native vLLM validates every ID, queue membership, refcount,
and uniqueness before it mutates the queue. Hash eviction and allocation remain
native. After bootstrap, readiness, owner, and LRU-sequence deltas maintain the
AgentInfer typed index.

The patch is required only for `mori.mode=full`. The explicit
`mori.mode=placement_only` ablation does not install native cache policies.
