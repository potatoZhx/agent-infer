# Trace provenance

`cc-2.1.152-ptkvc-64g-terminal-32_16.requests.jsonl` is the local AgentInfer request telemetry used by the
formal MORI benchmark. It was copied byte-for-byte from:

```text
/data/zx/AgentInfer/results/kvc/ablation/cc-2.1.152-ptkvc-64g-terminal-32_16/requests.jsonl
```

Integrity and shape:

- SHA-256: `d38b0afe97ea0dd0689d32bff6a888831637f3eb963660055d58a8b4d30c7b4a`
- size: 1,459,299 bytes
- records: 2,435
- Sessions: 32
- distinct global actor IDs: 105; distinct `(session_id, actor_id)` pairs: 136
- status: all 2,435 records are `success` with HTTP 200
- schema version: 1

Each line contains only these fields:

```text
actor_id, actor_role, cache_creation_tokens, cached_tokens, error,
finished_at, input_tokens, latency_seconds, output_tokens, request_id,
run_id, schema_version, session_id, started_at, status, status_code,
ttft_seconds, upstream
```

The trace contains no prompt, message, completion text, tool payload, credential, or remote upstream URL. AgentBench
constructs deterministic synthetic prompts from the token and identity telemetry; it does not reproduce original task
semantics or correctness.
