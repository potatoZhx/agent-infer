# MORI server-only isolated implementation

This directory packages the causal MORI implementation used by the AgentX real
benchmark as an out-of-tree vLLM scheduler. It is intentionally isolated from
the production `agentinfer` package: checking out this branch does not make MORI
the default scheduler and does not change Vanilla or Progress-TTL startup.

The supported runtime is vLLM `0.31.0+cu129`. The adapter uses vLLM private
cache and offload APIs, so another vLLM build must fail the startup gate instead
of being treated as compatible.

## Server-only contract

“Server-only” means the scheduling process obtains all online state from the
request and from events already committed inside EngineCore:

- Program identity comes from `vllm_xargs.agentic_context` on the arriving
  request.
- Request completion comes from EngineCore cleanup; Program terminal/continue
  lifecycle comes from the completed HTTP response and its request metadata.
- Current HBM and optional CPU-KV capacity come from the native vLLM managers.
- The MORI server neither receives nor opens a Replay plan and performs no
  future-return lookup.

The Replay client may still use a trace plan to generate workload arrivals.
That client-side workload input is not exported to the server. The only
client-to-server scheduling fields are the identity of the current request and
its live lifecycle declaration.

Startup fails closed when the preflight does not say
`plan_available_to_server=false`, a `KVC_MORI_PLAN` variable is present, a
candidate source hash changes, the imported AgentInfer checkout differs from
the sealed checkout, or vLLM is not exactly `0.31.0+cu129`.

## Directory layout

| Path | Purpose |
| --- | --- |
| `mori_http_mp.py` | vLLM `AsyncScheduler` adapter and final audit |
| `joint/mori.py` | k=5 relative-idleness placement and typed GPU/CPU eviction |
| `joint/eviction.py` | Native HBM safe-tail block selection |
| `joint/server_state.py` | Plan-free request-to-Program ownership and completion events |
| `adapters/private_pinned.py` | Optional single-worker native pinned CPU-KV allocation |
| `bootstrap/sitecustomize.py` | Early vLLM version gate |
| `inputs/policy-config.json` | Frozen public MORI policy parameters |
| `requirements-vllm-0.31.0-cu129.txt` | Exact server-environment freeze used by validation |
| `prepare_server.py` | Creates a fresh sealed preflight and launch environment |
| `VALIDATION.md` | Commands, exact runtime, results, and explicit not-run scope |

`joint/replay.py`, workload plans, benchmark traces, models, and result files are
deliberately not part of this server package.

## 1. Prepare the environment

The validated Linux x86_64 server environment is pinned as follows. These are
observed package versions from the real benchmark host, not broad compatibility
claims.

| Component | Validated version |
| --- | --- |
| Python | `3.12.13` |
| uv | `0.11.8` |
| vLLM distribution / module | `0.31.0+cu129` / `0.31.0` official manylinux 2.28 wheel |
| PyTorch | `2.13.0+cu129` |
| torchvision / torchaudio | `0.28.0+cu129` / `2.11.0+cu129` |
| CUDA Python/toolkit packages | `12.9.10` / `12.9.1` |
| Triton | `3.7.1` |
| FlashInfer Python | `0.7.0.post1` |
| Transformers / tokenizers | `5.17.0` / `0.23.3` |
| NumPy | `2.3.5` |
| FastAPI / Uvicorn | `0.136.3` / `0.54.0` |
| Pydantic / HTTPX | `2.14.1` / `0.28.1` |
| NVIDIA driver on the validated host | `570.133.20` |

`flashinfer-cubin` and `xformers` are not installed in the validated
environment. Do not add either package merely to make the package list look
more complete.

### Existing environment on the same machine

An existing AgentInfer/vLLM environment can be reused only when the checks
below print the pinned Python, vLLM, torch, and CUDA versions. Keep a shared
environment read-only while another experiment is using it.

```bash
git switch codex/mori-server-only

python - <<'PY'
import importlib.metadata
import sys

import torch
import vllm

assert sys.version_info[:3] == (3, 12, 13), sys.version
assert importlib.metadata.version("vllm") == "0.31.0+cu129"
assert vllm.__version__ == "0.31.0", vllm.__version__
assert torch.__version__ == "2.13.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
print(
    sys.version.split()[0],
    importlib.metadata.version("vllm"),
    vllm.__version__,
    torch.__version__,
    torch.version.cuda,
)
PY
```

### New machine: reinstall the pinned environment

A virtual environment directory is not portable between machines. On a new
machine, install `uv >= 0.11`, use glibc `>= 2.28`, ensure the NVIDIA driver can
run CUDA 12.9 binaries, and recreate the environment from the committed freeze.
The freeze contains the exact official vLLM wheel URL and all 197 transitive
pins used on the validated host.

```bash
git switch codex/mori-server-only

export MORI_ENV="$PWD/.venv-vllm-0.31.0-cu129"
uv venv --python 3.12.13 "$MORI_ENV"
uv pip install --python "$MORI_ENV/bin/python" \
  -r examples/mori/requirements-vllm-0.31.0-cu129.txt

# Install this checkout and its test-only dependencies after the binary stack.
uv pip install --python "$MORI_ENV/bin/python" -e '.[dev]'
source "$MORI_ENV/bin/activate"
```

Do not replace the URL-pinned `vllm-0.31.0+cu129` wheel with an unqualified
`vllm==0.31.0`: MORI reaches private vLLM scheduler/offload interfaces, and the
startup gate intentionally rejects a different build. The wheel is x86_64 and
manylinux 2.28; another architecture needs a separately validated build and is
not equivalent to this environment.

Verify the reconstructed stack before running tests or reserving a GPU:

```bash
python - <<'PY'
import importlib.metadata
import sys

import torch
import vllm

expected = {
    "flashinfer-python": "0.7.0.post1",
    "tokenizers": "0.23.3",
    "transformers": "5.17.0",
    "triton": "3.7.1",
}
assert sys.version_info[:3] == (3, 12, 13), sys.version
assert importlib.metadata.version("vllm") == "0.31.0+cu129"
assert vllm.__version__ == "0.31.0", vllm.__version__
assert torch.__version__ == "2.13.0+cu129", torch.__version__
assert torch.version.cuda == "12.9", torch.version.cuda
for package, version in expected.items():
    assert importlib.metadata.version(package) == version, package
assert torch.cuda.is_available(), "CUDA is unavailable"
print(
    "validated",
    sys.version.split()[0],
    importlib.metadata.version("vllm"),
    vllm.__version__,
    torch.__version__,
)
PY
```

Run the CPU-only contracts before reserving a GPU:

```bash
python -m pytest -q tests/examples/test_mori_server_only.py
python -m pytest -q tests/agentbench/replay/test_agentx.py
python -m pytest -q tests/agentcache/core/test_vllm_api_adapter.py \
  -k 'completion_lifecycle_uses_only_explicit_optional_xarg or completion_middleware_delivers_explicit_terminal_after_success'
python -m pytest -q tests/scheduling/test_progress_ttl_runtime.py \
  -k 'overlapping_terminal_response'
```

## 2. Seal one fresh server run

Every process must use a new audit directory. Do not reuse a directory after a
server has started.

```bash
export AGENTINFER_ROOT="$PWD"
export MORI_RUN_ROOT="/tmp/agentinfer-mori-$(date +%s)"

python examples/mori/prepare_server.py \
  --agentinfer-root "$AGENTINFER_ROOT" \
  --output "$MORI_RUN_ROOT" \
  > "$MORI_RUN_ROOT.env"

source "$MORI_RUN_ROOT.env"
test -z "${KVC_MORI_PLAN+x}"
```

The command creates `server-preflight.json` and `launch-env.json`, then prints
the exact environment required by the server. `PYTHONPATH` contains only the
isolated MORI roots and this AgentInfer checkout.

## 3. Start without CPU-KV offload

Start with zero CPU-KV capacity first; this validates request identity,
admission, k=5 scheduling, and typed HBM eviction without introducing transfer
or host-memory variables.

```bash
export MODEL_PATH=/absolute/path/to/model
export SERVED_MODEL_NAME=your-model-name

python -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.60 \
  --async-scheduling \
  --enforce-eager \
  --block-size 16 \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256 \
  --enable-prompt-tokens-details \
  --shutdown-timeout 60 \
  --scheduler-cls mori_http_mp.MoriAsyncScheduler \
  --middleware agentinfer.agentcache.core.api_adapter.AgentCacheIdentityMiddleware \
  --middleware agentinfer.agentcache.core.api_adapter.AgentCacheLifecycleMiddleware
```

Choose `--gpu-memory-utilization`, context length, batching limits, and model
according to the target machine. Keep `--tensor-parallel-size 1`; this isolated
adapter has only been validated for one EngineCore worker and one KV group.

## 4. Send a direct smoke request

The server accepts a Program that did not exist when the server started. The
identity below is parsed from this request; it is not registered by a plan.

```bash
curl -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${SERVED_MODEL_NAME}\",
    \"prompt\": \"Write one short sentence.\",
    \"max_tokens\": 16,
    \"stream\": true,
    \"stream_options\": {\"include_usage\": true},
    \"vllm_xargs\": {
      \"agentic_context\": {
        \"program_id\": \"portable-smoke-program\",
        \"task_id\": \"portable-smoke-task\",
        \"agent_id\": \"lead\",
        \"request_id\": \"portable-smoke-request-1\"
      },
      \"agentinfer_expected_lifecycle\": \"terminal\"
    }
  }"
```

Use `continue` instead of `terminal` when the same Program is expected to issue
another request. AgentBench AgentX Replay on this branch adds both fields
automatically. For an external client, stable `program_id` values and unique
`request_id` values are required.

## 5. Enable native CPU-KV offload

First stop the zero-capacity server and prepare another fresh run. The example
below requests 32 GiB; lower it to fit the machine while leaving enough host
memory for the model runtime.

```bash
export MORI_RUN_ROOT="/tmp/agentinfer-mori-cpu-$(date +%s)"

python examples/mori/prepare_server.py \
  --agentinfer-root "$AGENTINFER_ROOT" \
  --output "$MORI_RUN_ROOT" \
  --cpu-kv-gib 32 \
  > "$MORI_RUN_ROOT.env"

source "$MORI_RUN_ROOT.env"
KV_TRANSFER_CONFIG="$(tr -d '\n' < "$KVC_MORI_KV_TRANSFER_CONFIG")"
```

Append this option to the same server command:

```bash
--kv-transfer-config "$KV_TRANSFER_CONFIG"
```

The optional adapter deliberately rejects tensor/data/pipeline/context
parallelism, replicated/canonical layouts, pageable tensors, prompt-only
offload, and a capacity that cannot be represented by native chunks. Its audit
is written to `$MORI_RUN_ROOT/offload-allocation.jsonl`.

## 6. Run AgentX Trace Replay

Generate the Replay workload on the client side using the normal AgentBench
workflow documented in `docs/en/how-to/run-trace-replay.md` or
`docs/zh/how-to/run-trace-replay.md`. Point its backend to the server above.

Keep this boundary when building a custom harness:

- The client can read the trace and plan arrivals.
- Do not export a plan path or plan contents into the server environment.
- Do not pre-register Program IDs with the server.
- Each request must carry its own `agentic_context`.
- A successful final response must carry `agentinfer_expected_lifecycle=terminal`;
  earlier rounds carry `continue`.

The server accepts previously unseen Program IDs. Therefore a trace subset,
task count, concurrency, or arrival order can change without regenerating any
server-side identity table.

## 7. Verify the final audit

Stop vLLM gracefully so `MoriAsyncScheduler.shutdown()` can write the summary:

```bash
jq '{server_state, checks: .joint.checks}' "$MORI_RUN_ROOT/joint-summary.json"

jq -e '
  .joint.passed == true and
  .server_state.server_plan_reads == 0 and
  .server_state.future_oracle_reads == 0 and
  .server_state.identity_source == "request_metadata" and
  .server_state.lifecycle_source == "api_completion_signal" and
  .server_state.registered_requests == .server_state.completed_requests
' "$MORI_RUN_ROOT/joint-summary.json"
```

Also retain these files with any benchmark result:

- `server-preflight.json` and `launch-env.json`;
- `joint-policy.jsonl` and `joint-summary.json`;
- `offload-allocation.jsonl` when CPU-KV is enabled;
- the vLLM server log, Replay config, request facts, model revision, GPU UUID,
  and exact launch command.

## Known limitations

- This is an isolated research adapter, not the final production AgentInfer
  MORI API.
- It is pinned to the vLLM `0.31.0+cu129` build and its private interfaces.
- Only one EngineCore worker, one full-attention KV group, and TP/PP/DP/DCP/PCP
  equal to one are supported.
- The branch contains the hooks and client lifecycle transport used by the real
  benchmark, but does not bundle its model, AgentX dataset, or long-running
  matrix harness.
- Full GPU validation should be repeated on each new CUDA, PyTorch, vLLM, model,
  or hardware combination; passing the CPU-only contracts is not a performance
  result.
