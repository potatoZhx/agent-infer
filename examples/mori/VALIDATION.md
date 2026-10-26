# MORI isolated-branch validation

Validation date: 2026-10-26 (Asia/Shanghai).

## Frozen runtime

The server runtime used for this validation reported:

| Component | Observed value |
| --- | --- |
| Python | `3.12.13` |
| vLLM distribution / module | `0.31.0+cu129` / `0.31.0` |
| PyTorch | `2.13.0+cu129` |
| PyTorch CUDA build | `12.9` |
| NVIDIA driver / GPU | `570.133.20` / `NVIDIA L20X` |

The committed environment freeze has 197 lines and SHA-256
`c062681139cdbeeddb036d0106e72848fd63c1ad55eb946495d88e1b4651afa9`.

## Static checks

The following command completed successfully for every changed or added file:

```bash
pre-commit run --files $(git diff --name-only --diff-filter=ACMR; git ls-files --others --exclude-standard)
```

Results: Ruff check passed, Ruff format passed, typos passed,
markdownlint-cli2 passed, and suggestion passed.

## Focused contracts under vLLM 0.31.0

The shared vLLM environment intentionally had no pytest installation. The
system pytest package path was appended, not prepended, to the interpreter
path; the printed module paths confirmed that vLLM and PyTorch still loaded
from the frozen server environment. The selected command covered:

- the isolated MORI server package and plan-free preflight;
- AgentX request metadata and explicit lifecycle transport;
- completion middleware delivery; and
- overlapping terminal/continue runtime release.

Result after the exact-build gate was added: 50 passed, 14 warnings in 18.90s.
The warnings were PyTorch deprecations for `torch.jit.script_method`.

## Startup source gate

A new preflight was generated with `prepare_server.py`, then `_source_gate`
was invoked in a fresh Python 3.12.13 process from the frozen vLLM environment.
Its complete result was:

```json
{"bootstrap":"1","candidate_source_files_checked":12,"forbidden_plan_environment":[],"identity_source":"request_metadata","lifecycle_source":"api_completion_signal","plan_available_to_server":false,"status":"passed","vllm":"0.31.0+cu129"}
```

## Not run in this branch

No additional full-model GPU Trace Replay was started from this worktree. The
validation here establishes portability, causal server input boundaries, and
the exact vLLM compatibility gate; it does not create a new performance claim.
Before reporting throughput, JCT, cache-hit, or transfer results on another
machine, run the matched workload matrix and retain all artifacts listed in the
README.
