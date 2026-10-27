# AgentX replay trace subsets

Gzipped AgentX session subsets for `vllm bench serve --agentinfer replay --trace-type agentX`.
Gunzip before use (or keep a decompressed copy beside your results):

```bash
gunzip -k agentinfer/agentbench/data/traces-duration64-ctx120k-gap10min.jsonl.gz
vllm bench serve --agentinfer replay --trace-type agentX \
  --trace-path agentinfer/agentbench/data/traces-duration64-ctx120k-gap10min.jsonl ...
```

## Provenance

Source dataset: [`semianalysisai/cc-traces-weka-with-subagents-052726`](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-with-subagents-052726)
(`traces.jsonl`, 472 sessions, `block_size` 64, `hash_id_scope` local; proxy-tokenizer counts).
Every subset below is filtered to sessions whose largest single-request context is at most
120,000 tokens and whose maximum idle gap on the merged busy timeline is at most 600 seconds,
then ranked by wall-clock duration ascending. The subsets are deterministic selections from
that file; regenerate or vary them with `tools/select_agentx_traces.py`.

| File | Selection | Sessions | Requests | Total input tokens | Max request ctx |
| --- | --- | --- | --- | --- | --- |
| `traces-duration32-ctx120k-gap10min.jsonl.gz` | 32 shortest by duration | 32 | 938 | 67.5M | 111,805 |
| `traces-duration64-ctx120k-gap10min.jsonl.gz` | 64 shortest by duration (filter pool: 72) | 64 | 2,151 | 159.0M | 118,047 |

Set the backend `--max-model-len` to at least the subset's max request context
(plus a small margin) — e.g. 114,688 for the 32-session file and 118,784 for the
64-session file. The Qwen3-Coder-30B native window (262,144) covers all subsets.

## Regeneration

```bash
python tools/select_agentx_traces.py \
  --input weka-subagents-traces.jsonl \
  --output traces-duration64-ctx120k-gap10min.jsonl \
  --mode duration --count 64 --max-ctx 120000 --max-gap 600
```

Tie-breaks use the source line number, so repeated runs are reproducible.
