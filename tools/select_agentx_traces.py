#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Select AgentX (cc-traces-weka) sessions into a replay-ready JSONL subset.

Selection modes
---------------
duration: rank sessions by wall-clock span (first request start to last request
          end on the merged busy timeline), ascending.
tokens:   rank sessions by total input+output tokens, ascending.

Filters (applied before ranking)
--------------------------------
--max-ctx:   keep only sessions whose every leaf request has ``in + out`` at or
             below this many tokens (bounds the replay ``--max-model-len``).
--max-gap:   keep only sessions whose maximum idle interval between merged busy
             intervals is at or below this many seconds.

The source dataset is the public SemiAnalysis AgentX hash-block snapshot
``semianalysisai/cc-traces-weka-with-subagents-052726`` (JSONL, one session per
line, ``block_size`` 64, ``hash_id_scope`` local). Token counts in the dataset
are proxy-tokenizer counts; AgentX Replay re-renders every prompt from the
backend model tokenizer, so the selection stays backend-agnostic.

Example
-------
python tools/select_agentx_traces.py \
    --input weka-subagents-traces.jsonl \
    --output traces-duration64-ctx120k-gap10min.jsonl \
    --mode duration --count 64 --max-ctx 120000 --max-gap 600
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _leaves(requests: list[dict]) -> list[dict]:
    out: list[dict] = []
    for request in requests:
        if request.get("type") == "subagent":
            out.extend(_leaves(request["requests"]))
        else:
            out.append(request)
    return out


def _merged_timeline(intervals: list[tuple[float, float]]) -> tuple[float, float]:
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    gaps = [merged[k + 1][0] - merged[k][1] for k in range(len(merged) - 1)]
    return merged[-1][1] - merged[0][0], (max(gaps) if gaps else 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="source weka traces JSONL")
    parser.add_argument("--output", type=Path, required=True, help="selected subset JSONL (must not exist)")
    parser.add_argument("--count", type=int, required=True, help="number of sessions to keep")
    parser.add_argument("--mode", choices=("duration", "tokens"), default="duration")
    parser.add_argument("--max-ctx", type=int, default=120_000, help="max per-request in+out tokens (0 disables)")
    parser.add_argument("--max-gap", type=float, default=600.0, help="max idle gap seconds (0 disables)")
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"output exists: {args.output}")

    candidates = []
    total = 0
    with args.input.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream):
            session = json.loads(line)
            requests = _leaves(session["requests"])
            duration, max_gap = _merged_timeline([(r["t"], r["t"] + r["api_time"]) for r in requests])
            max_ctx = max(r["in"] + r["out"] for r in requests)
            tokens = sum(r["in"] + r["out"] for r in requests)
            total += 1
            if args.max_ctx and max_ctx > args.max_ctx:
                continue
            if args.max_gap and max_gap > args.max_gap:
                continue
            rank_key = duration if args.mode == "duration" else tokens
            candidates.append((rank_key, line_number, line, max_ctx, max_gap, len(requests)))

    candidates.sort(key=lambda item: (item[0], item[1]))
    selected = candidates[: args.count]

    with args.output.open("w", encoding="utf-8") as stream:
        for _, _, line, *_ in selected:
            stream.write(line)

    print(
        f"{total} sessions -> {len(candidates)} pass filters "
        f"(mode={args.mode}, max_ctx={args.max_ctx}, max_gap={args.max_gap}) "
        f"-> kept {len(selected)}"
    )
    for rank, (_, line_number, _, max_ctx, max_gap, n_requests) in enumerate(selected):
        print(f"  #{rank:3d} line={line_number:4d} reqs={n_requests:3d} max_ctx={max_ctx:7d} max_gap={max_gap:6.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
