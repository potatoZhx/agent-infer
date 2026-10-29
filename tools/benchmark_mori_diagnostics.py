# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Measure default MORI request diagnostics against the explicitly disabled path.

Run from the repository root with ``python -m tools.benchmark_mori_diagnostics``.
This CPU microbenchmark measures real scheduler callbacks, not GPU throughput.
It alternates paired variants, records every sample, writes actual slow logs,
and checks fixed absolute overhead budgets rather than noisy speedup ratios.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

from agentinfer.scheduling.backend import BackendInfo, BackendPoolInfo, DpRankInfo
from agentinfer.scheduling.domain import ProgramRef
from agentinfer.scheduling.identity import AgentIdentity
from agentinfer.scheduling.lifecycle import ProgramLifecycle
from agentinfer.scheduling.mori import MORIConfig, build_mori_strategy
from agentinfer.scheduling.request_diagnostics import RequestDiagnostics
from agentinfer.scheduling.runtime import ProgramScheduler

BUDGET_NS = {
    "request_lifecycle": 10_000,
    "native_before_first_output": 1_000,
    "native_after_first_output": 1_000,
    "stream_output": 1_000,
    "schedule_cycle_16_requests": 20_000,
}


def _backend(capacity: int = 1_000_000_000) -> BackendPoolInfo:
    return BackendPoolInfo(
        (
            BackendInfo(
                "local",
                "embedded://local",
                True,
                (
                    DpRankInfo(
                        0,
                        True,
                        True,
                        total_hbm_kv_tokens=capacity,
                        total_dram_kv_tokens=capacity * 2,
                        hbm_block_size_tokens=16,
                        dram_chunk_size_tokens=16,
                    ),
                ),
            ),
        )
    )


def _runtime(enabled: bool):
    pool = _backend()
    components = build_mori_strategy(MORIConfig(slow_request_threshold_seconds=30.0 if enabled else 0.0))
    runtime = ProgramScheduler(components.strategy, components.initial_factors, pool, schedule_interval_seconds=5)
    return runtime, pool


def _timed(operation: Callable[[], None], count: int) -> dict[str, float]:
    cpu = time.thread_time_ns()
    wall = time.perf_counter_ns()
    operation()
    wall = time.perf_counter_ns() - wall
    cpu = time.thread_time_ns() - cpu
    return {"thread_cpu_ns_per_operation": cpu / count, "wall_ns_per_operation": wall / count}


class _CountLogs(logging.Handler):
    def __init__(self):
        super().__init__()
        self.rows = 0

    def emit(self, record):
        self.rows += 1


def _sample(name: str, enabled: bool, count: int) -> dict[str, float]:
    runtime, pool = _runtime(enabled)
    identities = [AgentIdentity(f"bench-{i}") for i in range(16)]
    keys = [f"request-{i}" for i in range(count)]
    if name == "request_lifecycle":

        def operation():
            for key in keys:
                if not runtime.on_request_arrival(key, identities[0], 100, pool, key):
                    raise RuntimeError("normal request unexpectedly queued")
                runtime.on_native_execution(key, running=True)
                runtime.on_stream_output(key, 1)
                runtime.on_request_completion(key, 101)
    else:
        request_count = 16 if name.startswith("schedule_cycle") else 1
        for i in range(request_count):
            if not runtime.on_request_arrival(f"active-{i}", identities[i], 100, pool, "native"):
                raise RuntimeError("callback fixture unexpectedly queued")
        if name != "native_before_first_output":
            runtime.on_stream_output("active-0", 1)
        if name.startswith("native_"):

            def operation():
                for _ in range(count):
                    runtime.on_native_execution("active-0", running=True)
        elif name == "stream_output":

            def operation():
                for tokens in range(2, count + 2):
                    runtime.on_stream_output("active-0", tokens)
        else:

            def operation():
                for _ in range(count):
                    # Force an actual periodic plan instead of timing its early return.
                    runtime._next_periodic_check_at_monotonic_s = 0.0
                    runtime.schedule_cycle(pool)

    counter = _CountLogs()
    diagnostics_logger = logging.getLogger("agentinfer.scheduling.request_diagnostics")
    diagnostics_logger.addHandler(counter)
    try:
        sample = _timed(operation, count)
    finally:
        diagnostics_logger.removeHandler(counter)
    if counter.rows:
        raise RuntimeError("normal lifecycle/callback path emitted diagnostic logs")
    sample["log_rows_during_timed_fast_path"] = counter.rows
    for key in tuple(runtime._request_bindings):
        runtime.on_request_completion(key, 101)
    for identity in identities:
        runtime.on_response_completion(identity.program_id, ProgramLifecycle.TERMINAL)
    tracker = runtime._request_diagnostics
    if tracker is not None and tracker.active_count:
        raise RuntimeError("completed callback fixture leaked diagnostic request state")
    sample["active_diagnostics_after_cleanup"] = 0
    return sample


def _paired(name: str, count: int, repeats: int) -> dict:
    # Warm imports, dispatch paths, and allocation caches outside recorded samples.
    for enabled in (False, True):
        _sample(name, enabled, min(100, count))
    samples = []
    for repeat in range(repeats):
        variants = (False, True) if repeat % 2 == 0 else (True, False)
        pair = {"repeat": repeat}
        for enabled in variants:
            pair["enabled" if enabled else "disabled"] = _sample(name, enabled, count)
        pair["added_thread_cpu_ns_per_operation"] = (
            pair["enabled"]["thread_cpu_ns_per_operation"] - pair["disabled"]["thread_cpu_ns_per_operation"]
        )
        samples.append(pair)
    added = statistics.median(s["added_thread_cpu_ns_per_operation"] for s in samples)
    return {
        "operation_count_per_sample": count,
        "disabled_median_thread_cpu_ns_per_operation": statistics.median(
            s["disabled"]["thread_cpu_ns_per_operation"] for s in samples
        ),
        "enabled_median_thread_cpu_ns_per_operation": statistics.median(
            s["enabled"]["thread_cpu_ns_per_operation"] for s in samples
        ),
        "median_paired_added_thread_cpu_ns_per_operation": added,
        "added_cpu_budget_ns_per_operation": BUDGET_NS[name],
        "passed": added <= BUDGET_NS[name],
        "samples": samples,
    }


def _slow_burst(path: Path, pending: int) -> dict:
    logger = logging.getLogger("mori-diagnostics-benchmark")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    raw_path = path.with_suffix(".raw.tmp")
    handler = logging.FileHandler(raw_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    context_calls = 0

    def context(program: ProgramRef, now: float) -> dict:
        nonlocal context_calls
        context_calls += 1
        return {"program_id": program.program_id, "observed_at": now, "pending_requests": pending}

    tracker = RequestDiagnostics(context_provider=context, logger=logger)
    owners = [ProgramRef(f"pending-{i}", 0) for i in range(pending)]
    keys = [f"slow-{i}" for i in range(pending)]
    for key, owner in zip(keys, owners, strict=True):
        tracker.arrive(key, owner, 0.0, prompt_tokens=1024)
    poll_samples = []
    previous_rows = 0
    for poll in range(32):
        now = 31.0 + poll * 5.0
        sample = _timed(lambda now=now: tracker.poll(now), 1)
        handler.flush()
        rows = len(raw_path.read_text().splitlines())
        sample["new_log_rows"] = rows - previous_rows
        previous_rows = rows
        if sample["new_log_rows"] > 8:
            raise RuntimeError("slow poll exceeded the configured eight-event budget")
        poll_samples.append(sample)
    # Cancellation/release must remove pending state without retaining history.
    cleanup_wall = time.perf_counter_ns()
    cleanup_cpu = time.thread_time_ns()
    for owner in owners:
        tracker.release(owner, 200.0)
    cleanup_cpu = time.thread_time_ns() - cleanup_cpu
    cleanup_wall = time.perf_counter_ns() - cleanup_wall
    handler.flush()
    logger.removeHandler(handler)
    handler.close()
    if tracker.active_count or tracker._program_requests:
        raise RuntimeError("releasing pending programs leaked diagnostic state")
    rows = len(raw_path.read_text().splitlines())
    if previous_rows == 0:
        raise RuntimeError("slow-path benchmark did not write any log records")
    poll_median = statistics.median(s["thread_cpu_ns_per_operation"] for s in poll_samples)
    raw_bytes = raw_path.stat().st_size
    lines = raw_path.read_text().splitlines()
    sample_lines = lines[:16] + lines[-4:] if len(lines) > 20 else lines
    path.write_text("\n".join(sample_lines) + "\n")
    raw_path.unlink()
    return {
        "context_scope": "small synthetic context provider; actual production context measured separately",
        "pending_requests": pending,
        "polls": len(poll_samples),
        "max_events_per_poll": 8,
        "poll_median_thread_cpu_us": poll_median / 1000,
        "poll_max_thread_cpu_us": max(s["thread_cpu_ns_per_operation"] for s in poll_samples) / 1000,
        "poll_cpu_budget_us": 5000,
        "poll_log_rows": previous_rows,
        "all_log_rows_including_cleanup": rows,
        "log_bytes": raw_bytes,
        "log_sample_artifact": str(path),
        "log_sample_selection": "first 16 and last 4 rows; all rows if at most 20",
        "log_sample_rows": len(sample_lines),
        "terminal_summaries_share_poll_cap": False,
        "terminal_summary_rows": rows - previous_rows,
        "terminal_burst_thread_cpu_ms": cleanup_cpu / 1_000_000,
        "terminal_burst_wall_ms": cleanup_wall / 1_000_000,
        "terminal_cpu_budget_us_per_request": 200,
        "context_provider_calls": context_calls,
        "cleanup_thread_cpu_ns_per_request": cleanup_cpu / pending,
        "active_diagnostics_after_cleanup": tracker.active_count,
        "passed": poll_median <= 5_000_000 and cleanup_cpu / pending <= 200_000,
        "samples": poll_samples,
    }


def _production_slow_sample(enabled: bool, owners: int, path: Path) -> dict:
    pool = _backend(4096)
    components = build_mori_strategy(MORIConfig(slow_request_threshold_seconds=30.0 if enabled else 0.0))
    runtime = ProgramScheduler(components.strategy, components.initial_factors, pool, schedule_interval_seconds=5)
    logger = logging.getLogger(f"mori-production-slow-benchmark-{owners}-{enabled}")
    logger.propagate = False
    logger.disabled = False
    logger.setLevel(logging.WARNING)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    tracker = runtime._request_diagnostics
    if tracker is not None:
        tracker._logger = logger
    now = [0.0]
    with patch("agentinfer.scheduling.runtime.time.monotonic", lambda: now[0]):
        for i in range(owners):
            admitted = runtime.on_request_arrival(f"overdue-{i}", AgentIdentity(f"owner-{i}"), 1024, pool, "native")
            if admitted:
                runtime.on_native_execution(f"overdue-{i}", running=True)
        retained = runtime.retained_request_count
        now[0] = 31.0
        sample = _timed(lambda: runtime.schedule_cycle(pool), 1)
    handler.flush()
    lines = path.read_text().splitlines()
    sample["log_rows"] = len(lines)
    sample["log_bytes"] = path.stat().st_size
    sample["retained_requests_before_plan"] = retained
    sample["active_diagnostics_after_plan"] = tracker.active_count if tracker is not None else 0
    if len(lines) > 8 or (enabled and len(lines) == 0) or (not enabled and lines):
        raise RuntimeError("production overdue plan violated expected slow-log budget")
    # Retain bounded samples, while cleanup must drop all tracker indexes.
    logger.disabled = True
    if tracker is not None:
        for program in tuple(tracker._program_requests):
            tracker.release(program, 31.0)
        if tracker.active_count or tracker._program_requests:
            raise RuntimeError("production overdue fixture leaked diagnostic state")
    sample["active_diagnostics_after_cleanup"] = 0
    logger.removeHandler(handler)
    handler.close()
    logger.disabled = False
    return sample


def _production_slow_paired(owners: int, repeats: int, directory: Path) -> dict:
    samples = []
    for repeat in range(repeats):
        pair = {"repeat": repeat}
        variants = (False, True) if repeat % 2 == 0 else (True, False)
        for enabled in variants:
            variant = "enabled" if enabled else "disabled"
            path = directory / f"production-slow-{owners}-{variant}-{repeat}.log"
            pair[variant] = _production_slow_sample(enabled, owners, path)
        pair["added_thread_cpu_ns_per_plan"] = (
            pair["enabled"]["thread_cpu_ns_per_operation"] - pair["disabled"]["thread_cpu_ns_per_operation"]
        )
        samples.append(pair)
    added = statistics.median(s["added_thread_cpu_ns_per_plan"] for s in samples)
    return {
        "owners": owners,
        "scope": "real MORI plan and production diagnostic context; all requests overdue before first output",
        "disabled_median_thread_cpu_ms_per_plan": statistics.median(
            s["disabled"]["thread_cpu_ns_per_operation"] for s in samples
        )
        / 1_000_000,
        "enabled_median_thread_cpu_ms_per_plan": statistics.median(
            s["enabled"]["thread_cpu_ns_per_operation"] for s in samples
        )
        / 1_000_000,
        "median_paired_added_thread_cpu_ms_per_plan": added / 1_000_000,
        "added_cpu_budget_ms_per_plan": 5,
        "passed": added <= 5_000_000,
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--lifecycle-count", type=int, default=2000)
    parser.add_argument("--callback-count", type=int, default=100000)
    parser.add_argument("--cycle-count", type=int, default=500)
    parser.add_argument("--pending-count", type=int, default=4096)
    args = parser.parse_args()
    if min(args.repeats, args.lifecycle_count, args.callback_count, args.cycle_count, args.pending_count) <= 0:
        parser.error("all counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        Path(__file__),
        Path("agentinfer/scheduling/request_diagnostics.py"),
        Path("agentinfer/scheduling/runtime.py"),
        Path("agentinfer/scheduling/mori/strategy.py"),
        Path("agentinfer/scheduling/mori/config.py"),
        Path("agentinfer/scheduling/mori/diagnostics.py"),
    ]
    results = {
        "scope": "CPU callback overhead only; no GPU/end-to-end throughput benchmark was run.",
        "clock": "time.thread_time_ns (per-thread CPU); perf_counter_ns also recorded",
        "command": " ".join([sys.executable, "-m", "tools.benchmark_mori_diagnostics", *sys.argv[1:]]),
        "python": sys.version,
        "platform": platform.platform(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "budget_rationale": "Absolute incremental budget: 10 us per request lifecycle, 1 us per hot callback, 20 us per real plan; slow diagnostics at most eight rows/poll and median CPU below 5 ms per five-second poll; terminal summaries are outside that poll cap and budgeted at 200 us CPU per request. These are engineering acceptance budgets, not GPU performance claims.",
        "fast_paths": {},
    }
    for name in BUDGET_NS:
        count = (
            args.lifecycle_count
            if name == "request_lifecycle"
            else (args.cycle_count if name.startswith("schedule_cycle") else args.callback_count)
        )
        results["fast_paths"][name] = _paired(name, count, args.repeats)
        print(name, json.dumps({k: v for k, v in results["fast_paths"][name].items() if k != "samples"}), flush=True)
    results["slow_burst"] = _slow_burst(args.output_dir / "slow-burst.log", args.pending_count)
    results["slow_production_planning"] = {
        str(owners): _production_slow_paired(owners, args.repeats, args.output_dir) for owners in (32, 256)
    }
    results["source_sha256_after"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    results["source_unchanged"] = results["source_sha256"] == results["source_sha256_after"]
    results["passed"] = (
        results["source_unchanged"]
        and all(value["passed"] for value in results["slow_production_planning"].values())
        and all(v["passed"] for v in results["fast_paths"].values())
        and results["slow_burst"]["passed"]
    )
    (args.output_dir / "overhead.json").write_text(json.dumps(results, indent=2) + "\n")
    print("passed=" + str(results["passed"]), flush=True)
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
