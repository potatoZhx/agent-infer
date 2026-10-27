# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Compare formal and isolated MORI CPU replacement on a bounded real trace.

The D2H submissions model CPU cache fill and eviction batch sizes, assuming
each submitted block inserts a distinct resident chunk without failed stores
or independent removals. This is not a reconstruction of native victim events.
The benchmark keeps the recorded cache
capacity, executes a bounded prefix, and compares three implementations:

* the isolated MORI ``MORICPUTypedPolicy`` loaded from its frozen source;
* the formal policy through its original non-mutating proposal path; and
* the formal policy through the reservation path used in production.

Every implementation receives identical ownership, tier-change, recency, and
insertion events. A run is valid only when every victim digest is identical.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

from agentinfer.agentcache.mori.runtime_state import MORI_RUNTIME_STATE
from agentinfer.agentcache.mori_cpu_policy import MoriCPUCachePolicy
from agentinfer.scheduling import CacheClass, ProgramRef

VARIANTS = ("isolated_scan", "formal_proposal", "formal_reservation")
CLASSES = (CacheClass.INACTIVE, CacheClass.BUSY, CacheClass.IDLE)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_isolated(path: Path) -> ModuleType:
    name = "_agentinfer_isolated_mori_benchmark"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load isolated MORI source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _eviction_batches(path: Path, capacity: int) -> tuple[list[int], dict[str, int]]:
    resident = 0
    stores = 0
    stored_chunks = 0
    evictions: list[int] = []
    with path.open() as handle:
        for raw in handle:
            event = json.loads(raw)
            if event.get("event") != "transfer_submit" or event.get("direction") != "D2H":
                continue
            count = event["transfer_features"]["block_count"]
            stores += 1
            stored_chunks += count
            evicted = max(0, resident + count - capacity)
            resident = resident - evicted + count
            if evicted:
                evictions.append(evicted)
    if resident != capacity:
        raise ValueError(f"trace ended with {resident} CPU chunks, expected capacity {capacity}")
    return evictions, {
        "d2h_store_calls": stores,
        "stored_chunks": stored_chunks,
        "eviction_calls": len(evictions),
        "evicted_chunks": sum(evictions),
        "final_resident_chunks": resident,
    }


def _chunk(chunk_id: int) -> ChunkStatus:
    chunk = ChunkStatus(chunk_id)
    chunk.ref_cnt = 0
    return chunk


def _key(key_id: int) -> bytes:
    return key_id.to_bytes(8, "big")


def _owner_ids(key_id: int, program_count: int) -> tuple[int, ...]:
    primary = key_id % program_count
    if key_id % 11:
        return (primary,)
    return (primary, (key_id * 17 + 13) % program_count)


class _IsolatedState:
    def subscribe(self, callback: Any) -> None:
        self.callback = callback


class _IsolatedStrategy:
    def __init__(self, module: ModuleType, program_count: int) -> None:
        self._module = module
        tiers = (module.MORITier.WAITING, module.MORITier.GPU, module.MORITier.CPU)
        self.tiers = {f"p{i}": tiers[i % len(tiers)] for i in range(program_count)}
        self.tier_epoch = 0

    def tier_for_program(self, program_id: str) -> Any:
        return self.tiers.get(program_id, self._module.MORITier.WAITING)

    def rotate(self, program_id: str) -> None:
        tiers = (
            self._module.MORITier.WAITING,
            self._module.MORITier.GPU,
            self._module.MORITier.CPU,
        )
        current = self.tiers[program_id]
        self.tiers[program_id] = tiers[(tiers.index(current) + 1) % len(tiers)]
        self.tier_epoch += 1


class _FormalProvider:
    def __init__(self, programs: tuple[ProgramRef, ...]) -> None:
        self.classes = {program: CLASSES[i % len(CLASSES)] for i, program in enumerate(programs)}
        self.observers: list[Any] = []
        self._epoch = 0

    @property
    def cache_class_epoch(self) -> int:
        return self._epoch

    def cache_class_for_program(self, program: ProgramRef) -> CacheClass:
        return self.classes.get(program, CacheClass.INACTIVE)

    def subscribe_cache_class_observer(self, observer: Any) -> None:
        self.observers.append(observer)

    def unsubscribe_cache_class_observer(self, observer: Any) -> None:
        self.observers.remove(observer)

    def rotate(self, program: ProgramRef) -> None:
        previous = self.classes[program]
        current = CLASSES[(CLASSES.index(previous) + 1) % len(CLASSES)]
        self.classes[program] = current
        self._epoch += 1
        for observer in tuple(self.observers):
            observer.on_cache_class_changed(program, previous, current)


def _old_formal_evict(
    policy: MoriCPUCachePolicy,
    count: int,
) -> list[tuple[bytes, ChunkStatus]] | None:
    """Execute the formal policy's pre-reservation proposal/restore path."""
    proposed = policy.index.propose_exact(count, set())
    if proposed is None:
        return None
    result: list[tuple[bytes, ChunkStatus]] = []
    for key in proposed:
        chunk = policy.chunks.get(key)
        if chunk is None or chunk.ref_cnt != 0:
            if chunk is not None:
                policy.index.mark_ready(key, chunk.ref_cnt == 0)
            return None
        result.append((key, chunk))
    for key, _ in result:
        policy.remove(key)
    return result


def _victim_digest(victims: list[bytes]) -> str:
    digest = hashlib.sha256()
    for key in victims:
        digest.update(key)
    return digest.hexdigest()


def _run_isolated(
    module: ModuleType,
    *,
    capacity: int,
    batches: list[int],
    program_count: int,
    transition_interval: int,
    touch_count: int,
) -> dict[str, Any]:
    policy = LRUCachePolicy(capacity)
    for key_id in range(capacity):
        policy.insert(_key(key_id), _chunk(key_id))
    manager = type("_Manager", (), {"_policy": policy})()
    strategy = _IsolatedStrategy(module, program_count)
    typed = module.MORICPUTypedPolicy(manager, None, _IsolatedState(), strategy)
    for key_id in range(capacity):
        typed.key_programs[_key(key_id)].update(f"p{i}" for i in _owner_ids(key_id, program_count))

    next_key = capacity
    recent = [_key(key_id) for key_id in range(capacity - touch_count, capacity)]
    victims: list[bytes] = []
    started = time.thread_time_ns()
    try:
        for operation, count in enumerate(batches):
            if operation % transition_interval == 0:
                strategy.rotate(f"p{(operation * 17) % program_count}")
            policy._update_recency(recent)
            selected = typed.evict(count, set())
            if selected is None:
                raise RuntimeError("isolated policy returned an incomplete victim set")
            victims.extend(key for key, _ in selected)
            recent = []
            for _ in range(count):
                key = _key(next_key)
                policy.insert(key, _chunk(next_key))
                typed.key_programs[key].update(f"p{i}" for i in _owner_ids(next_key, program_count))
                recent.append(key)
                next_key += 1
            recent = recent[-touch_count:]
    finally:
        elapsed_ns = time.thread_time_ns() - started
        policy.evict = typed._original_evict
    return {
        "thread_cpu_seconds": elapsed_ns / 1_000_000_000,
        "victim_digest": _victim_digest(victims),
        "victim_count": len(victims),
        "stats": dict(typed.stats),
    }


def _run_formal(
    mode: str,
    *,
    capacity: int,
    batches: list[int],
    program_count: int,
    transition_interval: int,
    touch_count: int,
) -> dict[str, Any]:
    MORI_RUNTIME_STATE.reset()
    programs = tuple(ProgramRef(f"p{i}", 0) for i in range(program_count))
    provider = _FormalProvider(programs)
    MORI_RUNTIME_STATE.bind_provider(provider)
    policy = MoriCPUCachePolicy(capacity)
    for key_id in range(capacity):
        key = _key(key_id)
        policy._pending_owners[key] = {programs[i] for i in _owner_ids(key_id, program_count)}
        policy.insert(key, _chunk(key_id))

    next_key = capacity
    recent = [_key(key_id) for key_id in range(capacity - touch_count, capacity)]
    victims: list[bytes] = []
    started = time.thread_time_ns()
    try:
        for operation, count in enumerate(batches):
            if operation % transition_interval == 0:
                provider.rotate(programs[(operation * 17) % program_count])
            policy._update_recency(recent)
            selected = _old_formal_evict(policy, count) if mode == "formal_proposal" else policy.evict(count, set())
            if selected is None:
                raise RuntimeError(f"{mode} returned an incomplete victim set")
            victims.extend(key for key, _ in selected)
            recent = []
            for _ in range(count):
                key = _key(next_key)
                policy._pending_owners[key] = {programs[i] for i in _owner_ids(next_key, program_count)}
                policy.insert(key, _chunk(next_key))
                recent.append(key)
                next_key += 1
            recent = recent[-touch_count:]
        elapsed_ns = time.thread_time_ns() - started
        result = {
            "thread_cpu_seconds": elapsed_ns / 1_000_000_000,
            "victim_digest": _victim_digest(victims),
            "victim_count": len(victims),
            "stats": dict(policy.index.stats),
        }
    finally:
        policy.close()
        MORI_RUNTIME_STATE.unbind_provider(provider)
        MORI_RUNTIME_STATE.reset()
    return result


def _summary(trials: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = {variant: [] for variant in VARIANTS}
    for trial in trials:
        grouped[trial["variant"]].append(trial["thread_cpu_seconds"])
    medians = {variant: statistics.median(values) for variant, values in grouped.items()}
    return {
        "median_thread_cpu_seconds": medians,
        "formal_reservation_speedup_over_isolated": (medians["isolated_scan"] / medians["formal_reservation"]),
        "reservation_speedup_over_formal_proposal": (medians["formal_proposal"] / medians["formal_reservation"]),
        "formal_reservation_reduction_vs_isolated_percent": (
            100 * (1 - medians["formal_reservation"] / medians["isolated_scan"])
        ),
        "reservation_reduction_vs_formal_proposal_percent": (
            100 * (1 - medians["formal_reservation"] / medians["formal_proposal"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--isolated-source", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=10_922)
    parser.add_argument("--operations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--programs", type=int, default=192)
    parser.add_argument("--transition-interval", type=int, default=8)
    parser.add_argument("--touch-count", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        min(
            args.capacity,
            args.operations,
            args.repeats,
            args.programs,
            args.transition_interval,
            args.touch_count,
        )
        <= 0
    ):
        raise ValueError("all numeric arguments must be positive")

    batches, trace_summary = _eviction_batches(args.trace, args.capacity)
    if args.operations > len(batches):
        raise ValueError(f"requested {args.operations} operations, trace contains {len(batches)}")
    selected_batches = batches[: args.operations]
    isolated = _load_isolated(args.isolated_source)
    runners = {
        "isolated_scan": lambda: _run_isolated(
            isolated,
            capacity=args.capacity,
            batches=selected_batches,
            program_count=args.programs,
            transition_interval=args.transition_interval,
            touch_count=args.touch_count,
        ),
        "formal_proposal": lambda: _run_formal(
            "formal_proposal",
            capacity=args.capacity,
            batches=selected_batches,
            program_count=args.programs,
            transition_interval=args.transition_interval,
            touch_count=args.touch_count,
        ),
        "formal_reservation": lambda: _run_formal(
            "formal_reservation",
            capacity=args.capacity,
            batches=selected_batches,
            program_count=args.programs,
            transition_interval=args.transition_interval,
            touch_count=args.touch_count,
        ),
    }

    trials: list[dict[str, Any]] = []
    variant_stats: dict[str, dict[str, int]] = {}
    expected_digest: str | None = None
    for repeat in range(args.repeats):
        order = VARIANTS[repeat % len(VARIANTS) :] + VARIANTS[: repeat % len(VARIANTS)]
        for variant in order:
            result = runners[variant]()
            stats = result.pop("stats")
            if variant in variant_stats and stats != variant_stats[variant]:
                raise RuntimeError(f"operation counters changed across {variant} repeats")
            variant_stats.setdefault(variant, stats)
            result.update({"repeat": repeat + 1, "variant": variant})
            digest = result["victim_digest"]
            if expected_digest is None:
                expected_digest = digest
            elif digest != expected_digest:
                raise RuntimeError(f"victim mismatch for {variant} repeat {repeat + 1}")
            trials.append(result)

    report = {
        "schema": "agentinfer_mori_typed_lru_benchmark_v1",
        "valid": True,
        "comparison_scope": "CPU typed-LRU maintenance and eviction; excludes model execution and transfer time",
        "victim_sequences_identical": True,
        "host": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
        "inputs": {
            "trace": str(args.trace.resolve()),
            "trace_sha256": _digest(args.trace),
            "isolated_source": str(args.isolated_source.resolve()),
            "isolated_source_sha256": _digest(args.isolated_source),
            "capacity_chunks": args.capacity,
            "operations": args.operations,
            "operation_chunks": sum(selected_batches),
            "repeats": args.repeats,
            "programs": args.programs,
            "shared_owner_every_n_keys": 11,
            "transition_interval": args.transition_interval,
            "touch_count": args.touch_count,
        },
        "full_trace_reconstruction": trace_summary,
        "trial_order_rotates": True,
        "trials": trials,
        "variant_operation_counters": variant_stats,
        "summary": _summary(trials),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
