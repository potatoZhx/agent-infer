"""Causal MORI policy and typed two-tier eviction for the isolated replay.

This module is an independent implementation of the public MORI design.  It
uses only committed scheduler events and current cache state; the replay DAG's
future-return advice is deliberately ignored.
"""

from __future__ import annotations

import heapq
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from enum import Enum
from types import MethodType
from typing import Any

from agentinfer.scheduling.admission_outcome import AdmissionDisposition, AdmissionOutcome
from agentinfer.scheduling.domain import ProgramRef, ProgramState, ProgramStatus, ProgramView
from agentinfer.scheduling.events import SchedulingEvent, StrategyDiagnostic
from agentinfer.scheduling.factors import StrategyFactors
from agentinfer.scheduling.snapshot import SchedulingSnapshot
from agentinfer.scheduling.strategy import SchedulingStrategy
from agentinfer.scheduling.transitions import TransitionController, TransitionResult


class MORITier(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    WAITING = "waiting"


@dataclass(frozen=True)
class MORIConfig:
    """Public-paper parameters plus capacities supplied by the host adapter."""

    window_size: int = 5
    tick_seconds: float = 5.0
    cpu_capacity_tokens: int = 0
    decode_buffer_tokens: int = 256

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("MORI window_size must be positive")
        if not math.isfinite(self.tick_seconds) or self.tick_seconds <= 0:
            raise ValueError("MORI tick_seconds must be finite and positive")
        if self.cpu_capacity_tokens < 0 or self.decode_buffer_tokens < 0:
            raise ValueError("MORI capacities must be non-negative")


@dataclass(frozen=True)
class MORIGlobalFactors:
    schedule_cycles: int = 0


@dataclass(frozen=True)
class MORIProgramFactors:
    """Causal recent-cycle history and scheduler-owned logical placement."""

    tier: MORITier = MORITier.WAITING
    cycles: tuple[tuple[float, float], ...] = ()
    pending_reasoning_seconds: float = 0.0
    acting_started_at_monotonic_s: float | None = None
    reasoning_started_at_monotonic_s: float | None = None
    pending_tier: MORITier | None = None
    last_idleness: float = 0.0


MORIFactors = StrategyFactors[MORIGlobalFactors, MORIProgramFactors]


class MORIStrategy(SchedulingStrategy[MORIGlobalFactors, MORIProgramFactors]):
    """Sticky relative-idleness placement over GPU, CPU and Waiting tiers."""

    def __init__(self, config: MORIConfig) -> None:
        self.config = config
        self.counters: Counter[str] = Counter()
        self.decision_cpu_seconds = 0.0
        self._bound_factors: MORIFactors | None = None
        self._tiers_by_program: dict[str, MORITier] = {}
        self.tier_epoch = 0

    def bind_factors(self, factors: MORIFactors) -> None:
        self._bound_factors = factors
        self._tiers_by_program = {ref.program_id: sidecar.tier for ref, sidecar in factors.program_factors}
        self.tier_epoch += 1

    def tier_for_program(self, program_id: str) -> MORITier:
        cached = self._tiers_by_program.get(program_id)
        if cached is not None:
            return cached
        factors = self._bound_factors
        if factors is not None:
            for ref, sidecar in reversed(factors.program_factors):
                if ref.program_id == program_id:
                    self._tiers_by_program[program_id] = sidecar.tier
                    return sidecar.tier
        return MORITier.WAITING

    def idleness_for_program(self, ref: ProgramRef, now: float) -> float:
        factors = self._bound_factors
        if factors is None:
            return 0.0
        return self._idleness(self._factor(factors, ref), now)

    def handle_admission(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: MORIFactors,
        transitions: TransitionController,
        candidate: ProgramRef,
    ) -> AdmissionOutcome:
        started = time.thread_time()
        try:
            view = snapshot.program(candidate)
            if view is None:
                raise ValueError("MORI admission candidate is absent from the snapshot")
            sidecar = self._factor(strategy_factors, candidate)
            required = self._gpu_tokens(view)
            if view.state is ProgramState.ACTIVE and view.backend_id == snapshot.backend_id:
                self.counters["admission_already_gpu"] += 1
                return AdmissionOutcome(AdmissionDisposition.ADMITTED, "mori_already_gpu", required_tokens=required)

            remaining = self._remaining_gpu_tokens(snapshot, strategy_factors)
            if remaining is None or remaining >= required:
                kind = "resume" if view.step_count else "admit"
                result = (
                    transitions.resume(candidate, reason="mori_cpu_ready", backend_id=snapshot.backend_id)
                    if view.step_count
                    else transitions.admit(candidate, reason="mori_new_arrival", backend_id=snapshot.backend_id)
                )
                if result.applied:
                    self.counters[f"admission_{kind}_gpu"] += 1
                    return AdmissionOutcome(
                        AdmissionDisposition.ADMITTED,
                        f"mori_{kind}_gpu",
                        required_tokens=required,
                        transitioned_programs=(candidate,),
                    )

            if sidecar.tier not in (MORITier.CPU, MORITier.WAITING):
                self._set(strategy_factors, candidate, replace(sidecar, tier=MORITier.WAITING))
            result = transitions.queue(candidate, reason="mori_capacity_queue")
            self._require_applied(result)
            self.counters["admission_queued"] += 1
            deficit = max(0, required - max(0, remaining or 0)) if remaining is not None else 0
            return AdmissionOutcome(
                AdmissionDisposition.QUEUED,
                "mori_capacity_queue",
                required_tokens=required,
                deficit_tokens=deficit,
                transitioned_programs=(candidate,),
            )
        finally:
            self.decision_cpu_seconds += time.thread_time() - started

    def schedule_resume(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: MORIFactors,
        transitions: TransitionController,
    ) -> None:
        started = time.thread_time()
        try:
            now = snapshot.observed_at_monotonic_s
            remaining = self._remaining_gpu_tokens(snapshot, strategy_factors)
            if remaining is None:
                remaining = math.inf
            active = [
                program
                for program in snapshot.programs
                if program.state is ProgramState.ACTIVE
                and program.backend_id == snapshot.backend_id
                and not program.marked_for_pause
            ]
            waiting = [
                program
                for program in snapshot.programs
                if program.state is ProgramState.PAUSED and program.status is ProgramStatus.REASONING
            ]
            waiting.sort(key=lambda program: self._promotion_key(program, strategy_factors, now))
            marked: set[ProgramRef] = set()
            for candidate in waiting:
                required = self._gpu_tokens(candidate)
                if remaining < required:
                    freed = self._swap_out_acting_victims(
                        candidate,
                        snapshot,
                        active,
                        strategy_factors,
                        transitions,
                        now,
                        required - remaining,
                    )
                    remaining += freed
                if remaining < required:
                    self._mark_lazy_victims(
                        candidate,
                        snapshot,
                        active,
                        strategy_factors,
                        transitions,
                        now,
                        required - remaining,
                        marked,
                    )
                    self.counters["promotion_waited_for_lazy_demotion"] += 1
                    continue
                result = transitions.resume(candidate.ref, reason="mori_promote_gpu", backend_id=snapshot.backend_id)
                if not result.applied:
                    self.counters["promotion_transition_rejected"] += 1
                    continue
                remaining -= required
                active.append(candidate)
                self.counters[f"promotion_{self._factor(strategy_factors, candidate.ref).tier.value}_to_gpu"] += 1
        finally:
            self.decision_cpu_seconds += time.thread_time() - started

    def repair_capacity(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: MORIFactors,
        transitions: TransitionController,
    ) -> None:
        started = time.thread_time()
        try:
            now = snapshot.observed_at_monotonic_s
            self._enforce_cpu_capacity(snapshot, strategy_factors, now)
            remaining = self._remaining_gpu_tokens(snapshot, strategy_factors)
            if remaining is None or remaining >= 0:
                return
            deficit = -remaining
            victims = [
                program
                for program in snapshot.programs
                if program.state is ProgramState.ACTIVE
                and program.backend_id == snapshot.backend_id
                and not program.marked_for_pause
            ]
            victims.sort(key=lambda program: self._demotion_key(program, strategy_factors, now))
            for victim in victims:
                target = self._lower_tier(victim, snapshot, strategy_factors, now)
                sidecar = self._factor(strategy_factors, victim.ref)
                self._set(strategy_factors, victim.ref, replace(sidecar, pending_tier=target))
                result = (
                    transitions.pause(victim.ref, reason="mori_capacity_demotion")
                    if victim.status is ProgramStatus.ACTING
                    else transitions.mark_for_pause(victim.ref, reason="mori_lazy_capacity_demotion")
                )
                if not result.applied:
                    self.counters["capacity_demotion_rejected"] += 1
                    continue
                deficit -= self._gpu_tokens(victim)
                self.counters[f"capacity_demotion_to_{target.value}"] += 1
                if deficit <= 0:
                    break
        finally:
            self.decision_cpu_seconds += time.thread_time() - started

    def handle_request_completion(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: MORIFactors,
        transitions: TransitionController,
        completed: ProgramRef,
    ) -> None:
        del snapshot, strategy_factors, transitions, completed

    def on_schedule_cycle_start(self, snapshot: SchedulingSnapshot, strategy_factors: MORIFactors) -> None:
        del snapshot
        global_factors = strategy_factors.global_factors
        strategy_factors.set_global_factors(replace(global_factors, schedule_cycles=global_factors.schedule_cycles + 1))
        self.counters["schedule_cycles"] += 1

    def diagnostics(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: MORIFactors,
    ) -> tuple[StrategyDiagnostic, ...]:
        now = snapshot.observed_at_monotonic_s
        tiers = Counter(self._factor(strategy_factors, program.ref).tier.value for program in snapshot.programs)
        return (
            StrategyDiagnostic(
                "mori_placement",
                "periodic_relative_idleness",
                fields=(
                    ("gpu_programs", tiers[MORITier.GPU.value]),
                    ("cpu_programs", tiers[MORITier.CPU.value]),
                    ("waiting_programs", tiers[MORITier.WAITING.value]),
                    (
                        "mean_idleness",
                        sum(self._idleness(self._factor(strategy_factors, p.ref), now) for p in snapshot.programs)
                        / max(1, len(snapshot.programs)),
                    ),
                ),
            ),
        )

    def on_program_status_changed(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        if event.program is None or event.current_status is not ProgramStatus.REASONING:
            return
        sidecar = self._factor(strategy_factors, event.program)
        cycles = sidecar.cycles
        if sidecar.acting_started_at_monotonic_s is not None:
            acting = max(0.0, event.occurred_at_monotonic_s - sidecar.acting_started_at_monotonic_s)
            cycles = (*cycles, (sidecar.pending_reasoning_seconds, acting))[-self.config.window_size :]
            self.counters["cycles_completed"] += 1
        reasoning_started = event.occurred_at_monotonic_s if event.current_state is ProgramState.ACTIVE else None
        updated = replace(
            sidecar,
            cycles=cycles,
            pending_reasoning_seconds=0.0,
            acting_started_at_monotonic_s=None,
            reasoning_started_at_monotonic_s=reasoning_started,
        )
        self._set(
            strategy_factors,
            event.program,
            replace(updated, last_idleness=self._idleness(updated, event.occurred_at_monotonic_s)),
        )

    def on_program_admitted(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        self._on_gpu_transition(strategy_factors, event)

    def on_program_resumed(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        self._on_gpu_transition(strategy_factors, event)

    def _on_gpu_transition(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        if event.program is None:
            return
        sidecar = self._factor(strategy_factors, event.program)
        self._set(
            strategy_factors,
            event.program,
            replace(
                sidecar,
                tier=MORITier.GPU,
                pending_tier=None,
                reasoning_started_at_monotonic_s=(
                    event.occurred_at_monotonic_s
                    if event.current_status is ProgramStatus.REASONING
                    else sidecar.reasoning_started_at_monotonic_s
                ),
            ),
        )

    def on_request_finished(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        if event.program is None:
            return
        sidecar = self._factor(strategy_factors, event.program)
        latency = self._event_float(event.field("request_latency_seconds"))
        scheduler_wait = self._event_float(event.field("scheduler_queue_seconds"))
        reasoning = max(0.0, latency - scheduler_wait)
        pending = sidecar.pending_reasoning_seconds + reasoning
        acting_started = event.occurred_at_monotonic_s if event.current_status is ProgramStatus.ACTING else None
        updated = replace(
            sidecar,
            pending_reasoning_seconds=pending,
            acting_started_at_monotonic_s=acting_started,
            reasoning_started_at_monotonic_s=None
            if acting_started is not None
            else sidecar.reasoning_started_at_monotonic_s,
        )
        self._set(
            strategy_factors,
            event.program,
            replace(updated, last_idleness=self._idleness(updated, event.occurred_at_monotonic_s)),
        )

    def on_program_paused(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        if event.program is None:
            return
        sidecar = self._factor(strategy_factors, event.program)
        target = sidecar.pending_tier or MORITier.CPU
        self._set(
            strategy_factors,
            event.program,
            replace(sidecar, tier=target, pending_tier=None, reasoning_started_at_monotonic_s=None),
        )

    def on_program_released(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        if event.program is not None:
            previous = self._tiers_by_program.get(event.program.program_id)
            strategy_factors.remove_program_factors(event.program)
            self._tiers_by_program.pop(event.program.program_id, None)
            for ref, sidecar in reversed(strategy_factors.program_factors):
                if ref.program_id == event.program.program_id:
                    self._tiers_by_program[ref.program_id] = sidecar.tier
                    break
            if self._tiers_by_program.get(event.program.program_id) is not previous:
                self.tier_epoch += 1

    def _swap_out_acting_victims(
        self,
        candidate: ProgramView,
        snapshot: SchedulingSnapshot,
        active: list[ProgramView],
        factors: MORIFactors,
        transitions: TransitionController,
        now: float,
        deficit: float,
    ) -> int:
        # Status precedes idleness: a waiting Reasoning program displaces an
        # Acting GPU resident before same-status relative ranking applies.
        victims = [program for program in active if program.status is ProgramStatus.ACTING]
        victims.sort(key=lambda program: self._demotion_key(program, factors, now))
        freed = 0
        for victim in victims:
            target = self._lower_tier(victim, snapshot, factors, now)
            sidecar = self._factor(factors, victim.ref)
            self._set(factors, victim.ref, replace(sidecar, pending_tier=target))
            result = transitions.pause(victim.ref, reason="mori_relative_swap")
            if not result.applied:
                continue
            active.remove(victim)
            freed += self._gpu_tokens(victim)
            self.counters[f"relative_swap_to_{target.value}"] += 1
            if freed >= deficit:
                break
        return freed

    def _mark_lazy_victims(
        self,
        candidate: ProgramView,
        snapshot: SchedulingSnapshot,
        active: list[ProgramView],
        factors: MORIFactors,
        transitions: TransitionController,
        now: float,
        deficit: float,
        marked: set[ProgramRef],
    ) -> None:
        candidate_idle = self._idleness(self._factor(factors, candidate.ref), now)
        victims = [
            program
            for program in active
            if program.status is ProgramStatus.REASONING
            and program.ref not in marked
            and self._idleness(self._factor(factors, program.ref), now) > candidate_idle
        ]
        victims.sort(key=lambda program: self._demotion_key(program, factors, now))
        promised = 0
        for victim in victims:
            target = self._lower_tier(victim, snapshot, factors, now)
            sidecar = self._factor(factors, victim.ref)
            self._set(factors, victim.ref, replace(sidecar, pending_tier=target))
            result = transitions.mark_for_pause(victim.ref, reason="mori_lazy_relative_swap")
            if not result.applied:
                continue
            marked.add(victim.ref)
            promised += self._gpu_tokens(victim)
            self.counters[f"lazy_relative_swap_to_{target.value}"] += 1
            if promised >= deficit:
                return

    def _lower_tier(
        self,
        candidate: ProgramView,
        snapshot: SchedulingSnapshot,
        factors: MORIFactors,
        now: float,
    ) -> MORITier:
        need = self._cpu_tokens(candidate)
        cpu = [
            program
            for program in snapshot.programs
            if program.ref != candidate.ref and self._factor(factors, program.ref).tier is MORITier.CPU
        ]
        used = sum(self._cpu_tokens(program) for program in cpu)
        if used + need <= self.config.cpu_capacity_tokens:
            return MORITier.CPU
        candidate_idle = self._idleness(self._factor(factors, candidate.ref), now)
        victims = sorted(
            cpu,
            key=lambda program: (
                -self._idleness(self._factor(factors, program.ref), now),
                program.ref.program_id,
                program.ref.generation,
            ),
        )
        for victim in victims:
            victim_idle = self._idleness(self._factor(factors, victim.ref), now)
            if victim_idle <= candidate_idle:
                break
            sidecar = self._factor(factors, victim.ref)
            self._set(factors, victim.ref, replace(sidecar, tier=MORITier.WAITING))
            used -= self._cpu_tokens(victim)
            self.counters["cpu_to_waiting_replacement"] += 1
            if used + need <= self.config.cpu_capacity_tokens:
                return MORITier.CPU
        return MORITier.WAITING

    def _enforce_cpu_capacity(self, snapshot: SchedulingSnapshot, factors: MORIFactors, now: float) -> None:
        cpu = [program for program in snapshot.programs if self._factor(factors, program.ref).tier is MORITier.CPU]
        used = sum(self._cpu_tokens(program) for program in cpu)
        if used <= self.config.cpu_capacity_tokens:
            return
        cpu.sort(
            key=lambda program: (
                -self._idleness(self._factor(factors, program.ref), now),
                program.ref.program_id,
                program.ref.generation,
            )
        )
        for victim in cpu:
            sidecar = self._factor(factors, victim.ref)
            self._set(factors, victim.ref, replace(sidecar, tier=MORITier.WAITING))
            used -= self._cpu_tokens(victim)
            self.counters["cpu_capacity_to_waiting"] += 1
            if used <= self.config.cpu_capacity_tokens:
                return

    def _promotion_key(self, program: ProgramView, factors: MORIFactors, now: float) -> tuple[Any, ...]:
        tier = self._factor(factors, program.ref).tier
        priority = 0 if tier is MORITier.CPU else 1 if program.step_count else 2
        new_size = self._gpu_tokens(program) if priority == 2 else 0
        return (
            priority,
            new_size,
            self._idleness(self._factor(factors, program.ref), now),
            program.ref.program_id,
            program.ref.generation,
        )

    def _demotion_key(self, program: ProgramView, factors: MORIFactors, now: float) -> tuple[Any, ...]:
        return (
            0 if program.status is ProgramStatus.ACTING else 1,
            -self._idleness(self._factor(factors, program.ref), now),
            program.ref.program_id,
            program.ref.generation,
        )

    def _idleness(self, sidecar: MORIProgramFactors, now: float) -> float:
        partial_reasoning = partial_acting = 0.0
        partial = False
        if sidecar.acting_started_at_monotonic_s is not None:
            partial = True
            partial_reasoning = sidecar.pending_reasoning_seconds
            partial_acting = max(0.0, now - sidecar.acting_started_at_monotonic_s)
        elif sidecar.reasoning_started_at_monotonic_s is not None:
            partial = True
            partial_reasoning = max(0.0, now - sidecar.reasoning_started_at_monotonic_s)
        keep = max(0, self.config.window_size - int(partial))
        cycles = sidecar.cycles[-keep:] if keep else ()
        reasoning = sum(item[0] for item in cycles) + partial_reasoning
        acting = sum(item[1] for item in cycles) + partial_acting
        total = reasoning + acting
        return acting / total if total > 0 else 0.0

    def _remaining_gpu_tokens(self, snapshot: SchedulingSnapshot, factors: MORIFactors) -> int | None:
        if snapshot.total_kv_tokens is None:
            return None
        # MORI constrains logical GPU-queue ownership. Physical HBM can still
        # contain ref-zero blocks from CPU/Waiting programs; the typed engine
        # policy is responsible for evicting those blocks on allocation.
        used = sum(
            self._gpu_tokens(program)
            for program in snapshot.programs
            if self._factor(factors, program.ref).tier is MORITier.GPU
        )
        return snapshot.total_kv_tokens - used

    def _gpu_tokens(self, program: ProgramView) -> int:
        return self._cpu_tokens(program) + self.config.decode_buffer_tokens

    @staticmethod
    def _cpu_tokens(program: ProgramView) -> int:
        return max(0, program.tokens.estimated_context_tokens - program.tokens.shared_prefix_tokens)

    @staticmethod
    def _event_float(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0.0
        result = float(value)
        return max(0.0, result) if math.isfinite(result) else 0.0

    @staticmethod
    def _require_applied(result: TransitionResult) -> None:
        if not result.applied:
            raise RuntimeError(f"MORI transition rejected: {result.kind.value}: {result.reason}")

    @staticmethod
    def _factor(factors: MORIFactors, ref: ProgramRef) -> MORIProgramFactors:
        sidecar = factors.for_program(ref)
        if sidecar is None:
            sidecar = MORIProgramFactors()
            factors.set_program_factors(ref, sidecar)
        return sidecar

    def _set(self, factors: MORIFactors, ref: ProgramRef, sidecar: MORIProgramFactors) -> None:
        previous = self._tiers_by_program.get(ref.program_id)
        factors.set_program_factors(ref, sidecar)
        self._tiers_by_program[ref.program_id] = sidecar.tier
        if previous is not sidecar.tier:
            self.tier_epoch += 1

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        observed = time.monotonic() if now is None else now
        factors = self._bound_factors
        programs = [] if factors is None else list(factors.program_factors)
        tiers = Counter(sidecar.tier.value for _, sidecar in programs)
        expected_tiers = {ref.program_id: sidecar.tier for ref, sidecar in programs}
        return {
            "config": {
                "window_size": self.config.window_size,
                "tick_seconds": self.config.tick_seconds,
                "cpu_capacity_tokens": self.config.cpu_capacity_tokens,
                "decode_buffer_tokens": self.config.decode_buffer_tokens,
            },
            "counters": dict(self.counters),
            "decision_thread_cpu_seconds": self.decision_cpu_seconds,
            "live_program_factors": len(programs),
            "tier_counts": dict(tiers),
            "tier_index_entries": len(self._tiers_by_program),
            "tier_index_consistent": self._tiers_by_program == expected_tiers,
            "tier_epoch": self.tier_epoch,
            "idleness": {
                f"{ref.program_id}@{ref.generation}": self._idleness(sidecar, observed) for ref, sidecar in programs
            },
            "future_oracle_reads": 0,
        }


class MORIGPUEviction:
    """Factory returning a causal typed subclass of the existing safe-tail index."""

    @staticmethod
    def create(scheduler: Any, state: Any, strategy: MORIStrategy, *, emit: Any = None) -> Any:
        from .eviction import GapFarthestEviction

        class _TypedEviction(GapFarthestEviction):
            def __init__(self) -> None:
                self.program_by_request: dict[str, str] = {}
                super().__init__(scheduler, state, emit=emit)

            def observe(self, request: Any) -> None:
                self.program_by_request[request.request_id] = state.program_for(request)
                super().observe(request)

            def on_complete(self, request: Any, advice: Any) -> None:
                del advice
                self.observe(request)
                self.finished.add(request.request_id)
                self.stats["causal_completions"] += 1

            def _deadline(self, request: Any) -> float:
                if request.request_id not in self.finished:
                    return 0.0
                tier = strategy.tier_for_program(self.program_by_request.get(request.request_id, ""))
                if tier is MORITier.WAITING:
                    return math.inf
                if tier is MORITier.CPU:
                    return 1.0
                return 0.0

            def _preview(self, count: int, protected_ids: Any, *, allow_partial: bool) -> Any:
                before = self.stats["oracle_ranked_previews"]
                blocks = super()._preview(count, protected_ids, allow_partial=allow_partial)
                if self._last_preview[1] == "oracle":
                    self.stats["oracle_ranked_previews"] = before
                    self.stats["mori_typed_previews"] += 1
                    self._last_preview = (self._last_preview[0], "mori_typed", self._last_preview[2])
                return blocks

        return _TypedEviction()


class MORICPUTypedPolicy:
    """Reverse type priority for native CPU LRU: inactive, busy, then idle."""

    def __init__(self, manager: Any, native: Any, state: Any, strategy: MORIStrategy, *, emit: Any = None) -> None:
        from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

        policy = manager._policy
        if not isinstance(policy, LRUCachePolicy):
            raise TypeError("MORI CPU typed eviction requires native LRU policy")
        self.manager = manager
        self.native = native
        self.state = state
        self.strategy = strategy
        self.policy = policy
        self.emit = emit or (lambda row: None)
        self.key_programs: dict[Any, set[str]] = defaultdict(set)
        self.priority_cache: dict[Any, tuple[int, int]] = {}
        self.stats: Counter[str] = Counter()
        self.thread_cpu_seconds = 0.0
        self._original_evict = policy.evict
        owner = self

        def typed_evict(_policy: Any, n: int, protected: set[Any]) -> Any:
            return owner.evict(n, protected)

        policy.evict = MethodType(typed_evict, policy)
        state.subscribe(self.on_request_finished)
        self._closed = False

    def on_request_finished(self, request: Any, advice: Any) -> None:
        del advice
        from vllm.v1.kv_offload.base import make_offload_key

        started = time.thread_time()
        try:
            program_id = self.state.program_for(request)
            config = self.native.config
            group = config.kv_group_configs[0]
            committed = max(request.num_prompt_tokens, request.num_tokens - 1)
            status = self.native._req_status.get(request.request_id)
            if status is not None and status.max_offload_tokens is not None:
                committed = min(committed, status.max_offload_tokens)
            chunks = committed // group.tokens_per_chunk
            hashes = tuple(request.block_hashes[: chunks * group.hashes_per_chunk])
            chunks = min(chunks, len(hashes) // group.hashes_per_chunk)
            for index in range(chunks):
                block_hash = hashes[(index + 1) * group.hashes_per_chunk - 1]
                key = make_offload_key(block_hash, group.group_idx)
                owners = self.key_programs[key]
                if program_id not in owners:
                    owners.add(program_id)
                    self.priority_cache.pop(key, None)
            self.stats["completion_indexes"] += 1
            self.stats["completion_keys"] += chunks
        finally:
            self.thread_cpu_seconds += time.thread_time() - started

    def evict(self, n: int, protected: set[Any]) -> list[tuple[Any, Any]] | None:
        started = time.thread_time()
        try:
            if n == 0:
                return []
            candidate_count = len(self.policy._evictable) - sum(key in self.policy._evictable for key in protected)
            self.stats["keys_examined"] += candidate_count
            if candidate_count < n:
                self.stats["insufficient"] += 1
                return None
            # Native LRU ranks are unique and monotonically assigned. Retain at
            # most n lowest ranks per MORI type in bounded max-heaps. This
            # preserves (type, LRU) ordering while avoiding both an O(N log N)
            # global sort and O(N) temporary candidate tuples per eviction.
            heaps: tuple[list[tuple[int, Any]], ...] = ([], [], [])
            for key in self.policy._evictable:
                if key in protected:
                    continue
                priority = self._type_priority(key)
                rank = self.policy._ranks[key]
                heap = heaps[priority]
                if len(heap) < n:
                    heapq.heappush(heap, (-rank, key))
                    self.stats["selection_heap_pushes"] += 1
                elif rank < -heap[0][0]:
                    heapq.heapreplace(heap, (-rank, key))
                    self.stats["selection_heap_replacements"] += 1
            selected: list[tuple[int, Any]] = []
            for priority, heap in enumerate(heaps):
                needed = n - len(selected)
                if needed <= 0:
                    break
                ranked = sorted(heap, reverse=True)
                selected.extend((priority, key) for _, key in ranked[:needed])
            result = [(key, self.policy.chunks[key]) for _, key in selected]
            for priority, key in selected:
                self.stats[("inactive", "busy", "idle")[priority] + "_evicted"] += 1
                self.policy.remove(key)
                self.priority_cache.pop(key, None)
            self.stats["calls"] += 1
            self.stats["chunks"] += len(result)
            return result
        finally:
            self.thread_cpu_seconds += time.thread_time() - started

    def _type_priority(self, key: Any) -> int:
        epoch = self.strategy.tier_epoch
        cached = self.priority_cache.get(key)
        if cached is not None and cached[0] == epoch:
            self.stats["priority_cache_hits"] += 1
            return cached[1]
        priority = 0
        for program_id in self.key_programs.get(key, ()):
            tier = self.strategy.tier_for_program(program_id)
            priority = max(priority, 2 if tier is MORITier.CPU else 1 if tier is MORITier.GPU else 0)
            if priority == 2:
                break
        self.priority_cache[key] = (epoch, priority)
        self.stats["priority_cache_misses"] += 1
        return priority

    def snapshot(self) -> dict[str, Any]:
        return {
            "stats": dict(self.stats),
            "decision_thread_cpu_seconds": self.thread_cpu_seconds,
            "indexed_keys": len(self.key_programs),
            "priority_cache_entries": len(self.priority_cache),
            "future_oracle_reads": 0,
        }

    def close(self) -> None:
        """Restore the native method; process-local subscriptions die with Core."""
        if self._closed:
            return
        self._closed = True
        if self.policy.evict.__func__ is not self._original_evict.__func__:
            self.policy.evict = self._original_evict
