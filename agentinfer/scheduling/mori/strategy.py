# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""MORI sticky GPU/CPU/Waiting placement over committed server-side facts."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import replace

from agentinfer.scheduling.admission_outcome import AdmissionDisposition, AdmissionOutcome
from agentinfer.scheduling.cache_placement import CacheClass, ProgramCacheClassObserver
from agentinfer.scheduling.domain import ProgramRef, ProgramState, ProgramStatus, ProgramView
from agentinfer.scheduling.events import SchedulingEvent, StrategyDiagnostic
from agentinfer.scheduling.factors import StrategyFactors
from agentinfer.scheduling.mori.config import MORIConfig, MORIMode
from agentinfer.scheduling.mori.factors import MORIGlobalFactors, MORIProgramFactors, MORITier
from agentinfer.scheduling.snapshot import SchedulingSnapshot
from agentinfer.scheduling.strategy import SchedulingCycleOrder, SchedulingStrategy
from agentinfer.scheduling.transitions import TransitionController, TransitionResult

MORIFactors = StrategyFactors[MORIGlobalFactors, MORIProgramFactors]


class MORIStrategy(SchedulingStrategy[MORIGlobalFactors, MORIProgramFactors]):
    """Capacity-triggered placement with relative idleness used only for ordering.

    The strategy does not predict a future return, prefetch KV, or perform a
    transfer. It commits logical protection classes; native vLLM remains the
    owner of allocation, on-demand loading, offload jobs, fences, and physical
    safety validation.
    """

    def __init__(self, config: MORIConfig) -> None:
        self.config = config
        self.counters: Counter[str] = Counter()
        self.decision_cpu_seconds = 0.0
        self._factors: MORIFactors | None = None
        self._classes: dict[ProgramRef, CacheClass] = {}
        self._cache_class_epoch = 0
        self._observers: list[ProgramCacheClassObserver] = []

    @property
    def cycle_order(self) -> SchedulingCycleOrder:
        return SchedulingCycleOrder.REPAIR_THEN_RESUME

    @property
    def requires_program_identity(self) -> bool:
        return True

    @property
    def allows_overlapping_program_requests(self) -> bool:
        return False

    @property
    def observes_streaming_context_growth(self) -> bool:
        """Account for generated tokens when repairing MORI reservations."""
        return True

    @property
    def cache_class_epoch(self) -> int:
        return self._cache_class_epoch

    def bind_factors(self, factors: MORIFactors) -> None:
        """Bind the one factor collection owned by the containing runtime."""
        self._factors = factors
        self._classes = {ref: self._class_for_tier(sidecar.tier) for ref, sidecar in factors.program_factors}

    def subscribe_cache_class_observer(self, observer: ProgramCacheClassObserver) -> None:
        """Register an engine-thread observer once."""
        if observer not in self._observers:
            self._observers.append(observer)

    def unsubscribe_cache_class_observer(self, observer: ProgramCacheClassObserver) -> None:
        """Remove an observer without disturbing other native policy indexes."""
        if observer in self._observers:
            self._observers.remove(observer)

    def cache_class_for_program(self, program: ProgramRef) -> CacheClass:
        """Return the committed class, failing safe to INACTIVE for an unknown generation."""
        return self._classes.get(program, CacheClass.INACTIVE)

    def idleness_for_program(self, program: ProgramRef, now_monotonic_s: float | None = None) -> float:
        """Return causal relative idleness for diagnostics and native policy tests."""
        factors = self._factors
        if factors is None:
            return 0.0
        sidecar = factors.for_program(program)
        if sidecar is None:
            return 0.0
        now = time.monotonic() if now_monotonic_s is None else now_monotonic_s
        return self._idleness(sidecar, now)

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
            required = self._gpu_target(view, snapshot)

            if view.state is ProgramState.ACTIVE and sidecar.tier is MORITier.GPU:
                # The exact prompt was already committed to the Program view by
                # ProgramScheduler. Repair aggregate growth before returning,
                # but never pretend lazy relief is immediately free capacity.
                self._repair_gpu_capacity(snapshot, strategy_factors, transitions)
                self.counters["admission_existing_gpu"] += 1
                return AdmissionOutcome(
                    AdmissionDisposition.ADMITTED,
                    "mori_existing_gpu",
                    required_tokens=required,
                )

            available = self._remaining_gpu_tokens(snapshot, strategy_factors)
            waiting_refs = set(snapshot.waiting_programs)
            has_waiters = any(
                program.ref != candidate and program.ref in waiting_refs and program.status is ProgramStatus.REASONING
                for program in snapshot.programs
            )
            if available >= required and not has_waiters:
                result = (
                    transitions.resume(candidate, reason="mori_promote_gpu", backend_id=snapshot.backend_id)
                    if view.step_count > 0
                    else transitions.admit(candidate, reason="mori_new_gpu", backend_id=snapshot.backend_id)
                )
                if result.applied:
                    self.counters[f"admission_{sidecar.tier.value}_to_gpu"] += 1
                    return AdmissionOutcome(
                        AdmissionDisposition.ADMITTED,
                        result.reason,
                        required_tokens=required,
                        transitioned_programs=(candidate,),
                    )

            result = transitions.queue(candidate, reason="mori_capacity_queue")
            self._require_applied(result)
            self.counters["admission_queued"] += 1
            return AdmissionOutcome(
                AdmissionDisposition.QUEUED,
                "mori_capacity_queue",
                required_tokens=required,
                deficit_tokens=max(0, required - max(0, available)),
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
            available = self._remaining_gpu_tokens(snapshot, strategy_factors)
            if available < 0:
                self.counters["promotion_blocked_capacity_debt"] += 1
                return
            now = snapshot.observed_at_monotonic_s
            candidates = [
                program
                for program in snapshot.programs
                if program.state is ProgramState.PAUSED
                and program.status is ProgramStatus.REASONING
                and not program.marked_for_pause
            ]
            candidates.sort(key=lambda program: self._promotion_key(program, strategy_factors, snapshot, now))
            oldest = next(
                (
                    program
                    for program in candidates
                    if self._wait_seconds(program, now) >= self.config.starvation_timeout_seconds
                    and self._gpu_target(program, snapshot) <= self._gpu_quota(snapshot)
                ),
                None,
            )
            if oldest is not None and self._gpu_target(oldest, snapshot) > available:
                # Reserve the next turn for the oldest feasible overdue waiter.
                # Reasoning victims finish before their reservation is released;
                # do not fill that promised capacity with smaller newcomers.
                self._repair_gpu_capacity(
                    snapshot,
                    strategy_factors,
                    transitions,
                    waiting_reservation=self._gpu_target(oldest, snapshot),
                )
                self.counters["starvation_capacity_reclaims"] += 1
                return
            for candidate in candidates:
                required = self._gpu_target(candidate, snapshot)
                if required > available:
                    self.counters["promotion_no_fit"] += 1
                    continue
                result = (
                    transitions.resume(candidate.ref, reason="mori_cycle_promote", backend_id=snapshot.backend_id)
                    if candidate.step_count > 0
                    else transitions.admit(candidate.ref, reason="mori_cycle_admit", backend_id=snapshot.backend_id)
                )
                if not result.applied:
                    self.counters["promotion_rejected"] += 1
                    continue
                available -= required
                self.counters["promotion_applied"] += 1
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
            self._repair_cpu_capacity(snapshot, strategy_factors)
            self._repair_gpu_capacity(snapshot, strategy_factors, transitions)
        finally:
            self.decision_cpu_seconds += time.thread_time() - started

    def handle_request_completion(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: MORIFactors,
        transitions: TransitionController,
        completed: ProgramRef,
    ) -> None:
        """Keep sticky placement; lazy demotion is applied by ProgramScheduler first."""
        del snapshot, strategy_factors, transitions, completed

    def on_schedule_cycle_start(self, snapshot: SchedulingSnapshot, strategy_factors: MORIFactors) -> None:
        del snapshot
        current = strategy_factors.global_factors
        strategy_factors.set_global_factors(replace(current, schedule_cycles=current.schedule_cycles + 1))
        self.counters["schedule_cycles"] += 1

    def diagnostics(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: MORIFactors,
    ) -> tuple[StrategyDiagnostic, ...]:
        now = snapshot.observed_at_monotonic_s
        tiers = Counter(self._factor(strategy_factors, program.ref).tier.value for program in snapshot.programs)
        gpu_reserved = self._gpu_reserved_tokens(snapshot, strategy_factors)
        cpu_reserved = self._cpu_reserved_tokens(snapshot, strategy_factors)
        prompt_tokens = self.counters["cache_prompt_tokens"]
        prefix_hit_tokens = self.counters["cache_prefix_hit_tokens"]
        hbm_hit_tokens = self.counters["cache_hbm_hit_tokens"]
        mean_idleness = sum(
            self._idleness(self._factor(strategy_factors, program.ref), now) for program in snapshot.programs
        ) / max(1, len(snapshot.programs))
        return (
            StrategyDiagnostic(
                "mori_placement",
                "periodic_relative_idleness",
                (
                    ("mode", self.config.mode.value),
                    ("window_size", self.config.window_size),
                    ("decode_buffer_tokens", self.config.decode_buffer_tokens),
                    ("gpu_programs", tiers[MORITier.GPU.value]),
                    ("cpu_programs", tiers[MORITier.CPU.value]),
                    ("waiting_programs", tiers[MORITier.WAITING.value]),
                    ("gpu_reserved_tokens", gpu_reserved),
                    ("gpu_quota_tokens", snapshot.total_kv_tokens),
                    ("gpu_capacity_debt_tokens", max(0, gpu_reserved - self._gpu_quota(snapshot))),
                    ("cpu_reserved_tokens", cpu_reserved),
                    ("cpu_quota_tokens", snapshot.total_dram_kv_tokens),
                    ("native_hbm_used_tokens", snapshot.native_used_kv_tokens),
                    ("native_dram_used_tokens", snapshot.used_dram_kv_tokens),
                    ("pending_h2d_tokens", snapshot.pending_h2d_kv_tokens),
                    ("pending_d2h_tokens", snapshot.pending_d2h_kv_tokens),
                    ("cache_observed_requests", self.counters["cache_observed_requests"]),
                    ("cache_prompt_tokens", prompt_tokens),
                    ("cache_prefix_hit_tokens", prefix_hit_tokens),
                    ("cache_hbm_hit_tokens", hbm_hit_tokens),
                    ("cache_hbm_ref_zero_hit_tokens", self.counters["cache_hbm_ref_zero_hit_tokens"]),
                    ("cache_prefix_hit_ratio", prefix_hit_tokens / prompt_tokens if prompt_tokens else 0.0),
                    ("cache_hbm_hit_ratio", hbm_hit_tokens / prompt_tokens if prompt_tokens else 0.0),
                    ("mean_idleness", mean_idleness),
                    (
                        "max_wait_seconds",
                        max(
                            (
                                self._wait_seconds(program, now)
                                for program in snapshot.programs
                                if program.state is ProgramState.PAUSED and program.status is ProgramStatus.REASONING
                            ),
                            default=0.0,
                        ),
                    ),
                    ("cache_class_epoch", self.cache_class_epoch),
                    ("decision_cpu_seconds", self.decision_cpu_seconds),
                ),
            ),
        )

    def on_request_pending(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        if event.program is not None:
            self._factor(strategy_factors, event.program)

    def on_program_status_changed(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        if event.program is None or event.current_status is not ProgramStatus.REASONING:
            return
        sidecar = self._factor(strategy_factors, event.program)
        cycles = sidecar.cycles
        if sidecar.acting_started_at_monotonic_s is not None:
            acting = max(0.0, event.occurred_at_monotonic_s - sidecar.acting_started_at_monotonic_s)
            cycles = (*cycles, (sidecar.pending_reasoning_seconds, acting))[-self.config.window_size :]
            self.counters["cycles_completed"] += 1
        updated = replace(
            sidecar,
            cycles=cycles,
            pending_reasoning_seconds=0.0,
            acting_started_at_monotonic_s=None,
            reasoning_started_at_monotonic_s=(
                event.occurred_at_monotonic_s if event.current_state is ProgramState.ACTIVE else None
            ),
        )
        self._set_factor(
            strategy_factors,
            event.program,
            replace(updated, last_idleness=self._idleness(updated, event.occurred_at_monotonic_s)),
        )

    def on_program_admitted(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        self._on_gpu_transition(strategy_factors, event)

    def on_program_resumed(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        self._on_gpu_transition(strategy_factors, event)

    def on_request_finished(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        if event.program is None:
            return
        sidecar = self._factor(strategy_factors, event.program)
        latency = self._event_non_negative_float(event.field("request_latency_seconds"))
        scheduler_wait = self._event_non_negative_float(event.field("scheduler_queue_seconds"))
        reasoning = max(0.0, latency - scheduler_wait)
        prompt_tokens = self._event_non_negative_int(event.field("prompt_tokens"))
        prefix_hit_tokens = min(
            prompt_tokens,
            self._event_non_negative_int(event.field("cached_prefix_tokens")),
        )
        hbm_hit_tokens = min(
            prompt_tokens,
            self._event_non_negative_int(event.field("hbm_cached_prefix_tokens")),
        )
        hbm_ref_zero_hit_tokens = min(
            hbm_hit_tokens,
            self._event_non_negative_int(event.field("hbm_hit_ref_zero_tokens")),
        )
        self.counters["cache_observed_requests"] += 1
        self.counters["cache_prompt_tokens"] += prompt_tokens
        self.counters["cache_prefix_hit_tokens"] += prefix_hit_tokens
        self.counters["cache_hbm_hit_tokens"] += hbm_hit_tokens
        self.counters["cache_hbm_ref_zero_hit_tokens"] += hbm_ref_zero_hit_tokens
        acting_started = event.occurred_at_monotonic_s if event.current_status is ProgramStatus.ACTING else None
        updated = replace(
            sidecar,
            pending_reasoning_seconds=reasoning,
            acting_started_at_monotonic_s=acting_started,
            reasoning_started_at_monotonic_s=None
            if acting_started is not None
            else sidecar.reasoning_started_at_monotonic_s,
        )
        self._set_factor(
            strategy_factors,
            event.program,
            replace(updated, last_idleness=self._idleness(updated, event.occurred_at_monotonic_s)),
        )

    def on_program_paused(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        if event.program is None:
            return
        sidecar = self._factor(strategy_factors, event.program)
        target = sidecar.pending_tier or MORITier.WAITING
        self._set_factor(
            strategy_factors,
            event.program,
            replace(sidecar, tier=target, pending_tier=None, reasoning_started_at_monotonic_s=None),
        )

    def on_program_released(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        if event.program is None:
            return
        strategy_factors.remove_program_factors(event.program)
        self._classes.pop(event.program, None)
        self._cache_class_epoch += 1
        for observer in tuple(self._observers):
            observer.on_program_released(event.program)

    def _on_gpu_transition(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        if event.program is None:
            return
        sidecar = self._factor(strategy_factors, event.program)
        self._set_factor(
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

    def _repair_gpu_capacity(
        self,
        snapshot: SchedulingSnapshot,
        factors: MORIFactors,
        transitions: TransitionController,
        *,
        waiting_reservation: int = 0,
    ) -> None:
        """Reclaim quota debt or a feasible overdue waiter's next turn, without early relief."""
        deficit = max(0, self._gpu_reserved_tokens(snapshot, factors) + waiting_reservation - self._gpu_quota(snapshot))
        if deficit == 0:
            return
        promised = sum(
            self._gpu_target(program, snapshot)
            for program in snapshot.programs
            if (sidecar := self._factor(factors, program.ref)).tier is MORITier.GPU
            and sidecar.pending_tier in (MORITier.CPU, MORITier.WAITING)
        )
        needed = max(0, deficit - promised)
        if needed == 0:
            self.counters["gpu_debt_waiting_lazy_relief"] += 1
            return
        now = snapshot.observed_at_monotonic_s
        victims = [
            program
            for program in snapshot.programs
            if program.state is ProgramState.ACTIVE
            and program.backend_id == snapshot.backend_id
            and self._factor(factors, program.ref).tier is MORITier.GPU
            and self._factor(factors, program.ref).pending_tier is None
            and not program.marked_for_pause
        ]
        victims.sort(key=lambda program: self._demotion_key(program, factors, now))
        for victim in victims:
            target = self._lower_tier(victim, snapshot, factors)
            sidecar = self._factor(factors, victim.ref)
            self._set_factor(factors, victim.ref, replace(sidecar, pending_tier=target))
            result = (
                transitions.pause(victim.ref, reason="mori_capacity_demotion")
                if victim.status is ProgramStatus.ACTING
                else transitions.mark_for_pause(victim.ref, reason="mori_lazy_capacity_demotion")
            )
            if not result.applied:
                self._set_factor(factors, victim.ref, sidecar)
                self.counters["demotion_rejected"] += 1
                continue
            needed -= self._gpu_target(victim, snapshot)
            self.counters[f"demotion_to_{target.value}"] += 1
            if needed <= 0:
                return
        self.counters["gpu_capacity_debt_unresolved"] += 1

    def _repair_cpu_capacity(self, snapshot: SchedulingSnapshot, factors: MORIFactors) -> None:
        if self.config.mode is MORIMode.PLACEMENT_ONLY:
            return
        quota = self._dram_quota(snapshot)
        used = self._cpu_reserved_tokens(snapshot, factors)
        if used <= quota:
            return
        now = snapshot.observed_at_monotonic_s
        victims: list[tuple[ProgramView, bool]] = []
        for program in snapshot.programs:
            sidecar = self._factor(factors, program.ref)
            if sidecar.tier is MORITier.CPU:
                victims.append((program, False))
            elif sidecar.pending_tier is MORITier.CPU:
                victims.append((program, True))
        victims.sort(
            key=lambda item: (
                -self._idleness(self._factor(factors, item[0].ref), now),
                item[0].ref.program_id,
                item[0].ref.generation,
            )
        )
        for victim, pending in victims:
            sidecar = self._factor(factors, victim.ref)
            replacement = (
                replace(sidecar, pending_tier=MORITier.WAITING) if pending else replace(sidecar, tier=MORITier.WAITING)
            )
            self._set_factor(factors, victim.ref, replacement)
            used -= self._dram_target(victim, snapshot)
            self.counters["cpu_capacity_to_waiting"] += 1
            if used <= quota:
                return

    def _lower_tier(self, candidate: ProgramView, snapshot: SchedulingSnapshot, factors: MORIFactors) -> MORITier:
        if self.config.mode is MORIMode.PLACEMENT_ONLY:
            return MORITier.WAITING
        required = self._dram_target(candidate, snapshot)
        return (
            MORITier.CPU
            if self._cpu_reserved_tokens(snapshot, factors) + required <= self._dram_quota(snapshot)
            else MORITier.WAITING
        )

    def _promotion_key(
        self,
        program: ProgramView,
        factors: MORIFactors,
        snapshot: SchedulingSnapshot,
        now: float,
    ) -> tuple[int, float, int, int, float, str, int]:
        """Use oldest-first priority after the wait bound, then the normal MORI order."""
        tier = self._factor(factors, program.ref).tier
        priority = 0 if tier is MORITier.CPU else 1 if program.step_count > 0 else 2
        new_context = self._gpu_target(program, snapshot) if priority == 2 else 0
        waited = self._wait_seconds(program, now)
        overdue = waited >= self.config.starvation_timeout_seconds
        return (
            0 if overdue else 1,
            -waited if overdue else 0.0,
            priority,
            new_context,
            self._idleness(self._factor(factors, program.ref), now),
            program.ref.program_id,
            program.ref.generation,
        )

    @staticmethod
    def _wait_seconds(program: ProgramView, now: float) -> float:
        """Measure only the current pending request's wait, excluding idle time."""
        started = program.wait_started_at_monotonic_s
        return max(0.0, now - started) if started is not None else 0.0

    def _demotion_key(
        self,
        program: ProgramView,
        factors: MORIFactors,
        now: float,
    ) -> tuple[int, float, str, int]:
        return (
            0 if program.status is ProgramStatus.ACTING else 1,
            -self._idleness(self._factor(factors, program.ref), now),
            program.ref.program_id,
            program.ref.generation,
        )

    def _remaining_gpu_tokens(self, snapshot: SchedulingSnapshot, factors: MORIFactors) -> int:
        return self._gpu_quota(snapshot) - self._gpu_reserved_tokens(snapshot, factors)

    def _gpu_reserved_tokens(self, snapshot: SchedulingSnapshot, factors: MORIFactors) -> int:
        return sum(
            self._gpu_target(program, snapshot)
            for program in snapshot.programs
            if self._factor(factors, program.ref).tier is MORITier.GPU
        )

    def _cpu_reserved_tokens(self, snapshot: SchedulingSnapshot, factors: MORIFactors) -> int:
        return sum(
            self._dram_target(program, snapshot)
            for program in snapshot.programs
            if (sidecar := self._factor(factors, program.ref)).tier is MORITier.CPU
            or sidecar.pending_tier is MORITier.CPU
        )

    def _gpu_target(self, program: ProgramView, snapshot: SchedulingSnapshot) -> int:
        block_size = snapshot.hbm_block_size_tokens or program.tokens.block_size_tokens or 1
        context = max(0, program.tokens.estimated_context_tokens)
        return self._round_up(context + self.config.decode_buffer_tokens, block_size)

    @staticmethod
    def _dram_target(program: ProgramView, snapshot: SchedulingSnapshot) -> int:
        chunk_size = snapshot.dram_chunk_size_tokens or program.tokens.block_size_tokens or 1
        return MORIStrategy._round_up(max(0, program.tokens.estimated_context_tokens), chunk_size)

    @staticmethod
    def _gpu_quota(snapshot: SchedulingSnapshot) -> int:
        if snapshot.total_kv_tokens is None:
            raise RuntimeError("MORI requires a rank-local HBM token quota")
        return snapshot.total_kv_tokens

    @staticmethod
    def _dram_quota(snapshot: SchedulingSnapshot) -> int:
        if snapshot.total_dram_kv_tokens is None:
            raise RuntimeError("full MORI requires a rank-local DRAM token quota")
        return snapshot.total_dram_kv_tokens

    @staticmethod
    def _round_up(value: int, quantum: int) -> int:
        return ((value + quantum - 1) // quantum) * quantum

    def _idleness(self, sidecar: MORIProgramFactors, now: float) -> float:
        partial_reasoning = 0.0
        partial_acting = 0.0
        has_partial = False
        if sidecar.acting_started_at_monotonic_s is not None:
            has_partial = True
            partial_reasoning = sidecar.pending_reasoning_seconds
            partial_acting = max(0.0, now - sidecar.acting_started_at_monotonic_s)
        elif sidecar.reasoning_started_at_monotonic_s is not None:
            has_partial = True
            partial_reasoning = max(0.0, now - sidecar.reasoning_started_at_monotonic_s)
        keep = max(0, self.config.window_size - int(has_partial))
        cycles = sidecar.cycles[-keep:] if keep else ()
        reasoning = sum(item[0] for item in cycles) + partial_reasoning
        acting = sum(item[1] for item in cycles) + partial_acting
        total = reasoning + acting
        return acting / total if total > 0 else 0.0

    def _factor(self, factors: MORIFactors, ref: ProgramRef) -> MORIProgramFactors:
        sidecar = factors.for_program(ref)
        if sidecar is None:
            sidecar = MORIProgramFactors()
            factors.set_program_factors(ref, sidecar)
            self._classes.setdefault(ref, CacheClass.INACTIVE)
        return sidecar

    def _set_factor(self, factors: MORIFactors, ref: ProgramRef, sidecar: MORIProgramFactors) -> None:
        previous_class = self._classes.get(ref, CacheClass.INACTIVE)
        current_class = self._class_for_tier(sidecar.tier)
        factors.set_program_factors(ref, sidecar)
        self._classes[ref] = current_class
        if previous_class is current_class:
            return
        self._cache_class_epoch += 1
        for observer in tuple(self._observers):
            observer.on_cache_class_changed(ref, previous_class, current_class)

    @staticmethod
    def _class_for_tier(tier: MORITier) -> CacheClass:
        if tier is MORITier.GPU:
            return CacheClass.BUSY
        if tier is MORITier.CPU:
            return CacheClass.IDLE
        return CacheClass.INACTIVE

    @staticmethod
    def _event_non_negative_float(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0.0
        normalized = float(value)
        return max(0.0, normalized) if math.isfinite(normalized) else 0.0

    @staticmethod
    def _event_non_negative_int(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            return 0
        return max(0, value)

    @staticmethod
    def _require_applied(result: TransitionResult) -> None:
        if not result.applied:
            raise RuntimeError(f"MORI transition rejected: {result.kind.value}: {result.reason}")
