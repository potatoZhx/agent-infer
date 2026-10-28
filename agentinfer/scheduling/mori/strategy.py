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
from agentinfer.scheduling.mori.capacity import CapacityClaim, CapacityLedger, FootprintProvider
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
        self._footprints: FootprintProvider | None = None

    @property
    def uses_request_branches(self) -> bool:
        """Use an independent server-owned branch for each overlapping request."""
        return True

    def bind_footprints(self, provider: FootprintProvider | None) -> None:
        """Attach native block evidence, or detach it at adapter shutdown."""
        self._footprints = provider

    def on_context_replaced(self, program: ProgramRef) -> None:
        """Invalidate previous-prompt block proof and wake placement planning."""
        if self._footprints is not None:
            self._footprints.invalidate_footprint(program)
        if self._factors is not None:
            self._request_schedule(self._factors, time.monotonic())

    def on_context_growth(
        self, program: ProgramRef, now: float, *, previous_tokens: int, current_tokens: int, block_size: int
    ) -> None:
        """Wake only when rounded context plus decode headroom grows, preserving block proof."""
        buffer = self.config.decode_buffer_tokens
        if self._factors is not None and self._round_up(previous_tokens + buffer, block_size) < self._round_up(
            current_tokens + buffer, block_size
        ):
            self._request_schedule(self._factors, now)

    def on_execution_changed(self, factors: MORIFactors, program: ProgramRef, now: float, *, running: bool) -> None:
        """Accumulate native execution intervals, excluding admission and preemption waits."""
        sidecar = self._factor(factors, program)
        if running:
            if not sidecar.execution_observed:
                sidecar = replace(
                    sidecar,
                    execution_observed=True,
                    pending_reasoning_seconds=0.0,
                    reasoning_started_at_monotonic_s=now,
                )
            elif sidecar.reasoning_started_at_monotonic_s is None:
                sidecar = replace(sidecar, reasoning_started_at_monotonic_s=now)
        elif sidecar.execution_observed and sidecar.reasoning_started_at_monotonic_s is not None:
            sidecar = replace(
                sidecar,
                pending_reasoning_seconds=sidecar.pending_reasoning_seconds
                + max(0.0, now - sidecar.reasoning_started_at_monotonic_s),
                reasoning_started_at_monotonic_s=None,
            )
        self._set_factor(factors, program, sidecar)

    @property
    def cycle_order(self) -> SchedulingCycleOrder:
        return SchedulingCycleOrder.REPAIR_THEN_RESUME

    @property
    def requires_program_identity(self) -> bool:
        return True

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
                # GPU membership is the admission decision. Growth is repaired by
                # the placement loop; never retain an unstarted request behind a
                # second capacity gate while still charging it to the GPU tier.
                self._request_schedule(strategy_factors, snapshot.observed_at_monotonic_s)
                self.counters["admission_existing_gpu"] += 1
                return AdmissionOutcome(AdmissionDisposition.ADMITTED, "mori_existing_gpu", required_tokens=required)

            # Ingress uses only existing slack. Victim selection is coalesced
            # into the host's next planning cycle, never repeated per waiter.
            self._request_schedule(strategy_factors, snapshot.observed_at_monotonic_s)
            gpu, _cpu = self._ledgers(snapshot, strategy_factors)
            other_waiters = any(
                p.ref != candidate and p.state is ProgramState.PAUSED and p.status is ProgramStatus.REASONING
                for p in snapshot.programs
            )
            if not other_waiters and gpu.used + gpu.incremental(
                self._claim(view, snapshot, gpu=True)
            ) <= self._gpu_quota(snapshot):
                result = transitions.admit(candidate, reason="mori_free_capacity", backend_id=snapshot.backend_id)
                self._require_applied(result)
                return AdmissionOutcome(AdmissionDisposition.ADMITTED, "mori_free_capacity", required_tokens=required)

            available = self._gpu_quota(snapshot) - gpu.used
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
            self._select_pending(snapshot, strategy_factors, transitions)
        finally:
            self.decision_cpu_seconds += time.thread_time() - started

    def _select_pending(
        self,
        snapshot: SchedulingSnapshot,
        strategy_factors: MORIFactors,
        transitions: TransitionController,
    ) -> None:
        """Rank once, share a victim cursor, and skip candidates that cannot fit.

        Waiting inference can reclaim Acting reservations, highest idleness
        first. Running inference is drained only for actual GPU capacity debt;
        no candidate owns future releases or blocks unrelated ready work.
        """
        gpu, cpu = self._ledgers(snapshot, strategy_factors)
        quota = self._gpu_quota(snapshot)
        if gpu.used > quota:
            self.counters["promotion_blocked_capacity_debt"] += 1
            return
        now = snapshot.observed_at_monotonic_s
        candidates = [
            p
            for p in snapshot.programs
            if p.state is ProgramState.PAUSED and p.status is ProgramStatus.REASONING and not p.marked_for_pause
        ]
        candidates.sort(key=lambda p: self._promotion_key(p, strategy_factors, snapshot, now))
        victims = [
            p
            for p in snapshot.programs
            if p.ref in gpu.claims
            and p.status is ProgramStatus.ACTING
            and not p.marked_for_pause
            and self._factor(strategy_factors, p.ref).pending_tier is None
        ]
        victims.sort(key=lambda p: self._demotion_key(p, strategy_factors, now))
        # Test feasibility without repeatedly scanning owners or needlessly
        # demoting Acting programs for a candidate blocked by running inference.
        available = gpu.copy()
        for victim in victims:
            available.remove(victim.ref)
        victim_cursor = iter(victims)
        for candidate in candidates:
            self.counters["planning_candidates"] += 1
            claim = self._claim(candidate, snapshot, gpu=True)
            if claim.tokens > quota:
                self.counters["promotion_exceeds_total_quota"] += 1
                continue
            if available.used + available.incremental(claim) > quota:
                self.counters["promotion_no_fit"] += 1
                continue
            while gpu.used + gpu.incremental(claim) > quota:
                victim = next(victim_cursor, None)
                if victim is None:
                    break
                self.counters["planning_victims"] += 1
                if not self._demote(
                    victim, snapshot, strategy_factors, transitions, gpu, cpu, reason="mori_waiting_inference"
                ):
                    available.add(victim.ref, gpu.claims[victim.ref])
            if gpu.used + gpu.incremental(claim) > quota:
                continue
            result = (transitions.resume if candidate.step_count else transitions.admit)(
                candidate.ref, reason="mori_cycle_promote", backend_id=snapshot.backend_id
            )
            if result.applied:
                gpu.add(candidate.ref, claim)
                available.add(candidate.ref, claim)
                cpu.remove(candidate.ref)
                self.counters["promotion_applied"] += 1

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
        strategy_factors.set_global_factors(
            replace(current, schedule_cycles=current.schedule_cycles + 1, schedule_requested_at=None)
        )
        self.counters["schedule_cycles"] += 1

    def next_check_at(self, strategy_factors: MORIFactors) -> float | None:
        """Wake on demand or capacity changes without waiting for a periodic tick."""
        return strategy_factors.global_factors.schedule_requested_at

    @staticmethod
    def _request_schedule(strategy_factors: MORIFactors, now: float) -> None:
        current = strategy_factors.global_factors
        previous = current.schedule_requested_at
        strategy_factors.set_global_factors(
            replace(current, schedule_requested_at=now if previous is None else min(previous, now))
        )

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
                    *(
                        (name, self.counters[name])
                        for name in (
                            "planning_program_reads",
                            "planning_candidates",
                            "planning_victims",
                            "promotion_no_fit",
                            "promotion_exceeds_total_quota",
                            "promotion_applied",
                            "gpu_debt_waiting_lazy_relief",
                            "demotion_to_cpu",
                            "demotion_to_waiting",
                        )
                    ),
                ),
            ),
        )

    def on_request_pending(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        if event.program is not None:
            self._factor(strategy_factors, event.program)
            self._request_schedule(strategy_factors, event.occurred_at_monotonic_s)

    def on_program_status_changed(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        if (
            event.program is not None
            and event.current_status is ProgramStatus.ACTING
            and event.reason == "request_cancelled"
        ):
            sidecar = self._factor(strategy_factors, event.program)
            elapsed = (
                max(0.0, event.occurred_at_monotonic_s - sidecar.reasoning_started_at_monotonic_s)
                if sidecar.reasoning_started_at_monotonic_s is not None
                else 0.0
            )
            self._set_factor(
                strategy_factors,
                event.program,
                replace(
                    sidecar,
                    pending_reasoning_seconds=(sidecar.pending_reasoning_seconds if sidecar.execution_observed else 0.0)
                    + elapsed,
                    reasoning_started_at_monotonic_s=None,
                    acting_started_at_monotonic_s=event.occurred_at_monotonic_s,
                ),
            )
            self._request_schedule(strategy_factors, event.occurred_at_monotonic_s)
            return
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
            execution_observed=False,
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
        self._request_schedule(strategy_factors, event.occurred_at_monotonic_s)
        sidecar = self._factor(strategy_factors, event.program)
        latency = self._event_non_negative_float(event.field("request_latency_seconds"))
        scheduler_wait = self._event_non_negative_float(event.field("scheduler_queue_seconds"))
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
        reasoning_finished = event.current_status is ProgramStatus.ACTING
        reasoning = sidecar.pending_reasoning_seconds
        if reasoning_finished:
            reasoning = (sidecar.pending_reasoning_seconds if sidecar.execution_observed else 0.0) + (
                max(0.0, event.occurred_at_monotonic_s - sidecar.reasoning_started_at_monotonic_s)
                if sidecar.reasoning_started_at_monotonic_s is not None
                else 0.0
                if sidecar.execution_observed
                else max(0.0, latency - scheduler_wait)
            )
        acting_started = event.occurred_at_monotonic_s if reasoning_finished else sidecar.acting_started_at_monotonic_s
        updated = replace(
            sidecar,
            pending_reasoning_seconds=reasoning,
            acting_started_at_monotonic_s=acting_started,
            reasoning_started_at_monotonic_s=(None if reasoning_finished else sidecar.reasoning_started_at_monotonic_s),
        )
        self._set_factor(
            strategy_factors,
            event.program,
            replace(updated, last_idleness=self._idleness(updated, event.occurred_at_monotonic_s)),
        )

    def on_program_paused(self, strategy_factors: MORIFactors, event: SchedulingEvent) -> None:
        if event.program is None:
            return
        self._request_schedule(strategy_factors, event.occurred_at_monotonic_s)
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
        self._request_schedule(strategy_factors, event.occurred_at_monotonic_s)
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

    def _claim(self, program: ProgramView, snapshot: SchedulingSnapshot, *, gpu: bool) -> CapacityClaim:
        """Construct a rounded promise, discounting only plausible current-prompt block proof."""
        tokens = self._gpu_target(program, snapshot) if gpu else self._dram_target(program, snapshot)
        quantum = (snapshot.hbm_block_size_tokens if gpu else snapshot.dram_chunk_size_tokens) or 1
        blocks = self._footprints.footprint(program.ref, "gpu" if gpu else "cpu") if self._footprints else frozenset()
        # More evidence than this prompt can contain means it is stale or from an
        # unsupported multi-group layout. Do not guess which subset is current.
        if len(blocks) * quantum > self._round_up(program.tokens.estimated_context_tokens, quantum):
            blocks = frozenset()
        return CapacityClaim(tokens, blocks)

    def _ledgers(self, snapshot: SchedulingSnapshot, factors: MORIFactors) -> tuple[CapacityLedger, CapacityLedger]:
        """Build GPU commitments and CPU commitments plus pending destinations in one scan."""
        gpu = CapacityLedger(snapshot.hbm_block_size_tokens or 1)
        cpu = CapacityLedger(snapshot.dram_chunk_size_tokens or 1)
        for program in snapshot.programs:
            self.counters["planning_program_reads"] += 1
            sidecar = self._factor(factors, program.ref)
            if sidecar.tier is MORITier.GPU:
                gpu.add(program.ref, self._claim(program, snapshot, gpu=True))
            if sidecar.tier is MORITier.CPU or sidecar.pending_tier is MORITier.CPU:
                cpu.add(program.ref, self._claim(program, snapshot, gpu=False))
        return gpu, cpu

    def _demote(
        self,
        victim: ProgramView,
        snapshot: SchedulingSnapshot,
        factors: MORIFactors,
        transitions: TransitionController,
        gpu: CapacityLedger,
        cpu: CapacityLedger,
        *,
        reason: str,
    ) -> bool:
        """Reserve a destination; only finished Acting leases release GPU commitments now."""
        cpu_claim = self._claim(victim, snapshot, gpu=False)
        target = (
            MORITier.CPU
            if (
                self.config.mode is MORIMode.FULL
                and cpu.used + cpu.incremental(cpu_claim) <= self._dram_quota(snapshot)
            )
            else MORITier.WAITING
        )
        previous = self._factor(factors, victim.ref)
        self._set_factor(factors, victim.ref, replace(previous, pending_tier=target))
        operation = transitions.pause if victim.status is ProgramStatus.ACTING else transitions.mark_for_pause
        result = operation(victim.ref, reason=reason)
        if not result.applied:
            self._set_factor(factors, victim.ref, previous)
            self.counters["demotion_rejected"] += 1
            return False
        if target is MORITier.CPU:
            cpu.add(victim.ref, cpu_claim)
        if victim.status is ProgramStatus.ACTING:
            gpu.remove(victim.ref)
        self.counters[f"demotion_to_{target.value}"] += 1
        return True

    def _repair_gpu_capacity(
        self, snapshot: SchedulingSnapshot, factors: MORIFactors, transitions: TransitionController
    ) -> None:
        """Plan enough marginal relief to cover hard debt without spending pending releases."""
        gpu, cpu = self._ledgers(snapshot, factors)
        quota = self._gpu_quota(snapshot)
        if gpu.used <= quota:
            return
        projected = gpu.copy()
        for program in snapshot.programs:
            sidecar = self._factor(factors, program.ref)
            if sidecar.pending_tier is not None:
                projected.remove(program.ref)
        if projected.used <= quota:
            self.counters["gpu_debt_waiting_lazy_relief"] += 1
            return
        victims = [
            p
            for p in snapshot.programs
            if p.ref in gpu.claims and not p.marked_for_pause and self._factor(factors, p.ref).pending_tier is None
        ]
        victims.sort(key=lambda p: self._demotion_key(p, factors, snapshot.observed_at_monotonic_s))
        for victim in victims:
            self.counters["planning_victims"] += 1
            if self._demote(victim, snapshot, factors, transitions, gpu, cpu, reason="mori_capacity_debt"):
                projected.remove(victim.ref)
            if projected.used <= quota:
                return
        self.counters["gpu_capacity_debt_unresolved"] += 1

    def _repair_cpu_capacity(self, snapshot: SchedulingSnapshot, factors: MORIFactors) -> None:
        """Relinquish lower-priority CPU claims until the logical union fits its quota."""
        if self.config.mode is MORIMode.PLACEMENT_ONLY:
            return
        _gpu, cpu = self._ledgers(snapshot, factors)
        quota = self._dram_quota(snapshot)
        if cpu.used <= quota:
            return
        victims = [p for p in snapshot.programs if p.ref in cpu.claims]
        victims.sort(
            key=lambda p: (-self._idleness(self._factor(factors, p.ref), snapshot.observed_at_monotonic_s), p.ref)
        )
        for victim in victims:
            sidecar = self._factor(factors, victim.ref)
            replacement = (
                replace(sidecar, tier=MORITier.WAITING)
                if sidecar.tier is MORITier.CPU
                else replace(sidecar, pending_tier=MORITier.WAITING)
            )
            self._set_factor(factors, victim.ref, replacement)
            cpu.remove(victim.ref)
            self.counters["cpu_capacity_to_waiting"] += 1
            if cpu.used <= quota:
                return

    def _promotion_key(
        self,
        program: ProgramView,
        factors: MORIFactors,
        snapshot: SchedulingSnapshot,
        now: float,
    ) -> tuple[int, int, float, str, int]:
        """Order by MORI tier, new context size, idleness and stable identity."""
        tier = self._factor(factors, program.ref).tier
        priority = 0 if tier is MORITier.CPU else 1 if program.step_count > 0 else 2
        new_context = self._gpu_target(program, snapshot) if priority == 2 else 0
        return (
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
        return self._ledgers(snapshot, factors)[0].used

    def _cpu_reserved_tokens(self, snapshot: SchedulingSnapshot, factors: MORIFactors) -> int:
        return self._ledgers(snapshot, factors)[1].used

    def _gpu_target(self, program: ProgramView, snapshot: SchedulingSnapshot) -> int:
        block_size = snapshot.hbm_block_size_tokens or program.tokens.block_size_tokens or 1
        context = max(0, program.tokens.estimated_context_tokens)
        rounded_context = self._round_up(context, block_size)
        target = self._round_up(context + self.config.decode_buffer_tokens, block_size)
        # The speculative decode buffer alone must not make a physically
        # feasible prompt wait forever. Native max-model-length checks and
        # allocation remain authoritative for actual growth.
        return max(rounded_context, min(target, self._gpu_quota(snapshot)))

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
            if sidecar.execution_observed:
                partial_reasoning += sidecar.pending_reasoning_seconds
        elif sidecar.execution_observed:
            has_partial = True
            partial_reasoning = sidecar.pending_reasoning_seconds
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
