# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Ready service reservations must wait for committed request-boundary relief."""

from dataclasses import replace
from unittest.mock import patch

import pytest

from agentinfer.scheduling import ProgramState, ProgramStatus, TransitionController, TransitionKind, TransitionResult
from agentinfer.scheduling.identity import AgentIdentity
from agentinfer.scheduling.mori import MORIConfig, MORITier, build_mori_strategy
from agentinfer.scheduling.mori.capacity import CapacityClaim
from agentinfer.scheduling.runtime import ProgramScheduler
from tests.scheduling.test_mori_runtime import backend
from tests.scheduling.test_mori_strategy import accepting_controller, set_tier, snapshot, view

pytestmark = pytest.mark.cpu_test


def setup_case(limit=30.0):
    components = build_mori_strategy(MORIConfig(ready_wait_limit_seconds=limit, enable_wait_credit=False))
    resident = view("resident", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=700)
    oldest = replace(
        view("oldest", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=500, step_count=1),
        wait_started_at_monotonic_s=10.0,
    )
    younger = replace(
        view("younger", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100),
        wait_started_at_monotonic_s=99.0,
    )
    set_tier(components, resident, MORITier.GPU)
    set_tier(components, oldest, MORITier.WAITING)
    set_tier(components, younger, MORITier.CPU)
    calls = []
    controller = accepting_controller(components.strategy, components.initial_factors, calls)
    return components, resident, oldest, younger, calls, controller


def test_lazy_reservation_does_not_admit_or_discount_running_capacity():
    c, resident, oldest, younger, calls, controller = setup_case()
    fixed = snapshot(resident, oldest, younger)
    c.strategy.repair_capacity(fixed, c.initial_factors, controller)
    c.strategy.schedule_resume(fixed, c.initial_factors, controller)
    assert [(call.kind, call.program) for call in calls] == [(TransitionKind.MARK_FOR_PAUSE, resident.ref)]
    sidecar = c.initial_factors.for_program(resident.ref)
    assert sidecar.tier is MORITier.GPU and sidecar.pending_tier is MORITier.CPU
    assert c.strategy._ledgers(fixed, c.initial_factors)[0].used == 800
    assert c.strategy.counters["ready_reservation_blocked_cycles"] == 1
    # A second cycle must wait, rather than repeat or cancel preparation.
    c.strategy.repair_capacity(fixed, c.initial_factors, controller)
    assert len(calls) == 1


def test_committed_completion_relief_serves_oldest_before_younger_cpu_hit():
    c, resident, oldest, younger, calls, controller = setup_case()
    fixed = snapshot(resident, oldest, younger)
    c.strategy.repair_capacity(fixed, c.initial_factors, controller)
    # Model the backend's completion-boundary pause, not a projected release.
    controller.pause(resident.ref, reason="marked_pause_completion")
    paused = replace(resident, state=ProgramState.PAUSED, status=ProgramStatus.ACTING, backend_id=None)
    c.strategy.schedule_resume(snapshot(paused, oldest, younger), c.initial_factors, controller)
    admissions = [call.program for call in calls if call.kind in (TransitionKind.ADMIT, TransitionKind.RESUME)]
    assert admissions[0] == oldest.ref


def test_disabled_feature_preserves_younger_fit_behavior():
    c, resident, oldest, younger, calls, controller = setup_case(0)
    fixed = snapshot(resident, oldest, younger)
    c.strategy.repair_capacity(fixed, c.initial_factors, controller)
    c.strategy.schedule_resume(fixed, c.initial_factors, controller)
    assert [(call.kind, call.program) for call in calls] == [(TransitionKind.ADMIT, younger.ref)]


def test_reservation_reclaims_acting_even_when_idleness_favors_it():
    c, resident, oldest, younger, calls, controller = setup_case()
    resident = replace(resident, status=ProgramStatus.ACTING)
    set_tier(c, oldest, MORITier.WAITING, idleness=1.0)
    c.strategy.repair_capacity(snapshot(resident, oldest), c.initial_factors, controller)
    assert calls[0].kind is TransitionKind.PAUSE
    c.strategy.schedule_resume(
        snapshot(replace(resident, state=ProgramState.PAUSED), oldest), c.initial_factors, controller
    )
    assert any(call.program == oldest.ref and call.kind is TransitionKind.RESUME for call in calls)


def test_rejected_lazy_transition_does_not_create_relief_or_cpu_claim():
    c, resident, oldest, younger, calls, controller = setup_case()

    def reject(request):
        calls.append(request)
        return TransitionResult(request.kind, request.program, False, "injected rejection")

    c.strategy.repair_capacity(snapshot(resident, oldest), c.initial_factors, TransitionController(reject))
    sidecar = c.initial_factors.for_program(resident.ref)
    assert sidecar.pending_tier is None and sidecar.tier is MORITier.GPU
    assert c.strategy.counters["ready_reservation_marks"] == 0
    assert c.strategy._ledgers(snapshot(resident, oldest), c.initial_factors)[1].used == 0


def test_shared_blocks_require_both_owners_to_release_marginal_capacity():
    c, resident, oldest, younger, calls, controller = setup_case()
    peer = replace(resident, ref=younger.ref)
    set_tier(c, peer, MORITier.GPU)
    fixed = replace(snapshot(resident, peer, oldest, hbm=1050), hbm_block_size_tokens=100)

    # Both GPU claims own the same 700-token prefix plus 100 private growth.
    # Adding 600 needs two removals: one owner frees only its private 100.
    def claim(program, snapshot, *, gpu):
        if gpu and program.ref != oldest.ref:
            return CapacityClaim(800, frozenset(range(7)))
        return CapacityClaim(600)

    with patch.object(c.strategy, "_claim", side_effect=claim):
        c.strategy.repair_capacity(fixed, c.initial_factors, controller)
        assert c.strategy._ledgers(fixed, c.initial_factors)[0].used == 900
    assert len(calls) == 2 and all(call.kind is TransitionKind.MARK_FOR_PAUSE for call in calls)


def test_unaged_waiter_sets_future_wakeup_without_demoting():
    c, resident, oldest, younger, calls, controller = setup_case()
    oldest = replace(oldest, wait_started_at_monotonic_s=99.0)
    c.strategy.repair_capacity(snapshot(resident, oldest), c.initial_factors, controller)
    assert calls == []
    assert c.strategy.next_check_at(c.initial_factors) == 129.0


def test_infeasible_oldest_does_not_reserve_forever():
    c, resident, oldest, younger, calls, controller = setup_case()
    oldest = replace(oldest, tokens=replace(oldest.tokens, estimated_context_tokens=2000))
    c.strategy.repair_capacity(snapshot(resident, oldest, younger), c.initial_factors, controller)
    c.strategy.schedule_resume(snapshot(resident, oldest, younger), c.initial_factors, controller)
    assert not any(call.kind is TransitionKind.MARK_FOR_PAUSE for call in calls)
    assert any(call.program == younger.ref and call.kind is TransitionKind.ADMIT for call in calls)


def test_real_runtime_drains_at_completion_and_preserves_wait_clock():
    components = build_mori_strategy(MORIConfig(ready_wait_limit_seconds=30, enable_wait_credit=False))
    capacity = backend(1008)
    runtime = ProgramScheduler(components.strategy, components.initial_factors, capacity, schedule_interval_seconds=5)
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert runtime.on_request_arrival("resident", AgentIdentity("resident"), 700, capacity, "resident")
        runtime.schedule_cycle(capacity)
        clock.return_value = 1.0
        assert not runtime.on_request_arrival("old", AgentIdentity("old"), 500, capacity, "old")
        old = runtime.program_ref_for_request("old")
        resident = runtime.program_ref_for_request("resident")
        clock.return_value = 31.0
        runtime.schedule_cycle(capacity)
        assert runtime.registry.require(resident).marked_for_pause
        assert runtime.consume_admitted_requests() == ()
        assert runtime.registry.require(old).wait_started_at_monotonic_s == 1.0
        clock.return_value = 32.0
        runtime.on_request_completion("resident", 710)
        runtime.schedule_cycle(capacity)
        assert [entry.request_id for entry, _ in runtime.consume_admitted_requests()] == ["old"]
        assert runtime.registry.require(old).wait_started_at_monotonic_s is None
