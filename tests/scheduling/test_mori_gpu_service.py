# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Useful-service commitments must be bounded and preserve real native progress."""

from dataclasses import replace
from unittest.mock import patch

import pytest

from agentinfer.scheduling import ProgramState, ProgramStatus, TransitionKind
from agentinfer.scheduling.identity import AgentIdentity
from agentinfer.scheduling.lifecycle import ProgramLifecycle
from agentinfer.scheduling.mori import MORIConfig, MORITier, build_mori_strategy
from agentinfer.scheduling.runtime import ProgramScheduler
from tests.scheduling.test_mori_runtime import backend
from tests.scheduling.test_mori_strategy import accepting_controller, set_tier, snapshot, view

pytestmark = pytest.mark.cpu_test


def components():
    return build_mori_strategy(
        MORIConfig(gpu_service_requests=4, ready_wait_limit_seconds=30, enable_wait_credit=False)
    )


def protect(c, program, *, start=95.0, completed=1, acting_start=None):
    set_tier(c, program, MORITier.GPU, idleness=0.9)
    factor = c.initial_factors.for_program(program.ref)
    c.strategy._set_factor(
        c.initial_factors,
        program.ref,
        replace(
            factor,
            gpu_service_started_at_monotonic_s=start,
            gpu_service_completed_requests=completed,
            acting_started_at_monotonic_s=acting_start,
        ),
    )


@pytest.mark.parametrize("completed,now,expected", [(0, 100, True), (3, 100, True), (4, 100, False), (1, 125, False)])
def test_request_budget_and_hard_deadline(completed, now, expected):
    c = components()
    p = view("gpu", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=400)
    protect(c, p, completed=completed)
    assert c.strategy._gpu_service_protected(p, c.initial_factors, now) is expected


def test_admission_and_repeated_planning_cannot_earn_or_reset_service():
    c = components()
    p = view("gpu", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=400)
    set_tier(c, p, MORITier.GPU)
    assert not c.strategy._gpu_service_protected(p, c.initial_factors, 100)
    c.strategy.on_execution_changed(c.initial_factors, p.ref, 100, running=True)
    for now in [101, 102, 105]:
        assert c.strategy._gpu_service_protected(p, c.initial_factors, now)
        c.strategy.on_execution_changed(c.initial_factors, p.ref, now, running=True)
    f = c.initial_factors.for_program(p.ref)
    assert f.gpu_service_started_at_monotonic_s == 100 and f.gpu_service_completed_requests == 0
    assert c.strategy.next_check_at(c.initial_factors) == 130
    c.strategy.on_execution_changed(c.initial_factors, p.ref, 110, running=False)
    c.strategy.on_execution_changed(c.initial_factors, p.ref, 120, running=True)
    assert c.initial_factors.for_program(p.ref).gpu_service_started_at_monotonic_s == 100


def test_acting_grace_uses_only_elapsed_time_and_is_at_most_one_second():
    c = components()
    p = view("gpu", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=400)
    protect(c, p, acting_start=99.5)
    assert c.strategy._gpu_service_protected(p, c.initial_factors, 100)
    assert c.strategy.next_check_at(c.initial_factors) == 100.5
    assert not c.strategy._gpu_service_protected(p, c.initial_factors, 100.5)
    protect(c, p, acting_start=None)
    assert not c.strategy._gpu_service_protected(p, c.initial_factors, 100)


def test_normal_acting_exchange_respects_grace_then_reclaims():
    c = components()
    p = view("gpu", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=700)
    waiter = view("waiter", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=500, step_count=1)
    protect(c, p, acting_start=99.5)
    calls = []
    controller = accepting_controller(c.strategy, c.initial_factors, calls)
    fixed = snapshot(p, waiter)
    c.strategy.schedule_resume(fixed, c.initial_factors, controller)
    assert calls == []
    c.strategy.schedule_resume(replace(fixed, observed_at_monotonic_s=100.5), c.initial_factors, controller)
    assert [x.kind for x in calls][:2] == [TransitionKind.PAUSE, TransitionKind.RESUME]


def test_aged_reservation_cannot_bypass_protected_reasoning_but_capacity_debt_can():
    c = components()
    p = view("gpu", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=700)
    waiter = replace(
        view("waiter", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=500),
        wait_started_at_monotonic_s=0,
    )
    protect(c, p)
    calls = []
    controller = accepting_controller(c.strategy, c.initial_factors, calls)
    c.strategy.repair_capacity(snapshot(p, waiter), c.initial_factors, controller)
    assert calls == []
    c.strategy.repair_capacity(snapshot(p, waiter, hbm=600), c.initial_factors, controller)
    assert calls[0].kind is TransitionKind.MARK_FOR_PAUSE
    assert calls[0].reason == "mori_capacity_debt"


def test_real_completions_consume_budget_and_survive_fast_continuations():
    c = components()
    capacity = backend(1008)
    runtime = ProgramScheduler(c.strategy, c.initial_factors, capacity, schedule_interval_seconds=5)
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=100.0) as clock:
        assert runtime.on_request_arrival("one", AgentIdentity("owner"), 300, capacity, "one")
        ref = runtime.program_ref_for_request("one")
        assert c.initial_factors.for_program(ref).gpu_service_started_at_monotonic_s is None
        runtime.on_native_execution("one", running=True)
        clock.return_value = 100.2
        runtime.on_request_completion("one", 310)
        f = c.initial_factors.for_program(ref)
        assert f.gpu_service_completed_requests == 1 and f.gpu_service_started_at_monotonic_s == 100
        clock.return_value = 100.3
        assert runtime.on_request_arrival("two", AgentIdentity("owner"), 320, capacity, "two")
        runtime.on_native_execution("two", running=True)
        clock.return_value = 100.5
        runtime.on_request_completion("two", 330)
        f = c.initial_factors.for_program(ref)
        assert f.gpu_service_completed_requests == 2 and f.gpu_service_started_at_monotonic_s == 100
        assert c.strategy.counters["gpu_service_completed_requests"] == 2
        assert runtime.on_response_completion("owner", ProgramLifecycle.TERMINAL)
        clock.return_value = 101
        assert runtime.on_request_arrival("new", AgentIdentity("owner"), 100, capacity, "new")
        new_ref = runtime.program_ref_for_request("new")
        assert new_ref != ref
        factor = c.initial_factors.for_program(new_ref)
        assert factor.gpu_service_started_at_monotonic_s is None
        assert factor.gpu_service_completed_requests == 0


def test_zero_output_completion_is_not_counted_as_useful_service():
    c = components()
    capacity = backend(1008)
    runtime = ProgramScheduler(c.strategy, c.initial_factors, capacity, schedule_interval_seconds=5)
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=100.0):
        assert runtime.on_request_arrival("zero", AgentIdentity("zero"), 300, capacity, "zero")
        ref = runtime.program_ref_for_request("zero")
        runtime.on_native_execution("zero", running=True)
        runtime.on_request_completion("zero", 300)
        assert c.initial_factors.for_program(ref).gpu_service_completed_requests == 0
        assert c.strategy.counters["gpu_service_completed_requests"] == 0
