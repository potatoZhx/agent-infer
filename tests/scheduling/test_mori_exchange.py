# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Cross-tier exchanges preserve feasible destinations and rejected CPU owners."""

from dataclasses import replace
from unittest.mock import patch

import pytest

from agentinfer.scheduling import (
    ProgramRef,
    ProgramState,
    ProgramStatus,
    SchedulingEvent,
    SchedulingEventKind,
    TransitionController,
    TransitionKind,
    TransitionResult,
)
from agentinfer.scheduling.identity import AgentIdentity
from agentinfer.scheduling.mori import (
    MORIConfig,
    MORIGlobalFactors,
    MORIProgramFactors,
    MORITier,
    build_mori_strategy,
)
from tests.scheduling.test_mori_rebalance import pool, scheduler
from tests.scheduling.test_mori_strategy import accepting_controller, set_tier, snapshot, view

pytestmark = pytest.mark.cpu_test


def _controller(components, calls, reject=None):
    accepted = accepting_controller(components.strategy, components.initial_factors, calls)

    def apply(request):
        if reject is not None and reject(request):
            calls.append(request)
            return TransitionResult(request.kind, request.program, False, "test_rejected")
        return accepted.apply(request)

    return TransitionController(apply)


def _program(name, tokens, *, acting=False, gpu=False):
    return view(
        name,
        state=ProgramState.ACTIVE if gpu else ProgramState.PAUSED,
        status=ProgramStatus.ACTING if acting else ProgramStatus.REASONING,
        tokens=tokens,
        step_count=1,
    )


def _set_service_debt(components, program, tokens):
    components.strategy._set_service_debt_frontier(components.initial_factors, program.ref)
    current = components.initial_factors.for_program(program.ref)
    assert current is not None
    components.strategy._set_factor(
        components.initial_factors,
        program.ref,
        replace(current, service_bypass_tokens=tokens),
    )


def test_later_admission_charges_only_actual_gpu_tokens_to_highest_blocked_candidate():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    running = _program("running", 500, gpu=True)
    victim = _program("victim", 200, acting=True, gpu=True)
    blocked = _program("a-blocked", 400)
    lower = _program("b-lower", 350)
    first_backfill = _program("c-backfill", 50)
    second_backfill = _program("d-backfill", 50)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, victim, MORITier.GPU)
    set_tier(components, blocked, MORITier.CPU, idleness=0.0)
    set_tier(components, lower, MORITier.CPU, idleness=0.1)
    calls = []

    components.strategy.schedule_resume(
        snapshot(running, victim, blocked, lower, first_backfill, second_backfill, hbm=1000),
        components.initial_factors,
        _controller(components, calls),
    )

    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.RESUME, first_backfill.ref),
        (TransitionKind.RESUME, second_backfill.ref),
    ]
    assert components.initial_factors.for_program(blocked.ref).service_bypass_tokens == 100
    assert components.initial_factors.for_program(lower.ref).service_bypass_tokens == 0
    assert components.strategy.counters["service_debt_charged_tokens"] == 100


def test_election_skips_pinned_prefix_and_selects_first_actionable_blocked_candidate():
    components = build_mori_strategy(
        MORIConfig(
            decode_buffer_tokens=0,
            enable_bounded_service_debt=True,
            slow_request_threshold_seconds=1,
        )
    )
    running = _program("running", 600, gpu=True)
    victim = _program("victim", 200, acting=True, gpu=True)
    first_pinned = replace(_program("a-pinned", 450), wait_started_at_monotonic_s=0.0)
    second_pinned = replace(_program("b-pinned", 425), wait_started_at_monotonic_s=0.0)
    actionable = replace(
        _program("c-actionable", 350),
        ref=ProgramRef("c-actionable", 7),
        wait_started_at_monotonic_s=0.0,
    )
    later_fit = _program("z-later-fit", 100)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, victim, MORITier.GPU, idleness=0.5)
    set_tier(components, first_pinned, MORITier.CPU, idleness=0.0)
    set_tier(components, second_pinned, MORITier.CPU, idleness=0.1)
    set_tier(components, actionable, MORITier.CPU, idleness=0.9)
    calls = []

    components.strategy.schedule_resume(
        snapshot(running, victim, first_pinned, second_pinned, actionable, later_fit, hbm=1000),
        components.initial_factors,
        _controller(components, calls),
    )

    assert [(call.kind, call.program) for call in calls] == [(TransitionKind.RESUME, later_fit.ref)]
    assert components.initial_factors.global_factors.service_debt_frontier == actionable.ref
    assert components.initial_factors.for_program(first_pinned.ref).service_bypass_tokens == 0
    assert components.initial_factors.for_program(second_pinned.ref).service_bypass_tokens == 0
    assert components.initial_factors.for_program(actionable.ref).service_bypass_tokens == 100
    assert components.strategy.counters["service_debt_election_skipped_pinned"] == 2
    assert components.strategy.counters["service_debt_frontier_elections_actionable"] == 1
    assert components.strategy.counters["service_debt_frontier_elections"] == 1
    first_decision = components.strategy._slow_diagnostics.decisions[first_pinned.ref]["service_debt"]
    actionable_decision = components.strategy._slow_diagnostics.decisions[actionable.ref]["service_debt"]
    assert first_decision["capacity_state"] == "pinned_infeasible"
    assert first_decision["election_state"] == "skipped_pinned"
    assert actionable_decision["capacity_state"] == "fits_after_acting"
    assert actionable_decision["election_state"] == "elected_actionable"
    assert actionable_decision["frontier_generation"] == actionable.ref.generation


def test_rejected_exchange_does_not_elect_and_real_slack_still_admits_later_fit():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    victim = _program("victim", 600, acting=True, gpu=True)
    running = _program("running", 300, gpu=True)
    rejected = _program("a-rejected", 500)
    later_fit = _program("z-later-fit", 100)
    set_tier(components, victim, MORITier.GPU, idleness=0.2)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, rejected, MORITier.CPU, idleness=0.0)
    calls = []

    components.strategy.schedule_resume(
        snapshot(victim, running, rejected, later_fit, hbm=1000),
        components.initial_factors,
        _controller(components, calls, reject=lambda request: request.kind is TransitionKind.PAUSE),
    )

    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.PAUSE, victim.ref),
        (TransitionKind.RESUME, later_fit.ref),
    ]
    assert components.initial_factors.global_factors.service_debt_frontier is None
    assert components.strategy.counters["service_debt_frontier_elections"] == 0
    assert components.strategy.counters["service_debt_frontier_elections_actionable"] == 0
    assert components.strategy.counters["service_debt_election_skipped_pinned"] == 0


def test_admission_before_frontier_block_does_not_charge_it():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    running = _program("running", 700, gpu=True)
    earlier = _program("a-earlier", 100)
    owner = _program("b-owner", 400)
    later = _program("c-later", 50)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, earlier, MORITier.CPU, idleness=0.0)
    set_tier(components, owner, MORITier.CPU, idleness=0.5)
    _set_service_debt(components, owner, 0)
    calls = []

    components.strategy.schedule_resume(
        snapshot(running, earlier, owner, later, hbm=1000),
        components.initial_factors,
        _controller(components, calls),
    )

    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.RESUME, earlier.ref),
        (TransitionKind.RESUME, later.ref),
    ]
    assert components.initial_factors.for_program(owner.ref).service_bypass_tokens == 50
    assert components.strategy.counters["service_debt_charge_events"] == 1
    assert components.strategy.counters["service_debt_charged_tokens"] == 50


def test_frontier_and_credit_persist_across_fresh_cycles_without_splitting():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    running = _program("running", 500, gpu=True)
    victim = _program("victim", 200, acting=True, gpu=True)
    owner = _program("b-owner", 400)
    first_backfill = _program("c-first", 50)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, victim, MORITier.GPU)
    set_tier(components, owner, MORITier.CPU, idleness=0.5)
    first_calls = []

    components.strategy.schedule_resume(
        snapshot(running, victim, owner, first_backfill, hbm=1000),
        components.initial_factors,
        _controller(components, first_calls),
    )

    competing = _program("a-competing", 400)
    second_backfill = _program("d-second", 50)
    active_first = replace(first_backfill, state=ProgramState.ACTIVE, backend_id="backend")
    set_tier(components, competing, MORITier.CPU, idleness=0.0)
    second_calls = []
    components.strategy.schedule_resume(
        snapshot(running, victim, active_first, competing, owner, second_backfill, hbm=1000),
        components.initial_factors,
        _controller(components, second_calls),
    )

    assert [(call.kind, call.program) for call in first_calls] == [(TransitionKind.RESUME, first_backfill.ref)]
    assert [(call.kind, call.program) for call in second_calls] == [(TransitionKind.RESUME, second_backfill.ref)]
    assert components.initial_factors.global_factors.service_debt_frontier == owner.ref
    assert components.initial_factors.for_program(owner.ref).service_bypass_tokens == 100
    assert components.initial_factors.for_program(competing.ref).service_bypass_tokens == 0
    assert components.strategy.counters["service_debt_frontier_elections"] == 1
    assert components.strategy.counters["service_debt_frontier_retained_cycles"] == 1


def test_retained_owner_becoming_pinned_keeps_exact_debt_and_allows_later_fit_without_clock_advance():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    running = _program("running", 400, gpu=True)
    victim = _program("victim", 300, acting=True, gpu=True)
    owner = replace(_program("owner", 400), ref=ProgramRef("owner", 3))
    first_backfill = _program("first-backfill", 300)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, victim, MORITier.GPU, idleness=0.2)
    set_tier(components, owner, MORITier.CPU, idleness=0.9)
    first_calls = []
    first_snapshot = snapshot(running, victim, owner, first_backfill, hbm=1000)

    components.strategy.schedule_resume(
        first_snapshot,
        components.initial_factors,
        _controller(components, first_calls),
    )

    active_backfill = replace(first_backfill, state=ProgramState.ACTIVE, backend_id="backend")
    new_reasoning = _program("new-reasoning", 200, gpu=True)
    later_fit = _program("later-fit", 100)
    set_tier(components, new_reasoning, MORITier.GPU)
    second_calls = []
    second_snapshot = snapshot(running, active_backfill, new_reasoning, owner, later_fit, hbm=1000)
    components.strategy.schedule_resume(
        second_snapshot,
        components.initial_factors,
        _controller(components, second_calls),
    )

    assert [(call.kind, call.program) for call in first_calls] == [(TransitionKind.RESUME, first_backfill.ref)]
    assert [(call.kind, call.program) for call in second_calls] == [(TransitionKind.RESUME, later_fit.ref)]
    assert first_snapshot.observed_at_monotonic_s == second_snapshot.observed_at_monotonic_s
    assert components.initial_factors.global_factors.service_debt_frontier == owner.ref
    assert components.initial_factors.for_program(owner.ref).service_bypass_tokens == 400
    assert components.strategy.counters["service_debt_frontier_elections"] == 1
    assert components.strategy.counters["service_debt_frontier_elections_actionable"] == 1
    assert components.strategy.counters["service_debt_election_skipped_pinned"] == 0
    assert components.strategy.counters["service_debt_frontier_retained_pinned_cycles"] == 1
    assert components.strategy.counters["service_debt_reservations"] == 0


def test_saved_frontier_clears_absent_when_exact_factor_is_missing():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    running = _program("running", 900, gpu=True)
    owner = _program("owner", 200)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, owner, MORITier.CPU)
    _set_service_debt(components, owner, 200)
    components.initial_factors.remove_program_factors(owner.ref)
    calls = []

    components.strategy.schedule_resume(
        snapshot(running, owner, hbm=1000),
        components.initial_factors,
        _controller(components, calls),
    )

    assert calls == []
    assert components.initial_factors.global_factors.service_debt_frontier is None
    assert components.initial_factors.for_program(owner.ref) is None
    assert components.strategy.counters["service_debt_frontier_clears_absent"] == 1


def test_saved_frontier_clears_absent_when_exact_snapshot_view_is_missing():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    running = _program("running", 900, gpu=True)
    owner = _program("owner", 200)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, owner, MORITier.CPU)
    _set_service_debt(components, owner, 200)
    calls = []

    components.strategy.schedule_resume(
        snapshot(running, hbm=1000),
        components.initial_factors,
        _controller(components, calls),
    )

    assert calls == []
    assert components.initial_factors.global_factors.service_debt_frontier is None
    assert components.initial_factors.for_program(owner.ref).service_bypass_tokens == 0
    assert components.strategy.counters["service_debt_frontier_clears_absent"] == 1


@pytest.mark.parametrize(
    "ineligible_owner",
    [
        _program("owner", 200, acting=True),
        replace(_program("owner", 200), marked_for_pause=True),
    ],
    ids=["wrong-status", "marked-for-pause"],
)
def test_saved_frontier_clears_not_waiting_when_owner_is_ineligible(ineligible_owner):
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    set_tier(components, ineligible_owner, MORITier.CPU)
    _set_service_debt(components, ineligible_owner, 200)
    calls = []

    components.strategy.schedule_resume(
        snapshot(ineligible_owner, hbm=1000),
        components.initial_factors,
        _controller(components, calls),
    )

    assert calls == []
    assert components.initial_factors.global_factors.service_debt_frontier is None
    assert components.initial_factors.for_program(ineligible_owner.ref).service_bypass_tokens == 0
    assert components.strategy.counters["service_debt_frontier_clears_not_waiting"] == 1


def test_rejected_backfill_does_not_charge_service_debt():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    running = _program("running", 600, gpu=True)
    victim = _program("victim", 200, acting=True, gpu=True)
    blocked = _program("a-blocked", 400)
    backfill = _program("b-backfill", 100)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, victim, MORITier.GPU, idleness=0.2)
    set_tier(components, blocked, MORITier.CPU, idleness=0.9)
    calls = []

    components.strategy.schedule_resume(
        snapshot(running, victim, blocked, backfill, hbm=1000),
        components.initial_factors,
        _controller(components, calls, reject=lambda request: request.program == backfill.ref),
    )

    assert [(call.kind, call.program) for call in calls] == [(TransitionKind.RESUME, backfill.ref)]
    assert components.initial_factors.global_factors.service_debt_frontier == blocked.ref
    assert components.initial_factors.for_program(blocked.ref).service_bypass_tokens == 0
    assert components.strategy.counters["service_debt_frontier_elections_actionable"] == 1
    assert components.strategy.counters["service_debt_charge_events"] == 0
    assert components.strategy.counters["service_debt_charged_tokens"] == 0


def test_service_debt_stops_same_cycle_at_one_candidate_resource_quantum():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    running = _program("running", 700, gpu=True)
    blocked = _program("a-blocked", 400)
    first_backfill = _program("b-backfill", 50)
    second_backfill = _program("c-backfill", 50)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, blocked, MORITier.CPU)
    _set_service_debt(components, blocked, 350)
    calls = []

    components.strategy.schedule_resume(
        snapshot(running, blocked, first_backfill, second_backfill, hbm=1000),
        components.initial_factors,
        _controller(components, calls),
    )

    assert [(call.kind, call.program) for call in calls] == [(TransitionKind.RESUME, first_backfill.ref)]
    assert components.initial_factors.for_program(blocked.ref).service_bypass_tokens == 400
    assert components.strategy.counters["service_debt_bypass_limits"] == 1


def test_service_debt_charges_and_limits_shared_blocks_by_marginal_union_tokens():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    running = _program("running", 700, gpu=True)
    blocked = _program("a-blocked", 400)
    first_backfill = _program("b-backfill", 200)
    second_backfill = _program("c-backfill", 200)
    blocks = {
        running.ref: frozenset(range(300)),
        blocked.ref: frozenset(range(300)),
        first_backfill.ref: frozenset(range(150)),
        second_backfill.ref: frozenset(range(150)),
    }

    class SharedGPU:
        def footprint(self, ref, tier):
            return blocks.get(ref, frozenset()) if tier == "gpu" else frozenset()

    components.strategy.bind_footprints(SharedGPU())
    set_tier(components, running, MORITier.GPU)
    set_tier(components, blocked, MORITier.CPU)
    _set_service_debt(components, blocked, 50)
    calls = []

    components.strategy.schedule_resume(
        snapshot(running, blocked, first_backfill, second_backfill, hbm=750),
        components.initial_factors,
        _controller(components, calls),
    )

    assert [(call.kind, call.program) for call in calls] == [(TransitionKind.RESUME, first_backfill.ref)]
    assert components.initial_factors.for_program(blocked.ref).service_bypass_tokens == 100
    assert components.strategy.counters["service_debt_charged_tokens"] == 50
    assert components.strategy.counters["service_debt_bypass_limits"] == 1


def test_zero_marginal_backfill_does_not_charge_or_trigger_saturation_edge():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    running = _program("running", 700, gpu=True)
    owner = _program("a-owner", 400)
    shared_backfill = _program("b-shared", 50)
    charged_backfill = _program("c-charged", 50)
    blocks = {
        running.ref: frozenset(range(700)),
        shared_backfill.ref: frozenset(range(50)),
    }

    class SharedGPU:
        def footprint(self, ref, tier):
            return blocks.get(ref, frozenset()) if tier == "gpu" else frozenset()

    components.strategy.bind_footprints(SharedGPU())
    set_tier(components, running, MORITier.GPU)
    set_tier(components, owner, MORITier.CPU)
    _set_service_debt(components, owner, 399)
    calls = []

    components.strategy.schedule_resume(
        snapshot(running, owner, shared_backfill, charged_backfill, hbm=1000),
        components.initial_factors,
        _controller(components, calls),
    )

    assert [call.program for call in calls] == [shared_backfill.ref, charged_backfill.ref]
    assert components.initial_factors.for_program(owner.ref).service_bypass_tokens == 449
    assert components.strategy.counters["service_debt_charge_events"] == 1
    assert components.strategy.counters["service_debt_charged_tokens"] == 50
    assert components.strategy.counters["service_debt_saturations"] == 1


def test_saturated_pinned_frontier_does_not_reserve_or_convoy_backfill():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    running = _program("running", 800, gpu=True)
    blocked = _program("a-blocked", 400)
    first_backfill = _program("b-backfill", 50)
    second_backfill = _program("c-backfill", 50)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, blocked, MORITier.CPU)
    _set_service_debt(components, blocked, 400)
    calls = []

    components.strategy.schedule_resume(
        snapshot(running, blocked, first_backfill, second_backfill, hbm=1000),
        components.initial_factors,
        _controller(components, calls),
    )

    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.RESUME, first_backfill.ref),
        (TransitionKind.RESUME, second_backfill.ref),
    ]
    assert components.initial_factors.for_program(blocked.ref).service_bypass_tokens == 400
    assert components.initial_factors.global_factors.service_debt_frontier == blocked.ref
    assert components.strategy.counters["service_debt_reservations"] == 0
    assert components.strategy.counters["service_debt_pinned_infeasible_evaluations"] == 1
    assert components.strategy.counters["service_debt_pinned_backfill_admissions"] == 2


def test_pending_cpu_destination_is_not_a_release_certificate_for_pinned_frontier():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    running = _program("running", 800, gpu=True)
    owner = _program("owner", 400)
    backfill = _program("backfill", 100)
    set_tier(components, running, MORITier.GPU)
    running_factor = components.initial_factors.for_program(running.ref)
    components.strategy._set_factor(
        components.initial_factors,
        running.ref,
        replace(running_factor, pending_tier=MORITier.CPU),
    )
    set_tier(components, owner, MORITier.CPU)
    _set_service_debt(components, owner, 400)
    calls = []

    components.strategy.schedule_resume(
        snapshot(running, owner, backfill, hbm=1000),
        components.initial_factors,
        _controller(components, calls),
    )

    assert [call.program for call in calls] == [backfill.ref]
    assert components.initial_factors.global_factors.service_debt_frontier == owner.ref
    assert components.initial_factors.for_program(owner.ref).service_bypass_tokens == 400
    assert components.initial_factors.for_program(running.ref).pending_tier is MORITier.CPU
    assert components.strategy.counters["service_debt_reservations"] == 0
    assert components.strategy.counters["service_debt_pinned_infeasible_evaluations"] == 1
    assert components.strategy.counters["service_debt_pinned_backfill_admissions"] == 1


def test_over_quota_frontier_clears_and_does_not_convoy_feasible_work():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    running = _program("running", 700, gpu=True)
    oversized = _program("a-oversized", 1100)
    feasible = _program("b-feasible", 100)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, oversized, MORITier.CPU)
    _set_service_debt(components, oversized, 1100)
    calls = []

    components.strategy.schedule_resume(
        snapshot(running, oversized, feasible, hbm=1000),
        components.initial_factors,
        _controller(components, calls),
    )

    assert [call.program for call in calls] == [feasible.ref]
    assert components.initial_factors.global_factors.service_debt_frontier is None
    assert components.initial_factors.for_program(oversized.ref).service_bypass_tokens == 0
    assert components.strategy.counters["service_debt_frontier_clears_exceeds_quota"] == 1


@pytest.mark.parametrize(
    ("debt", "expected"),
    [
        (0, ("competitor", "owner")),
        (200, ("owner", "competitor")),
    ],
)
def test_only_saturated_frontier_gets_binary_priority(debt, expected):
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    running = _program("running", 700, gpu=True)
    owner = _program("owner", 200)
    competitor = _program("competitor", 100)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, owner, MORITier.WAITING)
    set_tier(components, competitor, MORITier.CPU)
    _set_service_debt(components, owner, debt)
    calls = []

    components.strategy.schedule_resume(
        snapshot(running, owner, competitor, hbm=1000),
        components.initial_factors,
        _controller(components, calls),
    )

    refs = {"owner": owner.ref, "competitor": competitor.ref}
    assert [call.program for call in calls] == [refs[name] for name in expected]


def test_capacity_release_serves_persistent_saturated_frontier_first_without_clock_advance():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    running = _program("running", 800, gpu=True)
    owner = _program("owner", 400)
    competitor = _program("competitor", 100)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, owner, MORITier.WAITING)
    set_tier(components, competitor, MORITier.CPU)
    _set_service_debt(components, owner, 400)
    first_calls = []
    first_snapshot = snapshot(running, owner, hbm=1000)

    components.strategy.schedule_resume(
        first_snapshot,
        components.initial_factors,
        _controller(components, first_calls),
    )
    components.strategy.handle_scheduling_event(
        components.initial_factors,
        SchedulingEvent(
            event_id="capacity-release",
            sequence=1,
            kind=SchedulingEventKind.PROGRAM_RELEASED,
            occurred_at_monotonic_s=first_snapshot.observed_at_monotonic_s,
            reason="released",
            program=running.ref,
        ),
    )
    second_calls = []
    second_snapshot = snapshot(owner, competitor, hbm=1000)
    components.strategy.schedule_resume(
        second_snapshot,
        components.initial_factors,
        _controller(components, second_calls),
    )

    assert first_calls == []
    assert first_snapshot.observed_at_monotonic_s == second_snapshot.observed_at_monotonic_s
    assert [call.program for call in second_calls] == [owner.ref, competitor.ref]
    assert components.initial_factors.global_factors.service_debt_frontier is None
    assert components.strategy.counters["service_debt_urgent_services"] == 1
    assert components.strategy.counters["service_debt_frontier_elections"] == 1


def test_synchronous_owner_service_clears_frontier_without_recharging_it():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    running = _program("running", 500, gpu=True)
    owner = _program("owner", 400)
    backfill = _program("backfill", 50)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, owner, MORITier.WAITING)
    _set_service_debt(components, owner, 400)
    calls = []

    components.strategy.schedule_resume(
        snapshot(running, owner, backfill, hbm=1000),
        components.initial_factors,
        _controller(components, calls),
    )

    assert [call.program for call in calls] == [owner.ref, backfill.ref]
    assert components.initial_factors.global_factors.service_debt_frontier is None
    assert components.initial_factors.for_program(owner.ref).service_bypass_tokens == 0
    assert components.strategy.counters["service_debt_urgent_services"] == 1
    assert components.strategy.counters["service_debt_frontier_clears_served"] == 1
    assert components.strategy.counters["service_debt_charged_tokens"] == 0


def test_served_owner_is_absent_from_later_slow_decisions_without_same_cycle_reelection():
    components = build_mori_strategy(
        MORIConfig(
            decode_buffer_tokens=0,
            enable_bounded_service_debt=True,
            slow_request_threshold_seconds=1,
        )
    )
    running = _program("running", 500, gpu=True)
    owner = replace(_program("owner", 400), wait_started_at_monotonic_s=0.0)
    blocked = replace(_program("a-blocked", 200), wait_started_at_monotonic_s=0.0)
    backfill = replace(_program("z-backfill", 50), wait_started_at_monotonic_s=0.0)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, owner, MORITier.WAITING)
    _set_service_debt(components, owner, 400)
    calls = []

    components.strategy.schedule_resume(
        snapshot(running, owner, blocked, backfill, hbm=1000),
        components.initial_factors,
        _controller(components, calls),
    )

    backfill_debt = components.strategy._slow_diagnostics.decisions[backfill.ref]["service_debt"]
    assert [call.program for call in calls] == [owner.ref, backfill.ref]
    assert backfill_debt["is_frontier"] is False
    assert backfill_debt["frontier_generation"] is None
    assert components.initial_factors.global_factors.service_debt_frontier is None
    assert components.initial_factors.for_program(blocked.ref) is None
    assert components.strategy.counters["service_debt_frontier_elections"] == 1


def test_mature_service_debt_uses_minimal_safe_acting_prefix_without_changing_idleness():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    first = _program("a-first", 200, acting=True, gpu=True)
    second = _program("b-second", 200, acting=True, gpu=True)
    retained = _program("c-retained", 200, acting=True, gpu=True)
    running = _program("running", 400, gpu=True)
    blocked = _program("blocked", 300)
    set_tier(components, first, MORITier.GPU, idleness=0.9)
    set_tier(components, second, MORITier.GPU, idleness=0.7)
    set_tier(components, retained, MORITier.GPU, idleness=0.6)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, blocked, MORITier.CPU, idleness=0.8)
    _set_service_debt(components, blocked, 300)
    before_idleness = components.strategy.idleness_for_program(blocked.ref, 100.0)
    calls = []

    components.strategy.schedule_resume(
        snapshot(first, second, retained, running, blocked, hbm=1000),
        components.initial_factors,
        _controller(components, calls),
    )

    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.PAUSE, first.ref),
        (TransitionKind.PAUSE, second.ref),
        (TransitionKind.RESUME, blocked.ref),
    ]
    assert before_idleness == pytest.approx(0.8)
    assert components.initial_factors.for_program(retained.ref).tier is MORITier.GPU
    assert components.initial_factors.for_program(blocked.ref).service_bypass_tokens == 0
    assert components.strategy.counters["service_debt_rebalances"] == 1


def test_rejected_pause_does_not_count_planned_service_debt_rebalance():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    victim = _program("victim", 600, acting=True, gpu=True)
    running = _program("running", 300, gpu=True)
    owner = _program("owner", 500)
    set_tier(components, victim, MORITier.GPU, idleness=0.2)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, owner, MORITier.CPU, idleness=0.8)
    _set_service_debt(components, owner, 500)
    calls = []

    components.strategy.schedule_resume(
        snapshot(victim, running, owner, hbm=1000),
        components.initial_factors,
        _controller(components, calls, reject=lambda request: request.kind is TransitionKind.PAUSE),
    )

    assert [(call.kind, call.program) for call in calls] == [(TransitionKind.PAUSE, victim.ref)]
    assert components.initial_factors.global_factors.service_debt_frontier == owner.ref
    assert components.initial_factors.for_program(owner.ref).service_bypass_tokens == 500
    assert components.strategy.counters["service_debt_rebalances"] == 0
    assert components.strategy.counters["service_debt_urgent_services"] == 0


def test_rejected_resume_does_not_count_planned_service_debt_rebalance():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    victim = _program("victim", 600, acting=True, gpu=True)
    running = _program("running", 300, gpu=True)
    owner = _program("owner", 500)
    set_tier(components, victim, MORITier.GPU, idleness=0.2)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, owner, MORITier.CPU, idleness=0.8)
    _set_service_debt(components, owner, 500)
    calls = []

    components.strategy.schedule_resume(
        snapshot(victim, running, owner, hbm=1000),
        components.initial_factors,
        _controller(components, calls, reject=lambda request: request.kind is TransitionKind.RESUME),
    )

    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.PAUSE, victim.ref),
        (TransitionKind.RESUME, owner.ref),
    ]
    assert components.initial_factors.global_factors.service_debt_frontier == owner.ref
    assert components.initial_factors.for_program(owner.ref).service_bypass_tokens == 500
    assert components.strategy.counters["service_debt_rebalances"] == 0
    assert components.strategy.counters["service_debt_urgent_services"] == 0


def test_disabled_service_debt_preserves_relative_idleness_blocking():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0))
    victim = _program("victim", 600, acting=True, gpu=True)
    running = _program("running", 100, gpu=True)
    blocked = _program("blocked", 400)
    set_tier(components, victim, MORITier.GPU, idleness=0.1)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, blocked, MORITier.CPU, idleness=0.9)
    _set_service_debt(components, blocked, 400)
    calls = []

    components.strategy.schedule_resume(
        snapshot(victim, running, blocked, hbm=800),
        components.initial_factors,
        _controller(components, calls),
    )

    assert calls == []
    assert components.strategy.idleness_for_program(blocked.ref, 100.0) == pytest.approx(0.9)
    assert components.initial_factors.global_factors.service_debt_frontier is None
    assert components.strategy.counters["service_debt_frontier_elections_actionable"] == 0
    assert components.strategy.counters["service_debt_election_skipped_pinned"] == 0
    assert components.strategy.counters["service_debt_frontier_retained_pinned_cycles"] == 0


def test_disabled_service_debt_ignores_seeded_frontier_and_preserves_legacy_order():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=False))
    running = _program("running", 700, gpu=True)
    owner = _program("owner", 200)
    competitor = _program("competitor", 100)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, owner, MORITier.WAITING)
    set_tier(components, competitor, MORITier.CPU)
    current = components.initial_factors.for_program(owner.ref)
    components.strategy._set_factor(
        components.initial_factors,
        owner.ref,
        replace(current, service_bypass_tokens=200),
    )
    global_factors = components.initial_factors.global_factors
    components.initial_factors.set_global_factors(replace(global_factors, service_debt_frontier=owner.ref))
    calls = []

    components.strategy.schedule_resume(
        snapshot(running, owner, competitor, hbm=1000),
        components.initial_factors,
        _controller(components, calls),
    )

    assert [call.program for call in calls] == [competitor.ref, owner.ref]
    assert components.initial_factors.global_factors.service_debt_frontier == owner.ref
    assert components.strategy.counters["service_debt_frontier_clears_served"] == 0


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_service_debt_factor_requires_non_negative_integer_tokens(value):
    with pytest.raises(ValueError, match="service_bypass_tokens"):
        MORIProgramFactors(service_bypass_tokens=value)


def test_service_debt_preserves_original_factor_positional_arguments():
    factor = MORIProgramFactors(MORITier.CPU, (), 1.0, 2.0, 3.0, 4.0, 5.0, MORITier.WAITING, True)

    assert factor.pending_tier is MORITier.WAITING
    assert factor.execution_observed is True
    assert factor.service_bypass_tokens == 0


def test_service_debt_preserves_original_global_factor_positional_arguments():
    factor = MORIGlobalFactors(7, 3.0)

    assert factor.schedule_cycles == 7
    assert factor.schedule_requested_at == 3.0
    assert factor.service_debt_frontier is None


@pytest.mark.parametrize(
    ("kind", "reason", "state", "status", "counter"),
    [
        (
            SchedulingEventKind.REQUEST_ADMITTED,
            "admitted",
            ProgramState.ACTIVE,
            ProgramStatus.REASONING,
            "service_debt_frontier_clears_served",
        ),
        (
            SchedulingEventKind.PROGRAM_STATUS_CHANGED,
            "request_cancelled",
            ProgramState.PAUSED,
            ProgramStatus.ACTING,
            "service_debt_frontier_clears_cancelled",
        ),
        (
            SchedulingEventKind.PROGRAM_STATUS_CHANGED,
            "new_reasoning",
            ProgramState.PAUSED,
            ProgramStatus.REASONING,
            "service_debt_frontier_clears_new_spell",
        ),
        (
            SchedulingEventKind.PROGRAM_RELEASED,
            "released",
            ProgramState.PAUSED,
            ProgramStatus.REASONING,
            "service_debt_frontier_clears_released",
        ),
    ],
)
def test_exact_frontier_clears_at_lifecycle_boundaries(kind, reason, state, status, counter):
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    owner = _program("owner", 200)
    set_tier(components, owner, MORITier.CPU)
    _set_service_debt(components, owner, 100)
    event = SchedulingEvent(
        event_id="lifecycle",
        sequence=1,
        kind=kind,
        occurred_at_monotonic_s=101.0,
        reason=reason,
        program=owner.ref,
        current_state=state,
        current_status=status,
    )

    components.strategy.handle_scheduling_event(components.initial_factors, event)

    assert components.initial_factors.global_factors.service_debt_frontier is None
    remaining = components.initial_factors.for_program(owner.ref)
    if kind is SchedulingEventKind.PROGRAM_RELEASED:
        assert remaining is None
    else:
        assert remaining.service_bypass_tokens == 0
    assert components.strategy.counters[counter] == 1


def test_late_old_generation_release_cannot_clear_new_frontier():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    old = _program("owner", 200)
    set_tier(components, old, MORITier.CPU)
    _set_service_debt(components, old, 100)
    old_release = SchedulingEvent(
        event_id="old-release",
        sequence=1,
        kind=SchedulingEventKind.PROGRAM_RELEASED,
        occurred_at_monotonic_s=101.0,
        reason="released",
        program=old.ref,
    )
    components.strategy.handle_scheduling_event(components.initial_factors, old_release)
    new = replace(old, ref=ProgramRef(old.ref.program_id, old.ref.generation + 1))
    set_tier(components, new, MORITier.CPU)
    _set_service_debt(components, new, 200)
    late_release = replace(old_release, event_id="late-old-release", sequence=2, occurred_at_monotonic_s=102.0)

    components.strategy.handle_scheduling_event(components.initial_factors, late_release)

    assert components.initial_factors.global_factors.service_debt_frontier == new.ref
    assert components.initial_factors.for_program(new.ref).service_bypass_tokens == 200
    assert components.strategy.counters["service_debt_frontier_clears_released"] == 1


def test_frontier_diagnostics_are_scalar_exact_and_reset_when_absent():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, enable_bounded_service_debt=True))
    running = _program("running", 800, gpu=True)
    owner = _program("owner", 400)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, owner, MORITier.CPU)
    _set_service_debt(components, owner, 400)
    observed = snapshot(running, owner, hbm=1000)

    fields = dict(components.strategy.diagnostics(observed, components.initial_factors)[0].fields)

    assert fields["service_debt_frontier_present"] is True
    assert fields["service_debt_frontier_program_id"] == owner.ref.program_id
    assert fields["service_debt_frontier_generation"] == owner.ref.generation
    assert fields["service_debt_frontier_tokens"] == 400
    assert fields["service_debt_frontier_claim_tokens"] == 400
    assert fields["service_debt_frontier_incremental_tokens"] == 400
    assert fields["service_debt_frontier_saturated"] is True
    assert fields["service_debt_frontier_capacity_state"] == "pinned_infeasible"
    assert fields["service_debt_frontier_all_acting_required_tokens"] == 1200
    assert fields["service_debt_frontier_elections"] == 1
    assert fields["service_debt_frontier_elections_actionable"] == 0
    assert fields["service_debt_election_skipped_pinned"] == 0
    assert fields["service_debt_frontier_retained_pinned_cycles"] == 0
    assert all(not isinstance(value, ProgramRef) for value in fields.values())

    components.strategy._clear_service_debt_frontier(components.initial_factors, owner.ref, "released")
    absent = dict(components.strategy.diagnostics(observed, components.initial_factors)[0].fields)
    assert absent["service_debt_frontier_present"] is False
    assert absent["service_debt_frontier_program_id"] is None
    assert absent["service_debt_frontier_generation"] is None
    assert absent["service_debt_frontier_tokens"] == 0
    assert absent["service_debt_frontier_capacity_state"] == "absent"


def test_exchange_keeps_victim_in_cpu_space_released_by_successful_promotion():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0))
    victim = _program("victim", 700, acting=True, gpu=True)
    running = _program("running", 100, gpu=True)
    waiting = _program("waiting", 400)
    set_tier(components, victim, MORITier.GPU, idleness=0.6)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, waiting, MORITier.CPU, idleness=0.45)
    fixed = snapshot(victim, running, waiting, hbm=800, dram=800)
    calls = []

    components.strategy.schedule_resume(fixed, components.initial_factors, _controller(components, calls))

    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.PAUSE, victim.ref),
        (TransitionKind.RESUME, waiting.ref),
    ]
    assert components.initial_factors.for_program(victim.ref).tier is MORITier.CPU
    assert components.initial_factors.for_program(waiting.ref).tier is MORITier.GPU
    gpu, cpu = components.strategy._ledgers(fixed, components.initial_factors)
    assert gpu.used == 500
    assert cpu.used == 700


def test_exchange_rejection_keeps_waiter_cpu_reservation():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0))
    victim = _program("victim", 700, acting=True, gpu=True)
    running = _program("running", 100, gpu=True)
    waiting = _program("waiting", 400)
    set_tier(components, victim, MORITier.GPU, idleness=0.6)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, waiting, MORITier.CPU, idleness=0.45)
    fixed = snapshot(victim, running, waiting, hbm=800, dram=800)
    calls = []
    controller = _controller(components, calls, reject=lambda request: request.kind is TransitionKind.RESUME)

    components.strategy.schedule_resume(fixed, components.initial_factors, controller)

    assert components.initial_factors.for_program(waiting.ref).tier is MORITier.CPU
    assert components.initial_factors.for_program(victim.ref) is None
    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.PAUSE, victim.ref),
        (TransitionKind.RESUME, waiting.ref),
        (TransitionKind.RETIRE, victim.ref),
    ]
    gpu, cpu = components.strategy._ledgers(fixed, components.initial_factors)
    assert gpu.used == 100
    assert cpu.used == 400


@pytest.mark.parametrize("dram", [650, 800])
def test_exchange_multiple_victims_share_the_released_cpu_capacity_once(dram):
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0))
    first = _program("a-victim", 350, acting=True, gpu=True)
    second = _program("b-victim", 350, acting=True, gpu=True)
    running = _program("running", 100, gpu=True)
    waiting = _program("waiting", 650)
    for victim in (first, second):
        set_tier(components, victim, MORITier.GPU, idleness=0.6)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, waiting, MORITier.CPU, idleness=0.45)
    fixed = snapshot(first, second, running, waiting, hbm=800, dram=dram)
    calls = []

    components.strategy.schedule_resume(fixed, components.initial_factors, _controller(components, calls))

    expected = [
        (TransitionKind.PAUSE, first.ref),
        (TransitionKind.PAUSE, second.ref),
        (TransitionKind.RESUME, waiting.ref),
    ]
    if dram == 650:
        expected.append((TransitionKind.RETIRE, second.ref))
    assert [(call.kind, call.program) for call in calls] == expected
    assert components.initial_factors.for_program(first.ref).tier is MORITier.CPU
    if dram == 800:
        assert components.initial_factors.for_program(second.ref).tier is MORITier.CPU
    else:
        assert components.initial_factors.for_program(second.ref) is None
    gpu, cpu = components.strategy._ledgers(fixed, components.initial_factors)
    assert gpu.used == 750
    assert cpu.used == (700 if dram == 800 else 350)


def test_rejected_second_pause_never_spends_unreleased_waiter_cpu_capacity():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0))
    first = _program("a-victim", 350, acting=True, gpu=True)
    second = _program("b-victim", 350, acting=True, gpu=True)
    running = _program("running", 100, gpu=True)
    waiting = _program("waiting", 650)
    for victim in (first, second):
        set_tier(components, victim, MORITier.GPU, idleness=0.6)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, waiting, MORITier.CPU, idleness=0.45)
    fixed = snapshot(first, second, running, waiting, hbm=800, dram=1000)
    calls = []
    controller = _controller(
        components, calls, reject=lambda request: request.kind is TransitionKind.PAUSE and request.program == second.ref
    )

    components.strategy.schedule_resume(fixed, components.initial_factors, controller)

    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.PAUSE, first.ref),
        (TransitionKind.PAUSE, second.ref),
    ]
    for program, expected in ((first, MORITier.CPU), (second, MORITier.GPU), (waiting, MORITier.CPU)):
        factors = components.initial_factors.for_program(program.ref)
        assert factors.tier is expected
        assert factors.pending_tier is None
    gpu, cpu = components.strategy._ledgers(fixed, components.initial_factors)
    assert gpu.used == 450
    assert cpu.used == 1000


def test_exchange_cpu_relief_counts_only_unique_blocks_and_preserves_other_owner():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0))
    victim = _program("victim", 600, acting=True, gpu=True)
    running = _program("running", 100, gpu=True)
    waiting = _program("waiting", 400)
    resident = _program("resident", 400, acting=True)
    blocks = {
        resident.ref: frozenset(range(400)),
        waiting.ref: frozenset((*range(100, 400), *range(900, 1000))),
        victim.ref: frozenset(range(200, 800)),
    }

    class SharedCPU:
        def footprint(self, ref, tier):
            return blocks.get(ref, frozenset()) if tier == "cpu" else frozenset()

    components.strategy.bind_footprints(SharedCPU())
    set_tier(components, victim, MORITier.GPU, idleness=0.6)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, waiting, MORITier.CPU, idleness=0.45)
    set_tier(components, resident, MORITier.CPU)
    fixed = snapshot(victim, running, waiting, resident, hbm=700, dram=800)
    calls = []
    assert components.strategy._ledgers(fixed, components.initial_factors)[1].used == 500

    components.strategy.schedule_resume(fixed, components.initial_factors, _controller(components, calls))

    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.PAUSE, victim.ref),
        (TransitionKind.RESUME, waiting.ref),
    ]
    assert components.initial_factors.for_program(resident.ref).tier is MORITier.CPU
    assert components.initial_factors.for_program(victim.ref).tier is MORITier.CPU
    gpu, cpu = components.strategy._ledgers(fixed, components.initial_factors)
    assert gpu.used == 500
    assert cpu.used == 800


def test_exchange_does_not_spend_cpu_space_of_a_later_unadmitted_candidate():
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0))
    first = _program("a-victim", 350, acting=True, gpu=True)
    second = _program("b-victim", 350, acting=True, gpu=True)
    running = _program("running", 100, gpu=True)
    waiting = _program("a-waiting", 500)
    later = _program("b-waiting", 300)
    for victim in (first, second):
        set_tier(components, victim, MORITier.GPU, idleness=0.6)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, waiting, MORITier.CPU, idleness=0.4)
    set_tier(components, later, MORITier.CPU, idleness=0.5)
    fixed = snapshot(first, second, running, waiting, later, hbm=800, dram=800)
    calls = []

    components.strategy.schedule_resume(fixed, components.initial_factors, _controller(components, calls))

    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.PAUSE, first.ref),
        (TransitionKind.PAUSE, second.ref),
        (TransitionKind.RESUME, waiting.ref),
        (TransitionKind.RETIRE, second.ref),
    ]
    assert components.initial_factors.for_program(first.ref).tier is MORITier.CPU
    assert components.initial_factors.for_program(later.ref).tier is MORITier.CPU
    gpu, cpu = components.strategy._ledgers(fixed, components.initial_factors)
    assert gpu.used == 600
    assert cpu.used == 650


@pytest.mark.parametrize("failure", ["first_pause", "second_pause", "resume"])
def test_exchange_exception_finishes_accepted_pauses_without_spending_candidate_capacity(failure):
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0))
    first = _program("a-victim", 350, acting=True, gpu=True)
    second = _program("b-victim", 350, acting=True, gpu=True)
    running = _program("running", 100, gpu=True)
    waiting = _program("waiting", 650)
    for victim in (first, second):
        set_tier(components, victim, MORITier.GPU, idleness=0.6)
    set_tier(components, running, MORITier.GPU)
    set_tier(components, waiting, MORITier.CPU, idleness=0.45)
    fixed = snapshot(first, second, running, waiting, hbm=800, dram=1000)
    calls = []
    accepted = _controller(components, calls)

    def apply(request):
        if (
            (failure == "first_pause" and request.kind is TransitionKind.PAUSE and request.program == first.ref)
            or (failure == "second_pause" and request.kind is TransitionKind.PAUSE and request.program == second.ref)
            or (failure == "resume" and request.kind is TransitionKind.RESUME)
        ):
            raise RuntimeError("transition failed before mutation")
        return accepted.apply(request)

    with pytest.raises(RuntimeError, match="transition failed before mutation"):
        components.strategy.schedule_resume(fixed, components.initial_factors, TransitionController(apply))

    assert components.initial_factors.for_program(waiting.ref).tier is MORITier.CPU
    expected = {
        "first_pause": (MORITier.GPU, MORITier.GPU, 800, 650),
        "second_pause": (MORITier.CPU, MORITier.GPU, 450, 1000),
        "resume": (MORITier.CPU, MORITier.WAITING, 100, 1000),
    }[failure]
    for program, tier in zip((first, second), expected[:2], strict=True):
        factor = components.initial_factors.for_program(program.ref)
        assert factor.tier is tier
        assert factor.pending_tier is None
    gpu, cpu = components.strategy._ledgers(fixed, components.initial_factors)
    assert (gpu.used, cpu.used) == expected[2:]


@pytest.mark.parametrize("rejected", [False, True])
def test_exchange_runtime_preserves_ownership_through_admission_boundary(rejected):
    runtime = scheduler()
    capacity = pool(hbm=800, dram=800)
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert runtime.on_request_arrival("first", AgentIdentity("waiting"), 400, capacity, "first")
        waiting = runtime.program_ref_for_request("first")
        runtime.on_native_execution("first", running=True)
        clock.return_value = 1.0
        runtime.on_request_completion("first", 400)
        runtime.schedule_cycle(pool(hbm=0, dram=800))
        assert runtime.strategy_factors.for_program(waiting).tier is MORITier.CPU

        clock.return_value = 2.0
        assert runtime.on_request_arrival("victim", AgentIdentity("victim"), 700, capacity, "victim")
        victim = runtime.program_ref_for_request("victim")
        runtime.on_native_execution("victim", running=True)
        assert runtime.on_request_arrival("running", AgentIdentity("running"), 100, capacity, "running")
        runtime.on_native_execution("running", running=True)
        clock.return_value = 3.0
        runtime.on_request_completion("victim", 700)
        assert not runtime.on_request_arrival("second", AgentIdentity("waiting"), 400, capacity, "second")
        for ref, idle in ((victim, 0.6), (waiting, 0.45)):
            runtime.strategy_factors.set_program_factors(
                ref,
                replace(
                    runtime.strategy_factors.for_program(ref),
                    cycles=((1.0 - idle, idle),),
                    pending_reasoning_seconds=0.0,
                    acting_started_at_monotonic_s=None,
                ),
            )
        apply = runtime._apply_transition

        def observed(request):
            if request.kind is TransitionKind.RESUME:
                # Neither owner is dropped while the CPU->GPU outcome is unknown.
                assert runtime.strategy_factors.for_program(waiting).tier is MORITier.CPU
                assert runtime.strategy_factors.for_program(victim).tier is MORITier.GPU
                assert runtime.registry.get(victim) is not None
                if rejected:
                    return TransitionResult(request.kind, request.program, False, "test_rejected")
            return apply(request)

        with patch.object(runtime, "_apply_transition", side_effect=observed):
            runtime.schedule_cycle(capacity)

        if rejected:
            assert runtime.registry.get(victim) is None
            assert runtime.strategy_factors.for_program(waiting).tier is MORITier.CPU
            assert runtime.request_pool.waiting_request_ids == ("second",)
            assert runtime.consume_admitted_requests() == ()
        else:
            assert runtime.registry.get(victim) is not None
            assert runtime.strategy_factors.for_program(victim).tier is MORITier.CPU
            assert runtime.strategy_factors.for_program(waiting).tier is MORITier.GPU
            assert runtime.registry.require(victim).state is ProgramState.PAUSED
            assert [entry.request_id for entry, _ in runtime.consume_admitted_requests()] == ["second"]
        gpu, cpu = runtime.strategy._ledgers(runtime._snapshot(), runtime.strategy_factors)
        assert gpu.used <= 800
        assert cpu.used <= 800
