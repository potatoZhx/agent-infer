# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Slow-request evidence preserves policy decisions and exact union accounting."""

from dataclasses import replace
from unittest.mock import patch

import pytest

from agentinfer.scheduling.domain import ProgramState, ProgramStatus
from agentinfer.scheduling.events import SchedulingEvent, SchedulingEventKind
from agentinfer.scheduling.mori import MORIConfig, MORIProgramFactors, MORITier, build_mori_strategy
from agentinfer.scheduling.mori.capacity import ReliefProjection
from agentinfer.scheduling.mori.diagnostics import score_components
from tests.scheduling.test_mori_strategy import accepting_controller, set_tier, snapshot, view

pytestmark = pytest.mark.cpu_test


def scenario(*, threshold=30.0, victim_tokens=600, running_tokens=300, waited=100.0):
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0, slow_request_threshold_seconds=threshold))
    victim = view("victim", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=victim_tokens)
    running = view("running", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=running_tokens)
    candidate = replace(
        view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=500, step_count=1),
        wait_started_at_monotonic_s=100.0 - waited,
    )
    for program, tier, score in [
        (victim, MORITier.GPU, 0.2),
        (running, MORITier.GPU, 0),
        (candidate, MORITier.CPU, 0.9),
    ]:
        set_tier(components, program, tier, idleness=score)
    return components, snapshot(victim, running, candidate), candidate


def plan(components, observed):
    calls = []
    components.strategy.schedule_resume(
        observed,
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )
    return calls


@pytest.mark.parametrize(
    ("victim_tokens", "running_tokens", "reason", "required"),
    [
        (600, 300, "idleness_limited", 800),
        (100, 800, "no_all_acting_fit", 1300),
    ],
)
def test_slow_decision_separates_relative_score_from_unreclaimable_capacity(
    victim_tokens, running_tokens, reason, required
):
    components, observed, candidate = scenario(victim_tokens=victim_tokens, running_tokens=running_tokens)
    assert plan(components, observed) == []
    decision = components.strategy._slow_diagnostics.decisions[candidate.ref]
    assert decision["reason"] == reason
    assert decision["relief"]["required_after_all_acting_tokens"] == required
    assert decision["relief"]["eligible_acting_prefix"] == 0
    assert decision["score"]["idleness"] == pytest.approx(0.9)


def test_shared_victims_do_not_overstate_diagnostic_reclaimable_space():
    components, observed, candidate = scenario(victim_tokens=700, running_tokens=600)
    victim, running, _ = observed.programs
    candidate = replace(candidate, tokens=replace(candidate.tokens, estimated_context_tokens=600))
    observed = snapshot(victim, running, candidate)
    set_tier(components, victim, MORITier.GPU, idleness=0.99)
    blocks = frozenset(range(400))

    class Footprints:
        def footprint(self, ref, tier):
            return blocks if tier == "gpu" and ref in (victim.ref, running.ref) else frozenset()

    components.strategy.bind_footprints(Footprints())
    assert plan(components, observed) == []
    decision = components.strategy._slow_diagnostics.decisions[candidate.ref]
    assert decision["capacity"]["gpu_committed_tokens"] == 900
    assert decision["relief"]["required_after_all_acting_tokens"] == 1200
    assert decision["reason"] == "no_all_acting_fit"


def test_fast_waiter_does_not_reconstruct_context_or_add_projection_queries():
    queries = []
    original = ReliefProjection.required

    def counted(projection, removed, claim):
        queries.append(removed)
        return original(projection, removed, claim)

    for threshold in (0, 30):
        components, observed, _ = scenario(threshold=threshold, waited=10)
        diagnostics = components.strategy._slow_diagnostics
        with patch.object(ReliefProjection, "required", counted):
            if diagnostics is None:
                plan(components, observed)
            else:
                with patch.object(diagnostics, "context", side_effect=AssertionError("fast context scan")):
                    plan(components, observed)
                assert diagnostics.decisions == {}
    assert queries == [0, 0]


def test_decision_sampling_is_bounded_and_release_cleans_generation():
    components, observed, candidate = scenario()
    plan(components, observed)
    diagnostics = components.strategy._slow_diagnostics
    first = diagnostics.decisions[candidate.ref]
    plan(components, replace(observed, observed_at_monotonic_s=104))
    assert diagnostics.decisions[candidate.ref] is first
    plan(components, replace(observed, observed_at_monotonic_s=105))
    assert diagnostics.decisions[candidate.ref] is not first
    event = SchedulingEvent(
        event_id="release",
        sequence=1,
        kind=SchedulingEventKind.PROGRAM_RELEASED,
        occurred_at_monotonic_s=106,
        reason="finished",
        program=candidate.ref,
    )
    components.strategy.on_program_released(components.initial_factors, event)
    assert candidate.ref not in diagnostics.decisions


def test_new_prompt_does_not_inherit_previous_request_decision():
    components, observed, candidate = scenario()
    plan(components, observed)
    state = components.strategy._slow_diagnostics
    assert candidate.ref in state.decisions
    components.strategy.on_context_replaced(candidate.ref)
    assert candidate.ref not in state.decisions


@pytest.mark.parametrize("reason", ["global_capacity_debt", "claim_exceeds_total_quota"])
def test_early_capacity_branches_record_distinct_reasons(reason):
    components, observed, candidate = scenario(victim_tokens=800 if reason == "global_capacity_debt" else 600)
    if reason == "claim_exceeds_total_quota":
        candidate = replace(candidate, tokens=replace(candidate.tokens, estimated_context_tokens=1200))
        observed = snapshot(*observed.programs[:-1], candidate)
    assert plan(components, observed) == []
    assert components.strategy._slow_diagnostics.decisions[candidate.ref]["reason"] == reason


def test_current_pending_slot_drops_old_gap_and_matches_live_score():
    components = build_mori_strategy()
    factor = MORIProgramFactors(
        cycles=((10, 1349), (4, 0.25), (83, 0.15), (45, 0.18), (173, 1.05)),
        reasoning_wait_started_at_monotonic_s=0,
    )
    fields = score_components(factor, 100, 5)
    assert fields["retained_cycles"] == 4
    assert fields["dropped_cycles"] == 1
    assert fields["completed_acting_seconds"] == pytest.approx(1.63)
    assert fields["current_wait_credit_seconds"] == 100
    assert fields["idleness"] == components.strategy._idleness(factor, 100)


def test_enabled_diagnostics_preserve_transitions_factors_and_policy_counters():
    results = []
    for threshold in (0, 30):
        components, observed, _ = scenario(threshold=threshold)
        set_tier(components, observed.programs[0], MORITier.GPU, idleness=0.99)
        calls = plan(components, observed)
        results.append(
            (
                [(call.kind, call.program, call.reason) for call in calls],
                tuple(components.initial_factors.program_factors),
                dict(components.strategy.counters),
            )
        )
    assert results[0] == results[1]


def test_context_pending_release_uses_union_and_does_not_change_policy_counters():
    components, observed, candidate = scenario(victim_tokens=700, running_tokens=600)
    victim, running, _ = observed.programs
    blocks = frozenset(range(400))

    class Footprints:
        def footprint(self, ref, tier):
            return blocks if tier == "gpu" and ref in (victim.ref, running.ref) else frozenset()

    components.strategy.bind_footprints(Footprints())
    for ref in (victim.ref, running.ref):
        factor = components.initial_factors.for_program(ref)
        components.strategy._set_factor(components.initial_factors, ref, replace(factor, pending_tier=MORITier.CPU))
    state = components.strategy._slow_diagnostics
    state.observe(observed)
    before = dict(components.strategy.counters)
    context = state.context(candidate.ref, 110)
    assert context["reconstructed_capacity"]["gpu_committed_tokens"] == 900
    assert context["reconstructed_capacity"]["gpu_committed_after_pending_release_tokens"] == 0
    assert context["reconstructed_capacity"]["pending_release_programs"] == 2
    assert context["latest_snapshot"]["age_seconds"] == 10
    assert dict(components.strategy.counters) == before
