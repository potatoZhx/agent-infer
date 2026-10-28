# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""State-machine regressions for serial leases and independently scheduled overlaps."""

from dataclasses import replace
from unittest.mock import patch

import pytest

from agentinfer.scheduling.domain import ProgramState, ProgramStatus
from agentinfer.scheduling.identity import AgentIdentity, encode_agent_identity, parse_agent_identity
from agentinfer.scheduling.lifecycle import ProgramLifecycle
from agentinfer.scheduling.mori import MORIConfig, MORITier, build_mori_strategy
from agentinfer.scheduling.runtime import ProgramScheduler
from agentinfer.scheduling.transitions import TransitionKind, TransitionRequest
from tests.scheduling.test_mori_runtime import backend

pytestmark = pytest.mark.cpu_test


def runtime(capacity=1008, **config):
    pool = backend(capacity)
    components = build_mori_strategy(MORIConfig(**config))
    return ProgramScheduler(components.strategy, components.initial_factors, pool), pool


def test_lineage_and_lifecycle_token_round_trip():
    identity = AgentIdentity("agent", lineage_id="research", lifecycle_token="attempt")
    assert parse_agent_identity(vllm_xargs={"agentic_context": encode_agent_identity(identity)}, headers={}) == identity


def test_short_overlap_never_replaces_long_working_set_or_status():
    r, pool = runtime(2016)
    assert r.on_request_arrival("long", AgentIdentity("agent"), 800, pool, "long")
    assert r.on_request_arrival("short", AgentIdentity("agent"), 100, pool, "short")
    long_ref = r.program_ref_for_request("long")
    short_ref = r.program_ref_for_request("short")
    assert long_ref != short_ref
    r.on_stream_output("long", 30)
    r.on_request_completion("short", 120)
    assert r.registry.require(long_ref).tokens.estimated_context_tokens == 830
    assert r.registry.require(long_ref).status is ProgramStatus.REASONING
    assert r.registry.get(short_ref) is None  # temporary protection ends, not native contents
    r.on_request_completion("short", 120)  # a late duplicate cannot touch the long branch
    assert r.registry.require(long_ref).tokens.estimated_context_tokens == 830


def test_named_lineage_serializes_and_uses_the_next_real_prompt():
    r, pool = runtime(2016)
    identity = AgentIdentity("agent", lineage_id="main")
    assert r.on_request_arrival("first", identity, 800, pool, "one")
    assert not r.on_request_arrival("next", identity, 100, pool, "two")
    ref = r.program_ref_for_request("first")
    assert r.program_ref_for_request("next") == ref
    assert r.registry.require(ref).tokens.estimated_context_tokens == 800
    r.on_request_completion("first", 900)
    assert r.registry.require(ref).tokens.estimated_context_tokens == 100
    assert [entry.request_id for entry, _ in r.consume_admitted_requests()] == ["next"]
    assert r.registry.require(ref).status is ProgramStatus.REASONING


def test_temporary_slot_reuse_advances_generation_and_rejects_old_events():
    r, pool = runtime(2016)
    assert r.on_request_arrival("main", AgentIdentity("agent"), 400, pool, "main")
    assert r.on_request_arrival("old", AgentIdentity("agent", lifecycle_token="old"), 100, pool, "old")
    old = r.program_ref_for_request("old")
    r.on_request_completion("old", 120)
    assert r.on_request_arrival("new", AgentIdentity("agent", lifecycle_token="new"), 100, pool, "new")
    new = r.program_ref_for_request("new")
    assert old.program_id == new.program_id
    assert new.generation > old.generation
    assert not r._apply_transition(TransitionRequest(TransitionKind.RELEASE, old, "late")).applied
    assert not r.on_scoped_response_completion("agent", ProgramLifecycle.TERMINAL, "old")
    r.on_request_completion("old", 120)
    assert r.registry.require(new).status is ProgramStatus.REASONING


def test_executing_lease_cannot_pause_even_if_phase_observation_is_wrong():
    r, pool = runtime()
    assert r.on_request_arrival("one", AgentIdentity("agent"), 400, pool, "one")
    ref = r.program_ref_for_request("one")
    r.registry.require(ref).status = ProgramStatus.ACTING
    assert not r._apply_transition(TransitionRequest(TransitionKind.PAUSE, ref, "unsafe")).applied


def test_gpu_tier_request_is_dispatched_and_growth_is_repaired_lazily():
    r, pool = runtime(1008)
    assert r.on_request_arrival("one", AgentIdentity("agent"), 400, pool, "one")
    assert r.on_request_arrival("peer", AgentIdentity("peer"), 300, pool, "peer")
    ref = r.program_ref_for_request("one")
    r.on_request_completion("one", 420)
    assert r.on_request_arrival("next", AgentIdentity("agent"), 800, pool, "next")
    assert r.registry.require(ref).state is ProgramState.ACTIVE
    assert r.registry.require(ref).tokens.estimated_context_tokens == 800
    assert r.consume_admitted_requests() == ()
    assert r.registry.require(r.program_ref_for_request("peer")).status is ProgramStatus.REASONING
    r.schedule_cycle(pool)
    assert (
        r.registry.require(ref).marked_for_pause
        or r.registry.require(r.program_ref_for_request("peer")).marked_for_pause
    )


def test_growth_wakes_on_reservation_boundary_including_decode_headroom():
    r, pool = runtime()
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        # 27 + 100 rounds to 128. The reservation grows at context 29, before
        # the raw context crosses its next block boundary at 32.
        assert r.on_request_arrival("one", AgentIdentity("agent"), 27, pool, "one")
        r.schedule_cycle(pool)
        clock.return_value = 0.1
        r.on_stream_output("one", 1)
        assert not r.needs_schedule_cycle(0.1)
        r.on_stream_output("one", 2)
        assert r.needs_schedule_cycle(0.1)


def test_drain_excludes_queued_requests_and_new_overlaps_cannot_extend_it():
    r, pool = runtime(1024)
    identity = AgentIdentity("agent", lineage_id="main")
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert r.on_request_arrival("running", identity, 400, pool, "one")
        assert r.on_request_arrival("peer", AgentIdentity("peer"), 400, pool, "peer")
        assert not r.on_request_arrival("serial", identity, 100, pool, "two")
        clock.return_value = 1.0
        assert not r.on_request_arrival("waiter", AgentIdentity("waiter"), 400, pool, "waiter")
        clock.return_value = 3.0
        r.on_stream_output("running", 50)
        r.schedule_cycle(pool)
        ref = r.program_ref_for_request("running")
        assert r.registry.require(ref).marked_for_pause
        assert r.strategy_factors.for_program(ref).tier is MORITier.GPU
        assert not r._apply_transition(TransitionRequest(TransitionKind.PAUSE, ref, "unsafe")).applied
        assert not r.on_request_arrival("later", identity, 150, pool, "three")
        r.on_request_completion("running", 400)
        assert r.registry.require(ref).state is ProgramState.PAUSED
        assert r.registry.require(ref).status is ProgramStatus.REASONING  # serial is waiting, not executing
        assert r.strategy_factors.for_program(ref).tier is MORITier.CPU
        r.schedule_cycle(pool)
        assert [entry.request_id for entry, _ in r.consume_admitted_requests()] == ["serial"]
        assert r.request_pool.waiting_request_ids == ("waiter", "later")


def test_cancel_deferred_request_does_not_end_running_lease():
    r, pool = runtime()
    identity = AgentIdentity("agent", lineage_id="main")
    assert r.on_request_arrival("one", identity, 400, pool, "one")
    assert not r.on_request_arrival("two", identity, 100, pool, "two")
    ref = r.program_ref_for_request("one")
    assert r.cancel_request("two").request_id == "two"
    assert r.registry.require(ref).status is ProgramStatus.REASONING
    assert r.registry.require(ref).tokens.estimated_context_tokens == 400
    assert not r._apply_transition(TransitionRequest(TransitionKind.PAUSE, ref, "unsafe")).applied


def test_cancel_confirmed_running_request_completes_pending_drain():
    r, pool = runtime(1024)
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert r.on_request_arrival("running", AgentIdentity("resident"), 400, pool, "one")
        assert r.on_request_arrival("peer", AgentIdentity("z-peer"), 400, pool, "peer")
        assert not r.on_request_arrival("waiting", AgentIdentity("waiting"), 400, pool, "two")
        clock.return_value = 2.0
        r.on_stream_output("running", 50)
        r.schedule_cycle(pool)
        ref = r.program_ref_for_request("running")
        assert r.registry.require(ref).marked_for_pause
        r.cancel_request("running")  # this runtime call is after backend cancellation confirmation
        assert r.registry.require(ref).state is ProgramState.PAUSED
        r.schedule_cycle(pool)
        assert [entry.request_id for entry, _ in r.consume_admitted_requests()] == ["waiting"]


def test_acting_placement_sticks_without_pressure_and_yields_to_waiting_inference():
    r, pool = runtime(512)
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert r.on_request_arrival("one", AgentIdentity("resident"), 400, pool, "one")
        ref = r.program_ref_for_request("one")
        clock.return_value = 0.1
        r.on_request_completion("one", 400)
        r.schedule_cycle(pool)
        assert r.strategy_factors.for_program(ref).tier is MORITier.GPU
        clock.return_value = 1000.0
        r.schedule_cycle(pool)
        assert r.strategy_factors.for_program(ref).tier is MORITier.GPU  # age alone does not demote
        assert not r.on_request_arrival("two", AgentIdentity("waiting"), 400, pool, "two")
        r.schedule_cycle(pool)
        assert [entry.request_id for entry, _ in r.consume_admitted_requests()] == ["two"]


def test_cancelled_waiter_cannot_leave_a_soft_drain_behind():
    r, pool = runtime(512)
    assert r.on_request_arrival("one", AgentIdentity("resident"), 400, pool, "one")
    assert not r.on_request_arrival("two", AgentIdentity("waiting"), 400, pool, "two")
    r.schedule_cycle(pool)
    ref = r.program_ref_for_request("one")
    assert not r.registry.require(ref).marked_for_pause
    r.cancel_request("two")
    r.schedule_cycle(pool)
    assert r.strategy_factors.for_program(ref).pending_tier is None


def test_late_terminal_receipt_does_not_release_newer_context_or_other_lineage():
    r, pool = runtime(2016)
    first = AgentIdentity("agent", lineage_id="main", lifecycle_token="first")
    assert r.on_request_arrival("one", first, 100, pool, "one")
    ref = r.program_ref_for_request("one")
    r.on_request_completion("one", 120)
    assert r.on_request_arrival("two", replace(first, lifecycle_token="second"), 200, pool, "two")
    r.on_request_completion("two", 220)
    assert not r.on_scoped_response_completion("agent", ProgramLifecycle.TERMINAL, "first")
    assert r.registry.get(ref) is not None
    assert r.on_scoped_response_completion("agent", ProgramLifecycle.TERMINAL, "second")
    assert r.registry.get(ref) is None


def test_family_terminal_waits_for_all_accepted_lineage_work():
    r, pool = runtime(2016)
    assert r.on_request_arrival("one", AgentIdentity("agent", lineage_id="a"), 100, pool, "one")
    assert r.on_request_arrival("two", AgentIdentity("agent", lineage_id="b"), 100, pool, "two")
    one = r.program_ref_for_request("one")
    two = r.program_ref_for_request("two")
    r.on_request_completion("one", 120)
    assert not r.on_response_completion("agent", ProgramLifecycle.TERMINAL)
    r.on_request_completion("two", 120)
    assert r.on_response_completion("agent", ProgramLifecycle.TERMINAL)
    assert r.registry.get(one) is None and r.registry.get(two) is None


def test_native_queue_and_preemption_wait_are_excluded_from_reasoning():
    r, pool = runtime()
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert r.on_request_arrival("one", AgentIdentity("agent"), 100, pool, "one")
        ref = r.program_ref_for_request("one")
        clock.return_value = 10.0
        r.on_native_execution("one", running=True)
        clock.return_value = 12.0
        r.on_native_execution("one", running=False)
        clock.return_value = 20.0
        r.on_native_execution("one", running=True)
        clock.return_value = 23.0
        r.on_request_completion("one", 120)
        assert r.strategy_factors.for_program(ref).pending_reasoning_seconds == 5.0


def test_finished_named_families_reuse_slots_without_unbounded_generation_keys():
    """Normal terminal lifecycles must not add one permanent UUID tombstone per request."""
    r, pool = runtime()
    previous = None
    for number in range(100):
        request_id = f"request-{number}"
        family = f"family-{number}"
        identity = AgentIdentity(family, lineage_id="main", lifecycle_token=request_id)
        assert r.on_request_arrival(request_id, identity, 100, pool, request_id)
        ref = r.program_ref_for_request(request_id)
        if previous is not None:
            assert ref.program_id == previous.program_id and ref.generation > previous.generation
            assert not r._apply_transition(TransitionRequest(TransitionKind.RELEASE, previous, "late")).applied
        r.on_request_completion(request_id, 120)
        assert r.on_scoped_response_completion(family, ProgramLifecycle.TERMINAL, request_id)
        previous = ref
    assert len(r.registry._last_generations) == 1
    assert len(r._lineages._internal_ids) == 1
    assert not r._lineages._named
