# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Current-cycle waiting credit through real scheduler and native execution events."""

from dataclasses import replace
from unittest.mock import patch

import pytest

from agentinfer.scheduling.backend import BackendInfo, BackendPoolInfo, DpRankInfo
from agentinfer.scheduling.domain import ProgramRef, ProgramState, ProgramStatus
from agentinfer.scheduling.identity import AgentIdentity
from agentinfer.scheduling.mori import MORIConfig, MORIProgramFactors, MORITier, build_mori_strategy
from agentinfer.scheduling.runtime import ProgramScheduler
from tests.scheduling.test_mori_strategy import accepting_controller, snapshot, view

pytestmark = pytest.mark.cpu_test


def pool(hbm=96, dram=96):
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
                        total_hbm_kv_tokens=hbm,
                        total_dram_kv_tokens=dram,
                        hbm_block_size_tokens=16,
                        dram_chunk_size_tokens=16,
                    ),
                ),
            ),
        )
    )


def runtime(window_size=3):
    components = build_mori_strategy(MORIConfig(window_size=window_size, decode_buffer_tokens=0))
    return ProgramScheduler(components.strategy, components.initial_factors, pool(), schedule_interval_seconds=1)


def test_waiting_survives_cpu_to_waiting_and_gpu_admission_but_never_enters_history():
    scheduler = runtime()
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert scheduler.on_request_arrival("first", AgentIdentity("p"), 96, pool(), "first")
        ref = scheduler.program_ref_for_request("first")
        scheduler.on_native_execution("first", running=True)
        clock.return_value = 2.0
        scheduler.on_request_completion("first", 96)
        clock.return_value = 100.0
        scheduler.schedule_cycle(pool(hbm=0))
        assert scheduler.strategy_factors.for_program(ref).tier is MORITier.CPU
        # Long tool time does not itself become waiting compensation.
        assert scheduler.strategy.idleness_for_program(ref, 100.0) == pytest.approx(98 / 100)

        clock.return_value = 101.0
        assert not scheduler.on_request_arrival("second", AgentIdentity("p"), 96, pool(hbm=0), "second")
        assert scheduler.strategy_factors.for_program(ref).cycles == ((2.0, 99.0),)
        assert scheduler.strategy.idleness_for_program(ref, 111.0) == pytest.approx(99 / 111)
        # Repeated reads/ticks do not accumulate the same elapsed wait again.
        for _ in range(2):
            clock.return_value = 111.0
            scheduler.schedule_cycle(pool(hbm=0, dram=0))
            assert scheduler.strategy.idleness_for_program(ref, 111.0) == pytest.approx(99 / 111)
        assert scheduler.strategy_factors.for_program(ref).tier is MORITier.WAITING
        assert scheduler.request_pool.waiting_request_ids == ("second",)

        clock.return_value = 121.0
        scheduler.schedule_cycle(pool())
        assert [entry.request_id for entry, _ in scheduler.consume_admitted_requests()] == ["second"]
        admitted = scheduler.strategy_factors.for_program(ref)
        assert admitted.reasoning_wait_seconds == 20.0
        assert admitted.reasoning_wait_started_at_monotonic_s is None
        assert scheduler.registry.require(ref).wait_started_at_monotonic_s is None
        assert scheduler.strategy.idleness_for_program(ref, 121.0) == pytest.approx(99 / 121)

        clock.return_value = 124.0
        scheduler.on_native_execution("second", running=True)
        clock.return_value = 126.0
        scheduler.on_native_execution("second", running=False)
        # Native preemption is neither real execution nor CPU/WAITING credit.
        assert scheduler.strategy.idleness_for_program(ref, 136.0) == pytest.approx(99 / 123)
        clock.return_value = 136.0
        scheduler.on_native_execution("second", running=True)
        clock.return_value = 139.0
        scheduler.on_request_completion("second", 96)
        completed = scheduler.strategy_factors.for_program(ref)
        assert completed.pending_reasoning_seconds == 5.0
        assert completed.reasoning_wait_seconds == 0.0
        assert completed.reasoning_wait_started_at_monotonic_s is None
        assert scheduler.strategy.idleness_for_program(ref, 139.0) == pytest.approx(99 / 106)

        clock.return_value = 149.0
        assert scheduler.on_request_arrival("third", AgentIdentity("p"), 96, pool(), "third")
        assert scheduler.strategy_factors.for_program(ref).cycles == ((2.0, 99.0), (5.0, 10.0))


@pytest.mark.parametrize("window,expected", [(1, 0.0), (2, 9 / 15), (3, 109 / 116)])
def test_waiting_current_cycle_uses_one_existing_window_slot(window, expected):
    components = build_mori_strategy(MORIConfig(window_size=window))
    owner = ProgramRef("waiting", 0)
    sidecar = MORIProgramFactors(
        tier=MORITier.CPU,
        cycles=((1.0, 100.0), (1.0, 9.0)),
        reasoning_wait_started_at_monotonic_s=10.0,
    )
    components.strategy._set_factor(components.initial_factors, owner, sidecar)
    assert components.strategy.idleness_for_program(owner, 15.0) == pytest.approx(expected)
    assert components.initial_factors.for_program(owner).cycles == ((1.0, 100.0), (1.0, 9.0))


@pytest.mark.parametrize("tier", [MORITier.CPU, MORITier.WAITING])
def test_uniform_waiting_score_selects_older_peer_and_matches_diagnostics(tier):
    components = build_mori_strategy(MORIConfig(decode_buffer_tokens=0))
    young = replace(
        view("a-young", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=96, step_count=1),
        wait_started_at_monotonic_s=99.0,
    )
    old = replace(young, ref=ProgramRef("z-old", 0), wait_started_at_monotonic_s=0.0)
    for program in (young, old):
        components.strategy._set_factor(
            components.initial_factors,
            program.ref,
            MORIProgramFactors(
                tier=tier,
                cycles=((10.0, 90.0),),
                reasoning_wait_started_at_monotonic_s=program.wait_started_at_monotonic_s,
            ),
        )
    fixed = snapshot(young, old, hbm=96)
    scores = [components.strategy.idleness_for_program(p.ref, 100.0) for p in (young, old)]
    assert scores[1] < scores[0]
    report = dict(components.strategy.diagnostics(fixed, components.initial_factors)[0].fields)
    assert report["mean_idleness"] == pytest.approx(sum(scores) / 2)
    calls = []
    components.strategy.schedule_resume(
        fixed, components.initial_factors, accepting_controller(components.strategy, components.initial_factors, calls)
    )
    assert [call.program for call in calls] == [old.ref]


def test_cancelled_cpu_wait_discards_credit_without_counting_wait_as_reasoning():
    scheduler = runtime()
    with patch("agentinfer.scheduling.runtime.time.monotonic", return_value=0.0) as clock:
        assert scheduler.on_request_arrival("first", AgentIdentity("p"), 96, pool(), "first")
        ref = scheduler.program_ref_for_request("first")
        clock.return_value = 2.0
        scheduler.on_request_completion("first", 96)
        clock.return_value = 3.0
        scheduler.schedule_cycle(pool(hbm=0))
        clock.return_value = 5.0
        assert not scheduler.on_request_arrival("second", AgentIdentity("p"), 96, pool(hbm=0), "second")
        clock.return_value = 105.0
        scheduler.cancel_request("second")
        factor = scheduler.strategy_factors.for_program(ref)
        assert scheduler.registry.require(ref).status is ProgramStatus.ACTING
        assert factor.tier is MORITier.CPU
        assert factor.reasoning_wait_seconds == 0.0
        assert factor.reasoning_wait_started_at_monotonic_s is None
        assert factor.pending_reasoning_seconds == 0.0
        clock.return_value = 110.0
        scheduler.on_request_arrival("third", AgentIdentity("p"), 96, pool(hbm=0), "third")
        assert scheduler.strategy_factors.for_program(ref).cycles == ((2.0, 3.0), (0.0, 5.0))


def test_reading_retired_owner_does_not_recreate_factors_or_classes():
    components = build_mori_strategy()
    stale = view("retired", state=ProgramState.PAUSED, status=ProgramStatus.ACTING, tokens=96)
    components.strategy.diagnostics(snapshot(stale), components.initial_factors)
    components.strategy.on_execution_changed(components.initial_factors, stale.ref, 100.0, running=True)
    assert components.initial_factors.for_program(stale.ref) is None
    assert stale.ref not in components.strategy._classes
