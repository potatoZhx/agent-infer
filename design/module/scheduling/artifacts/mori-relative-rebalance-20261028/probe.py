# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""CPU-only policy/scaling observations; not an inference performance benchmark."""

import json
from unittest.mock import patch

from agentinfer.scheduling.domain import ProgramState, ProgramStatus
from agentinfer.scheduling.mori import MORITier, build_mori_strategy
from agentinfer.scheduling.mori.capacity import CapacityLedger, ReliefProjection
from tests.scheduling.test_mori_strategy import accepting_controller, set_tier, snapshot, view


def exchange(resident_idle, candidate_idle):
    components = build_mori_strategy()
    resident = view("resident", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=700)
    candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=400, step_count=1)
    set_tier(components, resident, MORITier.GPU, idleness=resident_idle)
    set_tier(components, candidate, MORITier.CPU, idleness=candidate_idle)
    calls = []
    fixed = snapshot(resident, candidate, hbm=1000, dram=2000)
    components.strategy.schedule_resume(
        fixed, components.initial_factors, accepting_controller(components.strategy, components.initial_factors, calls)
    )
    return {
        "resident_idleness": resident_idle,
        "candidate_idleness": candidate_idle,
        "transitions": [
            {"kind": call.kind.value, "program": call.program.program_id, "reason": call.reason} for call in calls
        ],
        "final_gpu_commitment": components.strategy._ledgers(fixed, components.initial_factors)[0].used,
    }


def no_fit_scaling(size):
    components = build_mori_strategy()
    victims = [
        view(f"victim-{i}", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=1) for i in range(size)
    ]
    running = view("running", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=100 * size)
    waiters = [
        view(f"wait-{i}", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=101 * size)
        for i in range(size)
    ]
    for program in (*victims, running):
        set_tier(components, program, MORITier.GPU, idleness=0.9)
    calls = []
    fixed = snapshot(*victims, running, *waiters, hbm=201 * size + 100, dram=1000 * size)
    with (
        patch.object(components.strategy, "_factor", wraps=components.strategy._factor) as reads,
        patch.object(CapacityLedger, "remove", autospec=True, side_effect=CapacityLedger.remove) as removes,
        patch.object(ReliefProjection, "required", autospec=True, side_effect=ReliefProjection.required) as queries,
    ):
        components.strategy.schedule_resume(
            fixed,
            components.initial_factors,
            accepting_controller(components.strategy, components.initial_factors, calls),
        )
    assert not calls
    return {
        "candidates": size,
        "acting_victims": size,
        "reasoning_residents": 1,
        "factor_reads": reads.call_count,
        "projected_owner_removals": removes.call_count,
        "projection_queries": queries.call_count,
        "actual_transitions": len(calls),
        "counters": dict(components.strategy.counters),
    }


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "scope": "CPU policy and operation counts; no DMA, native inference, or latency measurement",
                "exchanges": [exchange(0.01, 0.99), exchange(0.9, 0.1), exchange(0.5, 0.5)],
                "no_fit_scaling": [no_fit_scaling(size) for size in (64, 128, 256, 512, 1024)],
            },
            indent=2,
        )
    )
