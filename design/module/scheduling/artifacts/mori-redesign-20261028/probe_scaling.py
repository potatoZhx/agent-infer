# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Reproduce instrumented planner CPU costs; no inference or GPU allocation."""

import json
import statistics
import time
from pathlib import Path

from agentinfer.scheduling import ProgramRef, ProgramState, ProgramStatus
from agentinfer.scheduling.mori import MORITier, build_mori_strategy
from agentinfer.scheduling.mori.factors import MORIProgramFactors
from agentinfer.scheduling.mori.strategy import MORIFactors
from tests.scheduling.test_mori_strategy import accepting_controller, set_tier, snapshot, view


def measure(waiters: int) -> tuple[float, int]:
    """Run one non-fitting waiter cycle and return thread milliseconds plus factor reads."""
    components = build_mori_strategy()
    resident = view("resident", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=900)
    pending = [
        view(f"w{i:04}", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100) for i in range(waiters)
    ]
    set_tier(components, resident, MORITier.GPU)
    for program in pending:
        set_tier(components, program, MORITier.WAITING)
    fixed = snapshot(resident, *pending, hbm=1000, dram=100000)
    reads = 0
    original = components.strategy._factor

    def counted(factors: MORIFactors, ref: ProgramRef) -> MORIProgramFactors:
        """Instrument all factor lookups, including those outside the ledger helper."""
        nonlocal reads
        reads += 1
        return original(factors, ref)

    components.strategy._factor = counted
    controller = accepting_controller(components.strategy, components.initial_factors, [])
    started = time.thread_time()
    components.strategy.schedule_resume(fixed, components.initial_factors, controller)
    return (time.thread_time() - started) * 1000, reads


def main() -> None:
    """Write every sample and its median beside this probe for reviewer inspection."""
    rows = []
    for count in (64, 128, 256, 512):
        samples = [measure(count) for _ in range(3)]
        rows.append(
            {
                "waiters": count,
                "median_thread_ms": statistics.median(sample[0] for sample in samples),
                "factor_reads": samples[0][1],
                "repetitions": len(samples),
                "samples": [{"thread_ms": duration, "factor_reads": reads} for duration, reads in samples],
            }
        )
    result = {
        "command": "PYTHONPATH=. python design/module/scheduling/artifacts/mori-redesign-20261028/probe_scaling.py",
        "scope": "Instrumented CPU scheduling only; no GPU/E2E claim",
        "scaling": rows,
    }
    encoded = json.dumps(result, indent=2) + "\n"
    Path(__file__).with_name("scaling.json").write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
