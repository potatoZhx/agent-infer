# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Runtime-level MORI identity, token, and overlap contracts."""

from __future__ import annotations

import pytest

from agentinfer.scheduling.backend import BackendInfo, BackendPoolInfo, DpRankInfo
from agentinfer.scheduling.domain import TokenObservationSource
from agentinfer.scheduling.identity import AgentIdentity
from agentinfer.scheduling.mori import MORIConfig, MORIMode, build_mori_strategy
from agentinfer.scheduling.runtime import ProgramScheduler

pytestmark = pytest.mark.cpu_test


def backend(capacity: int = 10_000) -> BackendPoolInfo:
    return BackendPoolInfo(
        (
            BackendInfo(
                "vllm-local",
                "embedded://vllm",
                True,
                (
                    DpRankInfo(
                        0,
                        True,
                        True,
                        total_hbm_kv_tokens=capacity,
                        total_dram_kv_tokens=capacity,
                        hbm_block_size_tokens=16,
                        dram_chunk_size_tokens=16,
                    ),
                ),
            ),
        )
    )


def scheduler() -> ProgramScheduler[str, object, object]:
    components = build_mori_strategy(MORIConfig(mode=MORIMode.FULL))
    return ProgramScheduler(
        components.strategy,
        components.initial_factors,
        backend(),
        schedule_interval_seconds=5,
    )


def test_server_observed_stream_and_final_usage_replace_prompt_estimate() -> None:
    runtime = scheduler()
    assert runtime.on_request_arrival("request", AgentIdentity("program"), 100, backend(), "native") is True
    ref = runtime.program_ref_for_request("request")
    assert ref is not None

    runtime.on_stream_output("request", 30)
    streamed = runtime.registry.require(ref).tokens
    assert streamed.estimated_context_tokens == 130
    assert streamed.source is TokenObservationSource.MIXED

    runtime.on_request_completion("request", 155)
    completed = runtime.registry.require(ref).tokens
    assert completed.estimated_context_tokens == 155
    assert completed.source is TokenObservationSource.BACKEND_USAGE


def test_same_program_cannot_overlap_but_subagent_is_a_distinct_program() -> None:
    runtime = scheduler()
    lead = AgentIdentity("task:lead", task_id="task", agent_id="lead", agent_role="lead")
    child = AgentIdentity(
        "task:child",
        task_id="task",
        agent_id="child",
        agent_role="subagent",
        parent_program_id="task:lead",
        blocks_parent=True,
    )

    assert runtime.on_request_arrival("lead-1", lead, 100, backend(), "lead") is True
    with pytest.raises(ValueError, match="forbids overlapping requests"):
        runtime.on_request_arrival(
            "lead-2",
            AgentIdentity("task:lead", expected_resume=False),
            100,
            backend(),
            "duplicate",
        )
    assert runtime.on_request_arrival("child-1", child, 100, backend(), "child") is True

    lead_ref = runtime.program_ref_for_request("lead-1")
    child_ref = runtime.program_ref_for_request("child-1")
    assert lead_ref is not None and child_ref is not None
    assert lead_ref != child_ref
    assert lead_ref.program_id == "task:lead"
    assert child_ref.program_id == "task:child"
    assert runtime.registry.require(lead_ref).expected_resume is True
