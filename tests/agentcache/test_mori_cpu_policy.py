# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Native CachePolicy contract tests for MORI CPU typed replacement."""

from __future__ import annotations

import pytest
from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus

from agentinfer.agentcache.mori.runtime_state import MORI_RUNTIME_STATE
from agentinfer.agentcache.mori_cpu_policy import MoriCPUCachePolicy
from agentinfer.scheduling import ProgramRef
from agentinfer.scheduling.mori import MORIProgramFactors, MORITier, build_mori_strategy

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def reset_runtime_state():
    MORI_RUNTIME_STATE.reset()
    yield
    MORI_RUNTIME_STATE.reset()


def test_cpu_policy_resolves_server_owners_and_uses_typed_order() -> None:
    components = build_mori_strategy()
    MORI_RUNTIME_STATE.bind_provider(components.strategy)
    policy = MoriCPUCachePolicy(3)
    programs = {
        MORITier.WAITING: ProgramRef("waiting", 0),
        MORITier.GPU: ProgramRef("gpu", 0),
        MORITier.CPU: ProgramRef("cpu", 0),
    }
    inserted = []

    for chunk_id, (tier, program) in enumerate(programs.items()):
        request_id = f"request-{tier.value}"
        key = make_offload_key(f"block-{tier.value}".encode(), 0)
        MORI_RUNTIME_STATE.bind_request(request_id, program)
        components.strategy._set_factor(
            components.initial_factors,
            program,
            MORIProgramFactors(tier=tier),
        )
        policy.on_store_miss((key,), ReqContext(request_id))
        chunk = ChunkStatus(chunk_id)
        chunk.ref_cnt = 0
        policy.insert(key, chunk)
        inserted.append((key, chunk))

    waiting = programs[MORITier.WAITING]
    assert policy.release_program(waiting) == 1
    assert policy.index.owners(inserted[0][0]) == frozenset()
    assert policy._owners[inserted[0][0]] == set()
    assert policy.evict(3, set()) == inserted
    policy.close()
    MORI_RUNTIME_STATE.unbind_provider(components.strategy)
