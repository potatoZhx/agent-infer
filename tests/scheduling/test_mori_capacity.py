# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Independent union arithmetic and planner scaling checks."""

from dataclasses import replace
from random import Random
from unittest.mock import patch

import pytest

from agentinfer.agentcache.mori.footprints import FootprintBook
from agentinfer.agentcache.mori.hbm_policy import MORIHBMPolicy
from agentinfer.agentcache.mori.runtime_state import MORIRuntimeState
from agentinfer.agentcache.mori.simple_cpu_policy import MORISimpleCPUOffloadPolicy
from agentinfer.scheduling.domain import ProgramRef, ProgramState, ProgramStatus
from agentinfer.scheduling.mori import MORITier, build_mori_strategy
from agentinfer.scheduling.mori.capacity import CapacityClaim, CapacityLedger
from tests.scheduling.test_mori_strategy import accepting_controller, set_tier, snapshot, view

pytestmark = pytest.mark.cpu_test


def test_union_and_marginal_reclaim_are_not_max_or_full_context_sum():
    a, b = ProgramRef("a", 0), ProgramRef("b", 0)
    ledger = CapacityLedger(16)
    ledger.add(a, CapacityClaim(40 * 16, frozenset(range(1, 41))))
    ledger.add(b, CapacityClaim(30 * 16, frozenset(range(21, 51))))
    assert ledger.used == 50 * 16
    assert ledger.remove(a) == 20 * 16
    assert ledger.used == 30 * 16


def test_union_ledger_matches_brute_force_with_private_growth_promises():
    random = Random(178)
    ledger = CapacityLedger(16)
    active = {}
    for _ in range(500):
        ref = ProgramRef(str(random.randrange(20)), 0)
        if random.randrange(3) == 0:
            ledger.remove(ref)
            active.pop(ref, None)
        else:
            keys = frozenset(random.sample(range(1, 100), random.randrange(20)))
            private = random.randrange(10) * 16
            claim = CapacityClaim(len(keys) * 16 + private, keys)
            ledger.add(ref, claim)
            active[ref] = claim
        union = set().union(*(claim.blocks for claim in active.values()))
        expected = len(union) * 16 + sum(claim.tokens - len(claim.blocks) * 16 for claim in active.values())
        assert ledger.used == expected


def test_recycled_block_and_new_prompt_invalidate_only_relevant_proof():
    book = FootprintBook()
    book.enabled = True
    a, b = ProgramRef("a", 0), ProgramRef("b", 0)
    for owner in (a, b):
        book.add("gpu", 1, owner)
    book.add("cpu", 1, a)
    book.remove_block("gpu", 1)
    assert book.get(a, "gpu") == book.get(b, "gpu") == frozenset()
    assert book.get(a, "cpu") == frozenset({1})
    book.add("gpu", 1, b)  # new contents, old owner proof cannot survive reuse
    book.invalidate(a)
    assert book.get(b, "gpu") == frozenset({1})
    assert not book.get(a, "cpu")


@pytest.mark.parametrize(("policy_type", "tier"), [(MORIHBMPolicy, "gpu"), (MORISimpleCPUOffloadPolicy, "cpu")])
def test_native_ownership_reestablishes_proof_after_prompt_change_and_clears_on_reuse(policy_type, tier):
    """The native ownership delta supplies proof even if the LRU owner was already known."""
    state = MORIRuntimeState()
    state.footprints.enabled = True
    owner = ProgramRef("agent", 0)
    state.bind_request("request", owner)
    policy = policy_type(state)
    policy.on_block_owner_added(8, "request")
    assert state.footprint(owner, tier) == frozenset({8})
    state.invalidate_footprint(owner)
    policy.on_block_owner_added(8, "request")
    assert state.footprint(owner, tier) == frozenset({8})
    policy.on_block_removed(8)
    assert not state.footprint(owner, tier)
    policy.on_block_owner_added(8, "request")
    policy.on_cache_reset()
    assert not state.footprint(owner, tier)
    policy.close()


def test_planner_does_not_claim_shared_blocks_as_full_victim_relief():
    components = build_mori_strategy()
    a = view("a", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=400)
    b = view("b", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=400)
    waiting = view("wait", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=400)
    waiting = replace(waiting, wait_started_at_monotonic_s=0.0)

    class Shared:
        def footprint(self, ref, tier):
            return frozenset(range(1, 401)) if tier == "gpu" and ref in (a.ref, b.ref) else frozenset()

    components.strategy.bind_footprints(Shared())
    set_tier(components, a, MORITier.GPU, idleness=0.9)
    set_tier(components, b, MORITier.GPU, idleness=0.8)
    calls = []
    fixed = snapshot(a, b, waiting, hbm=700, dram=1000)
    components.strategy.schedule_resume(
        fixed, components.initial_factors, accepting_controller(components.strategy, components.initial_factors, calls)
    )
    # Removing A frees only its private decode reserve; B must drain too.
    assert [call.program for call in calls] == [a.ref, b.ref]
    assert components.initial_factors.for_program(b.ref).tier is MORITier.GPU
    assert components.initial_factors.global_factors.relief_candidate == waiting.ref


def test_non_fitting_waiter_scan_grows_linearly_in_program_reads():
    counts = []
    for size in (64, 128, 256, 512):
        components = build_mori_strategy()
        resident = view("resident", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=900)
        waiters = [
            view(str(i), state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100) for i in range(size)
        ]
        set_tier(components, resident, MORITier.GPU)
        calls = []
        with patch.object(components.strategy, "_factor", wraps=components.strategy._factor) as factor_reads:
            components.strategy.schedule_resume(
                snapshot(resident, *waiters),
                components.initial_factors,
                accepting_controller(components.strategy, components.initial_factors, calls),
            )
        counts.append(factor_reads.call_count)
        assert components.strategy.counters["planning_candidates"] == size
        assert not calls
    assert all(after < before * 2.1 for before, after in zip(counts, counts[1:], strict=False))
