# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Independent union arithmetic and planner scaling checks."""

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
from agentinfer.scheduling.transitions import TransitionKind
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


def test_capacity_repair_uses_marginal_shared_relief_and_keeps_lazy_debt():
    components = build_mori_strategy()
    a = view("a", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=400)
    b = view("b", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=400)
    waiting = view("wait", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=100)

    class Shared:
        def footprint(self, ref, tier):
            return frozenset(range(1, 301)) if tier == "gpu" and ref in (a.ref, b.ref) else frozenset()

    components.strategy.bind_footprints(Shared())
    set_tier(components, a, MORITier.GPU, idleness=0.9)
    set_tier(components, b, MORITier.GPU, idleness=0.8)
    calls = []
    fixed = snapshot(a, b, waiting, hbm=350, dram=1000)
    controller = accepting_controller(components.strategy, components.initial_factors, calls)
    components.strategy.repair_capacity(fixed, components.initial_factors, controller)
    components.strategy.schedule_resume(fixed, components.initial_factors, controller)
    # Actual 400-token contexts cannot be capped below their footprint. A
    # releases only 100 tokens; the shared 300 remain owned by executing B.
    assert [(call.kind, call.program) for call in calls] == [
        (TransitionKind.PAUSE, a.ref),
        (TransitionKind.MARK_FOR_PAUSE, b.ref),
    ]
    assert components.initial_factors.for_program(b.ref).tier is MORITier.GPU
    assert components.initial_factors.for_program(b.ref).pending_tier is MORITier.CPU
    assert components.strategy._ledgers(fixed, components.initial_factors)[0].used == 400


@pytest.mark.parametrize("quota", [1200, 1600])
def test_shared_candidate_uses_only_existing_slack_without_replacing_owners(quota):
    components = build_mori_strategy()
    a = view("a", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=400)
    b = view("b", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=400)
    spare = view("spare", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=100)
    running = view("running", state=ProgramState.ACTIVE, status=ProgramStatus.REASONING, tokens=100)
    waiting = view("wait", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=700, step_count=1)

    class Shared:
        def footprint(self, ref, tier):
            return frozenset(range(1, 301)) if tier == "gpu" and ref in (a.ref, b.ref, waiting.ref) else frozenset()

    components.strategy.bind_footprints(Shared())
    for program in (a, b, spare, running):
        set_tier(components, program, MORITier.GPU, idleness=0.9)
    set_tier(components, waiting, MORITier.CPU, idleness=0.1)
    calls = []
    fixed = snapshot(a, b, spare, running, waiting, hbm=quota, dram=2000)
    components.strategy.schedule_resume(
        fixed, components.initial_factors, accepting_controller(components.strategy, components.initial_factors, calls)
    )
    expected = [] if quota == 1200 else [(TransitionKind.RESUME, waiting.ref)]
    assert [(call.kind, call.program) for call in calls] == expected
    for program in (a, b, spare, running):
        assert components.initial_factors.for_program(program.ref).tier is MORITier.GPU
        assert components.initial_factors.for_program(program.ref).pending_tier is None
    assert components.strategy._ledgers(fixed, components.initial_factors)[0].used == (1100 if quota == 1200 else 1600)


def test_non_fitting_candidates_do_not_rescan_or_demote_acting_residents():
    counts = []
    for size in (64, 128, 256, 512):
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
        with patch.object(components.strategy, "_factor", wraps=components.strategy._factor) as factor_reads:
            components.strategy.schedule_resume(
                snapshot(*victims, running, *waiters, hbm=201 * size + 100, dram=1000 * size),
                components.initial_factors,
                accepting_controller(components.strategy, components.initial_factors, calls),
            )
        counts.append(factor_reads.call_count)
        assert components.strategy.counters["planning_candidates"] == size
        assert not calls  # No candidate fits the existing slack.
    assert all(after < before * 2.1 for before, after in zip(counts, counts[1:], strict=False))


@pytest.mark.parametrize("seed", range(50))
def test_slack_promotions_match_brute_force_union_planner(seed):
    random = Random(912 + seed)
    components = build_mori_strategy()
    programs, claims, scores, tiers = {}, {}, {}, {}
    for index in range(20):
        gpu = index < 10
        program = view(
            str(index),
            state=ProgramState.ACTIVE if gpu else ProgramState.PAUSED,
            status=ProgramStatus.ACTING if index < 8 else ProgramStatus.REASONING,
            tokens=200 + random.randrange(400),
            step_count=random.randrange(2),
        )
        ref = program.ref
        programs[ref] = program
        blocks = frozenset(random.sample(range(1, 301), random.randrange(200)))
        claims[ref] = CapacityClaim(program.tokens.estimated_context_tokens + 100, blocks)
        scores[ref] = random.choice((0.0, 0.1, 0.5, 0.9, 1.0))
        tiers[ref] = MORITier.GPU if gpu else random.choice((MORITier.CPU, MORITier.WAITING))
        set_tier(components, program, tiers[ref], idleness=scores[ref])

    class Shared:
        def footprint(self, ref, tier):
            return claims[ref].blocks if tier == "gpu" else frozenset()

    components.strategy.bind_footprints(Shared())

    def union_tokens(refs):
        blocks = set().union(*(claims[ref].blocks for ref in refs))
        return len(blocks) + sum(claims[ref].tokens - len(claims[ref].blocks) for ref in refs)

    active = {ref for ref, tier in tiers.items() if tier is MORITier.GPU}
    quota = union_tokens(active) + random.randrange(400)

    def promotion_key(ref):
        group = 0 if tiers[ref] is MORITier.CPU else 1 if programs[ref].step_count else 2
        return group, claims[ref].tokens if group == 2 else 0, scores[ref], ref

    candidates = sorted(set(programs) - active, key=promotion_key)
    expected = []
    for candidate in candidates:
        projected = active | {candidate}
        if union_tokens(projected) <= quota:
            kind = TransitionKind.RESUME if programs[candidate].step_count else TransitionKind.ADMIT
            expected.append((kind, candidate))
            active = projected
    calls = []
    fixed = snapshot(*programs.values(), hbm=quota, dram=100_000)
    components.strategy.schedule_resume(
        fixed,
        components.initial_factors,
        accepting_controller(components.strategy, components.initial_factors, calls),
    )
    assert [(call.kind, call.program) for call in calls] == expected
    assert components.strategy._ledgers(fixed, components.initial_factors)[0].used == union_tokens(active)


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
