"""Replay read-only captured placement facts without a server or GPU."""

import ast
import json
import sys
from dataclasses import asdict
from pathlib import Path

from agentinfer.scheduling.domain import ProgramRef, ProgramState, ProgramStatus, ProgramTokenObservation, ProgramView
from agentinfer.scheduling.mori import MORIProgramFactors, MORITier, build_mori_strategy
from agentinfer.scheduling.mori.factors import MORIGlobalFactors
from agentinfer.scheduling.snapshot import SchedulingSnapshot
from tests.scheduling.test_mori_strategy import accepting_controller


def ref(value):
    return ProgramRef(**value) if value else None


raw = json.loads(Path(sys.argv[1]).read_text())
components = build_mori_strategy()
strategy, factors = components.strategy, components.initial_factors
programs = []
for value in raw["programs"].values():
    fields = {key: val for key, val in value.items() if key in ProgramView.__dataclass_fields__}
    fields.update(
        ref=ref(value["ref"]),
        state=ProgramState(value["state"]),
        status=ProgramStatus(value["status"]),
        tokens=ProgramTokenObservation(estimated_context_tokens=value["tokens"]["estimated_context_tokens"]),
    )
    programs.append(ProgramView(**fields))
for key, value in raw["factors"]["_program_factors"].items():
    value.update(
        tier=MORITier(value["tier"]),
        pending_tier=MORITier(value["pending_tier"]) if value["pending_tier"] else None,
        relief_for=ref(value["relief_for"]),
    )
    value = {k: v for k, v in value.items() if k in MORIProgramFactors.__dataclass_fields__}
    strategy._set_factor(factors, ref(ast.literal_eval(key)), MORIProgramFactors(**value))
global_facts = raw["factors"]["_global_factors"]
global_facts["relief_candidate"] = ref(global_facts["relief_candidate"])
old_beneficiary = global_facts["relief_candidate"]
global_facts = {k: v for k, v in global_facts.items() if k in MORIGlobalFactors.__dataclass_fields__}
factors.set_global_factors(MORIGlobalFactors(**global_facts))
footprints = {}
for key, blocks in raw["footprints"].items():
    tier, owner = ast.literal_eval(key)
    footprints[tier, ref(owner)] = frozenset(blocks)


class Evidence:
    def footprint(self, owner, tier):
        return footprints.get((tier, owner), frozenset())


strategy.bind_footprints(Evidence())
snapshot = SchedulingSnapshot(
    observed_at_monotonic_s=7170000.0,
    backend_id="vllm-local",
    total_kv_tokens=122192,
    total_dram_kv_tokens=174752,
    hbm_block_size_tokens=16,
    dram_chunk_size_tokens=16,
    programs=tuple(programs),
    waiting_programs=tuple(p.ref for p in programs if p.state is ProgramState.PAUSED),
)
gpu, _ = strategy._ledgers(snapshot, factors)
beneficiary = snapshot.program(old_beneficiary)
claim = strategy._claim(beneficiary, snapshot, gpu=True)
calls = []
strategy.schedule_resume(snapshot, factors, accepting_controller(strategy, factors, calls))
print(
    json.dumps(
        {
            "native_requests": len(raw["native_requests"]),
            "retained_bindings": len(raw["bindings"]),
            "gpu_commitment_tokens": gpu.used,
            "quota_tokens": snapshot.total_kv_tokens,
            "beneficiary": asdict(beneficiary.ref),
            "beneficiary_claim_tokens": claim.tokens,
            "incremental_beneficiary_tokens": gpu.incremental(claim),
            "deficit_tokens": gpu.used + gpu.incremental(claim) - snapshot.total_kv_tokens,
            "pending_gpu_victims": sum(
                f.pending_tier is not None and f.tier is MORITier.GPU for _, f in factors.program_factors
            ),
            "transitions": [asdict(call) for call in calls],
            "counters": dict(strategy.counters),
        },
        indent=2,
    )
)
