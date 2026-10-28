"""CPU-only audit probes; run from the repository root with PYTHONPATH=."""

import json

from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus

from agentinfer.agentcache.core.api_adapter import ApiEndpoint, _response_lifecycle
from agentinfer.agentcache.mori.hbm_policy import MORIHBMPolicy
from agentinfer.agentcache.mori.runtime_state import MORI_RUNTIME_STATE, MORIRuntimeState
from agentinfer.agentcache.mori_cpu_policy import MoriCPUCachePolicy
from agentinfer.scheduling import ProgramRef, ProgramState, ProgramStatus
from agentinfer.scheduling.identity import AgentIdentity
from agentinfer.scheduling.mori import MORIProgramFactors, MORITier, build_mori_strategy
from tests.scheduling.test_mori_branches import runtime
from tests.scheduling.test_mori_strategy import accepting_controller, set_tier, snapshot, view

results = {}

# Legacy connector's eviction call has no incoming protection class.
components = build_mori_strategy()
MORI_RUNTIME_STATE.bind_provider(components.strategy)
policy = MoriCPUCachePolicy(1)
idle, busy = ProgramRef("idle", 0), ProgramRef("busy", 0)
for ref, tier in ((idle, MORITier.CPU), (busy, MORITier.GPU)):
    MORI_RUNTIME_STATE.bind_request(ref.program_id, ref)
    components.strategy._set_factor(components.initial_factors, ref, MORIProgramFactors(tier=tier))
old, new = make_offload_key(b"old", 0), make_offload_key(b"new", 0)
policy.on_store_miss((old,), ReqContext("idle"))
chunk = ChunkStatus(0)
chunk.ref_cnt = 0
policy.insert(old, chunk)
policy.on_store_miss((new,), ReqContext("busy"))
victims = policy.evict(1, set())
results["legacy_cpu_policy"] = {
    "incoming_tier": "gpu",
    "resident_tier": "cpu",
    "idle_chunk_evicted": victims == [(old, chunk)],
    "scope": "real policy API, no DMA or full connector execution",
}
assert results["legacy_cpu_policy"]["idle_chunk_evicted"]
policy.close()
MORI_RUNTIME_STATE.reset()

# A model response's stop currently destroys Program continuity.
r, pool = runtime()
identity = AgentIdentity("session", lifecycle_token="attempt-1")
assert r.on_request_arrival("one", identity, 100, pool, "one")
ref = r.program_ref_for_request("one")
r.on_request_completion("one", 120)
lifecycle = _response_lifecycle(ApiEndpoint.OPENAI_CHAT_COMPLETIONS, {"choices": [{"finish_reason": "stop"}]})
released = r.on_scoped_response_completion("session", lifecycle, "attempt-1")
assert r.on_request_arrival("two", AgentIdentity("session"), 130, pool, "two")
next_ref = r.program_ref_for_request("two")
results["response_stop"] = {
    "mapped_lifecycle": lifecycle.value,
    "released_program": released,
    "old_generation": ref.generation,
    "next_generation": next_ref.generation,
    "old_factors_removed": r.strategy_factors.for_program(ref) is None,
}
assert released and next_ref != ref

# Prompt replacement clears shared-footprint evidence but not typed protection.
components = build_mori_strategy()
state = MORIRuntimeState()
state.bind_provider(components.strategy)
state.footprints.enabled = True
hbm = MORIHBMPolicy(state)
owner = ProgramRef("owner", 0)
state.bind_request("request", owner)
components.strategy._set_factor(components.initial_factors, owner, MORIProgramFactors(tier=MORITier.GPU))
hbm.on_block_owner_added(7, "request")
hbm.on_block_ready(7, 1)
components.strategy.on_context_replaced(owner)
results["prompt_replacement"] = {
    "footprint_blocks_after_replace": sorted(state.footprint(owner, "gpu")),
    "old_block_still_has_owner": owner in hbm.index.owners(7),
    "old_block_class": hbm.index.cache_class(7).value,
    "scope": "old prompt block protection, native contents are not modified",
}
assert not state.footprint(owner, "gpu") and owner in hbm.index.owners(7)
hbm.close()
state.reset()

# Waiting inference can displace a much busier Acting resident below quota.
components = build_mori_strategy()
resident = view("resident", state=ProgramState.ACTIVE, status=ProgramStatus.ACTING, tokens=700)
candidate = view("candidate", state=ProgramState.PAUSED, status=ProgramStatus.REASONING, tokens=400, step_count=1)
set_tier(components, resident, MORITier.GPU, idleness=0.01)
set_tier(components, candidate, MORITier.CPU, idleness=0.99)
calls = []
s = snapshot(resident, candidate, hbm=1000, dram=2000)
gpu_before = components.strategy._ledgers(s, components.initial_factors)[0].used
components.strategy.schedule_resume(
    s, components.initial_factors, accepting_controller(components.strategy, components.initial_factors, calls)
)
results["waiting_pressure_swap"] = {
    "gpu_commitment_before": gpu_before,
    "gpu_quota": 1000,
    "resident_idleness": 0.01,
    "candidate_idleness": 0.99,
    "actions": [{"kind": call.kind.value, "owner": call.program.program_id, "reason": call.reason} for call in calls],
}
assert gpu_before < 1000 and len(calls) == 2

# Admission starts a provisional Reasoning clock before the first native batch.
components = build_mori_strategy()
ref = ProgramRef("awaiting-first-native-batch", 0)
sidecar = MORIProgramFactors(tier=MORITier.GPU, cycles=((1.0, 9.0),), reasoning_started_at_monotonic_s=0.0)
components.strategy._set_factor(components.initial_factors, ref, sidecar)
before = components.strategy.idleness_for_program(ref, 0.0)
queued = components.strategy.idleness_for_program(ref, 100.0)
components.strategy.on_execution_changed(components.initial_factors, ref, 100.0, running=True)
started = components.strategy.idleness_for_program(ref, 100.0)
results["before_first_native_execution"] = {
    "idleness_at_admission": before,
    "idleness_after_100_seconds_queued": queued,
    "idleness_when_execution_starts": started,
    "scope": "strategy clock before first native execution notification",
}
assert before == started == 0.9 and queued < before

print(json.dumps(results, indent=2))
