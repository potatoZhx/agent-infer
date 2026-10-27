# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Capability and cleanup tests for MORI's vLLM integration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.request import RequestStatus

from agentinfer.agentcache.core.scheduler import (
    AgentCacheAsyncSchedulerBridge,
    AgentCacheSyncSchedulerBridge,
    _VllmAdmissionHooks,
)
from agentinfer.agentcache.mori.runtime_state import MORI_RUNTIME_STATE
from agentinfer.agentcache.mori.vllm_adapter import install_vllm_mori, observe_vllm_offload
from agentinfer.scheduling import ProgramRef
from agentinfer.scheduling.mori import MORIConfig, MORIMode, build_mori_strategy

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def reset_runtime_state():
    MORI_RUNTIME_STATE.reset()
    yield
    MORI_RUNTIME_STATE.reset()


def test_placement_only_adapter_can_close_and_rebind_in_process() -> None:
    first = build_mori_strategy(MORIConfig(mode=MORIMode.PLACEMENT_ONLY)).strategy
    second = build_mori_strategy(MORIConfig(mode=MORIMode.PLACEMENT_ONLY)).strategy
    owner = SimpleNamespace(kv_cache_manager=None)

    adapter = install_vllm_mori(owner, first)
    MORI_RUNTIME_STATE.bind_request("shutdown-request", ProgramRef("shutdown-program", 0))
    with pytest.raises(RuntimeError, match="already bound"):
        install_vllm_mori(owner, second)
    adapter.close()
    assert MORI_RUNTIME_STATE.program_for_request("shutdown-request") is None
    rebound = install_vllm_mori(owner, second)
    rebound.close()


def test_failed_mori_startup_does_not_install_prefix_observer() -> None:
    """Failed capability negotiation must leave the native lookup unchanged."""

    def lookup(request):
        return None

    owner = SimpleNamespace(kv_cache_manager=SimpleNamespace(get_computed_blocks=lookup))
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_index=0),
        additional_config={"agentcache": {"strategy": "mori", "mori": {"mode": "placement_only"}}},
    )
    with patch("agentinfer.agentcache.core.scheduler.install_vllm_mori", side_effect=RuntimeError("missing hook")):
        with pytest.raises(RuntimeError, match="missing hook"):
            _VllmAdmissionHooks(owner, config, SimpleNamespace(num_blocks=10), 16)
    assert owner.kv_cache_manager.get_computed_blocks is lookup


def test_full_adapter_rejects_missing_hbm_hook_without_leaking_provider() -> None:
    full = build_mori_strategy(MORIConfig(mode=MORIMode.FULL)).strategy
    placement = build_mori_strategy(MORIConfig(mode=MORIMode.PLACEMENT_ONLY)).strategy

    class CPUPolicy:
        MORI_CPU_POLICY = True

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    cpu_policy = CPUPolicy()
    owner = SimpleNamespace(
        connector=SimpleNamespace(connector_scheduler=SimpleNamespace(manager=SimpleNamespace(_policy=cpu_policy))),
        kv_cache_manager=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="register_mori_block_selection_provider"):
        install_vllm_mori(owner, full)
    assert cpu_policy.closed is True

    adapter = install_vllm_mori(SimpleNamespace(kv_cache_manager=None), placement)
    adapter.close()


def test_full_adapter_unregisters_hbm_and_cpu_indexes_on_close() -> None:
    full = build_mori_strategy(MORIConfig(mode=MORIMode.FULL)).strategy
    calls: list[tuple[str, object]] = []

    class CPUPolicy:
        MORI_CPU_POLICY = True

        def close(self) -> None:
            calls.append(("cpu_close", self))

    cpu_policy = CPUPolicy()

    class KVManager:
        def register_mori_block_selection_provider(self, provider):
            calls.append(("register", provider))
            return {"hook_api_version": 1}

        def unregister_mori_block_selection_provider(self, provider):
            calls.append(("unregister", provider))

    owner = SimpleNamespace(
        connector=SimpleNamespace(connector_scheduler=SimpleNamespace(manager=SimpleNamespace(_policy=cpu_policy))),
        kv_cache_manager=KVManager(),
    )

    adapter = install_vllm_mori(owner, full)
    adapter.close()
    adapter.close()

    assert [name for name, _ in calls] == ["register", "unregister", "cpu_close"]


def test_full_adapter_installs_and_uninstalls_simple_cpu_provider() -> None:
    full = build_mori_strategy(MORIConfig(mode=MORIMode.FULL)).strategy
    calls: list[tuple[str, object]] = []

    class SimpleCPUScheduler:
        def register_mori_offload_tier_provider(self, version, provider):
            calls.append(("cpu_register", (version, provider)))
            return {"hook_api_version": version}

        def unregister_mori_offload_tier_provider(self, provider):
            calls.append(("cpu_unregister", provider))

    class KVManager:
        def register_mori_block_selection_provider(self, provider):
            calls.append(("hbm_register", provider))
            return {"hook_api_version": provider.HOOK_API_VERSION}

        def unregister_mori_block_selection_provider(self, provider):
            calls.append(("hbm_unregister", provider))

    owner = SimpleNamespace(
        connector=SimpleNamespace(scheduler_manager=SimpleCPUScheduler()),
        kv_cache_manager=KVManager(),
    )

    adapter = install_vllm_mori(owner, full)
    cpu_policy = adapter.cpu_policy
    assert getattr(cpu_policy, "HOOK_API_VERSION", None) == 1
    adapter.close()
    adapter.close()

    assert [name for name, _ in calls] == [
        "cpu_register",
        "hbm_register",
        "hbm_unregister",
        "cpu_unregister",
    ]


def test_simple_cpu_snapshot_is_normalized_to_tokens() -> None:
    snapshot = SimpleNamespace(
        total_blocks=100,
        resident_blocks=40,
        pending_store_blocks=3,
        pending_load_blocks=2,
        scheduler_block_size=16,
    )
    owner = SimpleNamespace(
        connector=SimpleNamespace(
            scheduler_manager=SimpleNamespace(
                get_mori_offload_tier_snapshot=lambda: snapshot,
            )
        )
    )

    observed = observe_vllm_offload(owner)

    assert observed.total_dram_kv_tokens == 1600
    assert observed.used_dram_kv_tokens == 640
    assert observed.pending_d2h_kv_tokens == 48
    assert observed.pending_h2d_kv_tokens == 32
    assert observed.dram_chunk_size_tokens == 16


def test_simple_cpu_snapshot_rejects_impossible_residency() -> None:
    snapshot = SimpleNamespace(
        total_blocks=4,
        resident_blocks=5,
        pending_store_blocks=0,
        pending_load_blocks=0,
        scheduler_block_size=16,
    )
    owner = SimpleNamespace(
        connector=SimpleNamespace(
            scheduler_manager=SimpleNamespace(
                get_mori_offload_tier_snapshot=lambda: snapshot,
            )
        )
    )

    assert observe_vllm_offload(owner).total_dram_kv_tokens is None


def test_simple_cpu_registration_rolls_back_on_hbm_failure() -> None:
    full = build_mori_strategy(MORIConfig(mode=MORIMode.FULL)).strategy
    calls: list[str] = []

    class SimpleCPUScheduler:
        def register_mori_offload_tier_provider(self, version, provider):
            calls.append("cpu_register")
            return {"hook_api_version": version}

        def unregister_mori_offload_tier_provider(self, provider):
            calls.append("cpu_unregister")

    owner = SimpleNamespace(
        connector=SimpleNamespace(scheduler_manager=SimpleCPUScheduler()),
        kv_cache_manager=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="register_mori_block_selection_provider"):
        install_vllm_mori(owner, full)

    assert calls == ["cpu_register", "cpu_unregister"]
    rebound = install_vllm_mori(
        SimpleNamespace(kv_cache_manager=None),
        build_mori_strategy(MORIConfig(mode=MORIMode.PLACEMENT_ONLY)).strategy,
    )
    rebound.close()


def test_full_adapter_cleans_process_state_when_native_unregister_fails() -> None:
    full = build_mori_strategy(MORIConfig(mode=MORIMode.FULL)).strategy

    class CPUPolicy:
        MORI_CPU_POLICY = True

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class KVManager:
        @staticmethod
        def register_mori_block_selection_provider(provider):
            return {"hook_api_version": provider.HOOK_API_VERSION}

        @staticmethod
        def unregister_mori_block_selection_provider(provider):
            raise RuntimeError("native unregister failed")

    cpu_policy = CPUPolicy()
    owner = SimpleNamespace(
        connector=SimpleNamespace(connector_scheduler=SimpleNamespace(manager=SimpleNamespace(_policy=cpu_policy))),
        kv_cache_manager=KVManager(),
    )
    adapter = install_vllm_mori(owner, full)
    MORI_RUNTIME_STATE.bind_request("request", ProgramRef("program", 0))

    with pytest.raises(RuntimeError, match="native unregister failed"):
        adapter.close()

    assert cpu_policy.closed is True
    assert MORI_RUNTIME_STATE.program_for_request("request") is None
    rebound = install_vllm_mori(
        SimpleNamespace(kv_cache_manager=None),
        build_mori_strategy(MORIConfig(mode=MORIMode.PLACEMENT_ONLY)).strategy,
    )
    rebound.close()


@pytest.mark.parametrize("bridge_type", [AgentCacheSyncSchedulerBridge, AgentCacheAsyncSchedulerBridge])
@pytest.mark.parametrize("invalid", ["missing", "malformed", "overlap"])
def test_mori_rejects_only_invalid_request_and_delivers_terminal_output(bridge_type, invalid) -> None:
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=bridge_type is AgentCacheAsyncSchedulerBridge),
        parallel_config=SimpleNamespace(data_parallel_index=0),
        additional_config={"agentcache": {"strategy": "mori", "mori": {"mode": "placement_only"}}},
    )

    def initialize(owner, *args, **kwargs):
        owner.requests = {}
        owner.kv_cache_manager = None

    def native_add(owner, request):
        owner.requests[request.request_id] = request

    def request(name, extra):
        return SimpleNamespace(
            request_id=name,
            client_index=7,
            num_prompt_tokens=100,
            num_tokens=100,
            num_output_tokens=0,
            sampling_params=SimpleNamespace(extra_args=extra),
            trace_headers={},
            events=[],
        )

    identity = {"agentic_context": {"program_id": "program"}}
    good = request("good", identity)
    bad = request(
        "bad", None if invalid == "missing" else {"agentic_context": "{"} if invalid == "malformed" else identity
    )
    with (
        patch.object(Scheduler, "__init__", initialize),
        patch.object(AsyncScheduler, "__init__", initialize),
        patch.object(Scheduler, "add_request", native_add),
        patch.object(Scheduler, "get_request_counts", return_value=(0, 0)),
        patch.object(Scheduler, "get_num_unfinished_requests", return_value=0),
        patch.object(Scheduler, "update_from_output", side_effect=[{}, {}]),
        patch.object(AsyncScheduler, "update_from_output", side_effect=[{}, {}]),
    ):
        bridge = bridge_type(config, SimpleNamespace(num_blocks=100), object(), 16, 16)
        if invalid == "overlap":
            bridge.add_request(good)
            binding = MORI_RUNTIME_STATE.program_for_request("good")
            assert binding is not None
        bridge.add_request(bad)
        assert bad.status is RequestStatus.FINISHED_ERROR
        assert "bad" not in bridge.requests
        assert bridge._agentcache.controller.program_ref_for_request("bad") is None
        assert MORI_RUNTIME_STATE.program_for_request("bad") is None
        # A lone rejected request must keep EngineCore polling until its error is sent.
        assert bridge.has_unfinished_requests()
        outputs = bridge.update_from_output(SimpleNamespace(), SimpleNamespace(req_ids=[]))
        assert outputs[7].outputs[0].finish_reason is FinishReason.ERROR
        assert outputs[7].finished_requests == {"bad"}
        assert bridge.update_from_output(SimpleNamespace(), SimpleNamespace(req_ids=[])) == {}
        assert bridge._agentcache._rejected_requests == []
        if invalid == "overlap":
            assert MORI_RUNTIME_STATE.program_for_request("good") == binding
        else:
            bridge.add_request(good)
        assert bridge.requests["good"] is good
        terminal = {7: EngineCoreOutputs(outputs=[EngineCoreOutput("good", [], finish_reason=FinishReason.STOP)])}
        bridge._agentcache.after_output(bridge, terminal, {})
        assert MORI_RUNTIME_STATE.program_for_request("good") is None
        assert bridge._agentcache.controller.program_ref_for_request("good") is None
        bridge._agentcache.close()


def test_simple_cpu_bind_failure_does_not_leak_an_index_or_unbind_existing_owner() -> None:
    first = build_mori_strategy(MORIConfig(mode=MORIMode.PLACEMENT_ONLY)).strategy
    adapter = install_vllm_mori(SimpleNamespace(kv_cache_manager=None), first)
    MORI_RUNTIME_STATE.bind_request("existing", ProgramRef("existing", 0))
    second = build_mori_strategy().strategy
    owner = SimpleNamespace(
        connector=SimpleNamespace(
            scheduler_manager=SimpleNamespace(
                register_mori_offload_tier_provider=lambda *args: {"hook_api_version": 1},
                unregister_mori_offload_tier_provider=lambda provider: None,
            )
        )
    )
    with pytest.raises(RuntimeError, match="already bound"):
        install_vllm_mori(owner, second)
    assert MORI_RUNTIME_STATE._indexes == []
    assert MORI_RUNTIME_STATE._provider is first
    assert MORI_RUNTIME_STATE.program_for_request("existing") is not None
    adapter.close()


@pytest.mark.parametrize("returns_requests", [False, True])
def test_retained_abort_uses_host_contract_and_releases_program_binding(monkeypatch, returns_requests) -> None:
    """Retained-only aborts preserve the installed host's return contract and clean bindings."""
    monkeypatch.setattr("agentinfer.agentcache.core.scheduler._NATIVE_FINISH_RETURNS_REQUESTS", returns_requests)
    owner = SimpleNamespace(requests={}, kv_cache_manager=None)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_index=0),
        additional_config={"agentcache": {"strategy": "mori", "mori": {"mode": "placement_only"}}},
    )
    hooks = _VllmAdmissionHooks(owner, config, SimpleNamespace(num_blocks=1), 16)
    request = SimpleNamespace(
        request_id="retained",
        client_index=7,
        num_prompt_tokens=100,
        sampling_params=SimpleNamespace(extra_args={"agentic_context": {"program_id": "program"}}),
        trace_headers={},
        events=[],
    )
    native = []
    hooks.add_request(owner, request, native.append)
    assert not native
    assert MORI_RUNTIME_STATE.program_for_request("retained") is not None
    finished = hooks.finish_requests(owner, "retained", RequestStatus.FINISHED_ABORTED, lambda ids, status: [])
    assert finished == ([request] if returns_requests else [("retained", 7)])
    assert request.status is RequestStatus.FINISHED_ABORTED
    assert hooks.controller.retained_request_count == 0
    assert hooks.controller.program_ref_for_request("retained") is None
    assert MORI_RUNTIME_STATE.program_for_request("retained") is None
    hooks.close()
