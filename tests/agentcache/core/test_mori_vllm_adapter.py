# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Capability and cleanup tests for MORI's vLLM integration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentinfer.agentcache.core.scheduler import _VllmAdmissionHooks
from agentinfer.agentcache.mori.runtime_state import MORI_RUNTIME_STATE
from agentinfer.agentcache.mori.vllm_adapter import install_vllm_mori
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


def test_mori_rejects_missing_server_resolved_identity_before_native_admission() -> None:
    hooks = object.__new__(_VllmAdmissionHooks)
    hooks.lifecycle_receiver = None
    hooks.controller = SimpleNamespace(requires_program_identity=True)
    native_requests: list[object] = []
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(extra_args=None),
        trace_headers=None,
    )

    with pytest.raises(ValueError, match="server-resolved Program identity"):
        hooks.add_request(SimpleNamespace(), request, native_requests.append)

    assert native_requests == []
