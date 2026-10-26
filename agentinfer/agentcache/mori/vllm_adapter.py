# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Capability-checked vLLM integration for full MORI cache management."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from agentinfer.agentcache.mori.hbm_policy import MORIHBMPolicy
from agentinfer.agentcache.mori.runtime_state import MORI_RUNTIME_STATE
from agentinfer.scheduling.mori import MORIMode, MORIStrategy


@dataclass(frozen=True)
class VllmOffloadObservation:
    """Rank-local CPU-offload capacity and transfer facts."""

    total_dram_kv_tokens: int | None = None
    used_dram_kv_tokens: int | None = None
    pending_h2d_kv_tokens: int | None = None
    pending_d2h_kv_tokens: int | None = None
    dram_chunk_size_tokens: int | None = None


@dataclass
class VllmMORIAdapter:
    """Installed native policy handles retained by the scheduler bridge."""

    hbm_policy: MORIHBMPolicy | None
    cpu_policy: object | None
    owner: object
    strategy: MORIStrategy
    closed: bool = False

    def close(self) -> None:
        """Unregister native hooks and process-local indexes idempotently."""
        if self.closed:
            return
        self.closed = True
        kv_cache_manager = getattr(self.owner, "kv_cache_manager", None)
        try:
            if self.hbm_policy is not None:
                unregister = getattr(kv_cache_manager, "unregister_mori_block_selection_provider", None)
                if callable(unregister):
                    unregister(self.hbm_policy)
        finally:
            try:
                if self.hbm_policy is not None:
                    self.hbm_policy.close()
            finally:
                try:
                    close_cpu = getattr(self.cpu_policy, "close", None)
                    if callable(close_cpu):
                        close_cpu()
                finally:
                    MORI_RUNTIME_STATE.unbind_provider(self.strategy)
                    MORI_RUNTIME_STATE.clear_requests()


def observe_vllm_offload(owner: object) -> VllmOffloadObservation:
    """Read vLLM 0.31 scheduler-side CPU cache facts without mutating it."""
    scheduler = _offload_scheduler(owner)
    if scheduler is None:
        return VllmOffloadObservation()
    manager = getattr(scheduler, "manager", None)
    config = getattr(scheduler, "config", None)
    groups = tuple(getattr(config, "kv_group_configs", ()) or ())
    if manager is None or len(groups) != 1:
        return VllmOffloadObservation()
    tokens_per_chunk = _non_negative_int(getattr(groups[0], "tokens_per_chunk", None), positive=True)
    num_chunks = _non_negative_int(getattr(manager, "_num_chunks", None))
    allocated = _non_negative_int(getattr(manager, "_num_allocated_chunks", None))
    free_list = getattr(manager, "_free_list", None)
    if tokens_per_chunk is None or num_chunks is None or allocated is None or not isinstance(free_list, list):
        return VllmOffloadObservation()
    used_chunks = max(0, allocated - len(free_list))
    h2d_keys: set[object] = set()
    d2h_keys: set[object] = set()
    jobs = getattr(scheduler, "_jobs", {})
    if isinstance(jobs, dict):
        for job in jobs.values():
            keys = set(getattr(job, "keys", ()) or ())
            (d2h_keys if bool(getattr(job, "is_store", False)) else h2d_keys).update(keys)
    return VllmOffloadObservation(
        total_dram_kv_tokens=num_chunks * tokens_per_chunk,
        used_dram_kv_tokens=used_chunks * tokens_per_chunk,
        pending_h2d_kv_tokens=len(h2d_keys) * tokens_per_chunk,
        pending_d2h_kv_tokens=len(d2h_keys) * tokens_per_chunk,
        dram_chunk_size_tokens=tokens_per_chunk,
    )


def install_vllm_mori(owner: object, strategy: MORIStrategy) -> VllmMORIAdapter:
    """Bind MORI and validate every native capability required by its mode."""
    if strategy.config.mode is MORIMode.PLACEMENT_ONLY:
        MORI_RUNTIME_STATE.bind_provider(strategy)
        return VllmMORIAdapter(hbm_policy=None, cpu_policy=None, owner=owner, strategy=strategy)

    scheduler = _offload_scheduler(owner)
    manager = getattr(scheduler, "manager", None) if scheduler is not None else None
    cpu_policy = getattr(manager, "_policy", None)
    if not bool(getattr(cpu_policy, "MORI_CPU_POLICY", False)):
        raise RuntimeError(
            "full MORI requires vLLM CPU offload configured with "
            "eviction_policy=MoriCPUCachePolicy and "
            "cache_policy_module_path=agentinfer.agentcache.mori_cpu_policy"
        )

    kv_cache_manager = getattr(owner, "kv_cache_manager", None)
    register = getattr(kv_cache_manager, "register_mori_block_selection_provider", None)
    if not callable(register):
        _close_failed_cpu_policy(cpu_policy)
        raise RuntimeError(
            "full MORI requires the vLLM MORI HBM hook API register_mori_block_selection_provider(version=1)"
        )
    unregister = getattr(kv_cache_manager, "unregister_mori_block_selection_provider", None)
    if not callable(unregister):
        _close_failed_cpu_policy(cpu_policy)
        raise RuntimeError(
            "full MORI requires the vLLM MORI HBM hook API unregister_mori_block_selection_provider(version=1)"
        )
    MORI_RUNTIME_STATE.bind_provider(strategy)
    hbm_policy = MORIHBMPolicy(MORI_RUNTIME_STATE)
    try:
        capabilities = register(hbm_policy)
    except BaseException:
        with suppress(Exception):
            unregister(hbm_policy)
        hbm_policy.close()
        MORI_RUNTIME_STATE.unbind_provider(strategy)
        _close_failed_cpu_policy(cpu_policy)
        raise
    version = (
        capabilities.get("hook_api_version")
        if isinstance(capabilities, dict)
        else getattr(capabilities, "hook_api_version", None)
    )
    if version != MORIHBMPolicy.HOOK_API_VERSION:
        with suppress(Exception):
            unregister(hbm_policy)
        hbm_policy.close()
        MORI_RUNTIME_STATE.unbind_provider(strategy)
        _close_failed_cpu_policy(cpu_policy)
        raise RuntimeError(
            f"unsupported vLLM MORI HBM hook API version: {version!r}; expected {MORIHBMPolicy.HOOK_API_VERSION}"
        )
    return VllmMORIAdapter(
        hbm_policy=hbm_policy,
        cpu_policy=cpu_policy,
        owner=owner,
        strategy=strategy,
    )


def _offload_scheduler(owner: object) -> object | None:
    connector = getattr(owner, "connector", None)
    if connector is None:
        return None
    direct = getattr(connector, "connector_scheduler", None)
    if direct is not None:
        return direct
    try:
        return getattr(connector, "scheduler", None)
    except (AssertionError, AttributeError):
        return None


def _close_failed_cpu_policy(cpu_policy: object) -> None:
    """Rollback a policy that registered process-local indexes during startup."""
    close = getattr(cpu_policy, "close", None)
    if callable(close):
        close()
    MORI_RUNTIME_STATE.clear_requests()


def _non_negative_int(value: Any, *, positive: bool = False) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < (1 if positive else 0):
        return None
    return value
