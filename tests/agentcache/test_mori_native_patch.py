# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Exercise the shipped HBM patch against an explicitly supplied clean vLLM source tree."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from agentinfer.agentcache.mori.hbm_policy import MORIHBMPolicy
from agentinfer.agentcache.mori.runtime_state import MORIRuntimeState
from agentinfer.scheduling import ProgramRef

pytestmark = pytest.mark.cpu_test


@pytest.fixture(scope="module")
def native_pool_class(tmp_path_factory):
    """Apply the packaged patch to pristine sources without changing the installed vLLM."""
    source = os.environ.get("MORI_VLLM_SOURCE")
    if source is None:
        pytest.skip("set MORI_VLLM_SOURCE to a clean vLLM v0.31.0 source tree to validate the native patch")
    root = tmp_path_factory.mktemp("mori-native")
    for name in ("block_pool.py", "kv_cache_manager.py"):
        target = root / "vllm/v1/core" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(source) / "vllm/v1/core" / name, target)
    patch = Path(__file__).resolve().parents[2] / "agentinfer/agentcache/patches/vllm-0.31-mori-hbm-provider.patch"
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "apply", str(patch)], cwd=root, check=True, capture_output=True)
    spec = importlib.util.spec_from_file_location("mori_native_test_pool", root / "vllm/v1/core/block_pool.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BlockPool


@pytest.mark.parametrize("advice", [None, (3,), (3, 3), (3, -1), (3, 100), (3, 0), (3, True), (3, "bad"), "raise"])
def test_invalid_advice_falls_back_atomically_and_updates_provider(native_pool_class, advice) -> None:
    """Invalid second victims must not remove a valid first victim before native fallback."""
    pool = native_pool_class(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    policy = MORIHBMPolicy(MORIRuntimeState())
    pool.register_mori_block_selection_provider(policy)

    def propose(count, protected):
        if advice == "raise":
            raise RuntimeError("injected provider failure")
        return advice

    policy.propose = propose
    blocks = pool.get_new_blocks(2)
    assert [block.block_id for block in blocks] == [1, 2]
    assert all(block.ref_cnt == 1 for block in blocks)
    assert [block.block_id for block in pool.free_block_queue.get_all_free_blocks()] == [3, 4, 5, 6, 7]
    assert policy.index.propose_exact(5) == (3, 4, 5, 6, 7)
    assert pool._mori_fallback_count == 1
    pool.unregister_mori_block_selection_provider(policy)
    policy.close()


def test_allocated_advice_falls_back_and_valid_advice_preserves_native_safety(native_pool_class) -> None:
    """Reject in-use blocks, then accept valid reordered victims and retain normal exhaustion errors."""
    pool = native_pool_class(num_gpu_blocks=8, enable_caching=False, hash_block_size=16)
    policy = MORIHBMPolicy(MORIRuntimeState())
    pool.register_mori_block_selection_provider(policy)
    occupied = pool.get_new_blocks(1)[0]
    policy.propose = lambda count, protected: (4, occupied.block_id)
    assert [block.block_id for block in pool.get_new_blocks(2)] == [2, 3]
    assert occupied.ref_cnt == 1
    policy.propose = lambda count, protected: (6, 4)
    assert [block.block_id for block in pool.get_new_blocks(2)] == [6, 4]
    assert policy.index.propose_exact(2) == (5, 7)
    with pytest.raises(ValueError, match="Cannot get"):
        pool.get_new_blocks(3)
    pool.unregister_mori_block_selection_provider(policy)
    policy.close()


def test_native_reallocation_invalidates_old_footprint_before_new_owner(native_pool_class) -> None:
    """The shipped native callback prevents shared-capacity proof surviving ID reuse."""
    pool = native_pool_class(num_gpu_blocks=3, enable_caching=False, hash_block_size=16)
    state = MORIRuntimeState()
    state.footprints.enabled = True
    old, new = ProgramRef("old", 0), ProgramRef("new", 0)
    state.bind_request("old", old)
    state.bind_request("new", new)
    policy = MORIHBMPolicy(state)
    pool.register_mori_block_selection_provider(policy)
    block = pool.get_new_blocks(1)[0]
    pool.observe_mori_block_owner("old", [block])
    assert state.footprint(old, "gpu") == frozenset({block.block_id})
    pool.free_blocks([block])
    policy.propose = lambda count, protected: (block.block_id,)
    assert pool.get_new_blocks(1) == [block]
    assert not state.footprint(old, "gpu")
    pool.observe_mori_block_owner("new", [block])
    assert state.footprint(new, "gpu") == frozenset({block.block_id})
    pool.unregister_mori_block_selection_provider(policy)
    policy.close()


@pytest.fixture(scope="module")
def native_cpu_module(tmp_path_factory, native_pool_class):
    """Apply the CPU patch too; exercise actual native methods with small CPU pools."""
    import sys

    root = tmp_path_factory.mktemp("mori-native-cpu")
    name = "vllm/v1/simple_kv_offload/manager.py"
    target = root / name
    target.parent.mkdir(parents=True)
    shutil.copyfile(Path(os.environ["MORI_VLLM_SOURCE"]) / name, target)
    patch = (
        Path(__file__).resolve().parents[2] / "agentinfer/agentcache/patches/vllm-0.31-mori-simple-cpu-provider.patch"
    )
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "apply", str(patch)], cwd=root, check=True, capture_output=True)
    spec = importlib.util.spec_from_file_location("mori_native_test_cpu", target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cpu_store(native_pool_class, native_cpu_module):
    from agentinfer.agentcache.mori.simple_cpu_policy import MORISimpleCPUOffloadPolicy
    from agentinfer.scheduling.mori import MORIProgramFactors, MORITier, build_mori_strategy

    components = build_mori_strategy()
    state = MORIRuntimeState()
    state.bind_provider(components.strategy)
    policy = MORISimpleCPUOffloadPolicy(state)
    for req, tier in [("idle", MORITier.CPU), ("busy", MORITier.GPU)]:
        owner = ProgramRef(req, 0)
        state.bind_request(req, owner)
        components.strategy._set_factor(components.initial_factors, owner, MORIProgramFactors(tier=tier))
    # Construct only the scheduler-side portion; no tensors, GPU or DMA involved.
    manager = native_cpu_module.SimpleCPUOffloadScheduler.__new__(native_cpu_module.SimpleCPUOffloadScheduler)
    manager.cpu_block_pool = native_pool_class(num_gpu_blocks=3, enable_caching=True, hash_block_size=16)
    manager._mori_cpu_backend = True
    manager._lazy_mode = False
    manager._mori_offload_tier_provider = None
    manager._mori_resident_block_ids = set()
    assert manager.register_mori_offload_tier_provider(2, policy) == {"hook_api_version": 2}
    yield manager, policy
    manager.unregister_mori_offload_tier_provider(policy)
    policy.close()
    state.unbind_provider(components.strategy)


def test_cpu_busy_backup_cannot_evict_idle_cache_even_when_native_calls_it_free(cpu_store):
    from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

    manager, policy = cpu_store
    pool = manager.cpu_block_pool
    blocks = manager._allocate_store_blocks(2, "idle")
    keys = [make_block_hash_with_group_id(BlockHash(bytes([i]) * 32), 0) for i in (1, 2)]
    for block, key in zip(blocks, keys, strict=True):
        pool._insert_block_hash(key, block, num_tokens=16)
    manager._observe_mori_owner("idle", blocks)
    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks() == 2  # ref-zero cached is not empty
    assert manager._mori_store_capacity("busy") == 0
    assert manager._allocate_store_blocks(1, "busy") == []
    assert all(pool.cached_block_hash_to_block.get_one_block(key) is not None for key in keys)
    assert pool._mori_fallback_count == 0
    assert manager._mori_store_capacity("idle") == 2
    assert len(manager._allocate_store_blocks(1, "idle")) == 1


@pytest.mark.parametrize("advice", [None, (1,), (1, 1), (1, 0), (1, 99), "raise"])
def test_rejected_cpu_store_is_atomic_and_has_no_lru_fallback(cpu_store, advice):
    manager, policy = cpu_store
    pool = manager.cpu_block_pool
    before = [b.block_id for b in pool.free_block_queue.get_all_free_blocks()]

    def propose(count, request_id):
        if advice == "raise":
            raise RuntimeError("injected failure")
        return advice

    policy.propose_for_store = propose
    assert manager._allocate_store_blocks(2, "busy") == []
    assert [b.block_id for b in pool.free_block_queue.get_all_free_blocks()] == before
    assert all(pool.blocks[i].ref_cnt == 0 for i in before)
    assert pool._mori_fallback_count == 0
    assert policy.store_capacity("busy") == 2


def test_finished_flush_rejection_does_not_pin_gpu_or_publish_transfer(cpu_store, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    manager, policy = cpu_store
    request = SimpleNamespace(request_id="busy")
    manager._reqs_to_store = {"busy": object()}
    manager._gpu_block_pool = SimpleNamespace(touch=Mock(), blocks={7: object()})
    manager._pending_finished_stores = []
    manager._in_flight_store_gpu_blocks = set()
    monkeypatch.setattr(manager, "_select_eager_blocks_to_store", lambda *args: ([7], [1], None))
    monkeypatch.setattr(manager, "_find_fa_partial_tail_source", lambda *args: None)
    policy.propose_for_store = lambda *args: None
    manager._queue_finished_eager_store(request, ([7],))
    assert manager._pending_finished_stores == []
    assert manager._in_flight_store_gpu_blocks == set()
    manager._gpu_block_pool.touch.assert_not_called()


def test_positional_store_rejection_keeps_cursor_for_retry(cpu_store, native_cpu_module, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    manager, policy = cpu_store
    state = SimpleNamespace(finished=False, block_ids=([7],), num_stored_blocks=[0])
    manager._reqs_to_store = {"busy": state}
    manager._gpu_block_pool = SimpleNamespace(touch=Mock(), blocks={7: object()})
    manager._in_flight_store_gpu_blocks = set()
    manager.enable_kv_cache_events = False
    manager.cpu_kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    manager.boundary_store_stats = native_cpu_module.BoundaryStoreStats()
    output = SimpleNamespace(kv_connector_block_state=None, preempted_req_ids=set(), num_scheduled_tokens={"busy": 1})
    monkeypatch.setattr(native_cpu_module, "yield_req_data", lambda _: [("busy", None, False)])
    monkeypatch.setattr(manager, "_select_eager_blocks_to_store", lambda *args: ([7], [1], None))
    policy.propose_for_store = lambda *args: None
    assert manager._prepare_eager_store_specs(output) == ([], [], [], None)
    assert state.num_stored_blocks == [0]
    assert manager._in_flight_store_gpu_blocks == set()
    manager._gpu_block_pool.touch.assert_not_called()


def test_boundary_store_rejection_does_not_pin_gpu(cpu_store, native_cpu_module, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    manager, policy = cpu_store
    manager._reqs_to_store = {"busy": SimpleNamespace(finished=False)}
    manager._gpu_block_pool = SimpleNamespace(touch=Mock(), blocks={7: object()})
    manager._in_flight_store_gpu_blocks = set()
    manager.enable_kv_cache_events = False
    manager.cpu_kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    manager.boundary_store_stats = native_cpu_module.BoundaryStoreStats()
    output = SimpleNamespace(
        kv_connector_block_state=SimpleNamespace(boundary_state_offloads={"busy": [(16, 7, 0)]}),
        preempted_req_ids=set(),
        finished_req_ids=set(),
    )
    monkeypatch.setattr(native_cpu_module, "yield_req_data", lambda _: [])
    monkeypatch.setattr(manager, "_classify_store_candidate", lambda *args: native_cpu_module._StoreAdmission.READY)
    policy.propose_for_store = lambda *args: None
    assert manager._prepare_eager_store_specs(output) == ([], [], [], None)
    assert manager.boundary_store_stats.dropped_cpu_full == 1
    assert manager._in_flight_store_gpu_blocks == set()
    manager._gpu_block_pool.touch.assert_not_called()


def test_eager_selection_uses_typed_capacity_instead_of_native_free_count(cpu_store):
    from types import SimpleNamespace

    manager, policy = cpu_store
    pool = manager.cpu_block_pool
    blocks = manager._allocate_store_blocks(2, "idle")
    manager._observe_mori_owner("idle", blocks)
    pool.free_blocks(blocks)
    manager._gpu_block_pool = object()
    manager.cpu_kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    manager.cpu_coordinator = SimpleNamespace(enable_partial_hash_hits=False)
    manager.block_size = 16
    manager.enable_kv_cache_events = False
    manager.prefix_cacheable_group_ids = {0}
    request = SimpleNamespace(request_id="busy", num_computed_tokens=16, num_output_placeholders=0)
    state = SimpleNamespace(request=request)
    assert manager._select_eager_blocks_to_store(state, ([7],)) == ([], [0], None)
    assert pool.get_num_free_blocks() == 2


def test_cpu_registration_rejects_old_blockpool_without_strict_store_support(cpu_store, monkeypatch):
    manager, policy = cpu_store
    manager.unregister_mori_offload_tier_provider(policy)
    pool = manager.cpu_block_pool
    register = pool.register_mori_block_selection_provider

    def old_register(provider):
        register(provider)
        return {"hook_api_version": 1}

    monkeypatch.setattr(pool, "register_mori_block_selection_provider", old_register)
    with pytest.raises(RuntimeError, match="unsupported MORI BlockPool"):
        manager.register_mori_offload_tier_provider(2, policy)
    assert pool._mori_block_selection_provider is None
    assert manager._mori_offload_tier_provider is None
