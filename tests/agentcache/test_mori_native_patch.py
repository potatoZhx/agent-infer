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
