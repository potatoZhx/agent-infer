# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Typed victim provider for vLLM's SimpleCPUOffloadConnector."""

from __future__ import annotations

from collections.abc import Iterable

from agentinfer.agentcache.mori.runtime_state import MORIRuntimeState
from agentinfer.agentcache.mori.typed_lru import TypedLRUIndex
from agentinfer.scheduling.cache_placement import CacheClass


class MORISimpleCPUOffloadPolicy:
    """Propose SimpleCPU blocks in INACTIVE, BUSY, IDLE order.

    vLLM remains authoritative for free-queue membership, refcounts, cached
    hashes, allocation, and transfer completion. This provider only mirrors
    owner/class/readiness deltas and proposes an exact ordered victim set.
    """

    HOOK_API_VERSION = 1

    def __init__(self, runtime_state: MORIRuntimeState) -> None:
        self.runtime_state = runtime_state
        self.index = TypedLRUIndex[int](
            eviction_order=(CacheClass.INACTIVE, CacheClass.BUSY, CacheClass.IDLE),
            protection_rank={
                CacheClass.INACTIVE: 0,
                CacheClass.BUSY: 1,
                CacheClass.IDLE: 2,
            },
            class_for_program=runtime_state.cache_class_for_program,
        )
        runtime_state.register_index(self.index)

    def on_block_ready(self, block_id: int, native_lru_seq: int) -> None:
        """Observe a ref-zero CPU block after native queue order is committed."""
        self.index.upsert(
            block_id,
            owners=self.index.owners(block_id),
            native_lru_seq=native_lru_seq,
            ready=True,
        )

    def on_block_owner_added(self, block_id: int, owner_request_id: str) -> None:
        """Resolve a native request ID through server-maintained Program state."""
        owner = self.runtime_state.program_for_request(owner_request_id)
        if owner is None:
            return
        owners = set(self.index.owners(block_id))
        if owner in owners:
            return
        owners.add(owner)
        if self.index.cache_class(block_id) is None:
            # Store destinations are observed while pinned. Their exact native
            # sequence arrives only after transfer completion frees the block.
            self.index.upsert(block_id, owners=owners, native_lru_seq=0, ready=False)
        else:
            self.index.update_owners(block_id, owners)

    def on_block_evictability_changed(self, block_id: int, ready: bool) -> None:
        self.index.mark_ready(block_id, ready)

    def on_block_removed(self, block_id: int) -> None:
        self.index.remove(block_id)

    def on_cache_reset(self) -> None:
        self.index.clear()

    def propose(
        self,
        count: int,
        protected_block_ids: Iterable[int] = (),
    ) -> tuple[int, ...] | None:
        """Return advice only; SimpleCPU validates and performs allocation."""
        return self.index.propose_exact(count, protected_block_ids)

    def close(self) -> None:
        """Detach the SimpleCPU index from Program class notifications."""
        self.index.clear()
        self.runtime_state.unregister_index(self.index)


__all__ = ["MORISimpleCPUOffloadPolicy"]
