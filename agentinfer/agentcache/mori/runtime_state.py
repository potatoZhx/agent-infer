# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Process-local bridge from native request ids to MORI Program generations."""

from __future__ import annotations

from typing import Protocol

from agentinfer.scheduling.cache_placement import (
    CacheClass,
    ProgramCacheClassObserver,
    ProgramCacheClassProvider,
)
from agentinfer.scheduling.domain import ProgramRef


class _OwnerIndex(Protocol):
    def reclassify_program(self, program: ProgramRef) -> int: ...

    def release_program(self, program: ProgramRef) -> int: ...


class MORIRuntimeState(ProgramCacheClassObserver):
    """Serialized engine-thread state shared with out-of-tree native policies."""

    def __init__(self) -> None:
        self._provider: ProgramCacheClassProvider | None = None
        self._requests: dict[str, ProgramRef] = {}
        self._indexes: list[_OwnerIndex] = []

    def bind_provider(self, provider: ProgramCacheClassProvider) -> None:
        if self._provider is not None and self._provider is not provider:
            raise RuntimeError("MORI runtime state is already bound to another scheduler")
        self._provider = provider
        provider.subscribe_cache_class_observer(self)

    def register_index(self, index: _OwnerIndex) -> None:
        if index not in self._indexes:
            self._indexes.append(index)

    def unregister_index(self, index: _OwnerIndex) -> None:
        if index in self._indexes:
            self._indexes.remove(index)

    def unbind_provider(self, provider: ProgramCacheClassProvider) -> None:
        if self._provider is not provider:
            return
        provider.unsubscribe_cache_class_observer(self)
        self._provider = None

    def bind_request(self, request_id: str, program: ProgramRef) -> None:
        current = self._requests.get(request_id)
        if current is not None and current != program:
            raise RuntimeError("native request id was rebound to another Program generation")
        self._requests[request_id] = program

    def unbind_request(self, request_id: str) -> None:
        self._requests.pop(request_id, None)

    def program_for_request(self, request_id: str) -> ProgramRef | None:
        return self._requests.get(request_id)

    def clear_requests(self) -> None:
        """Drop native request bindings when the owning scheduler shuts down."""
        self._requests.clear()

    def cache_class_for_program(self, program: ProgramRef) -> CacheClass:
        provider = self._provider
        return provider.cache_class_for_program(program) if provider is not None else CacheClass.INACTIVE

    def on_cache_class_changed(
        self,
        program: ProgramRef,
        previous: CacheClass,
        current: CacheClass,
    ) -> None:
        del previous, current
        for index in tuple(self._indexes):
            index.reclassify_program(program)

    def on_program_released(self, program: ProgramRef) -> None:
        for index in tuple(self._indexes):
            index.release_program(program)
        stale = [request_id for request_id, owner in self._requests.items() if owner == program]
        for request_id in stale:
            self._requests.pop(request_id, None)

    def reset(self) -> None:
        """Reset process-local registrations after shutdown or in unit tests."""
        if self._provider is not None:
            self._provider.unsubscribe_cache_class_observer(self)
        self._provider = None
        self._requests.clear()
        self._indexes.clear()


MORI_RUNTIME_STATE = MORIRuntimeState()
