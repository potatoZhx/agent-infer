# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Server-owned scheduling branches; no inference about application context adoption."""

from __future__ import annotations

import uuid
from dataclasses import replace

from agentinfer.scheduling.domain import ProgramRef
from agentinfer.scheduling.identity import AgentIdentity


class LineageRegistry:
    """Map external families to independent cache owners without inventing blocking edges."""

    def __init__(self) -> None:
        self._named: dict[tuple[str, str], str] = {}
        self._families: dict[str, set[ProgramRef]] = {}
        self._external: dict[ProgramRef, str] = {}
        self._temporary: set[ProgramRef] = set()
        self._internal_ids: set[str] = set()
        self._named_by_family: dict[str, set[tuple[str, str]]] = {}
        self._free_temporary: list[str] = []
        self._free_named: list[str] = []

    def resolve(self, metadata: AgentIdentity, *, default_busy: bool) -> tuple[AgentIdentity, bool]:
        """Resolve a serial owner or allocate a temporary slot; reject external collisions."""
        if metadata.program_id in self._internal_ids:
            raise ValueError("external Program identity collides with an internal lineage")
        temporary = metadata.lineage_id is None and default_busy
        if metadata.lineage_id is None and not temporary:
            return metadata, False
        key = (metadata.program_id, metadata.lineage_id or "")
        internal = None if temporary else self._named.get(key)
        if internal is None:
            free_slots = self._free_temporary if temporary else self._free_named
            internal = free_slots.pop() if free_slots else f"mori-lineage:{uuid.uuid4().hex}"
            self._internal_ids.add(internal)
            if not temporary:
                self._named[key] = internal
                self._named_by_family.setdefault(metadata.program_id, set()).add(key)
        return replace(metadata, program_id=internal, parent_program_id=None, blocks_parent=False), temporary

    def bind(self, external: str, ref: ProgramRef, *, temporary: bool) -> None:
        """Index a materialized exact generation under its external family."""
        self._families.setdefault(external, set()).add(ref)
        self._external[ref] = external
        if temporary:
            self._temporary.add(ref)

    def temporary(self, ref: ProgramRef) -> bool:
        """Return whether protection ends when this generation has no accepted work."""
        return ref in self._temporary

    def family(self, external: str) -> tuple[ProgramRef, ...]:
        """Return live branch generations; callers must not infer an ordering."""
        return tuple(self._families.get(external, ()))

    def release(self, ref: ProgramRef) -> None:
        """Recycle completed slots, retaining generation tombstones only for peak occupancy."""
        external = self._external.pop(ref, None)
        if ref in self._temporary:
            self._free_temporary.append(ref.program_id)
        self._temporary.discard(ref)
        if external is not None:
            family = self._families[external]
            family.discard(ref)
            if not family:
                del self._families[external]
                for key in self._named_by_family.pop(external, ()):
                    self._free_named.append(self._named.pop(key))
