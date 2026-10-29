# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Per-plan union accounting. Physical allocation remains owned by the backend."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from agentinfer.scheduling.domain import ProgramRef


class FootprintProvider(Protocol):
    """Supply current-prompt native block IDs; never promise physical readiness."""

    def footprint(self, program: ProgramRef, tier: str) -> frozenset[int]: ...

    def invalidate_footprint(self, program: ProgramRef) -> None: ...


@dataclass(frozen=True)
class CapacityClaim:
    """Logical token promise containing a proven subset of unique native blocks."""

    tokens: int
    blocks: frozenset[int] = frozenset()


class CapacityLedger:
    """Count known native blocks once and charge all unproven context conservatively.

    A claim contains only valid blocks from the current prompt version. Unknown
    blocks and speculative decode headroom are private promises. Marginal removal
    counts a shared block only after its last protection claim disappears.
    """

    def __init__(self, quantum: int) -> None:
        self.quantum = quantum
        self.claims: dict[ProgramRef, CapacityClaim] = {}
        self.owners: Counter[int] = Counter()
        self.used = 0

    def incremental(self, claim: CapacityClaim) -> int:
        """Return the additional tokens charged if a new owner takes this claim."""
        return claim.tokens - self.quantum * sum(key in self.owners for key in claim.blocks)

    def add(self, ref: ProgramRef, claim: CapacityClaim) -> None:
        """Insert or replace one owner claim, updating shared block multiplicities."""
        if ref in self.claims:
            self.remove(ref)
        self.used += self.incremental(claim)
        self.claims[ref] = claim
        self.owners.update(claim.blocks)

    def reclaimable(self, ref: ProgramRef) -> int:
        """Return marginal logical relief, excluding blocks still claimed by peers."""
        claim = self.claims[ref]
        return claim.tokens - self.quantum * sum(self.owners[key] > 1 for key in claim.blocks)

    def remove(self, ref: ProgramRef) -> int:
        """Remove a claim and return marginal relief; unknown owners release zero."""
        if ref not in self.claims:
            return 0
        freed = self.reclaimable(ref)
        claim = self.claims.pop(ref)
        self.used -= freed
        for key in claim.blocks:
            self.owners[key] -= 1
            if not self.owners[key]:
                del self.owners[key]
        return freed

    def copy(self) -> CapacityLedger:
        """Copy accounting for a speculative plan without modifying committed claims."""
        other = CapacityLedger(self.quantum)
        other.claims = self.claims.copy()
        other.owners = self.owners.copy()
        other.used = self.used
        return other
