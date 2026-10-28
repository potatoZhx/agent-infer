# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Current-context native block evidence with removal-safe reverse indexes."""

from __future__ import annotations

from agentinfer.scheduling.domain import ProgramRef


class FootprintBook:
    """A pool removes proof before reusing an ID; prompt replacement clears old proof."""

    def __init__(self) -> None:
        self.enabled = False
        self._by_owner: dict[tuple[str, ProgramRef], set[int]] = {}
        self._by_block: dict[tuple[str, int], set[ProgramRef]] = {}

    def add(self, tier: str, block: int, owner: ProgramRef) -> None:
        """Record a native owner delta when single-group accounting is enabled."""
        if not self.enabled or block == 0:
            return
        self._by_owner.setdefault((tier, owner), set()).add(block)
        self._by_block.setdefault((tier, block), set()).add(owner)

    def remove_block(self, tier: str, block: int) -> None:
        """Invalidate all evidence for a block before native reuse of its ID."""
        for owner in self._by_block.pop((tier, block), ()):
            blocks = self._by_owner[(tier, owner)]
            blocks.discard(block)
            if not blocks:
                del self._by_owner[(tier, owner)]

    def invalidate(self, owner: ProgramRef) -> None:
        """Drop both tiers of proof on prompt replacement or generation release."""
        for tier in ("gpu", "cpu"):
            for block in self._by_owner.pop((tier, owner), ()):
                owners = self._by_block[(tier, block)]
                owners.discard(owner)
                if not owners:
                    del self._by_block[(tier, block)]

    def get(self, owner: ProgramRef, tier: str) -> frozenset[int]:
        """Return immutable evidence; absence requires conservative accounting."""
        return frozenset(self._by_owner.get((tier, owner), ()))

    def clear(self, tier: str | None = None) -> None:
        """Drop proof for one reset native pool, or all pools at shutdown."""
        if tier is None:
            self._by_owner.clear()
            self._by_block.clear()
        else:
            for pool, block in tuple(self._by_block):
                if pool == tier:
                    self.remove_block(pool, block)
