# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Committed history and placement factors maintained by MORI."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from agentinfer.scheduling.domain import ProgramRef


class MORITier(str, Enum):
    """Logical cache protection tier, independent of Program execution state."""

    GPU = "gpu"
    CPU = "cpu"
    WAITING = "waiting"


@dataclass(frozen=True)
class MORIGlobalFactors:
    """Policy-wide committed counters used by diagnostics."""

    schedule_cycles: int = 0
    schedule_requested_at: float | None = None
    relief_candidate: ProgramRef | None = None

    def __post_init__(self) -> None:
        if self.schedule_cycles < 0:
            raise ValueError("MORI schedule_cycles must be non-negative")
        if self.schedule_requested_at is not None and (
            not math.isfinite(self.schedule_requested_at) or self.schedule_requested_at < 0
        ):
            raise ValueError("MORI schedule request time must be finite and non-negative")


@dataclass(frozen=True)
class MORIProgramFactors:
    """Recent reasoning/acting cycles and generation-safe logical placement."""

    tier: MORITier = MORITier.WAITING
    cycles: tuple[tuple[float, float], ...] = ()
    pending_reasoning_seconds: float = 0.0
    acting_started_at_monotonic_s: float | None = None
    reasoning_started_at_monotonic_s: float | None = None
    pending_tier: MORITier | None = None
    last_idleness: float = 0.0
    resident_since: float | None = None
    relief_for: ProgramRef | None = None
    execution_observed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.tier, MORITier):
            raise ValueError("MORI tier must be a MORITier")
        if self.pending_tier is not None and not isinstance(self.pending_tier, MORITier):
            raise ValueError("MORI pending_tier must be a MORITier when supplied")
        values = [self.pending_reasoning_seconds, self.last_idleness]
        values.extend(value for cycle in self.cycles for value in cycle)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("MORI timing and idleness values must be finite and non-negative")
        if self.last_idleness > 1:
            raise ValueError("MORI last_idleness must be at most one")
        timestamps = (self.acting_started_at_monotonic_s, self.reasoning_started_at_monotonic_s, self.resident_since)
        if any(value is not None and (not math.isfinite(value) or value < 0) for value in timestamps):
            raise ValueError("MORI timestamps must be finite and non-negative")
