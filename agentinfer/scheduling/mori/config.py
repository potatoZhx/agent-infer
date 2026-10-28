# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Validated configuration for MORI relative-idleness placement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class MORIMode(str, Enum):
    """Runtime capability level selected explicitly by the operator."""

    FULL = "full"
    PLACEMENT_ONLY = "placement_only"


@dataclass(frozen=True)
class MORIConfig:
    """Policy parameters whose values are independent of native cache capacity."""

    window_size: int = 5
    mode: MORIMode = MORIMode.FULL
    decode_buffer_tokens: int = 100
    min_residency_seconds: float = 1.0
    idleness_margin: float = 0.05
    max_wait_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("MORI window_size must be positive")
        if not isinstance(self.mode, MORIMode):
            raise ValueError("MORI mode must be a MORIMode")
        if self.decode_buffer_tokens < 0:
            raise ValueError("MORI decode_buffer_tokens must be non-negative")
        for name in ("min_residency_seconds", "idleness_margin", "max_wait_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"MORI {name} must be finite and non-negative")
        if self.idleness_margin > 1 or self.max_wait_seconds <= 0:
            raise ValueError("MORI requires idleness_margin <= 1 and max_wait_seconds > 0")
