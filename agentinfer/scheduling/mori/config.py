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
    starvation_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("MORI window_size must be positive")
        if not isinstance(self.mode, MORIMode):
            raise ValueError("MORI mode must be a MORIMode")
        if self.decode_buffer_tokens < 0:
            raise ValueError("MORI decode_buffer_tokens must be non-negative")
        if not math.isfinite(self.starvation_timeout_seconds) or self.starvation_timeout_seconds <= 0:
            raise ValueError("MORI starvation_timeout_seconds must be finite and positive")
