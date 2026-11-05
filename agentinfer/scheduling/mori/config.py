# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Validated configuration for MORI relative-idleness placement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class MORIMode(str, Enum):
    """Supported MORI runtime mode."""

    FULL = "full"


@dataclass(frozen=True)
class MORIConfig:
    """Policy parameters whose values are independent of native cache capacity."""

    window_size: int = 5
    mode: MORIMode = MORIMode.FULL
    enable_wait_credit: bool = True
    ready_wait_limit_seconds: float = 0.0
    gpu_service_requests: int = 0
    history_gated_gpu_service: bool = False
    reasoning_preempt_acting: bool = False
    drain_wait_threshold_seconds: float = 0.0
    drain_persist_cycles: int = 3
    drain_min_dwell_seconds: float = 60.0
    drain_max_victims_per_cycle: int = 2
    utility_victim_order: bool = False
    utility_min_samples: int = 20
    utility_jitter_seconds: float = 2.0
    utility_prefill_min_tokens_per_second: float = 2000.0
    utility_prefill_max_tokens_per_second: float = 50000.0
    decode_batch_mode: str = "static"
    decode_batch_tpot_guard_ratio: float = 1.3
    decode_batch_floor_tokens: int = 512
    decode_batch_tokens: int = 0
    audit_writeback_sources: bool = False
    repair_writeback_endpoints: bool = False
    fast_capacity_accounting: bool = False
    decode_buffer_tokens: int = 100
    slow_request_threshold_seconds: float = 30.0
    slow_request_repeat_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("MORI window_size must be positive")
        if not isinstance(self.mode, MORIMode):
            raise ValueError("MORI mode must be a MORIMode")
        if not isinstance(self.enable_wait_credit, bool):
            raise ValueError("MORI enable_wait_credit must be a boolean")
        if not isinstance(self.audit_writeback_sources, bool):
            raise ValueError("MORI audit_writeback_sources must be a boolean")
        if not isinstance(self.repair_writeback_endpoints, bool):
            raise ValueError("MORI repair_writeback_endpoints must be a boolean")
        if not isinstance(self.fast_capacity_accounting, bool):
            raise ValueError("MORI fast_capacity_accounting must be a boolean")
        if (
            isinstance(self.gpu_service_requests, bool)
            or not isinstance(self.gpu_service_requests, int)
            or self.gpu_service_requests < 0
        ):
            raise ValueError("MORI gpu_service_requests must be a non-negative integer")
        if not isinstance(self.history_gated_gpu_service, bool):
            raise ValueError("MORI history_gated_gpu_service must be a boolean")
        if not isinstance(self.reasoning_preempt_acting, bool):
            raise ValueError("MORI reasoning_preempt_acting must be a boolean")
        if (
            isinstance(self.drain_wait_threshold_seconds, bool)
            or not math.isfinite(self.drain_wait_threshold_seconds)
            or self.drain_wait_threshold_seconds < 0
        ):
            raise ValueError("MORI drain_wait_threshold_seconds must be finite and non-negative")
        if (
            isinstance(self.drain_persist_cycles, bool)
            or not isinstance(self.drain_persist_cycles, int)
            or self.drain_persist_cycles < 1
        ):
            raise ValueError("MORI drain_persist_cycles must be a positive integer")
        if (
            isinstance(self.drain_min_dwell_seconds, bool)
            or not math.isfinite(self.drain_min_dwell_seconds)
            or self.drain_min_dwell_seconds < 0
        ):
            raise ValueError("MORI drain_min_dwell_seconds must be finite and non-negative")
        if (
            isinstance(self.drain_max_victims_per_cycle, bool)
            or not isinstance(self.drain_max_victims_per_cycle, int)
            or self.drain_max_victims_per_cycle < 1
        ):
            raise ValueError("MORI drain_max_victims_per_cycle must be a positive integer")
        if not isinstance(self.utility_victim_order, bool):
            raise ValueError("MORI utility_victim_order must be a boolean")
        if (
            isinstance(self.utility_min_samples, bool)
            or not isinstance(self.utility_min_samples, int)
            or self.utility_min_samples < 0
        ):
            raise ValueError("MORI utility_min_samples must be a non-negative integer")
        if (
            isinstance(self.utility_jitter_seconds, bool)
            or not math.isfinite(self.utility_jitter_seconds)
            or self.utility_jitter_seconds < 0
        ):
            raise ValueError("MORI utility_jitter_seconds must be finite and non-negative")
        if (
            not math.isfinite(self.utility_prefill_min_tokens_per_second)
            or self.utility_prefill_min_tokens_per_second <= 0
            or not math.isfinite(self.utility_prefill_max_tokens_per_second)
            or self.utility_prefill_max_tokens_per_second <= self.utility_prefill_min_tokens_per_second
        ):
            raise ValueError("MORI utility prefill bounds must satisfy 0 < min < max")
        if self.decode_batch_mode not in ("static", "adaptive"):
            raise ValueError("MORI decode_batch_mode must be 'static' or 'adaptive'")
        if (
            isinstance(self.decode_batch_tpot_guard_ratio, bool)
            or not math.isfinite(self.decode_batch_tpot_guard_ratio)
            or self.decode_batch_tpot_guard_ratio <= 1.0
        ):
            raise ValueError("MORI decode_batch_tpot_guard_ratio must be finite and greater than 1")
        if (
            isinstance(self.decode_batch_floor_tokens, bool)
            or not isinstance(self.decode_batch_floor_tokens, int)
            or self.decode_batch_floor_tokens < 1
        ):
            raise ValueError("MORI decode_batch_floor_tokens must be a positive integer")
        if self.history_gated_gpu_service and self.gpu_service_requests <= 0:
            raise ValueError("MORI history_gated_gpu_service requires full mode and a positive service budget")
        if (
            isinstance(self.decode_batch_tokens, bool)
            or not isinstance(self.decode_batch_tokens, int)
            or self.decode_batch_tokens < 0
        ):
            raise ValueError("MORI decode_batch_tokens must be a non-negative integer")
        if (
            isinstance(self.ready_wait_limit_seconds, bool)
            or not math.isfinite(self.ready_wait_limit_seconds)
            or self.ready_wait_limit_seconds < 0
        ):
            raise ValueError("MORI ready_wait_limit_seconds must be finite and non-negative")
        if self.decode_buffer_tokens < 0:
            raise ValueError("MORI decode_buffer_tokens must be non-negative")
        if (
            isinstance(self.slow_request_threshold_seconds, bool)
            or not math.isfinite(self.slow_request_threshold_seconds)
            or self.slow_request_threshold_seconds < 0
        ):
            raise ValueError("MORI slow_request_threshold_seconds must be finite and non-negative")
        if (
            isinstance(self.slow_request_repeat_seconds, bool)
            or not math.isfinite(self.slow_request_repeat_seconds)
            or self.slow_request_repeat_seconds <= 0
        ):
            raise ValueError("MORI slow_request_repeat_seconds must be finite and positive")
