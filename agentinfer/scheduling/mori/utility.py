# SPDX-License-Identifier: Apache-2.0
# FileCopyrightText: Copyright contributors to the AgentInfer project
"""Self-calibrating recompute-cost utilities for MORI victim ordering.

``κ`` (effective prefill tokens/s) and ``η`` (average retained prefix
fraction) are EWMA-calibrated from per-request completion events. Victim
ordering converts each candidate eviction into an expected recomputation
cost in seconds, normalized by the marginal capacity it frees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentinfer.scheduling.domain import ProgramRef, ProgramStatus


@dataclass
class UtilityRuntimeState:
    """EWMA containers for the two calibrated physical quantities."""

    samples: int = 0
    prefill_tokens_per_second: float = 0.0
    retained_fraction: float = 0.0
    _recompute_tokens_total: int = 0
    _service_seconds_total: float = 0.0
    _alpha: float = 0.05
    calibration_history: list[tuple[int, float, float]] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.samples > 0 and self.prefill_tokens_per_second > 0.0

    def observe_request_finished(
        self,
        uncached_tokens: int,
        service_seconds: float,
        prompt_tokens: int,
        prefix_hit_tokens: int,
    ) -> None:
        """Fold one completed request into the κ/η estimates."""
        self.samples += 1
        if uncached_tokens > 0 and service_seconds > 0:
            self._recompute_tokens_total += uncached_tokens
            self._service_seconds_total += service_seconds
            if self._service_seconds_total > 0:
                observed = self._recompute_tokens_total / self._service_seconds_total
                if self.prefill_tokens_per_second <= 0:
                    self.prefill_tokens_per_second = observed
                else:
                    self.prefill_tokens_per_second = (
                        1 - self._alpha
                    ) * self.prefill_tokens_per_second + self._alpha * observed
        if prompt_tokens > 0:
            retained = prefix_hit_tokens / prompt_tokens
            if self.samples <= 1:
                self.retained_fraction = retained
            else:
                self.retained_fraction = (1 - self._alpha) * self.retained_fraction + self._alpha * retained

    def victim_cost_seconds(
        self,
        context_tokens: int,
        is_acting: bool,
        *,
        jitter_seconds: float,
        min_prefill_tokens_per_second: float,
        max_prefill_tokens_per_second: float,
    ) -> float:
        """Expected recomputation cost of evicting one owner, in seconds."""
        kappa = min(max(self.prefill_tokens_per_second, min_prefill_tokens_per_second), max_prefill_tokens_per_second)
        retained = min(max(self.retained_fraction, 0.0), 0.95)
        recompute = max(0, context_tokens) * (1.0 - retained) / max(kappa, 1.0)
        jitter = jitter_seconds if is_acting else 0.0
        return recompute + jitter

    def utility_victim_key(
        self,
        context_tokens: int,
        reclaimable_tokens: int,
        idleness: float,
        is_acting: bool,
        ref: ProgramRef,
        *,
        jitter_seconds: float,
        min_prefill_tokens_per_second: float,
        max_prefill_tokens_per_second: float,
    ) -> tuple[float, float, str, int]:
        """Order victims by seconds of recomputation per freed token, ascending."""
        cost = self.victim_cost_seconds(
            context_tokens,
            is_acting,
            jitter_seconds=jitter_seconds,
            min_prefill_tokens_per_second=min_prefill_tokens_per_second,
            max_prefill_tokens_per_second=max_prefill_tokens_per_second,
        )
        relief = max(1, reclaimable_tokens)
        return (cost / relief, -idleness, ref.program_id, ref.generation)


def is_acting_status(status: ProgramStatus | None) -> bool:
    return status is ProgramStatus.ACTING
