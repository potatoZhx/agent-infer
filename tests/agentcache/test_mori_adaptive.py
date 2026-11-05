# SPDX-License-Identifier: Apache-2.0
# FileCopyrightText: Copyright contributors to the AgentInfer project
"""Adaptive scheduling guardrails: structural drain, utility ordering, decode guard."""

from __future__ import annotations

import pytest

from agentinfer.scheduling import ProgramRef
from agentinfer.scheduling.mori import build_mori_strategy
from agentinfer.scheduling.mori.config import MORIConfig
from agentinfer.scheduling.mori.utility import UtilityRuntimeState

pytestmark = pytest.mark.cpu_test


def _strategy(**config_kwargs) -> object:
    components = build_mori_strategy(MORIConfig(**config_kwargs))
    return components.strategy


class TestStructuralDrain:
    def test_disabled_by_default_keeps_reservation_gate_closed(self):
        strategy = _strategy()
        assert strategy.config.drain_wait_threshold_seconds == 0.0
        strategy._update_drain_state(True, now=100.0)
        assert strategy._drain_persist == 0
        assert strategy._drain_armed_until is None
        assert strategy._effective_ready_wait_limit(100.0) == 0.0

    def test_arming_requires_persistence_streak(self):
        strategy = _strategy(drain_wait_threshold_seconds=180.0, drain_persist_cycles=3, drain_min_dwell_seconds=60.0)
        strategy._update_drain_state(True, now=10.0)
        strategy._update_drain_state(True, now=20.0)
        assert strategy._effective_ready_wait_limit(20.0) == 0.0
        strategy._update_drain_state(True, now=30.0)
        assert strategy._effective_ready_wait_limit(30.0) == 180.0
        assert strategy.counters["drain_armed_transitions"] == 1

    def test_hold_survives_brief_condition_loss_and_rearm_needs_streak(self):
        strategy = _strategy(drain_wait_threshold_seconds=180.0, drain_persist_cycles=1, drain_min_dwell_seconds=60.0)
        strategy._update_drain_state(True, now=10.0)
        assert strategy._effective_ready_wait_limit(50.0) == 180.0  # inside dwell hold
        assert strategy._effective_ready_wait_limit(71.0) == 0.0  # dwell expired
        strategy._update_drain_state(False, now=72.0)
        strategy._update_drain_state(True, now=73.0)
        assert strategy._effective_ready_wait_limit(73.0) == 180.0

    def test_streak_reset_on_gap(self):
        strategy = _strategy(drain_wait_threshold_seconds=180.0, drain_persist_cycles=3)
        strategy._update_drain_state(True, now=10.0)
        strategy._update_drain_state(True, now=20.0)
        strategy._update_drain_state(False, now=30.0)
        strategy._update_drain_state(True, now=40.0)
        strategy._update_drain_state(True, now=50.0)
        assert strategy._effective_ready_wait_limit(50.0) == 0.0

    def test_static_ready_wait_limit_takes_precedence(self):
        strategy = _strategy(ready_wait_limit_seconds=5.0, drain_wait_threshold_seconds=180.0)
        assert strategy._effective_ready_wait_limit(0.0) == 5.0


class TestUtilityOrdering:
    def test_calibration_updates_kappa_and_eta(self):
        state = UtilityRuntimeState()
        for _ in range(5):
            state.observe_request_finished(
                uncached_tokens=100_000, service_seconds=10.0, prompt_tokens=200_000, prefix_hit_tokens=100_000
            )
        assert 8_000 < state.prefill_tokens_per_second <= 12_000
        assert 0.4 < state.retained_fraction <= 0.6

    def test_cost_grows_with_context_and_acts_cheap_without_retention(self):
        state = UtilityRuntimeState()
        state.observe_request_finished(uncached_tokens=10_000, service_seconds=1.0, prompt_tokens=10_000, prefix_hit_tokens=0)
        cheap = state.victim_cost_seconds(
            10_000, False, jitter_seconds=2.0, min_prefill_tokens_per_second=2_000, max_prefill_tokens_per_second=50_000
        )
        expensive = state.victim_cost_seconds(
            100_000, False, jitter_seconds=2.0, min_prefill_tokens_per_second=2_000, max_prefill_tokens_per_second=50_000
        )
        assert expensive > cheap

    def test_key_prefers_cheap_recompute_per_freed_token(self):
        state = UtilityRuntimeState()
        state.observe_request_finished(uncached_tokens=10_000, service_seconds=1.0, prompt_tokens=10_000, prefix_hit_tokens=0)
        kwargs = dict(jitter_seconds=2.0, min_prefill_tokens_per_second=2_000, max_prefill_tokens_per_second=50_000)
        small = state.utility_victim_key(10_000, 10_000, 0.5, False, ProgramRef("a", 0), **kwargs)
        large = state.utility_victim_key(100_000, 10_000, 0.5, False, ProgramRef("b", 0), **kwargs)
        shared = state.utility_victim_key(100_000, 50_000, 0.5, False, ProgramRef("c", 0), **kwargs)
        assert small < large  # cheaper context first at equal relief
        assert shared < large  # shared-prefix victim frees more per unit cost

    def test_sort_victims_warmup_uses_legacy_key(self):
        strategy = _strategy(utility_victim_order=True, utility_min_samples=20)
        assert strategy._utility.samples == 0
        assert not (strategy._utility.samples >= strategy.config.utility_min_samples and strategy._utility.ready)


class TestDecodeGuardConfig:
    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="decode_batch_mode"):
            MORIConfig(decode_batch_mode="turbo")

    def test_guard_ratio_must_exceed_one(self):
        with pytest.raises(ValueError, match="decode_batch_tpot_guard_ratio"):
            MORIConfig(decode_batch_mode="adaptive", decode_batch_tpot_guard_ratio=1.0)

    def test_defaults_are_static_and_off(self):
        config = MORIConfig()
        assert config.decode_batch_mode == "static"
        assert config.decode_batch_tokens == 0
        assert config.drain_wait_threshold_seconds == 0.0
        assert config.utility_victim_order is False

    def test_drain_validation(self):
        with pytest.raises(ValueError, match="drain_persist_cycles"):
            MORIConfig(drain_wait_threshold_seconds=180.0, drain_persist_cycles=0)
        with pytest.raises(ValueError, match="drain_max_victims_per_cycle"):
            MORIConfig(drain_wait_threshold_seconds=180.0, drain_max_victims_per_cycle=0)
