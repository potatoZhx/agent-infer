# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Controller selection and strict MORI configuration tests."""

from __future__ import annotations

import pytest

from agentinfer.agentcache.core.controller import build_controller, build_mori_controller
from agentinfer.scheduling.backend import BackendInfo, BackendPoolInfo, DpRankInfo
from agentinfer.scheduling.mori import MORIMode, MORIStrategy

pytestmark = pytest.mark.cpu_test


def backend(*, dram: int | None = 2000, chunk: int | None = 16) -> BackendPoolInfo:
    return BackendPoolInfo(
        (
            BackendInfo(
                "vllm-local",
                "embedded://vllm",
                True,
                (
                    DpRankInfo(
                        0,
                        True,
                        True,
                        total_hbm_kv_tokens=1000,
                        total_dram_kv_tokens=dram,
                        hbm_block_size_tokens=16,
                        dram_chunk_size_tokens=chunk,
                    ),
                ),
            ),
        )
    )


def test_controller_selects_mori_without_a_dynamic_factory() -> None:
    controller = build_controller(backend(), {"strategy": "mori", "mori": {"window_size": 5}})

    assert isinstance(controller.strategy, MORIStrategy)
    assert controller.strategy.config.mode is MORIMode.FULL
    assert controller.strategy.config.decode_buffer_tokens == 100
    assert controller._schedule_interval_seconds == 5.0


def test_full_mori_fails_closed_without_dram_capacity() -> None:
    with pytest.raises(ValueError, match="DRAM quota"):
        build_mori_controller(backend(dram=None, chunk=None), {"mori": {}})


def test_placement_only_is_an_explicit_ablation_without_dram() -> None:
    controller = build_mori_controller(
        backend(dram=None, chunk=None),
        {"mori": {"mode": "placement_only"}},
    )

    assert controller.strategy.config.mode is MORIMode.PLACEMENT_ONLY


def test_decode_buffer_cannot_be_overridden() -> None:
    with pytest.raises(ValueError, match="fixed implementation"):
        build_mori_controller(backend(), {"mori": {"decode_buffer_tokens": 999}})


@pytest.mark.parametrize(
    "setting", ["starvation_timeout_seconds", "min_residency_seconds", "idleness_margin", "max_wait_seconds"]
)
def test_controller_rejects_removed_policy_extensions(setting) -> None:
    with pytest.raises(ValueError, match=f"not supported: {setting}"):
        build_mori_controller(backend(), {"mori": {setting: 60.0}})
