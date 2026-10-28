# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

from agentinfer.agentbench.replay.config import load_replay_config

SUITE = Path(__file__).resolve().parents[2] / "benchmarks" / "mori-formal" / "swe-cc-local-trace"
TRACE = SUITE / "traces" / "cc-2.1.152-ptkvc-64g-terminal-32_16.requests.jsonl"
CONFIG = SUITE / "config" / "replay16-8.yaml"
TRACE_SHA256 = "d38b0afe97ea0dd0689d32bff6a888831637f3eb963660055d58a8b4d30c7b4a"


def test_mori_formal_replay_config_resolves_bundled_trace() -> None:
    config = load_replay_config(CONFIG)

    assert config.experiment.task_num == 16
    assert config.experiment.max_concurrency == 8
    assert config.replay.trace_type == "agentinfer"
    assert config.replay.trace_path == TRACE.resolve()
    assert config.replay.interval_mode == "trace"
    assert config.replay.sample_seed == 0
    assert config.backend.endpoint == "/v1/messages"


def test_mori_formal_trace_integrity_and_shape() -> None:
    assert hashlib.sha256(TRACE.read_bytes()).hexdigest() == TRACE_SHA256

    records = [json.loads(line) for line in TRACE.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2435
    assert len({record["session_id"] for record in records}) == 32
    assert all(record["status"] == "success" and record["status_code"] == 200 for record in records)
    assert not {"prompt", "messages", "content", "text"} & set().union(*(record.keys() for record in records))


def test_mori_formal_plan_keeps_common_simplecpu_settings() -> None:
    env = os.environ | {
        "MODEL_PATH": "/models/qwen3-coder",
        "BASELINE_PYTHON": "/venvs/native/bin/python",
        "MORI_PYTHON": "/venvs/mori/bin/python",
        "RESULT_ROOT": "/results/swe-cc-local-trace",
    }
    output = subprocess.check_output(["bash", str(SUITE / "run_matrix.sh"), "plan"], env=env, text=True)
    plan = dict(line.split("=", 1) for line in output.splitlines())
    commands = {arm: shlex.split(plan[f"server[{arm}]"]) for arm in ("vanilla", "pt", "mori")}

    kv_configs = []
    for command in commands.values():
        assert "--async-scheduling" in command
        assert "--enable-prefix-caching" in command
        assert "--enforce-eager" not in command
        kv_configs.append(json.loads(command[command.index("--kv-transfer-config") + 1]))

    assert kv_configs[0] == kv_configs[1] == kv_configs[2]
    assert kv_configs[0]["kv_connector"] == "SimpleCPUOffloadConnector"
    assert kv_configs[0]["kv_connector_extra_config"] == {
        "cpu_bytes_to_use": 17179869184,
        "kv_offload_backend": "cpu",
        "lazy_offload": False,
    }
    assert "--agentinfer" not in commands["vanilla"]
    assert "--agentinfer" in commands["pt"]
    assert "--agentinfer" in commands["mori"]
