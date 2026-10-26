from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
MORI_ROOT = REPO / "examples/mori"


def _load_prepare_server() -> ModuleType:
    path = MORI_ROOT / "prepare_server.py"
    spec = importlib.util.spec_from_file_location("mori_prepare_server", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(program_id: str = "previously-unseen-program") -> SimpleNamespace:
    return SimpleNamespace(
        request_id="native-new-request",
        sampling_params=SimpleNamespace(
            extra_args={
                "agentic_context": json.dumps(
                    {
                        "program_id": program_id,
                        "task_id": "previously-unseen-task",
                        "request_id": "previously-unseen-wire-request",
                    }
                )
            }
        ),
    )


def test_server_state_accepts_unknown_request_without_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(MORI_ROOT))
    from joint.server_state import ServerRequestState

    request = _request()
    state = ServerRequestState()
    completed: list[tuple[object, object]] = []
    state.subscribe(lambda observed, advice: completed.append((observed, advice)))

    assert state.register_ingress(request) == "previously-unseen-program"
    assert state.program_for(request) == "previously-unseen-program"
    state.complete(request)

    assert completed == [(request, None)]
    assert state.summary() == {
        "identity_source": "request_metadata",
        "lifecycle_source": "api_completion_signal",
        "server_plan_reads": 0,
        "future_oracle_reads": 0,
        "registered_requests": 1,
        "completed_requests": 1,
        "programs_seen": 1,
        "counters": {
            "request_identities_parsed": 1,
            "request_ingress": 1,
            "request_identity_rechecks": 2,
            "request_completions": 1,
        },
        "errors": [],
    }


def test_server_state_rejects_identity_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(MORI_ROOT))
    from joint.server_state import ServerRequestState

    request = _request("program-a")
    state = ServerRequestState()
    state.register_ingress(request)
    request.sampling_params.extra_args = {
        "agentic_context": json.dumps({"program_id": "program-b", "request_id": "wire-b"})
    }

    with pytest.raises(ValueError, match="Program identity changed"):
        state.program_for(request)


@pytest.mark.parametrize("cpu_kv_gib", [0, 2])
def test_prepare_server_seals_only_plan_free_server_sources(tmp_path: Path, cpu_kv_gib: int) -> None:
    prepare_server = _load_prepare_server()
    output = tmp_path / "runtime"

    launch = prepare_server.prepare(output, REPO, cpu_kv_gib)

    preflight = json.loads((output / "server-preflight.json").read_text())
    contract = preflight["runtime_contract"]
    assert contract["plan_available_to_server"] is False
    assert contract["identity_source"] == "request_metadata"
    assert contract["lifecycle_source"] == "api_completion_signal"
    paths = [Path(item["path"]) for item in preflight["candidate_files"]]
    assert paths
    assert all(path.is_relative_to(MORI_ROOT) or path.is_relative_to(REPO / "agentinfer") for path in paths)
    assert not any(path.name in {"planner.py", "replay.py"} for path in paths)
    assert "KVC_MORI_PLAN" not in launch["environment"]
    assert launch["environment"]["KVC_AGENTINFER_ROOT"] == str(REPO)
    if cpu_kv_gib:
        transfer = json.loads((output / "kv-transfer-config.json").read_text())
        assert transfer["kv_connector_extra_config"]["cpu_bytes_to_use"] == cpu_kv_gib * 1024**3
        assert transfer["kv_connector_extra_config"]["offload_prompt_only"] is False
    else:
        assert "KVC_MORI_KV_TRANSFER_CONFIG" not in launch["environment"]


def test_portable_adapter_has_no_benchmark_host_path() -> None:
    adapter = (MORI_ROOT / "mori_http_mp.py").read_text()
    bootstrap = (MORI_ROOT / "bootstrap/sitecustomize.py").read_text()

    assert "/data/zx" not in adapter
    assert "/data/zx" not in bootstrap
    assert 'os.environ["KVC_MORI_PYTHON_ENV"]' in adapter
    assert 'version != "0.31.0+cu129"' in adapter
    assert 'version != "0.31.0+cu129"' in bootstrap


def test_mori_server_package_does_not_contain_replay_plan_code() -> None:
    relative_files = {path.relative_to(MORI_ROOT).as_posix() for path in MORI_ROOT.rglob("*") if path.is_file()}

    assert "joint/replay.py" not in relative_files
    assert not any(path.endswith("plan.json") for path in relative_files)
    assert "inputs/policy-config.json" in relative_files


@pytest.fixture(autouse=True)
def clear_isolated_joint_modules() -> None:
    yield
    for name in tuple(sys.modules):
        if name == "joint" or name.startswith("joint."):
            sys.modules.pop(name, None)
