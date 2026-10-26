#!/usr/bin/env python3
"""Seal one fresh, plan-free MORI server runtime and print its shell environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

CANDIDATE_ROOT = Path(__file__).resolve().parent
DEFAULT_AGENTINFER_ROOT = CANDIDATE_ROOT.parents[1]
AGENTINFER_SERVER_FILES = (
    "agentinfer/agentcache/core/api_adapter.py",
    "agentinfer/agentcache/core/scheduler.py",
    "agentinfer/scheduling/identity.py",
    "agentinfer/scheduling/lifecycle.py",
    "agentinfer/scheduling/runtime.py",
)
MORI_SERVER_FILES = (
    "bootstrap/sitecustomize.py",
    "inputs/policy-config.json",
    "joint/__init__.py",
    "joint/eviction.py",
    "joint/mori.py",
    "joint/server_state.py",
    "mori_http_mp.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": _sha256(resolved)}


def _server_files(agentinfer_root: Path) -> list[Path]:
    files = [CANDIDATE_ROOT / name for name in MORI_SERVER_FILES]
    files.extend(agentinfer_root / name for name in AGENTINFER_SERVER_FILES)
    return files


def _kv_transfer_config(cpu_kv_gib: int) -> dict[str, object]:
    return {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "spec_name": "PrivatePinnedCPUOffloadingSpec",
            "spec_module_path": "private_pinned",
            "cpu_bytes_to_use": cpu_kv_gib * 1024**3,
            "eviction_policy": "lru",
            "store_threshold": 0,
            "offload_prompt_only": False,
        },
    }


def prepare(output: Path, agentinfer_root: Path, cpu_kv_gib: int = 0) -> dict[str, Any]:
    """Create immutable startup inputs for one fresh server process."""
    if cpu_kv_gib < 0:
        raise ValueError("cpu_kv_gib must be non-negative")
    output = output.expanduser().resolve()
    agentinfer_root = agentinfer_root.expanduser().resolve(strict=True)
    if not (agentinfer_root / "agentinfer").is_dir():
        raise ValueError(f"not an AgentInfer checkout: {agentinfer_root}")
    files = _server_files(agentinfer_root)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing MORI server sources: {missing}")

    output.mkdir(parents=True, exist_ok=False)
    preflight = {
        "schema_version": 1,
        "runtime_contract": {
            "candidate_root": str(CANDIDATE_ROOT),
            "agentinfer_root": str(agentinfer_root),
            "python_env": str(Path(sys.prefix).resolve()),
            "plan_available_to_server": False,
            "identity_source": "request_metadata",
            "lifecycle_source": "api_completion_signal",
        },
        "candidate_files": [_record(path) for path in files],
    }
    preflight_path = output / "server-preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    socket_tag = hashlib.sha256(str(output).encode()).hexdigest()[:12]
    python_path = os.pathsep.join(
        (
            str(CANDIDATE_ROOT / "bootstrap"),
            str(CANDIDATE_ROOT),
            str(CANDIDATE_ROOT / "adapters"),
            str(agentinfer_root),
        )
    )
    environment = {
        "AGENTCACHE_VLLM_LIFECYCLE_SOCKET": f"/tmp/agentinfer-mori-{socket_tag}.sock",
        "KVC_AGENTINFER_ROOT": str(agentinfer_root),
        "KVC_MORI_AUDIT_DIR": str(output),
        "KVC_MORI_PYTHON_ENV": str(Path(sys.prefix).resolve()),
        "KVC_MORI_ROOT": str(CANDIDATE_ROOT),
        "KVC_MORI_SERVER_PREFLIGHT": str(preflight_path),
        "LATEST_OFFLOAD_ALLOC_AUDIT": str(output / "offload-allocation.jsonl"),
        "PYTHONPATH": python_path,
    }
    if cpu_kv_gib:
        transfer_path = output / "kv-transfer-config.json"
        transfer_path.write_text(
            json.dumps(_kv_transfer_config(cpu_kv_gib), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        environment["KVC_MORI_KV_TRANSFER_CONFIG"] = str(transfer_path)

    launch = {
        "schema_version": 1,
        "cpu_kv_gib": cpu_kv_gib,
        "environment": environment,
        "server_preflight": str(preflight_path),
    }
    (output / "launch-env.json").write_text(json.dumps(launch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return launch


def shell_exports(environment: dict[str, str]) -> str:
    """Render deterministic exports suitable for redirecting to a fresh file."""
    return "\n".join(f"export {key}={shlex.quote(value)}" for key, value in sorted(environment.items())) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="fresh per-run audit directory")
    parser.add_argument(
        "--agentinfer-root",
        type=Path,
        default=DEFAULT_AGENTINFER_ROOT,
        help="checkout containing this MORI branch",
    )
    parser.add_argument("--cpu-kv-gib", type=int, default=0, help="native CPU KV capacity; zero disables offload")
    args = parser.parse_args()
    launch = prepare(args.output, args.agentinfer_root, args.cpu_kv_gib)
    sys.stdout.write(shell_exports(launch["environment"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
