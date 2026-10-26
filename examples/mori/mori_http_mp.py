"""Causal MORI policy at the real vLLM 0.31 HTTP/MP EngineCore boundary."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import joint
import vllm
from joint.mori import MORIConfig, MORICPUTypedPolicy, MORIGlobalFactors, MORIGPUEviction, MORIStrategy
from joint.server_state import ServerRequestState

import agentinfer
from agentinfer.agentcache.core.scheduler import AgentCacheAsyncSchedulerBridge
from agentinfer.scheduling.factors import StrategyFactors


class _Runtime:
    def __init__(self, path: Path):
        self._stream = path.open("x", buffering=1024 * 1024)

    def emit(self, row):
        self._stream.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")

    def close(self):
        self._stream.close()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(module, directory: Path) -> str:
    source = Path(module.__file__).resolve(strict=True)
    if not source.is_relative_to(directory):
        raise RuntimeError(f"isolated MORI import escaped {directory}: {source}")
    return str(source)


def _source_gate(root: Path, out: Path) -> dict:
    if os.environ.get("KVC_MORI_BOOTSTRAP_ACTIVE") != "1":
        raise RuntimeError("MORI bootstrap did not validate the real server environment")
    if Path(__file__).resolve().parent != root:
        raise RuntimeError("MORI scheduler was imported from outside the candidate root")
    expected_env = Path(os.environ["KVC_MORI_PYTHON_ENV"]).resolve(strict=True)
    if Path(sys.prefix).resolve() != expected_env:
        raise RuntimeError(f"wrong MORI server Python environment: {sys.prefix}")
    version = importlib.metadata.version("vllm")
    if version != "0.31.0+cu129":
        raise RuntimeError(f"MORI requires vLLM 0.31.0+cu129, found {version}")
    agentinfer_root = Path(os.environ["KVC_AGENTINFER_ROOT"]).resolve(strict=True)
    imported = {
        "agentinfer": _inside(agentinfer, agentinfer_root / "agentinfer"),
        "joint": _inside(joint, root / "joint"),
        "vllm": str(Path(vllm.__file__).resolve(strict=True)),
    }
    if not Path(imported["vllm"]).is_relative_to(expected_env):
        raise RuntimeError("vLLM import did not come from the v0.31 benchmark environment")
    forbidden_plan_environment = sorted(key for key in os.environ if key == "KVC_MORI_PLAN")
    if forbidden_plan_environment:
        raise RuntimeError(f"MORI server received forbidden plan environment: {forbidden_plan_environment}")
    server_preflight_path = Path(os.environ["KVC_MORI_SERVER_PREFLIGHT"]).resolve(strict=True)
    if server_preflight_path != out / "server-preflight.json":
        raise RuntimeError("MORI server preflight must be the isolated server-only manifest")
    preflight = json.loads(server_preflight_path.read_text())
    if set(preflight) != {"schema_version", "runtime_contract", "candidate_files"}:
        raise RuntimeError("MORI server preflight contains non-server runtime material")
    if preflight["runtime_contract"]["candidate_root"] != str(root):
        raise RuntimeError("MORI preflight candidate root differs from Core import root")
    if preflight["runtime_contract"].get("plan_available_to_server") is not False:
        raise RuntimeError("MORI server runtime must explicitly forbid plan availability")
    checked = 0
    for item in preflight["candidate_files"]:
        path = Path(item["path"]).resolve(strict=True)
        if not (path.is_relative_to(root) or path.is_relative_to(agentinfer_root)):
            raise RuntimeError(f"MORI server preflight escaped approved source roots: {path}")
        if _sha(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"MORI candidate source changed after preparation: {path}")
        checked += 1
    if checked == 0:
        raise RuntimeError("MORI preflight did not freeze any candidate source")
    return {
        "version": version,
        "imports": imported,
        "candidate_source_files_checked": checked,
        "server_preflight": str(server_preflight_path),
        "forbidden_plan_environment": forbidden_plan_environment,
    }


def _policy(path: Path) -> dict:
    value = json.loads(path.read_text())
    mori = value.get("mori", {})
    expected = (
        value.get("name") == "mori-public-causal-k5-tick5"
        and mori == {"window_size": 5, "tick_seconds": 5.0, "decode_buffer_tokens": 256}
        and value.get("causality")
        == {
            "future_return_reads": False,
            "server_plan_reads": False,
            "identity_source": "request_metadata",
            "lifecycle_source": "api_completion_signal",
        }
        and value.get("gpu_eviction", {}).get("type_order") == ["inactive", "idle", "busy"]
        and value.get("cpu_eviction", {}).get("type_order") == ["inactive", "busy", "idle"]
        and value.get("gpu_eviction", {}).get("native_ownership") is True
        and value.get("cpu_eviction", {}).get("native_ownership") is True
    )
    if not expected:
        raise RuntimeError("MORI frozen causal policy contract changed")
    return value


def register_arrival(state, request):
    """Observe true Core ingress and parse identity from this request only."""
    return state.register_ingress(request)


def _offload_components(scheduler):
    """Resolve optional native CPU offload without making MORI depend on a connector."""
    connector = getattr(scheduler, "connector", None)
    native = getattr(connector, "scheduler", None)
    if native is None:
        return None, None, None, 0
    if native.config.offload_prompt_only:
        raise RuntimeError("MORI real benchmark requires full-KV CPU offload")
    if len(native.config.kv_group_configs) != 1:
        raise RuntimeError("MORI real benchmark requires one native KV group")
    manager = native.manager
    group = native.config.kv_group_configs[0]
    return native, manager, group, manager._num_chunks * group.tokens_per_chunk


class MoriAsyncScheduler(AgentCacheAsyncSchedulerBridge):
    """Attach MORI only to this candidate's v0.31 EngineCore process."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        root = Path(os.environ["KVC_MORI_ROOT"]).resolve(strict=True)
        out = Path(os.environ["KVC_MORI_AUDIT_DIR"]).resolve(strict=True)
        source = _source_gate(root, out)
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise RuntimeError("MORI real benchmark requires a reserved visible GPU")
        native, manager, _group, cpu_capacity_tokens = _offload_components(self)
        policy = _policy(root / "inputs/policy-config.json")
        state = ServerRequestState()
        runtime = _Runtime(out / "joint-policy.jsonl")
        controller = self._agentcache.controller
        config = MORIConfig(
            cpu_capacity_tokens=cpu_capacity_tokens,
            **policy["mori"],
        )
        strategy = MORIStrategy(config)
        factors = StrategyFactors(MORIGlobalFactors())
        strategy.bind_factors(factors)
        controller.strategy = strategy
        controller.strategy_factors = factors
        controller._schedule_interval_seconds = config.tick_seconds
        controller._next_periodic_check_at_monotonic_s = 0.0
        controller._next_check_at_monotonic_s = None
        controller._next_lightweight_check_at_monotonic_s = None
        state.install_completion_hook(self)
        eviction = MORIGPUEviction.create(self, state, strategy, emit=runtime.emit)
        cpu_typed = (
            MORICPUTypedPolicy(manager, native, state, strategy, emit=runtime.emit) if manager is not None else None
        )
        self._mori_state = state
        self._mori_runtime = runtime
        self._mori_strategy = strategy
        self._mori_eviction = eviction
        self._mori_cpu_typed = cpu_typed
        self._mori_closed = False
        runtime.emit(
            {
                "kind": "mori_installed",
                "native_gpu": True,
                "execution_path": "http_mp",
                "source_gate": source,
                "policy": policy,
                "resolved_mori_config": asdict(config),
                "enabled_components": {
                    "placement": True,
                    "gpu_typed_eviction": True,
                    "cpu_typed_eviction": cpu_typed is not None,
                    "oracle_prefetch": False,
                },
                "server_runtime_contract": {
                    "identity_source": state.identity_source,
                    "lifecycle_source": state.lifecycle_source,
                    "server_plan_reads": state.server_plan_reads,
                },
                "future_oracle_reads": 0,
            }
        )

    def add_request(self, request):
        register_arrival(self._mori_state, request)
        self._mori_eviction.observe(request)
        return super().add_request(request)

    def _summary(self) -> dict:
        strategy = self._mori_strategy.snapshot()
        cpu_typed = self._mori_cpu_typed.snapshot() if self._mori_cpu_typed is not None else None
        connector = getattr(self, "connector", None)
        native = getattr(connector, "scheduler", None)
        manager = native.manager if native is not None else None
        pool = self.kv_cache_manager.block_pool
        usable = [block for block in pool.blocks if not block.is_null]
        entries = list(manager._policy.chunks.values()) if manager is not None else []
        checks = {
            "mori_future_oracle_reads_zero": strategy.get("future_oracle_reads") == 0
            and (cpu_typed is None or cpu_typed.get("future_oracle_reads") == 0),
            "mori_factor_index_consistent": strategy.get("tier_index_consistent") is True,
            "gpu_typed_policy_installed": self._mori_eviction is not None,
            "cpu_typed_policy_matches_capacity": (self._mori_cpu_typed is not None) == (manager is not None),
            "hbm_refs_nonnegative": all(block.ref_cnt >= 0 for block in usable),
            "cpu_capacity_respected": manager is None or len(entries) <= manager._num_chunks,
            "cpu_all_ready": all(chunk.is_ready for chunk in entries),
            "cpu_no_transfer_refs": all(chunk.ref_cnt == 0 for chunk in entries),
            "native_jobs_drained": native is None or not native._jobs,
            "native_fences_drained": native is None or not any(native._block_id_to_pending_jobs.values()),
            "ordinary_requests_drained": not self.requests,
            "server_state_errors_zero": not self._mori_state.errors,
            "server_plan_reads_zero": self._mori_state.server_plan_reads == 0,
            "identity_source_request_metadata": self._mori_state.identity_source == "request_metadata",
        }
        return {
            "server_state": self._mori_state.summary(),
            "joint": {
                "passed": all(checks.values()),
                "native_gpu": True,
                "checks": checks,
                "strategy": {
                    "mori": strategy,
                    "gpu_typed": {
                        "stats": dict(self._mori_eviction.stats),
                        "decision_elapsed_seconds": self._mori_eviction.seconds,
                        "decision_thread_cpu_seconds": self._mori_eviction.thread_cpu_seconds,
                        "tracked_requests": len(self._mori_eviction.requests),
                    },
                    "cpu_typed": cpu_typed,
                },
                "native_ownership": {
                    "cpu_chunks": len(entries),
                    "cpu_capacity_chunks": manager._num_chunks if manager is not None else 0,
                    "native_jobs": len(native._jobs) if native is not None else 0,
                    "native_fences": (
                        sum(bool(value) for value in native._block_id_to_pending_jobs.values())
                        if native is not None
                        else 0
                    ),
                    "ordinary_requests": len(self.requests),
                },
            },
        }

    def shutdown(self):
        if not getattr(self, "_mori_closed", True):
            self._mori_closed = True
            try:
                out = Path(os.environ["KVC_MORI_AUDIT_DIR"])
                (out / "joint-summary.json").write_text(json.dumps(self._summary(), indent=2, default=str) + "\n")
            finally:
                if self._mori_cpu_typed is not None:
                    self._mori_cpu_typed.close()
                self._mori_eviction.close()
                self._mori_runtime.close()
        super().shutdown()
