"""Select native private pinned CPU tensors for this single-worker experiment.

The installed connector, manager, worker, and transfer implementations remain
unchanged. This out-of-tree spec only selects the existing non-mmap allocation
branch and validates its result. It is not the default CUDA allocation policy.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import inspect
import json
import os
import time
from pathlib import Path

from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

AUDIT_ENV = "LATEST_OFFLOAD_ALLOC_AUDIT"


def _class_name(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


def _emit(path, row):
    with path.open("a") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        handle.flush()


class PrivatePinnedCPUOffloadingSpec(CPUOffloadingSpec):
    """Restricted to one worker and the ordinary non-replicated KV layout."""

    def __init__(self, config):
        parallel = config.parallel
        dimensions = ("world_size", "tp_size", "pp_size", "pcp_size", "dcp_size", "data_parallel_size")
        if any(getattr(parallel, name) != 1 for name in dimensions):
            raise ValueError("Private pinned experiment requires world/TP/PP/PCP/DCP/DP sizes all equal to one")
        if (
            parallel.rank != 0
            or parallel.data_parallel_index != 0
            or parallel.data_parallel_rank_local not in (None, 0)
        ):
            raise ValueError("Private pinned experiment requires rank and DP indices zero")
        if config.canonical_layout or config.replicated_layout:
            raise ValueError("Private pinned experiment rejects canonical and replicated layouts")
        budget = config.extra_config.get("cpu_bytes_to_use")
        if type(budget) is not int or budget <= 0:
            raise ValueError("cpu_bytes_to_use must be a positive integer byte budget")
        super().__init__(config)

    def _uses_shared_region(self):
        return False

    def create_worker(self, kv_caches):
        audit_name = os.environ.get(AUDIT_ENV)
        if not audit_name:
            raise RuntimeError(f"{AUDIT_ENV} is required for allocation verification")
        audit_path = Path(audit_name)
        if not audit_path.parent.is_dir():
            raise RuntimeError("Allocation audit parent directory must already exist")
        requested = self.extra_config["cpu_bytes_to_use"]
        expected_pages = [tensor.page_size_bytes * self.blocks_per_chunk for tensor in kv_caches.tensors]
        expected_bytes = sum(expected_pages) * self.num_chunks
        row = {
            "event": "private_pinned_offload_allocation",
            "status": "started",
            "clock": time.monotonic(),
            "engine_id": self.config.engine_id,
            "pid": os.getpid(),
            "vllm_version": importlib.metadata.version("vllm"),
            "allocator_policy": "native_private_pinned_tensor_via_out_of_tree_spec",
            "parallel": dataclasses.asdict(self.config.parallel),
            "canonical_layout": self.config.canonical_layout,
            "replicated_layout": self.config.replicated_layout,
            "requested_bytes": requested,
            "num_chunks": self.num_chunks,
            "kv_bytes_per_chunk": self.kv_bytes_per_chunk,
            "expected_tensor_bytes": expected_bytes,
            "bytes_scope": "Tensor/storage byte sizes; not RSS or allocator-reserved physical bytes",
            "worker_class": _class_name(CPUOffloadingWorker),
            "worker_source": inspect.getfile(CPUOffloadingWorker),
            "manager_class": _class_name(CPUOffloadingManager),
            "manager_source": inspect.getfile(CPUOffloadingManager),
        }
        worker = None
        try:
            if not expected_pages or self.num_chunks <= 0:
                raise RuntimeError("Expected a nonempty CPU KV pool")
            if sum(expected_pages) != self.config.worker_kv_bytes_per_block * self.blocks_per_chunk:
                raise RuntimeError("Worker tensor page bytes differ from the capacity configuration")
            worker = super().create_worker(kv_caches)
            if type(worker) is not CPUOffloadingWorker:
                raise RuntimeError("Unexpected replacement of the native CPU offloading worker")
            store_tensors = worker._store_handler.dst_tensors
            load_tensors = worker._load_handler.src_tensors
            row["mmap_used"] = worker._mmap_region is not None
            row["tensors"] = [
                {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "bytes": tensor.numel() * tensor.element_size(),
                    "storage_bytes": tensor.untyped_storage().nbytes(),
                    "pinned": tensor.is_pinned(),
                }
                for tensor in store_tensors
            ]
            row["allocated_tensor_bytes"] = sum(item["bytes"] for item in row["tensors"])
            row["unused_budget_bytes"] = requested - row["allocated_tensor_bytes"]
            row["all_tensors_pinned"] = bool(store_tensors) and all(item["pinned"] for item in row["tensors"])
            if row["mmap_used"] or not row["all_tensors_pinned"]:
                raise RuntimeError("Expected private pinned tensors; mmap or pageable allocation is forbidden")
            if len(store_tensors) != len(expected_pages) or len(load_tensors) != len(store_tensors):
                raise RuntimeError("CPU tensor count differs from native KV tensor count")
            for index, (tensor, page_bytes) in enumerate(zip(store_tensors, expected_pages, strict=True)):
                if load_tensors[index] is not tensor:
                    raise RuntimeError("Store and load handlers do not share the same native CPU tensor")
                if list(tensor.shape) != [self.num_chunks, page_bytes] or str(tensor.device) != "cpu":
                    raise RuntimeError("CPU tensor shape/device does not match the native capacity model")
            if row["allocated_tensor_bytes"] != expected_bytes or expected_bytes > requested:
                raise RuntimeError("Allocated CPU tensor bytes do not match the requested capacity model")
            row.update(status="passed", finished_clock=time.monotonic())
            _emit(audit_path, row)
            return worker
        except BaseException as exc:
            row.update(status="failed", finished_clock=time.monotonic(), error=f"{type(exc).__name__}: {exc}")
            if worker is not None:
                try:
                    worker.shutdown()
                except Exception as cleanup_error:
                    row["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
            _emit(audit_path, row)
            raise
