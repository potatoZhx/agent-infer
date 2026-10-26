"""Fail closed when the isolated MORI server starts in the wrong vLLM env.

MORI has no oracle prefetch. No EngineCore WAKEUP shim is installed here;
policy hooks are attached by the candidate scheduler itself.
"""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path

if root_string := os.environ.get("KVC_MORI_ROOT"):
    root = Path(root_string).resolve(strict=True)
    if Path(__file__).resolve().parent != root / "bootstrap":
        raise RuntimeError("MORI bootstrap escaped the isolated candidate root")
    version = importlib.metadata.version("vllm")
    if version != "0.31.0+cu129":
        raise RuntimeError(f"MORI requires vLLM 0.31.0+cu129, found {version}")
    os.environ["KVC_MORI_BOOTSTRAP_ACTIVE"] = "1"
