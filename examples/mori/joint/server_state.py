"""Plan-free causal request ownership observed inside the real EngineCore."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from types import MethodType
from typing import Any

CompletionSubscriber = Callable[[Any, None], None]


class ServerRequestState:
    """Track only identities and completions already observed by EngineCore.

    The state deliberately has no constructor argument or method that can accept
    a Replay plan.  Program ownership is parsed from each native request's
    ``agentic_context`` and then pinned to that native request id.
    """

    identity_source = "request_metadata"
    lifecycle_source = "api_completion_signal"
    server_plan_reads = 0
    future_oracle_reads = 0

    def __init__(self) -> None:
        self.clock = time
        self.program_by_request: dict[str, str] = {}
        self.wire_request_by_request: dict[str, str | None] = {}
        self.completed_requests: set[str] = set()
        self.programs_seen: set[str] = set()
        self.counters: Counter[str] = Counter()
        self.errors: list[str] = []
        self._subscribers: list[CompletionSubscriber] = []

    @staticmethod
    def _identity(request: Any) -> Any:
        from agentinfer.scheduling.identity import parse_agent_identity

        identity = parse_agent_identity(vllm_xargs=request.sampling_params.extra_args, headers={})
        if identity is None:
            raise ValueError("native request has no AgentInfer Program identity")
        return identity

    def register_ingress(self, request: Any) -> str:
        """Parse and retain one previously unknown, legal request identity."""
        native_request_id = request.request_id
        if native_request_id in self.program_by_request:
            raise ValueError(f"duplicate native request ingress: {native_request_id}")
        identity = self._identity(request)
        self.program_by_request[native_request_id] = identity.program_id
        self.wire_request_by_request[native_request_id] = identity.request_id
        self.programs_seen.add(identity.program_id)
        self.counters["request_identities_parsed"] += 1
        self.counters["request_ingress"] += 1
        return identity.program_id

    def program_for(self, request: Any) -> str:
        """Return the request-derived owner and reject metadata mutation."""
        native_request_id = request.request_id
        expected = self.program_by_request.get(native_request_id)
        if expected is None:
            raise ValueError(f"request was not registered at EngineCore ingress: {native_request_id}")
        actual = self._identity(request).program_id
        self.counters["request_identity_rechecks"] += 1
        if actual != expected:
            raise ValueError(f"request Program identity changed after ingress: {native_request_id}")
        return expected

    def subscribe(self, callback: CompletionSubscriber) -> None:
        self._subscribers.append(callback)

    def complete(self, request: Any) -> None:
        native_request_id = request.request_id
        self.program_for(request)
        if native_request_id in self.completed_requests:
            raise ValueError(f"duplicate native request completion: {native_request_id}")
        self.completed_requests.add(native_request_id)
        self.counters["request_completions"] += 1
        for callback in tuple(self._subscribers):
            callback(request, None)

    def install_completion_hook(self, scheduler: Any) -> None:
        """Publish completion before native free/offload metadata is retired."""
        original = scheduler._free_request
        state = self

        def before_free(instance: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
            state.complete(request)
            return original(request, *args, **kwargs)

        scheduler._free_request = MethodType(before_free, scheduler)

    def summary(self) -> dict[str, Any]:
        return {
            "identity_source": self.identity_source,
            "lifecycle_source": self.lifecycle_source,
            "server_plan_reads": self.server_plan_reads,
            "future_oracle_reads": self.future_oracle_reads,
            "registered_requests": len(self.program_by_request),
            "completed_requests": len(self.completed_requests),
            "programs_seen": len(self.programs_seen),
            "counters": dict(self.counters),
            "errors": list(self.errors),
        }
