"""
Project Argus - Tool-call telemetry

Lightweight, dependency-free observability for agent tool calls. Each call is
recorded with its latency, success, and result size, and a tracer summarises a
run. Useful for diagnostics and per-run cost tracking.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_ARGS_PREVIEW = 200


def _preview_args(args: Any) -> str:
    s = str(args)
    return s if len(s) <= _MAX_ARGS_PREVIEW else s[:_MAX_ARGS_PREVIEW] + "…"


@dataclass
class ToolCallRecord:
    name: str
    args_preview: str
    latency_ms: float
    success: bool
    result_count: Optional[int] = None
    error: Optional[str] = None


@dataclass
class ToolCallTracer:
    """Collects ToolCallRecords for a single gathering session/run."""
    records: List[ToolCallRecord] = field(default_factory=list)

    def record(
        self,
        name: str,
        args: Any,
        latency_ms: float,
        success: bool,
        result_count: Optional[int] = None,
        error: Optional[str] = None,
    ) -> ToolCallRecord:
        rec = ToolCallRecord(
            name=name,
            args_preview=_preview_args(args),
            latency_ms=round(latency_ms, 1),
            success=success,
            result_count=result_count,
            error=error,
        )
        self.records.append(rec)
        return rec

    def summary(self) -> Dict[str, Any]:
        by_tool: Dict[str, Dict[str, Any]] = {}
        for r in self.records:
            t = by_tool.setdefault(r.name, {"calls": 0, "failures": 0, "results": 0, "latency_ms": 0.0})
            t["calls"] += 1
            t["failures"] += 0 if r.success else 1
            t["results"] += r.result_count or 0
            t["latency_ms"] += r.latency_ms
        return {
            "total_calls": len(self.records),
            "total_failures": sum(1 for r in self.records if not r.success),
            "total_latency_ms": round(sum(r.latency_ms for r in self.records), 1),
            "by_tool": by_tool,
        }

    def backend_health(self) -> Dict[str, Dict[str, Any]]:
        """Per-retrieval-backend health for the whole run.

        A backend that raises is already visible as a failure. The dangerous
        case is one that succeeds and returns nothing: the pipeline then falls
        through to whatever other backend answered, which may be the wrong
        corpus entirely, and the run still reports high credibility. `degraded`
        marks a backend that was called and never returned a single result.
        """
        health: Dict[str, Dict[str, Any]] = {}
        for r in self.records:
            h = health.setdefault(r.name, {
                "calls": 0, "failures": 0, "empty_calls": 0,
                "results": 0, "latency_ms": 0.0,
            })
            h["calls"] += 1
            h["failures"] += 0 if r.success else 1
            h["results"] += r.result_count or 0
            h["latency_ms"] += r.latency_ms
            if r.success and not (r.result_count or 0):
                h["empty_calls"] += 1

        for h in health.values():
            h["latency_ms"] = round(h["latency_ms"], 1)
            h["degraded"] = h["results"] == 0 and h["calls"] > 0
        return health


class timed_tool_call:
    """Async context manager that records one tool call into a tracer.

    Usage:
        async with timed_tool_call(tracer, name, args) as call:
            result = await tool.ainvoke(args)
            call.set_result(result_count=len(result))
    """

    def __init__(self, tracer: Optional[ToolCallTracer], name: str, args: Any):
        self.tracer = tracer
        self.name = name
        self.args = args
        self._start = 0.0
        self._result_count: Optional[int] = None
        self._success = True
        self._error: Optional[str] = None

    def set_result(self, result_count: Optional[int] = None) -> None:
        self._result_count = result_count

    async def __aenter__(self) -> "timed_tool_call":
        self._start = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        latency_ms = (time.perf_counter() - self._start) * 1000.0
        if exc is not None:
            self._success = False
            self._error = f"{exc_type.__name__}: {exc}"
        if self.tracer is not None:
            self.tracer.record(
                self.name, self.args, latency_ms, self._success,
                result_count=self._result_count, error=self._error,
            )
        return False  # never suppress exceptions
