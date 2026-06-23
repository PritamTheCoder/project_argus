"""
Project Argus - Tool telemetry tests
"""

import pytest

from src.tools.tool_telemetry import ToolCallTracer, timed_tool_call


def test_tracer_records_and_summarizes():
    t = ToolCallTracer()
    t.record("web_search", {"query": "x"}, 12.3, True, result_count=5)
    t.record("web_search", {"query": "y"}, 8.0, True, result_count=3)
    t.record("arxiv_search", {"query": "z"}, 50.0, False, error="boom")

    s = t.summary()
    assert s["total_calls"] == 3
    assert s["total_failures"] == 1
    assert s["by_tool"]["web_search"]["calls"] == 2
    assert s["by_tool"]["web_search"]["results"] == 8
    assert s["by_tool"]["arxiv_search"]["failures"] == 1


def test_args_preview_truncated():
    t = ToolCallTracer()
    rec = t.record("x", {"q": "a" * 500}, 1.0, True)
    assert len(rec.args_preview) <= 210
    assert rec.args_preview.endswith("…")


@pytest.mark.asyncio
async def test_timed_tool_call_success():
    t = ToolCallTracer()
    async with timed_tool_call(t, "web_search", {"q": "x"}) as call:
        call.set_result(result_count=4)
    assert len(t.records) == 1
    rec = t.records[0]
    assert rec.success is True
    assert rec.result_count == 4
    assert rec.latency_ms >= 0


@pytest.mark.asyncio
async def test_timed_tool_call_records_exception_and_reraises():
    t = ToolCallTracer()
    with pytest.raises(ValueError):
        async with timed_tool_call(t, "arxiv_search", {"q": "x"}):
            raise ValueError("kaboom")
    assert len(t.records) == 1
    assert t.records[0].success is False
    assert "ValueError" in t.records[0].error


@pytest.mark.asyncio
async def test_timed_tool_call_tolerates_no_tracer():
    # Must not crash when tracer is None.
    async with timed_tool_call(None, "x", {}) as call:
        call.set_result(result_count=1)
