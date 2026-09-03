import pytest
from unittest.mock import MagicMock, patch

from src.api.worker import _run


async def _fake_stream_success(query, thread_id=None, callbacks=None):
    yield "librarian", {"plan": [{"query": "q1", "mode": "MIXED"}]}
    yield "scout", {"scraped_data": [{"url": "http://a.com"}]}
    yield "__final__", {
        "report": "final report text",
        "source_map": {"[1]": {"url": "http://a.com"}},
        "quality_score": {"coverage": 0.9},
        "node_seconds": {"librarian": 1.2, "scout": 3.4},
    }


async def _fake_stream_failure(query, thread_id=None, callbacks=None):
    yield "librarian", {"plan": []}
    raise RuntimeError("provider outage")


@pytest.mark.asyncio
@patch("src.api.worker.astream_research", _fake_stream_success)
@patch("src.api.worker.job_store")
async def test_run_updates_job_progress_then_marks_done(mock_store):
    await _run("job-1", "test query", "thread-1")

    calls = [c.kwargs for c in mock_store.update_job.call_args_list]
    assert {"status": "running"} in calls
    assert {"active_node": "librarian"} in calls
    assert {"active_node": "scout"} in calls

    final_call = calls[-1]
    assert final_call["status"] == "done"
    assert final_call["report"] == "final report text"
    assert final_call["source_map"] == {"[1]": {"url": "http://a.com"}}
    assert final_call["quality_score"] == {"coverage": 0.9}
    assert final_call["usage"]["node_seconds"] == {"librarian": 1.2, "scout": 3.4}
    assert "llm" in final_call["usage"]


@pytest.mark.asyncio
@patch("src.api.worker.astream_research", _fake_stream_failure)
@patch("src.api.worker.job_store")
async def test_run_marks_job_error_on_exception(mock_store):
    await _run("job-2", "test query", "thread-2")

    calls = [c.kwargs for c in mock_store.update_job.call_args_list]
    error_call = calls[-1]
    assert error_call["status"] == "error"
    assert "provider outage" in error_call["error"]
    assert "llm" in error_call["usage"]
