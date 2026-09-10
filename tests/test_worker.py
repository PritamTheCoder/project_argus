import pytest
from unittest.mock import MagicMock, patch

from src.api.worker import _run


async def _fake_stream_success(query, thread_id=None, callbacks=None, resume=False, resume_value=None, require_approval=False):
    yield "librarian", {"plan": [{"query": "q1", "mode": "MIXED"}]}
    yield "scout", {"scraped_data": [{"url": "http://a.com"}]}
    yield "__final__", {
        "report": "final report text",
        "source_map": {"[1]": {"url": "http://a.com"}},
        "quality_score": {"coverage": 0.9},
        "node_seconds": {"librarian": 1.2, "scout": 3.4},
    }


async def _fake_stream_failure(query, thread_id=None, callbacks=None, resume=False, resume_value=None, require_approval=False):
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


# ── Branch resume path (Phase 8B) ────────────────────────────────────────────

@pytest.mark.asyncio
@patch("src.api.worker.astream_research", _fake_stream_success)
@patch("src.api.worker.job_store")
async def test_run_with_resume_passes_resume_true_and_no_query(mock_store):
    """A branch's run has no query of its own — the forked state already
    carries it — and must tell astream_research to resume, not restart."""
    captured = {}

    async def _capture(query, thread_id=None, callbacks=None, resume=False, resume_value=None, require_approval=False):
        captured["query"] = query
        captured["resume"] = resume
        yield "__final__", {"report": "r", "source_map": {}, "quality_score": {}, "node_seconds": {}}

    with patch("src.api.worker.astream_research", _capture):
        await _run("job-3", None, "forked-thread", resume=True)

    assert captured["query"] is None
    assert captured["resume"] is True


def test_run_branch_in_background_starts_a_resume_run():
    from src.api.worker import run_branch_in_background
    with patch("src.api.worker._start") as mock_start:
        run_branch_in_background("job-3", "forked-thread")
    mock_start.assert_called_once_with("job-3", None, "forked-thread", resume=True)


def test_run_job_in_background_still_starts_a_fresh_run():
    from src.api.worker import run_job_in_background
    with patch("src.api.worker._start") as mock_start:
        run_job_in_background("job-1", "a query", "thread-1")
    mock_start.assert_called_once_with("job-1", "a query", "thread-1", resume=False, require_approval=False)


# ── Plan approval gate ───────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("src.api.worker.job_store")
async def test_run_marks_awaiting_approval_on_interrupt(mock_store):
    """A paused run must not be marked done — status flips to
    awaiting_approval and the pending plan is stored for polling."""
    async def _fake_interrupted(query, thread_id=None, callbacks=None, resume=False, resume_value=None, require_approval=False):
        yield "librarian", {"plan": [{"query": "q1", "mode": "MIXED"}]}
        yield "__interrupt__", {"pending_plan": [{"query": "q1", "mode": "MIXED"}]}

    with patch("src.api.worker.astream_research", _fake_interrupted):
        await _run("job-4", "test query", "thread-4", require_approval=True)

    calls = [c.kwargs for c in mock_store.update_job.call_args_list]
    final_call = calls[-1]
    assert final_call["status"] == "awaiting_approval"
    assert final_call["active_node"] == "plan_gate"
    assert final_call["pending_plan"] == [{"query": "q1", "mode": "MIXED"}]
    assert not any(c.get("status") == "done" for c in calls)


def test_run_job_in_background_passes_require_approval_through():
    from src.api.worker import run_job_in_background
    with patch("src.api.worker._start") as mock_start:
        run_job_in_background("job-1", "a query", "thread-1", require_approval=True)
    mock_start.assert_called_once_with("job-1", "a query", "thread-1", resume=False, require_approval=True)


def test_run_approval_resume_in_background_sends_resume_value():
    from src.api.worker import run_approval_resume_in_background
    approved = [{"query": "q1", "mode": "MIXED"}]
    with patch("src.api.worker._start") as mock_start:
        run_approval_resume_in_background("job-4", "thread-4", approved)
    mock_start.assert_called_once_with("job-4", None, "thread-4", resume=False, resume_value=approved)
