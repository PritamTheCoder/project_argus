"""Runs one research job in its own OS thread + event loop (same trick as
src/ui/runner.py), since Playwright needs a ProactorEventLoop on Windows
that uvicorn's own loop doesn't guarantee."""

import asyncio
import logging
import sys
import threading
from typing import Optional

from eval.metrics import RunMetrics
from src.api.jobs import job_store
from src.graph.pipeline import astream_research

logger = logging.getLogger(__name__)


async def _run(
    job_id: str, query: Optional[str], thread_id: str, resume: bool = False,
    resume_value=None, require_approval: bool = False,
) -> None:
    job_store.update_job(job_id, status="running")
    metrics = RunMetrics()  # same tracker the eval harness uses
    try:
        async for node_name, state_update in astream_research(
            query, thread_id=thread_id, callbacks=[metrics], resume=resume,
            resume_value=resume_value, require_approval=require_approval,
        ):
            if node_name == "__final__":
                job_store.update_job(
                    job_id,
                    status="done",
                    active_node="done",
                    report=state_update.get("report", ""),
                    source_map=state_update.get("source_map", {}),
                    quality_score=state_update.get("quality_score", {}),
                    usage={"llm": metrics.summary(), "node_seconds": state_update.get("node_seconds", {})},
                )
            elif node_name == "__interrupt__":
                job_store.update_job(
                    job_id, status="awaiting_approval", active_node="plan_gate",
                    pending_plan=state_update.get("pending_plan"),
                )
            else:
                job_store.update_job(job_id, active_node=node_name)
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Research job {job_id} failed")
        job_store.update_job(job_id, status="error", error=str(e), usage={"llm": metrics.summary(), "node_seconds": {}})


def _start(
    job_id: str, query: Optional[str], thread_id: str, resume: bool,
    resume_value=None, require_approval: bool = False,
) -> threading.Thread:
    def _worker():
        loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_run(
                job_id, query, thread_id, resume=resume,
                resume_value=resume_value, require_approval=require_approval,
            ))
        finally:
            loop.close()

    # Named so log lines (threadName in the format) are traceable to this job.
    t = threading.Thread(target=_worker, daemon=True, name=f"job-{job_id[:8]}")
    t.start()
    return t


def run_job_in_background(job_id: str, query: str, thread_id: str, require_approval: bool = False) -> threading.Thread:
    return _start(job_id, query, thread_id, resume=False, require_approval=require_approval)


def run_branch_in_background(job_id: str, thread_id: str) -> threading.Thread:
    """Continue an already-forked thread (see ``fork_thread``) rather than
    starting a fresh run — the state is already seeded."""
    return _start(job_id, None, thread_id, resume=True)


def run_approval_resume_in_background(job_id: str, thread_id: str, approved_plan: list) -> threading.Thread:
    """Resume a thread paused at plan_gate, answering its interrupt() with
    the human-approved (possibly edited) plan."""
    return _start(job_id, None, thread_id, resume=False, resume_value=approved_plan)
