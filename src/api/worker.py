"""Runs one research job in a dedicated OS thread with its own event loop —
same trick as src/ui/runner.py. Needed because Playwright (via the scraper)
requires a ProactorEventLoop for subprocess support on Windows, which the
event loop uvicorn/FastAPI run the API on does not guarantee."""

import asyncio
import logging
import sys
import threading

from src.api.jobs import job_store
from src.graph.pipeline import astream_research

logger = logging.getLogger(__name__)


async def _run(job_id: str, query: str, thread_id: str) -> None:
    job_store.update_job(job_id, status="running")
    try:
        async for node_name, state_update in astream_research(query, thread_id=thread_id):
            if node_name == "__final__":
                job_store.update_job(
                    job_id,
                    status="done",
                    active_node="done",
                    report=state_update.get("report", ""),
                    source_map=state_update.get("source_map", {}),
                    quality_score=state_update.get("quality_score", {}),
                )
            else:
                job_store.update_job(job_id, active_node=node_name)
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Research job {job_id} failed")
        job_store.update_job(job_id, status="error", error=str(e))


def run_job_in_background(job_id: str, query: str, thread_id: str) -> threading.Thread:
    def _worker():
        loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_run(job_id, query, thread_id))
        finally:
            loop.close()

    # Named so log lines (threadName in the format) are traceable to this job.
    t = threading.Thread(target=_worker, daemon=True, name=f"job-{job_id[:8]}")
    t.start()
    return t
