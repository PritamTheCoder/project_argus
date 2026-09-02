"""Thin HTTP API in front of the research pipeline — product-track step 1
(deploy + API). POST /research kicks off a background job; GET /research/{id}
polls status/result; GET /research/ lists the caller's past jobs.

Run with: uvicorn src.api.app:app --reload
"""

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils.logging_setup import configure_logging

configure_logging()  # before importing anything that logs, so nothing gets dropped

from src.api.auth import require_api_key  # noqa: E402
from src.api.jobs import job_store  # noqa: E402
from src.api.worker import run_job_in_background  # noqa: E402

app = FastAPI(title="Project Argus API", version="0.1.0")


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class ResearchJobOut(BaseModel):
    job_id: str
    status: str


@app.get("/")
async def root():
    return {
        "name": "Project Argus API",
        "version": app.version,
        "docs": "/docs",
        "endpoints": {
            "POST /research": "submit a research query, returns a job_id (requires an API key)",
            "GET /research/{job_id}": "poll job status/progress/result (requires an API key)",
            "GET /research/": "list your past research jobs (requires an API key)",
            "GET /health": "liveness check",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/research", response_model=ResearchJobOut, status_code=202)
async def create_research(req: ResearchRequest, owner: dict = Depends(require_api_key)):
    job = job_store.create_job(req.query, owner["key_hash"])
    run_job_in_background(job["job_id"], req.query, job["thread_id"])
    return {"job_id": job["job_id"], "status": job["status"]}


@app.get("/research/")
async def list_research(owner: dict = Depends(require_api_key)):
    return {"results": job_store.list_jobs(owner["key_hash"])}


@app.get("/research/{job_id}")
async def get_research(job_id: str, owner: dict = Depends(require_api_key)):
    job = job_store.get_job(job_id)
    # Same 404 for "doesn't exist" and "belongs to someone else" so a valid
    # key can't be used to probe for other users' job ids.
    if job is None or job["owner_key_hash"] != owner["key_hash"]:
        raise HTTPException(status_code=404, detail="job not found")
    job.pop("thread_id", None)  # internal id, not part of the public contract
    job.pop("owner_key_hash", None)
    return job
