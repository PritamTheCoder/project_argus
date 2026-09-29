"""Thin HTTP API in front of the research pipeline — product-track step 1
(deploy + API). POST /research kicks off a background job; GET /research/{id}
polls status/result; GET /research/ lists the caller's past jobs.

Run with: uvicorn src.api.app:app --reload
"""

from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils.logging_setup import configure_logging

configure_logging()  # before importing anything that logs, so nothing gets dropped

from src.api.auth import require_api_key  # noqa: E402
from src.api.jobs import job_store  # noqa: E402
from src.api.worker import (  # noqa: E402
    run_job_in_background, run_branch_in_background, run_approval_resume_in_background,
)

app = FastAPI(title="Project Argus API", version="0.1.0")


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    require_approval: bool = Field(
        default=False, description="Pause after planning for human review of the sub-queries before Scout runs"
    )


class PlanItem(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    mode: str = Field(default="MIXED", description="TRUSTED_ONLY, TRUSTED_FIRST, or MIXED")


class ApprovePlanRequest(BaseModel):
    plan: list[PlanItem] = Field(..., min_length=1, description="The final sub-query list to research — approved as-is, edited, or extended")


class BranchRequest(BaseModel):
    checkpoint_id: str = Field(..., description="A checkpoint_id from GET /research/{job_id}/history")
    query: str = Field(..., min_length=1, max_length=2000, description="The new direction to research from this point")
    mode: str = Field(default="MIXED", description="TRUSTED_ONLY, TRUSTED_FIRST, or MIXED")


class DigDeeperRequest(BaseModel):
    fact_id: Optional[int] = Field(default=None, description="Target a specific fact from GET .../facts/{fact_id}")
    gap_id: Optional[int] = Field(default=None, description="Target a specific gap from GET .../graph")
    query: Optional[str] = Field(default=None, max_length=2000, description="Override: research this instead of deriving a query from fact_id/gap_id")
    mode: str = Field(default="MIXED", description="TRUSTED_ONLY, TRUSTED_FIRST, or MIXED")


class FlagFactRequest(BaseModel):
    reason: str = Field(..., min_length=1, max_length=1000, description="Why this fact is disputed")
    trigger_reverify: bool = Field(
        default=True, description="Also fork a targeted re-verification run on this claim"
    )


class ResearchJobOut(BaseModel):
    job_id: str
    status: str


class FlagFactOut(BaseModel):
    flagged: bool
    job_id: Optional[str] = None
    status: Optional[str] = None


@app.get("/")
async def root():
    return {
        "name": "Project Argus API",
        "version": app.version,
        "docs": "/docs",
        "endpoints": {
            "POST /research": "submit a research query, returns a job_id (requires an API key)",
            "GET /research/{job_id}": "poll status, progress, report, and citation_audit (requires an API key)",
            "GET /research/{job_id}/graph": "the evidence graph for a run: facts, sources, contradictions, consensus findings, gaps (requires an API key)",
            "GET /research/{job_id}/facts/{fact_id}": "one fact's quote, source, and contradictions (requires an API key)",
            "GET /research/{job_id}/history": "the run's checkpoint history, for picking a branch point (requires an API key)",
            "POST /research/{job_id}/branch": "fork a new run from a checkpoint with an injected direction; the source run is untouched (requires an API key)",
            "POST /research/{job_id}/dig-deeper": "fork a targeted follow-up on one fact or gap from a finished run (requires an API key)",
            "POST /research/{job_id}/approve-plan": "submit the approved/edited plan for a job awaiting_approval (requires an API key)",
            "POST /research/{job_id}/facts/{fact_id}/flag": "mark a fact disputed and optionally fork a targeted re-verification (requires an API key)",
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
    run_job_in_background(job["job_id"], req.query, job["thread_id"],
                          require_approval=req.require_approval, owner_key_hash=owner["key_hash"])
    return {"job_id": job["job_id"], "status": job["status"]}


@app.get("/research/")
async def list_research(owner: dict = Depends(require_api_key)):
    return {"results": job_store.list_jobs(owner["key_hash"])}


def _get_owned_job_or_404(job_id: str, owner: dict) -> dict:
    """Same 404 for "doesn't exist" and "belongs to someone else" so a valid
    key can't be used to probe for other users' job ids. Returns the raw job
    row (thread_id intact) for endpoints that need it internally."""
    job = job_store.get_job(job_id)
    if job is None or job["owner_key_hash"] != owner["key_hash"]:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/research/{job_id}")
async def get_research(job_id: str, owner: dict = Depends(require_api_key)):
    job = _get_owned_job_or_404(job_id, owner)
    job.pop("thread_id", None)  # internal id, not part of the public contract
    job.pop("owner_key_hash", None)
    return job


@app.get("/research/{job_id}/graph")
async def get_research_graph(job_id: str, owner: dict = Depends(require_api_key)):
    """The evidence graph for one run: facts, sources, contradictions,
    consensus findings, and gaps. Populated incrementally as the Verifier,
    Consensus, and Critic nodes run — available before the job finishes."""
    job = _get_owned_job_or_404(job_id, owner)
    from src.graph.kg import kg_store
    return kg_store.get_evidence_graph(job["thread_id"])


@app.get("/research/{job_id}/facts/{fact_id}")
async def get_research_fact(job_id: str, fact_id: int, owner: dict = Depends(require_api_key)):
    """One fact's full evidence: verbatim quote, source credibility and
    relevance, and any contradictions naming its claim."""
    job = _get_owned_job_or_404(job_id, owner)
    from src.graph.kg import kg_store
    fact = kg_store.get_fact_detail(job["thread_id"], fact_id)
    if fact is None:
        raise HTTPException(status_code=404, detail="fact not found")
    return fact


@app.get("/research/{job_id}/history")
async def get_research_history(job_id: str, owner: dict = Depends(require_api_key)):
    """The run's checkpoint history — a branch point is a checkpoint_id from
    this list."""
    job = _get_owned_job_or_404(job_id, owner)
    from src.graph.builder import build_graph
    from src.graph.persistence import get_checkpointer, list_checkpoints

    async with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        return {"checkpoints": await list_checkpoints(graph, job["thread_id"])}


@app.post("/research/{job_id}/branch", response_model=ResearchJobOut, status_code=202)
async def branch_research(job_id: str, req: BranchRequest, owner: dict = Depends(require_api_key)):
    """Fork a new run from one of this job's checkpoints, with an extra
    sub-query injected into the plan. The source run is untouched — this
    creates a new job the caller polls exactly like any other."""
    job = _get_owned_job_or_404(job_id, owner)
    from src.graph.builder import build_graph
    from src.graph.kg import kg_store
    from src.graph.persistence import fork_thread, get_checkpointer, get_run_config

    try:
        async with get_checkpointer() as checkpointer:
            graph = build_graph(checkpointer=checkpointer)
            source_config = {"configurable": {"thread_id": job["thread_id"], "checkpoint_id": req.checkpoint_id}}
            snapshot = await graph.aget_state(source_config)
            if snapshot is None or not snapshot.values:
                raise HTTPException(status_code=404, detail="checkpoint not found")

            # Appended, not replaced — the checkpoint's own plan (e.g. the
            # Critic's follow-up queries) must survive alongside the injected
            # direction, not be silently overwritten by it.
            existing_plan = snapshot.values.get("plan", []) or []
            injected = {"plan": existing_plan + [{"query": req.query, "mode": req.mode}]}

            new_thread_id = await fork_thread(graph, job["thread_id"], injected, checkpoint_id=req.checkpoint_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="checkpoint not found")

    kg_store.copy_session(job["thread_id"], new_thread_id)

    new_job = job_store.create_job(job["query"], owner["key_hash"], thread_id=new_thread_id)
    run_branch_in_background(new_job["job_id"], new_thread_id)
    return {"job_id": new_job["job_id"], "status": new_job["status"]}


def _dig_deeper_query(kg_store, thread_id: str, req: DigDeeperRequest) -> str:
    """Resolve the request to a single query string. Priority: an explicit
    override, then a fact, then a gap — checked in that order and the first
    one present wins, since a caller passing more than one is asking to
    override rather than to be told it's ambiguous."""
    if req.query:
        return req.query
    if req.fact_id is not None:
        fact = kg_store.get_fact_detail(thread_id, req.fact_id)
        if fact is None:
            raise HTTPException(status_code=404, detail="fact not found")
        return f"Find additional corroborating or contradicting evidence for: {fact['claim']}"
    if req.gap_id is not None:
        gaps = kg_store.get_evidence_graph(thread_id)["gaps"]
        gap = next((g for g in gaps if g["id"] == req.gap_id), None)
        if gap is None:
            raise HTTPException(status_code=404, detail="gap not found")
        return f"Research and find evidence for: {gap['description']}"
    raise HTTPException(status_code=422, detail="one of fact_id, gap_id, or query is required")


async def _fork_followup_job(job: dict, owner: dict, query_text: str, mode: str = "MIXED") -> dict:
    """Fork a targeted follow-up from a finished run into its own job — shared
    by dig-deeper and flag-as-wrong's re-verify. A finished run's checkpoint
    has no natural "next node" to resume into, so this always re-enters at
    Scout instead, by forking as if resuming from Reflector (whose only
    outgoing edge goes straight to Scout)."""
    from src.graph.builder import build_graph
    from src.graph.kg import kg_store
    from src.graph.persistence import fork_thread, get_checkpointer

    injected = {"plan": [{"query": query_text, "mode": mode}]}
    async with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        new_thread_id = await fork_thread(graph, job["thread_id"], injected, as_node="reflector")

    kg_store.copy_session(job["thread_id"], new_thread_id)

    new_job = job_store.create_job(job["query"], owner["key_hash"], thread_id=new_thread_id)
    run_branch_in_background(new_job["job_id"], new_thread_id)
    return new_job


@app.post("/research/{job_id}/dig-deeper", response_model=ResearchJobOut, status_code=202)
async def dig_deeper(job_id: str, req: DigDeeperRequest, owner: dict = Depends(require_api_key)):
    """Fork a targeted follow-up on one fact or gap from a finished run —
    the human-triggered counterpart to the Reflector's automatic gap-fill."""
    job = _get_owned_job_or_404(job_id, owner)
    from src.graph.kg import kg_store

    query_text = _dig_deeper_query(kg_store, job["thread_id"], req)
    new_job = await _fork_followup_job(job, owner, query_text, req.mode)
    return {"job_id": new_job["job_id"], "status": new_job["status"]}


@app.post("/research/{job_id}/approve-plan", response_model=ResearchJobOut, status_code=202)
async def approve_plan(job_id: str, req: ApprovePlanRequest, owner: dict = Depends(require_api_key)):
    """Submit the final sub-query list for a job paused at plan_gate —
    approved as-is, edited, or extended. Resumes the same job; unlike branch/
    dig-deeper this does not create a new one, since nothing has run yet."""
    job = _get_owned_job_or_404(job_id, owner)
    if job["status"] != "awaiting_approval":
        raise HTTPException(status_code=409, detail=f"job is '{job['status']}', not awaiting_approval")

    approved_plan = [item.model_dump() for item in req.plan]
    run_approval_resume_in_background(job_id, job["thread_id"], approved_plan)
    return {"job_id": job_id, "status": "running"}


@app.post("/research/{job_id}/facts/{fact_id}/flag", response_model=FlagFactOut, status_code=202)
async def flag_fact(job_id: str, fact_id: int, req: FlagFactRequest, owner: dict = Depends(require_api_key)):
    """Mark a fact disputed — its confidence drops to 0, so existing
    confidence thresholds already stop trusting it, no new filtering needed —
    and, by default, fork a targeted re-verification of the claim."""
    job = _get_owned_job_or_404(job_id, owner)
    from src.graph.kg import kg_store

    fact = kg_store.get_fact_detail(job["thread_id"], fact_id)
    if fact is None:
        raise HTTPException(status_code=404, detail="fact not found")
    kg_store.flag_fact(job["thread_id"], fact_id, req.reason)

    if not req.trigger_reverify:
        return {"flagged": True}

    query_text = (
        f"This claim was flagged as disputed: \"{fact['claim']}\" "
        f"(reason: {req.reason}). Verify it against the evidence, and note any "
        f"contradicting evidence explicitly."
    )
    new_job = await _fork_followup_job(job, owner, query_text)
    return {"flagged": True, "job_id": new_job["job_id"], "status": new_job["status"]}
