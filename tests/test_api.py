import os
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.auth import ApiKeyStore
from src.api.jobs import JobStore


@pytest.fixture()
def test_store():
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_jobs.db")
    store = JobStore(db_path)
    yield store
    store.db.close()


@pytest.fixture()
def test_key_store():
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_auth.db")
    store = ApiKeyStore(db_path)
    yield store
    store.db.close()


@pytest.fixture()
def client(test_store, test_key_store):
    api_key = test_key_store.create_key("test user")
    owner_key_hash = test_key_store.validate(api_key)["key_hash"]
    with patch("src.api.app.job_store", test_store), \
         patch("src.api.auth.api_key_store", test_key_store), \
         patch("src.api.app.run_job_in_background") as mock_bg:
        test_client = TestClient(app, headers={"Authorization": f"Bearer {api_key}"})
        yield test_client, mock_bg, test_store, owner_key_hash


def test_health():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_root_lists_endpoints():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Project Argus API"
    assert "POST /research" in body["endpoints"]


def test_create_research_returns_202_with_job_id(client):
    test_client, mock_bg, store, _ = client
    resp = test_client.post("/research", json={"query": "solid state batteries 2026"})

    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert body["job_id"]

    # Background execution was kicked off, not run inline.
    mock_bg.assert_called_once()
    args = mock_bg.call_args.args
    assert args[0] == body["job_id"]
    assert args[1] == "solid state batteries 2026"


def test_create_research_rejects_empty_query(client):
    test_client, _, _, _ = client
    resp = test_client.post("/research", json={"query": ""})
    assert resp.status_code == 422


def test_get_research_not_found(client):
    test_client, _, _, _ = client
    resp = test_client.get("/research/does-not-exist")
    assert resp.status_code == 404


def test_get_research_returns_job_and_hides_internal_fields(client):
    test_client, _, store, owner_key_hash = client
    job = store.create_job("test query", owner_key_hash)

    resp = test_client.get(f"/research/{job['job_id']}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == job["job_id"]
    assert body["status"] == "queued"
    assert "thread_id" not in body
    assert "owner_key_hash" not in body


def test_get_research_reflects_progress_updates(client):
    test_client, _, store, owner_key_hash = client
    job = store.create_job("test query", owner_key_hash)
    store.update_job(job["job_id"], status="running", active_node="scout")

    body = test_client.get(f"/research/{job['job_id']}").json()
    assert body["status"] == "running"
    assert body["active_node"] == "scout"


def test_get_research_hides_another_owners_job(client):
    test_client, _, store, _ = client
    someone_elses_job = store.create_job("not yours", owner_key_hash="someone-else")

    resp = test_client.get(f"/research/{someone_elses_job['job_id']}")
    assert resp.status_code == 404


def test_create_research_without_api_key_is_rejected(client):
    test_client, _, _, _ = client
    resp = test_client.post(
        "/research", json={"query": "q"}, headers={"Authorization": ""}
    )
    assert resp.status_code == 401


def test_create_research_with_wrong_api_key_is_rejected(client):
    test_client, _, _, _ = client
    resp = test_client.post(
        "/research", json={"query": "q"}, headers={"Authorization": "Bearer sk-argus-not-a-real-key"}
    )
    assert resp.status_code == 401


def test_list_research_returns_only_the_callers_jobs(client):
    test_client, _, store, owner_key_hash = client
    store.create_job("mine", owner_key_hash)
    store.create_job("also not yours", owner_key_hash="someone-else")

    body = test_client.get("/research/").json()

    assert len(body["results"]) == 1
    assert body["results"][0]["query"] == "mine"


def test_list_research_empty_for_a_new_key(client):
    test_client, _, _, _ = client
    body = test_client.get("/research/").json()
    assert body["results"] == []


def test_list_research_without_api_key_is_rejected(client):
    test_client, _, _, _ = client
    resp = test_client.get("/research/", headers={"Authorization": ""})
    assert resp.status_code == 401


# ── Evidence graph endpoints ──────────────────────────────────────────────────

def test_get_graph_not_found(client):
    test_client, _, _, _ = client
    resp = test_client.get("/research/does-not-exist/graph")
    assert resp.status_code == 404


def test_get_graph_hides_another_owners_job(client):
    test_client, _, store, _ = client
    job = store.create_job("not yours", owner_key_hash="someone-else")
    resp = test_client.get(f"/research/{job['job_id']}/graph")
    assert resp.status_code == 404


def test_get_graph_returns_evidence_for_the_jobs_thread(client):
    test_client, _, store, owner_key_hash = client
    job = store.create_job("test query", owner_key_hash)

    with patch("src.graph.kg.kg_store") as mock_kg:
        mock_kg.get_evidence_graph.return_value = {
            "facts": [{"id": 1, "claim": "x"}], "sources": [], "contradictions": [],
            "consensus_findings": [], "gaps": [],
        }
        resp = test_client.get(f"/research/{job['job_id']}/graph")

    assert resp.status_code == 200
    body = resp.json()
    assert body["facts"] == [{"id": 1, "claim": "x"}]
    # Queried by the job's internal thread_id, not the public job_id.
    mock_kg.get_evidence_graph.assert_called_once_with(job["thread_id"])


def test_get_fact_not_found(client):
    test_client, _, store, owner_key_hash = client
    job = store.create_job("test query", owner_key_hash)

    with patch("src.graph.kg.kg_store") as mock_kg:
        mock_kg.get_fact_detail.return_value = None
        resp = test_client.get(f"/research/{job['job_id']}/facts/999")

    assert resp.status_code == 404


def test_get_fact_hides_another_owners_job(client):
    test_client, _, store, _ = client
    job = store.create_job("not yours", owner_key_hash="someone-else")
    resp = test_client.get(f"/research/{job['job_id']}/facts/1")
    assert resp.status_code == 404


def test_get_fact_returns_detail(client):
    test_client, _, store, owner_key_hash = client
    job = store.create_job("test query", owner_key_hash)

    with patch("src.graph.kg.kg_store") as mock_kg:
        mock_kg.get_fact_detail.return_value = {
            "id": 1, "claim": "SpaceX valued at $350B", "source": {"credibility_score": 0.9},
            "contradictions": [],
        }
        resp = test_client.get(f"/research/{job['job_id']}/facts/1")

    assert resp.status_code == 200
    assert resp.json()["claim"] == "SpaceX valued at $350B"


# ── Checkpoint history + branch endpoints (Phase 8B) ─────────────────────────

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock


def _fake_checkpointer_ctx():
    @asynccontextmanager
    async def _ctx():
        yield MagicMock()
    return _ctx()


def test_get_history_not_found(client):
    test_client, _, _, _ = client
    resp = test_client.get("/research/does-not-exist/history")
    assert resp.status_code == 404


def test_get_history_hides_another_owners_job(client):
    test_client, _, store, _ = client
    job = store.create_job("not yours", owner_key_hash="someone-else")
    resp = test_client.get(f"/research/{job['job_id']}/history")
    assert resp.status_code == 404


def test_get_history_returns_checkpoints_for_the_jobs_thread(client):
    test_client, _, store, owner_key_hash = client
    job = store.create_job("test query", owner_key_hash)

    with patch("src.graph.persistence.get_checkpointer", return_value=_fake_checkpointer_ctx()), \
         patch("src.graph.builder.build_graph", return_value=MagicMock()), \
         patch("src.graph.persistence.list_checkpoints", new_callable=AsyncMock) as mock_list:
        mock_list.return_value = [
            {"checkpoint_id": "c1", "step": 0, "next": ["scout"], "iteration_count": 0},
        ]
        resp = test_client.get(f"/research/{job['job_id']}/history")

    assert resp.status_code == 200
    body = resp.json()
    assert body["checkpoints"][0]["checkpoint_id"] == "c1"
    # Queried against the job's internal thread_id.
    assert mock_list.call_args.args[1] == job["thread_id"]


def test_branch_not_found(client):
    test_client, _, _, _ = client
    resp = test_client.post("/research/does-not-exist/branch",
                            json={"checkpoint_id": "c1", "query": "dig deeper"})
    assert resp.status_code == 404


def test_branch_hides_another_owners_job(client):
    test_client, _, store, _ = client
    job = store.create_job("not yours", owner_key_hash="someone-else")
    resp = test_client.post(f"/research/{job['job_id']}/branch",
                            json={"checkpoint_id": "c1", "query": "dig deeper"})
    assert resp.status_code == 404


def test_branch_creates_a_new_job_from_the_forked_thread(client):
    test_client, _, store, owner_key_hash = client
    job = store.create_job("original query", owner_key_hash)

    fake_snapshot = MagicMock(values={"plan": [{"query": "existing", "mode": "MIXED"}]})
    fake_graph = MagicMock()
    fake_graph.aget_state = AsyncMock(return_value=fake_snapshot)

    with patch("src.graph.persistence.get_checkpointer", return_value=_fake_checkpointer_ctx()), \
         patch("src.graph.builder.build_graph", return_value=fake_graph), \
         patch("src.graph.persistence.fork_thread", new_callable=AsyncMock) as mock_fork, \
         patch("src.graph.kg.kg_store") as mock_kg, \
         patch("src.api.app.run_branch_in_background") as mock_run_branch:
        mock_fork.return_value = "forked-thread-id"
        resp = test_client.post(
            f"/research/{job['job_id']}/branch",
            json={"checkpoint_id": "c1", "query": "dig into Starlink revenue", "mode": "MIXED"},
        )

    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    new_job_id = body["job_id"]
    assert new_job_id != job["job_id"]

    # The new job row points at the forked thread, not a fresh one.
    stored = store.get_job(new_job_id)
    assert stored["thread_id"] == "forked-thread-id"
    assert stored["query"] == "original query"  # inherited from the source job

    # Evidence graph copied forward so the branch's graph isn't empty at birth.
    mock_kg.copy_session.assert_called_once_with(job["thread_id"], "forked-thread-id")
    mock_run_branch.assert_called_once_with(new_job_id, "forked-thread-id")


def test_branch_appends_to_existing_plan_rather_than_replacing_it(client):
    """The checkpoint's own plan (e.g. the Critic's follow-up queries) must
    survive alongside the injected direction, not be silently overwritten."""
    test_client, _, store, owner_key_hash = client
    job = store.create_job("q", owner_key_hash)

    fake_snapshot = MagicMock(values={"plan": [{"query": "critics follow-up", "mode": "MIXED"}]})
    fake_graph = MagicMock()
    fake_graph.aget_state = AsyncMock(return_value=fake_snapshot)

    with patch("src.graph.persistence.get_checkpointer", return_value=_fake_checkpointer_ctx()), \
         patch("src.graph.builder.build_graph", return_value=fake_graph), \
         patch("src.graph.persistence.fork_thread", new_callable=AsyncMock) as mock_fork, \
         patch("src.graph.kg.kg_store"), \
         patch("src.api.app.run_branch_in_background"):
        mock_fork.return_value = "forked-thread-id"
        test_client.post(f"/research/{job['job_id']}/branch",
                         json={"checkpoint_id": "c1", "query": "injected", "mode": "MIXED"})

    injected_values = mock_fork.call_args.args[2]
    queries = [q["query"] for q in injected_values["plan"]]
    assert queries == ["critics follow-up", "injected"]


def test_branch_checkpoint_not_found_returns_404(client):
    test_client, _, store, owner_key_hash = client
    job = store.create_job("q", owner_key_hash)

    fake_graph = MagicMock()
    fake_graph.aget_state = AsyncMock(return_value=None)

    with patch("src.graph.persistence.get_checkpointer", return_value=_fake_checkpointer_ctx()), \
         patch("src.graph.builder.build_graph", return_value=fake_graph):
        resp = test_client.post(f"/research/{job['job_id']}/branch",
                                json={"checkpoint_id": "does-not-exist", "query": "x"})

    assert resp.status_code == 404


# ── dig-deeper endpoint ───────────────────────────────────────────────────────

def test_dig_deeper_not_found(client):
    test_client, _, _, _ = client
    resp = test_client.post("/research/does-not-exist/dig-deeper", json={"query": "x"})
    assert resp.status_code == 404


def test_dig_deeper_hides_another_owners_job(client):
    test_client, _, store, _ = client
    job = store.create_job("not yours", owner_key_hash="someone-else")
    resp = test_client.post(f"/research/{job['job_id']}/dig-deeper", json={"query": "x"})
    assert resp.status_code == 404


def test_dig_deeper_requires_one_of_fact_id_gap_id_or_query(client):
    test_client, _, store, owner_key_hash = client
    job = store.create_job("q", owner_key_hash)
    resp = test_client.post(f"/research/{job['job_id']}/dig-deeper", json={})
    assert resp.status_code == 422


def test_dig_deeper_with_explicit_query_forces_scout_reentry(client):
    test_client, _, store, owner_key_hash = client
    job = store.create_job("original query", owner_key_hash)

    with patch("src.graph.persistence.get_checkpointer", return_value=_fake_checkpointer_ctx()), \
         patch("src.graph.builder.build_graph", return_value=MagicMock()), \
         patch("src.graph.persistence.fork_thread", new_callable=AsyncMock) as mock_fork, \
         patch("src.graph.kg.kg_store") as mock_kg, \
         patch("src.api.app.run_branch_in_background") as mock_run_branch:
        mock_fork.return_value = "forked-thread-id"
        resp = test_client.post(f"/research/{job['job_id']}/dig-deeper",
                                json={"query": "look into Starlink revenue"})

    assert resp.status_code == 202
    assert mock_fork.call_args.kwargs["as_node"] == "reflector"
    injected = mock_fork.call_args.args[2]
    assert injected["plan"] == [{"query": "look into Starlink revenue", "mode": "MIXED"}]
    mock_kg.copy_session.assert_called_once_with(job["thread_id"], "forked-thread-id")
    mock_run_branch.assert_called_once_with(resp.json()["job_id"], "forked-thread-id")


def test_dig_deeper_with_fact_id_builds_a_query_from_the_claim(client):
    test_client, _, store, owner_key_hash = client
    job = store.create_job("q", owner_key_hash)

    with patch("src.graph.persistence.get_checkpointer", return_value=_fake_checkpointer_ctx()), \
         patch("src.graph.builder.build_graph", return_value=MagicMock()), \
         patch("src.graph.persistence.fork_thread", new_callable=AsyncMock) as mock_fork, \
         patch("src.graph.kg.kg_store") as mock_kg, \
         patch("src.api.app.run_branch_in_background"):
        mock_kg.get_fact_detail.return_value = {"id": 5, "claim": "SpaceX valued at $350B"}
        mock_fork.return_value = "forked-thread-id"
        resp = test_client.post(f"/research/{job['job_id']}/dig-deeper", json={"fact_id": 5})

    assert resp.status_code == 202
    mock_kg.get_fact_detail.assert_called_once_with(job["thread_id"], 5)
    injected = mock_fork.call_args.args[2]
    assert "SpaceX valued at $350B" in injected["plan"][0]["query"]


def test_dig_deeper_with_unknown_fact_id_returns_404(client):
    test_client, _, store, owner_key_hash = client
    job = store.create_job("q", owner_key_hash)

    with patch("src.graph.kg.kg_store") as mock_kg:
        mock_kg.get_fact_detail.return_value = None
        resp = test_client.post(f"/research/{job['job_id']}/dig-deeper", json={"fact_id": 999})

    assert resp.status_code == 404


def test_dig_deeper_with_gap_id_builds_a_query_from_the_description(client):
    test_client, _, store, owner_key_hash = client
    job = store.create_job("q", owner_key_hash)

    with patch("src.graph.persistence.get_checkpointer", return_value=_fake_checkpointer_ctx()), \
         patch("src.graph.builder.build_graph", return_value=MagicMock()), \
         patch("src.graph.persistence.fork_thread", new_callable=AsyncMock) as mock_fork, \
         patch("src.graph.kg.kg_store") as mock_kg, \
         patch("src.api.app.run_branch_in_background"):
        mock_kg.get_evidence_graph.return_value = {"gaps": [
            {"id": 3, "gap_type": "coverage_gap", "description": "no data on Starlink revenue"},
        ]}
        mock_fork.return_value = "forked-thread-id"
        resp = test_client.post(f"/research/{job['job_id']}/dig-deeper", json={"gap_id": 3})

    assert resp.status_code == 202
    injected = mock_fork.call_args.args[2]
    assert "no data on Starlink revenue" in injected["plan"][0]["query"]


def test_dig_deeper_with_unknown_gap_id_returns_404(client):
    test_client, _, store, owner_key_hash = client
    job = store.create_job("q", owner_key_hash)

    with patch("src.graph.kg.kg_store") as mock_kg:
        mock_kg.get_evidence_graph.return_value = {"gaps": []}
        resp = test_client.post(f"/research/{job['job_id']}/dig-deeper", json={"gap_id": 999})

    assert resp.status_code == 404


def test_dig_deeper_query_overrides_fact_id_when_both_given(client):
    test_client, _, store, owner_key_hash = client
    job = store.create_job("q", owner_key_hash)

    with patch("src.graph.persistence.get_checkpointer", return_value=_fake_checkpointer_ctx()), \
         patch("src.graph.builder.build_graph", return_value=MagicMock()), \
         patch("src.graph.persistence.fork_thread", new_callable=AsyncMock) as mock_fork, \
         patch("src.graph.kg.kg_store") as mock_kg, \
         patch("src.api.app.run_branch_in_background"):
        mock_fork.return_value = "forked-thread-id"
        resp = test_client.post(f"/research/{job['job_id']}/dig-deeper",
                                json={"fact_id": 5, "query": "explicit override"})

    assert resp.status_code == 202
    mock_kg.get_fact_detail.assert_not_called()
    injected = mock_fork.call_args.args[2]
    assert injected["plan"][0]["query"] == "explicit override"


# ── Plan approval gate ───────────────────────────────────────────────────────

def test_create_research_passes_require_approval_through(client):
    test_client, mock_bg, store, _ = client
    resp = test_client.post("/research", json={"query": "q", "require_approval": True})

    assert resp.status_code == 202
    mock_bg.assert_called_once()
    assert mock_bg.call_args.kwargs["require_approval"] is True


def test_create_research_defaults_require_approval_false(client):
    test_client, mock_bg, store, _ = client
    test_client.post("/research", json={"query": "q"})
    assert mock_bg.call_args.kwargs["require_approval"] is False


def test_approve_plan_not_found(client):
    test_client, _, _, _ = client
    resp = test_client.post("/research/does-not-exist/approve-plan",
                            json={"plan": [{"query": "q1", "mode": "MIXED"}]})
    assert resp.status_code == 404


def test_approve_plan_hides_another_owners_job(client):
    test_client, _, store, _ = client
    job = store.create_job("not yours", owner_key_hash="someone-else")
    resp = test_client.post(f"/research/{job['job_id']}/approve-plan",
                            json={"plan": [{"query": "q1", "mode": "MIXED"}]})
    assert resp.status_code == 404


def test_approve_plan_rejects_a_job_not_awaiting_approval(client):
    test_client, _, store, owner_key_hash = client
    job = store.create_job("q", owner_key_hash)  # status: queued
    resp = test_client.post(f"/research/{job['job_id']}/approve-plan",
                            json={"plan": [{"query": "q1", "mode": "MIXED"}]})
    assert resp.status_code == 409


def test_approve_plan_requires_a_non_empty_plan(client):
    test_client, _, store, owner_key_hash = client
    job = store.create_job("q", owner_key_hash)
    store.update_job(job["job_id"], status="awaiting_approval")
    resp = test_client.post(f"/research/{job['job_id']}/approve-plan", json={"plan": []})
    assert resp.status_code == 422


def test_approve_plan_resumes_with_the_edited_plan(client):
    test_client, _, store, owner_key_hash = client
    job = store.create_job("q", owner_key_hash)
    store.update_job(job["job_id"], status="awaiting_approval",
                      pending_plan=[{"query": "q1", "mode": "MIXED"}])

    edited = [{"query": "q1", "mode": "MIXED"}, {"query": "human added", "mode": "TRUSTED_FIRST"}]
    with patch("src.api.app.run_approval_resume_in_background") as mock_resume:
        resp = test_client.post(f"/research/{job['job_id']}/approve-plan", json={"plan": edited})

    assert resp.status_code == 202
    assert resp.json()["job_id"] == job["job_id"]  # same job, not a new one
    mock_resume.assert_called_once_with(job["job_id"], job["thread_id"], edited)


def test_get_research_exposes_pending_plan_when_awaiting_approval(client):
    test_client, _, store, owner_key_hash = client
    job = store.create_job("q", owner_key_hash)
    store.update_job(job["job_id"], status="awaiting_approval",
                      pending_plan=[{"query": "q1", "mode": "MIXED"}])

    body = test_client.get(f"/research/{job['job_id']}").json()
    assert body["status"] == "awaiting_approval"
    assert body["pending_plan"] == [{"query": "q1", "mode": "MIXED"}]
