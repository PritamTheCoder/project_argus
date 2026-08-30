import os
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.jobs import JobStore


@pytest.fixture()
def test_store():
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_jobs.db")
    store = JobStore(db_path)
    yield store
    store.db.close()


@pytest.fixture()
def client(test_store):
    with patch("src.api.app.job_store", test_store), \
         patch("src.api.app.run_job_in_background") as mock_bg:
        yield TestClient(app), mock_bg, test_store


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
    test_client, mock_bg, store = client
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
    test_client, _, _ = client
    resp = test_client.post("/research", json={"query": ""})
    assert resp.status_code == 422


def test_get_research_not_found(client):
    test_client, _, _ = client
    resp = test_client.get("/research/does-not-exist")
    assert resp.status_code == 404


def test_get_research_returns_job_and_hides_thread_id(client):
    test_client, _, store = client
    job = store.create_job("test query")

    resp = test_client.get(f"/research/{job['job_id']}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == job["job_id"]
    assert body["status"] == "queued"
    assert "thread_id" not in body


def test_get_research_reflects_progress_updates(client):
    test_client, _, store = client
    job = store.create_job("test query")
    store.update_job(job["job_id"], status="running", active_node="scout")

    body = test_client.get(f"/research/{job['job_id']}").json()
    assert body["status"] == "running"
    assert body["active_node"] == "scout"
