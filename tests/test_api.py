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
