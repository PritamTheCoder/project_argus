import gc
import os
import tempfile

from src.api.jobs import JobStore


def _make_store():
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_jobs.db")
    return JobStore(db_path), tmp_dir, db_path


def _cleanup(store, tmp_dir, db_path):
    store.db.close()
    del store
    gc.collect()
    try:
        os.remove(db_path)
        os.rmdir(tmp_dir)
    except PermissionError:
        pass


def test_create_job_starts_queued():
    store, tmp_dir, db_path = _make_store()
    job = store.create_job("battery energy density", owner_key_hash="owner-1")
    assert job["status"] == "queued"
    assert job["job_id"] and job["thread_id"]
    assert job["job_id"] != job["thread_id"]
    _cleanup(store, tmp_dir, db_path)


def test_get_job_returns_stored_fields():
    store, tmp_dir, db_path = _make_store()
    job = store.create_job("q", owner_key_hash="owner-1")
    fetched = store.get_job(job["job_id"])
    assert fetched["query"] == "q"
    assert fetched["status"] == "queued"
    assert fetched["report"] is None
    assert fetched["owner_key_hash"] == "owner-1"
    _cleanup(store, tmp_dir, db_path)


def test_get_job_missing_returns_none():
    store, tmp_dir, db_path = _make_store()
    assert store.get_job("does-not-exist") is None
    _cleanup(store, tmp_dir, db_path)


def test_update_job_patches_fields_and_json_encodes_dicts():
    store, tmp_dir, db_path = _make_store()
    job = store.create_job("q", owner_key_hash="owner-1")

    store.update_job(job["job_id"], active_node="scout")
    assert store.get_job(job["job_id"])["active_node"] == "scout"

    store.update_job(
        job["job_id"],
        status="done",
        report="final report text",
        source_map={"[1]": {"url": "http://a.com"}},
        quality_score={"coverage": 0.9},
        citation_audit=[{"sentence": "x", "verdict": "SUPPORTED"}],
    )
    fetched = store.get_job(job["job_id"])
    assert fetched["status"] == "done"
    assert fetched["report"] == "final report text"
    assert fetched["source_map"] == {"[1]": {"url": "http://a.com"}}
    assert fetched["quality_score"] == {"coverage": 0.9}
    assert fetched["citation_audit"] == [{"sentence": "x", "verdict": "SUPPORTED"}]
    _cleanup(store, tmp_dir, db_path)


def test_update_job_on_unknown_id_is_a_noop_not_an_error():
    store, tmp_dir, db_path = _make_store()
    store.update_job("does-not-exist", status="done")  # must not raise
    assert store.get_job("does-not-exist") is None
    _cleanup(store, tmp_dir, db_path)


def test_list_jobs_returns_only_the_given_owners_jobs():
    store, tmp_dir, db_path = _make_store()
    store.create_job("alice's query", owner_key_hash="owner-a")
    store.create_job("bob's query", owner_key_hash="owner-b")

    results = store.list_jobs("owner-a")
    assert len(results) == 1
    assert results[0]["query"] == "alice's query"
    _cleanup(store, tmp_dir, db_path)


def test_list_jobs_orders_newest_first():
    store, tmp_dir, db_path = _make_store()
    first = store.create_job("first", owner_key_hash="owner-a")
    second = store.create_job("second", owner_key_hash="owner-a")

    results = store.list_jobs("owner-a")
    assert [r["job_id"] for r in results] == [second["job_id"], first["job_id"]]
    _cleanup(store, tmp_dir, db_path)


def test_list_jobs_empty_for_unknown_owner():
    store, tmp_dir, db_path = _make_store()
    assert store.list_jobs("nobody") == []
    _cleanup(store, tmp_dir, db_path)


def test_create_job_accepts_an_explicit_thread_id():
    """A branch already forked its thread before creating the job row — the
    job must point at that thread, not generate its own."""
    store, tmp_dir, db_path = _make_store()
    job = store.create_job("query", owner_key_hash="owner-1", thread_id="already-forked-thread")
    assert job["thread_id"] == "already-forked-thread"
    _cleanup(store, tmp_dir, db_path)


def test_create_job_without_thread_id_still_autogenerates():
    store, tmp_dir, db_path = _make_store()
    job = store.create_job("query", owner_key_hash="owner-1")
    assert job["thread_id"]
    assert job["thread_id"] != job["job_id"]
    _cleanup(store, tmp_dir, db_path)
