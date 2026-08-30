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
    job = store.create_job("battery energy density")
    assert job["status"] == "queued"
    assert job["job_id"] and job["thread_id"]
    assert job["job_id"] != job["thread_id"]
    _cleanup(store, tmp_dir, db_path)


def test_get_job_returns_stored_fields():
    store, tmp_dir, db_path = _make_store()
    job = store.create_job("q")
    fetched = store.get_job(job["job_id"])
    assert fetched["query"] == "q"
    assert fetched["status"] == "queued"
    assert fetched["report"] is None
    _cleanup(store, tmp_dir, db_path)


def test_get_job_missing_returns_none():
    store, tmp_dir, db_path = _make_store()
    assert store.get_job("does-not-exist") is None
    _cleanup(store, tmp_dir, db_path)


def test_update_job_patches_fields_and_json_encodes_dicts():
    store, tmp_dir, db_path = _make_store()
    job = store.create_job("q")

    store.update_job(job["job_id"], active_node="scout")
    assert store.get_job(job["job_id"])["active_node"] == "scout"

    store.update_job(
        job["job_id"],
        status="done",
        report="final report text",
        source_map={"[1]": {"url": "http://a.com"}},
        quality_score={"coverage": 0.9},
    )
    fetched = store.get_job(job["job_id"])
    assert fetched["status"] == "done"
    assert fetched["report"] == "final report text"
    assert fetched["source_map"] == {"[1]": {"url": "http://a.com"}}
    assert fetched["quality_score"] == {"coverage": 0.9}
    _cleanup(store, tmp_dir, db_path)


def test_update_job_on_unknown_id_is_a_noop_not_an_error():
    store, tmp_dir, db_path = _make_store()
    store.update_job("does-not-exist", status="done")  # must not raise
    assert store.get_job("does-not-exist") is None
    _cleanup(store, tmp_dir, db_path)
