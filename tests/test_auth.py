import gc
import os
import tempfile

from src.api.auth import ApiKeyStore


def _make_store():
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_auth.db")
    return ApiKeyStore(db_path), tmp_dir, db_path


def _cleanup(store, tmp_dir, db_path):
    store.db.close()
    del store
    gc.collect()
    try:
        os.remove(db_path)
        os.rmdir(tmp_dir)
    except PermissionError:
        pass


def test_create_key_starts_with_prefix():
    store, tmp_dir, db_path = _make_store()
    key = store.create_key("alice")
    assert key.startswith("sk-argus-")
    _cleanup(store, tmp_dir, db_path)


def test_validate_recognizes_a_created_key():
    store, tmp_dir, db_path = _make_store()
    key = store.create_key("alice")
    owner = store.validate(key)
    assert owner["label"] == "alice"
    assert owner["key_hash"]
    _cleanup(store, tmp_dir, db_path)


def test_validate_returns_the_same_key_hash_for_the_same_key():
    store, tmp_dir, db_path = _make_store()
    key = store.create_key("alice")
    assert store.validate(key)["key_hash"] == store.validate(key)["key_hash"]
    _cleanup(store, tmp_dir, db_path)


def test_validate_rejects_unknown_key():
    store, tmp_dir, db_path = _make_store()
    assert store.validate("sk-argus-not-a-real-key") is None
    _cleanup(store, tmp_dir, db_path)


def test_each_key_is_unique():
    store, tmp_dir, db_path = _make_store()
    assert store.create_key("alice") != store.create_key("bob")
    _cleanup(store, tmp_dir, db_path)


def test_plaintext_key_is_not_stored():
    store, tmp_dir, db_path = _make_store()
    key = store.create_key("alice")
    row = store.db.execute("SELECT key_hash FROM api_keys").fetchone()
    assert row[0] != key
    _cleanup(store, tmp_dir, db_path)
