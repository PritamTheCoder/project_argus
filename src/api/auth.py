"""API key auth for the research endpoints.

Keys are created by an admin via the CLI at the bottom of this file, not
through self-service signup. Clients send them as `Authorization: Bearer <key>`.
"""

import hashlib
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import JOBS_DB_PATH

_KEY_PREFIX = "sk-argus-"
_bearer_scheme = HTTPBearer(auto_error=False)


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


class ApiKeyStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.RLock()
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            key_hash TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            created_at REAL NOT NULL,
            last_used_at REAL
        )
        """)
        self.db.commit()

    def create_key(self, label: str) -> str:
        """Generate a key, store only its hash, return the plaintext (shown once)."""
        raw_key = _KEY_PREFIX + secrets.token_urlsafe(32)
        with self._lock:
            self.db.execute(
                "INSERT INTO api_keys (key_hash, label, created_at) VALUES (?, ?, ?)",
                (_hash_key(raw_key), label, time.time()),
            )
            self.db.commit()
        return raw_key

    def validate(self, raw_key: str) -> Optional[dict]:
        """Return {"key_hash", "label"} if the key exists, else None."""
        key_hash = _hash_key(raw_key)
        with self._lock:
            row = self.db.execute(
                "SELECT label FROM api_keys WHERE key_hash = ?", (key_hash,)
            ).fetchone()
            if row is None:
                return None
            self.db.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE key_hash = ?", (time.time(), key_hash)
            )
            self.db.commit()
        return {"key_hash": key_hash, "label": row[0]}


api_key_store = ApiKeyStore(JOBS_DB_PATH)


def require_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> dict:
    """FastAPI dependency: validates the Authorization header, returns {"key_hash", "label"}."""
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing API key")

    owner = api_key_store.validate(credentials.credentials)
    if owner is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")

    return owner


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create a new Argus API key")
    parser.add_argument("label", help="who or what this key is for, e.g. a username")
    args = parser.parse_args()

    new_key = api_key_store.create_key(args.label)
    print(f"API key for '{args.label}':\n{new_key}\n(save it now, it won't be shown again)")
