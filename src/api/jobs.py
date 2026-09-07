"""SQLite-backed job store for async research runs. Status/result survive a
server restart (unlike an in-memory dict), and each job is tagged with the
API key that created it, so a user's report history can be listed later."""

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from src.config import JOBS_DB_PATH

_JSON_FIELDS = ("source_map", "quality_score", "usage")


class JobStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.RLock()
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            query TEXT NOT NULL,
            status TEXT NOT NULL,
            active_node TEXT,
            report TEXT,
            source_map TEXT,
            quality_score TEXT,
            usage TEXT,
            error TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """)
        self._ensure_column("owner_key_hash", "TEXT")
        self._ensure_column("usage", "TEXT")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_jobs_owner ON jobs(owner_key_hash)")
        self.db.commit()

    def _ensure_column(self, column: str, col_type: str) -> None:
        """Add a column to the jobs table if an older database predates it."""
        existing = {row[1] for row in self.db.execute("PRAGMA table_info(jobs)").fetchall()}
        if column not in existing:
            self.db.execute(f"ALTER TABLE jobs ADD COLUMN {column} {col_type}")

    def create_job(self, query: str, owner_key_hash: str, thread_id: Optional[str] = None) -> dict:
        """``thread_id`` is normally auto-generated; a branch passes an
        already-forked one so the job row points at the forked run, not a
        fresh empty thread."""
        job_id = str(uuid.uuid4())
        thread_id = thread_id or str(uuid.uuid4())
        now = time.time()
        with self._lock:
            self.db.execute(
                "INSERT INTO jobs (job_id, thread_id, owner_key_hash, query, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'queued', ?, ?)",
                (job_id, thread_id, owner_key_hash, query, now, now),
            )
            self.db.commit()
        return {"job_id": job_id, "thread_id": thread_id, "status": "queued"}

    def update_job(self, job_id: str, **fields) -> None:
        """Patch arbitrary columns: status, active_node, report, source_map,
        quality_score, usage, error. Dict-valued fields are JSON-encoded on write."""
        if not fields:
            return
        fields = dict(fields)
        for key in _JSON_FIELDS:
            if key in fields and fields[key] is not None:
                fields[key] = json.dumps(fields[key])
        fields["updated_at"] = time.time()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self.db.execute(f"UPDATE jobs SET {set_clause} WHERE job_id = ?", (*fields.values(), job_id))
            self.db.commit()

    def list_jobs(self, owner_key_hash: str) -> list[dict]:
        """Summary view of a user's past jobs, newest first. Leaves out the
        report body — callers fetch that per-job via get_job() when needed."""
        with self._lock:
            cursor = self.db.execute(
                "SELECT job_id, query, status, created_at, updated_at FROM jobs "
                "WHERE owner_key_hash = ? ORDER BY created_at DESC",
                (owner_key_hash,),
            )
            rows = cursor.fetchall()
            columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def get_job(self, job_id: str) -> Optional[dict]:
        with self._lock:
            cursor = self.db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            columns = [d[0] for d in cursor.description]
        if not row:
            return None
        job = dict(zip(columns, row))
        for key in _JSON_FIELDS:
            if job.get(key):
                job[key] = json.loads(job[key])
        return job


job_store = JobStore(JOBS_DB_PATH)
