"""SQLite-backed job store for async research runs. Status/result survive a
server restart (unlike an in-memory dict) and this table is the natural seed
for persistent per-user history later."""

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from src.config import JOBS_DB_PATH

_JSON_FIELDS = ("source_map", "quality_score")


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
            error TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """)
        self.db.commit()

    def create_job(self, query: str) -> dict:
        job_id = str(uuid.uuid4())
        thread_id = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            self.db.execute(
                "INSERT INTO jobs (job_id, thread_id, query, status, created_at, updated_at) "
                "VALUES (?, ?, ?, 'queued', ?, ?)",
                (job_id, thread_id, query, now, now),
            )
            self.db.commit()
        return {"job_id": job_id, "thread_id": thread_id, "status": "queued"}

    def update_job(self, job_id: str, **fields) -> None:
        """Patch arbitrary columns: status, active_node, report, source_map,
        quality_score, error. Dict-valued fields are JSON-encoded on write."""
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
