"""
Project Argus - Persistence Layer (Phase 3)

Provides SQLite-backed state persistence for the LangGraph graph,
enabling "time travel" (rewinding to any previous state checkpoint).
"""

import uuid
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from src.config import DB_PATH

logger = logging.getLogger(__name__)


@asynccontextmanager
async def get_checkpointer():
    """
    Async context manager that yields an AsyncSqliteSaver checkpointer.

    Usage:
        async with get_checkpointer() as cp:
            graph = build_graph(checkpointer=cp)

    The database file is stored at ``data/db/argus_checkpoints.db``
    (configured via ``DB_PATH`` in config.py).
    The parent directory is created automatically if it doesn't exist.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # lazy import

    db_path = Path(DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Persistence: Opening SQLite checkpoint DB at {db_path}")
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
        yield saver


def generate_thread_id() -> str:
    """Generate a unique thread ID (UUID4) for a new research session."""
    return str(uuid.uuid4())


def get_run_config(thread_id: str) -> dict:
    """
    Build the LangGraph run config dict for a given thread.

    Args:
        thread_id: Unique identifier for the conversation / research session.

    Returns:
        Config dict with ``{"configurable": {"thread_id": thread_id}}``.
    """
    return {"configurable": {"thread_id": thread_id}}
