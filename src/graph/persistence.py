"""
Project Argus - Persistence Layer

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
    """Async context manager yielding an AsyncSqliteSaver checkpointer,
    backed by the file at ``DB_PATH`` (config.py)."""
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
    """Build the LangGraph run config dict for a given thread."""
    return {"configurable": {"thread_id": thread_id}}


async def list_checkpoints(graph, thread_id: str) -> list[dict]:
    """The checkpoint history for a thread, oldest first, each entry labelled
    with the node about to run next and the loop-safety iteration it belongs to."""
    config = get_run_config(thread_id)
    snapshots = [s async for s in graph.aget_state_history(config)]
    snapshots.reverse()
    return [
        {
            "checkpoint_id": s.config["configurable"]["checkpoint_id"],
            "step": s.metadata.get("step"),
            "next": list(s.next),
            "iteration_count": s.values.get("iteration_count"),
        }
        for s in snapshots
    ]


async def fork_thread(
    graph, source_thread_id: str, injected_values: dict,
    checkpoint_id: str | None = None, as_node: str | None = None,
) -> str:
    """Fork a new thread from one checkpoint of an existing run, with
    ``injected_values`` merged into its state, and return the new thread_id.
    ``checkpoint_id=None`` forks from the run's current/latest state.

    ``aupdate_state`` without ``as_node`` does not resume at the checkpoint's
    position — it resets ``.next`` to the graph's entry point, silently
    restarting the run and discarding the injected values as soon as the
    first node overwrites them. By default ``as_node`` is derived as
    ``.next[0]`` of the checkpoint's parent, preserving its natural routing
    (a checkpoint with no parent is the run's very first, where a full
    restart is correct). Pass ``as_node`` explicitly to force resumption at a
    specific node regardless of the checkpoint's own routing.

    The new thread gets a fresh ``session_id``; callers wanting its evidence
    graph to include what came before the fork should call
    ``kg_store.copy_session`` separately.
    """
    source_config = {"configurable": {"thread_id": source_thread_id}}
    if checkpoint_id is not None:
        source_config["configurable"]["checkpoint_id"] = checkpoint_id
    snapshot = await graph.aget_state(source_config)
    if snapshot is None or not snapshot.values:
        raise ValueError(f"checkpoint {checkpoint_id!r} not found for thread {source_thread_id}")

    if as_node is None and snapshot.parent_config is not None:
        parent = await graph.aget_state(snapshot.parent_config)
        if parent.next:
            as_node = parent.next[0]

    new_thread_id = generate_thread_id()
    new_config = get_run_config(new_thread_id)
    seeded_values = {**snapshot.values, **injected_values, "session_id": new_thread_id}

    if as_node is not None:
        await graph.aupdate_state(new_config, seeded_values, as_node=as_node)
    else:
        await graph.aupdate_state(new_config, seeded_values)

    return new_thread_id
