"""Shared pipeline entrypoint — the one place that builds the graph, seeds
initial state, and drives graph.astream(). main.py (CLI) and src/api (HTTP)
both call this instead of duplicating it."""

import time
from typing import Any, AsyncGenerator, Optional

from langgraph.types import Command

from src.graph.builder import build_graph
from src.graph.persistence import get_checkpointer, generate_thread_id, get_run_config
from src.schema.state import AgentState


def _initial_state(
    query: str, session_id: str, require_approval: bool = False, owner_key_hash: str = "",
) -> AgentState:
    return {
        "session_id": session_id,
        "query": query,
        "plan": [],
        "original_plan": [],
        "scraped_data": [],
        "structured_evidence": [],
        "source_map": {},
        "backend_health": {},
        "critique": "",
        "report": "",
        "re_search_required": False,
        "knowledge_gap_detected": False,
        "knowledge_gaps": [],
        "coverage_gaps": [],
        "gap_queries": [],
        "verified_facts": [],
        "iteration_count": 0,
        "active_node": "",
        "require_approval": require_approval,
        "owner_key_hash": owner_key_hash,
    }


async def astream_research(
    query: Optional[str] = None, thread_id: Optional[str] = None,
    callbacks: Optional[list[Any]] = None, resume: bool = False,
    resume_value: Any = None, require_approval: bool = False, owner_key_hash: str = "",
) -> AsyncGenerator[tuple[str, dict], None]:
    """Yield (node_name, state_update) per node, then either ("__final__",
    full_state) or ("__interrupt__", {"pending_plan": [...]}) if the run
    paused for plan approval instead of finishing. callbacks is passed to the
    graph run config (e.g. a token-usage tracker).

    ``resume=True`` continues an already-seeded thread (e.g. one written by
    ``fork_thread``) — passing ``None`` as the graph input tells LangGraph to
    read the existing checkpoint. ``resume_value`` instead sends a
    ``Command(resume=...)``, answering a pending ``interrupt()`` (the
    plan_gate case). ``query``/``require_approval`` are unused either way —
    the seeded state already carries them."""
    thread_id = thread_id or generate_thread_id()
    config = get_run_config(thread_id)
    if callbacks:
        config["callbacks"] = callbacks
    if resume_value is not None:
        graph_input = Command(resume=resume_value)
    elif resume:
        graph_input = None
    else:
        graph_input = _initial_state(query, thread_id, require_approval, owner_key_hash)

    node_seconds: dict[str, float] = {}
    node_started_at = time.perf_counter()

    async with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        async for event in graph.astream(graph_input, config=config):
            now = time.perf_counter()
            elapsed = now - node_started_at
            for node_name, state_update in event.items():
                if node_name == "__interrupt__":
                    continue  # handled below via snapshot.tasks, not this raw shape
                # += , not =, since a loop (e.g. reflector -> scout) can visit
                # the same node more than once in one run.
                node_seconds[node_name] = round(node_seconds.get(node_name, 0.0) + elapsed, 2)
                yield node_name, state_update
            node_started_at = now

        snapshot = await graph.aget_state(config)
        if snapshot.next:
            # Paused, not finished — e.g. plan_gate awaiting approval. A
            # "__final__" here would mark an unfinished run done.
            pending = snapshot.tasks[0].interrupts[0].value if snapshot.tasks and snapshot.tasks[0].interrupts else {}
            yield "__interrupt__", pending
        else:
            final_state = dict(snapshot.values)
            final_state["node_seconds"] = node_seconds
            yield "__final__", final_state


async def run_research(query: str, thread_id: Optional[str] = None) -> dict:
    """Convenience wrapper: drain ``astream_research`` and return the full final state."""
    final_state: dict = {}
    async for node_name, state_update in astream_research(query, thread_id):
        if node_name == "__final__":
            final_state = state_update
    return final_state
