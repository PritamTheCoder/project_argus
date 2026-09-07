"""Shared pipeline entrypoint — the one place that builds the graph, seeds
initial state, and drives graph.astream(). main.py (CLI), src/ui/runner.py
(Streamlit), and src/api (HTTP) all call this instead of duplicating it."""

import time
from typing import Any, AsyncGenerator, Optional

from src.graph.builder import build_graph
from src.graph.persistence import get_checkpointer, generate_thread_id, get_run_config
from src.schema.state import AgentState


def _initial_state(query: str, session_id: str) -> AgentState:
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
    }


async def astream_research(
    query: Optional[str] = None, thread_id: Optional[str] = None,
    callbacks: Optional[list[Any]] = None, resume: bool = False,
) -> AsyncGenerator[tuple[str, dict], None]:
    """Yield (node_name, state_update) per node, then ("__final__", full_state)
    with the complete AgentState plus a node_seconds timing breakdown.
    callbacks is passed to the graph run config (e.g. a token-usage tracker).

    ``resume=True`` continues an already-seeded thread (e.g. one written by
    ``fork_thread``) instead of starting fresh: passing ``None`` as the graph
    input tells LangGraph to read the existing checkpoint rather than
    overwrite it with a new initial state. ``query`` is unused in that mode —
    the seeded state already carries it."""
    thread_id = thread_id or generate_thread_id()
    config = get_run_config(thread_id)
    if callbacks:
        config["callbacks"] = callbacks
    graph_input = None if resume else _initial_state(query, thread_id)

    node_seconds: dict[str, float] = {}
    node_started_at = time.perf_counter()

    async with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        async for event in graph.astream(graph_input, config=config):
            now = time.perf_counter()
            elapsed = now - node_started_at
            for node_name, state_update in event.items():
                # += , not =, since a loop (e.g. reflector -> scout) can visit
                # the same node more than once in one run.
                node_seconds[node_name] = round(node_seconds.get(node_name, 0.0) + elapsed, 2)
                yield node_name, state_update
            node_started_at = now

        snapshot = await graph.aget_state(config)
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
