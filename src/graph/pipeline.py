"""Shared pipeline entrypoint — the one place that builds the graph, seeds
initial state, and drives graph.astream(). main.py (CLI), src/ui/runner.py
(Streamlit), and src/api (HTTP) all call this instead of duplicating it."""

from typing import AsyncGenerator, Optional

from src.graph.builder import build_graph
from src.graph.persistence import get_checkpointer, generate_thread_id, get_run_config
from src.schema.state import AgentState


def _initial_state(query: str, session_id: str) -> AgentState:
    return {
        "session_id": session_id,
        "query": query,
        "plan": [],
        "scraped_data": [],
        "structured_evidence": [],
        "source_map": {},
        "critique": "",
        "report": "",
        "re_search_required": False,
        "knowledge_gap_detected": False,
        "knowledge_gaps": [],
        "gap_queries": [],
        "verified_facts": [],
        "iteration_count": 0,
        "active_node": "",
    }


async def astream_research(query: str, thread_id: Optional[str] = None) -> AsyncGenerator[tuple[str, dict], None]:
    """Yield ``(node_name, state_update)`` per node, then a final
    ``("__final__", full_state)`` with the complete merged AgentState (node
    updates are partial). Docs/chunks/facts are session-scoped by
    ``thread_id``, so concurrent calls with different thread_ids are safe."""
    thread_id = thread_id or generate_thread_id()
    config = get_run_config(thread_id)
    initial_state = _initial_state(query, thread_id)

    async with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        async for event in graph.astream(initial_state, config=config):
            for node_name, state_update in event.items():
                yield node_name, state_update

        snapshot = await graph.aget_state(config)
        yield "__final__", dict(snapshot.values)


async def run_research(query: str, thread_id: Optional[str] = None) -> dict:
    """Convenience wrapper: drain ``astream_research`` and return the full final state."""
    final_state: dict = {}
    async for node_name, state_update in astream_research(query, thread_id):
        if node_name == "__final__":
            final_state = state_update
    return final_state
