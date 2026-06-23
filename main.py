"""
Project Argus - Command-line entrypoint

Runs the full LangGraph research pipeline (cyclic fact-checker loop, SQLite
state checkpointing) and prints node transitions and the final report.

Usage:
    python main.py
    python main.py "your research query here"
"""

import sys
import asyncio
import logging

from src.graph.builder import build_graph
from src.graph.persistence import get_checkpointer, generate_thread_id, get_run_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("argus")


async def run_query(query: str):
    print("\n" + "=" * 80)
    print("Project Argus")
    print(f"Query: {query}")
    print("=" * 80 + "\n")

    # ── 1. Build graph and clear vector scratchpad ─────────────────────────
    print("[1] Building graph and initializing Knowledge Graph...", flush=True)
    from src.graph.kg import kg_store
    kg_store.clear_scratchpad()
    
    async with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)

        thread_id = generate_thread_id()
        config = get_run_config(thread_id)
        print(f"Session: {thread_id}\n", flush=True)

        initial_state = {
            "session_id": thread_id,
            "query": query,
            "plan": [],
            "scraped_data": [],
            "structured_evidence": [],
            "source_map": {},
            "critique": "",
            "report": "",
            "re_search_required": False,
            "knowledge_gap_detected": False,
            "verified_facts": [],
            "iteration_count": 0,
            "active_node": "start",
        }

        print("Executing pipeline...\n", flush=True)

        previous_node = None
        final_state = None

        async for event in graph.astream(initial_state, config=config):
            for node_name, node_output in event.items():
                if node_name != previous_node:
                    print(f"  -> {node_name}", flush=True)
                    previous_node = node_name

                if "plan" in node_output and node_output["plan"]:
                    print(f"       generated {len(node_output['plan'])} search queries", flush=True)
                if "scraped_data" in node_output:
                    print(f"       scraped {len(node_output['scraped_data'])} pages", flush=True)
                if "structured_evidence" in node_output:
                    print(f"       extracted {len(node_output['structured_evidence'])} facts", flush=True)
                if "re_search_required" in node_output:
                    re_search = node_output["re_search_required"]
                    iteration = node_output.get("iteration_count", "?")
                    print(f"       re-search: {re_search} (iteration {iteration})", flush=True)
                if "report" in node_output and node_output["report"]:
                    print(f"       report generated ({len(node_output['report'])} chars)", flush=True)

                final_state = node_output

        print("\n" + "=" * 80)
        print("FINAL REPORT")
        print("=" * 80 + "\n")

        if final_state and "report" in final_state:
            print(final_state["report"])
        else:
            print("(No report generated — check logs for errors)")

        print("\n" + "=" * 80)
        print(f"State persisted to SQLite (thread: {thread_id})")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What are the latest breakthroughs in Solid State Batteries as of 2024-2025?"
    )
    asyncio.run(run_query(query))
