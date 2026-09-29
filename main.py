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

from src.graph.pipeline import astream_research
from src.graph.persistence import generate_thread_id
from src.utils.logging_setup import configure_logging

configure_logging()
logger = logging.getLogger("argus")


async def run_query(query: str):
    print("\n" + "=" * 80)
    print("Project Argus")
    print(f"Query: {query}")
    print("=" * 80 + "\n")

    thread_id = generate_thread_id()
    print(f"Session: {thread_id}\n", flush=True)
    print("Executing pipeline...\n", flush=True)

    previous_node = None
    final_state = None

    async for node_name, node_output in astream_research(query, thread_id=thread_id):
        if node_name == "__final__":
            final_state = node_output
            continue

        if node_name != previous_node:
            print(f"  -> {node_name}", flush=True)
            previous_node = node_name

        # A node that writes nothing (plan_gate with approval off) streams None.
        node_output = node_output or {}
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

    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80 + "\n")

    if final_state and final_state.get("report"):
        print(final_state["report"])
    else:
        print("(No report generated — check logs for errors)")

    node_seconds = (final_state or {}).get("node_seconds") or {}
    if node_seconds:
        print("\nTime per node:")
        for node, seconds in sorted(node_seconds.items(), key=lambda kv: -kv[1]):
            print(f"  {node:<15} {seconds:>6.1f}s")

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
