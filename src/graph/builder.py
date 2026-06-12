"""
Project Argus - Graph Builder

Wires agent nodes into a cyclic LangGraph StateGraph:
    START → librarian → scout → refiner → verifier → fact_checker →(conditional)→
        ├→ reflector → scout   (targeted gap-fill loop when gaps detected)
        ├→ scout               (broad re-search loop when no specific gaps)
        └→ ghostwriter → END

The conditional edge honours MAX_RESEARCH_LOOPS to prevent infinite loops.
"""

import logging
from langgraph.graph import StateGraph, START, END

from src.schema.state import AgentState
from src.config import MAX_RESEARCH_LOOPS

# Agent Node Imports
from src.agents.librarian import librarian_node
from src.agents.scout import scout_node
from src.agents.refiner import refiner_node
from src.agents.verifier import verifier_node
from src.agents.critic import critic_node
from src.agents.reflector import reflector_node
from src.agents.consensus import consensus_node
from src.agents.writer import writer_node

logger = logging.getLogger(__name__)


# Conditional Routing

def route_after_critic(state: AgentState) -> str:
    """
    Decide what happens after the fact-checker:
      • gaps + specific claims detected → "reflector" (targeted sub-queries)
      • re-search requested but no specific gaps → "scout" (broad re-search)
      • loop cap hit or sufficient evidence → "consensus" (then Writer)
    """
    re_search = state.get("re_search_required", False)
    gap_detected = state.get("knowledge_gap_detected", False)
    knowledge_gaps = state.get("knowledge_gaps", [])
    iteration = state.get("iteration_count", 0)

    if re_search and iteration < MAX_RESEARCH_LOOPS:
        if gap_detected and knowledge_gaps:
            logger.info(
                f"Router: {len(knowledge_gaps)} specific gap(s) detected "
                f"(iteration {iteration}/{MAX_RESEARCH_LOOPS}). "
                "Routing to Reflector for targeted queries."
            )
            return "reflector"
        logger.info(
            f"Router: Re-search requested (iteration {iteration}/{MAX_RESEARCH_LOOPS}). "
            "No specific gaps — looping back to Scout with critic's queries."
        )
        return "scout"

    if re_search and iteration >= MAX_RESEARCH_LOOPS:
        logger.warning(
            f"Router: Loop cap ({MAX_RESEARCH_LOOPS}) reached. Proceeding to synthesis."
        )

    # Done researching — run consensus/contradiction analysis before the Writer.
    return "consensus"


# Wrapper to increment iteration_count

def _critic_with_counter(state: AgentState) -> dict:
    """
    Wraps the real critic_node and bumps iteration_count
    so the router knows how many loops we've done.
    """
    result = critic_node(state)
    current_count = state.get("iteration_count", 0)
    result["iteration_count"] = current_count + 1
    return result


# Graph Construction

def build_graph(checkpointer=None):
    """
    Construct and compile the Project Argus StateGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. SqliteSaver)
                      for state persistence / time-travel.

    Returns:
        A compiled, runnable LangGraph graph.
    """
    builder = StateGraph(AgentState)

    # Register Nodes
    builder.add_node("librarian", librarian_node)
    builder.add_node("scout", scout_node)            # async — LangGraph handles it
    builder.add_node("refiner", refiner_node)
    builder.add_node("verifier", verifier_node)
    builder.add_node("fact_checker", _critic_with_counter)
    builder.add_node("reflector", reflector_node)
    builder.add_node("consensus", consensus_node)
    builder.add_node("ghostwriter", writer_node)

    # Linear Edges
    builder.add_edge(START, "librarian")
    builder.add_edge("librarian", "scout")
    builder.add_edge("scout", "refiner")
    builder.add_edge("refiner", "verifier")
    builder.add_edge("verifier", "fact_checker")

    # Reflector feeds back into Scout with targeted queries
    builder.add_edge("reflector", "scout")

    # Conditional Edge (the Research Loop)
    builder.add_conditional_edges(
        "fact_checker",
        route_after_critic,
        {
            "reflector": "reflector",
            "scout": "scout",
            "consensus": "consensus",
        },
    )

    # Consensus runs once on the final evidence, then the Writer synthesizes.
    builder.add_edge("consensus", "ghostwriter")

    # Terminal Edge
    builder.add_edge("ghostwriter", END)

    # Compile
    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    graph = builder.compile(**compile_kwargs)

    logger.info("Graph compiled successfully.")
    return graph
