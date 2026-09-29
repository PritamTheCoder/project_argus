"""
Project Argus - Graph Builder

Wires agent nodes into a cyclic LangGraph StateGraph:
    START → librarian → plan_gate → scout → refiner → verifier → fact_checker →(conditional)→
        ├→ scout               (avg source credibility too low — broaden, don't chase claims)
        ├→ reflector → scout   (targeted gap-fill loop when gaps detected)
        ├→ scout               (broad re-search loop when no specific gaps)
        └→ ghostwriter → citation_auditor → END

The conditional edge honours MAX_RESEARCH_LOOPS to prevent infinite loops.
"""

import logging
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from src.schema.state import AgentState
from src.config import MAX_RESEARCH_LOOPS, LOW_SOURCE_CREDIBILITY_THRESHOLD

from src.agents.librarian import librarian_node
from src.agents.scout import scout_node
from src.agents.refiner import refiner_node
from src.agents.verifier import verifier_node
from src.agents.critic import critic_node
from src.agents.reflector import reflector_node
from src.agents.consensus import consensus_node
from src.agents.writer import writer_node
from src.agents.citation_auditor import citation_auditor_node

logger = logging.getLogger(__name__)


def route_after_critic(state: AgentState) -> str:
    """
    Decide what happens after the fact-checker:
      • sources have been consistently low-credibility → "scout" (broad
        re-search with the Critic's trusted-mode queries, not the Reflector's
        claim-chasing ones — chasing an unverified claim tends to just re-find
        the low-credibility source that made it)
      • gaps + specific claims detected → "reflector" (targeted sub-queries)
      • re-search requested but no specific gaps → "scout" (broad re-search)
      • loop cap hit or sufficient evidence → "consensus" (then Writer,
        then the citation auditor)
    """
    re_search = state.get("re_search_required", False)
    gap_detected = state.get("knowledge_gap_detected", False)
    knowledge_gaps = state.get("knowledge_gaps", [])
    iteration = state.get("iteration_count", 0)
    avg_credibility = (state.get("quality_score") or {}).get("avg_source_credibility", 1.0)

    if re_search and iteration < MAX_RESEARCH_LOOPS:
        if avg_credibility < LOW_SOURCE_CREDIBILITY_THRESHOLD:
            logger.info(
                f"Router: avg source credibility {avg_credibility:.2f} is below "
                f"{LOW_SOURCE_CREDIBILITY_THRESHOLD} (iteration {iteration}/{MAX_RESEARCH_LOOPS}). "
                "Broadening search toward trusted sources instead of chasing claims."
            )
            return "scout"
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


def _critic_with_counter(state: AgentState) -> dict:
    """Wraps critic_node and bumps iteration_count so the router can track loop count."""
    result = critic_node(state)
    current_count = state.get("iteration_count", 0)
    result["iteration_count"] = current_count + 1
    return result


def plan_gate_node(state: AgentState) -> dict:
    """Pauses for plan approval when ``require_approval`` is set, else a
    pass-through. LangGraph re-runs a node's code from the top on every
    resume, so nothing before ``interrupt()`` may have a side effect."""
    if not state.get("require_approval"):
        return {}
    approved_plan = interrupt({"pending_plan": state.get("plan", [])})
    return {"plan": approved_plan}


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

    builder.add_node("librarian", librarian_node)
    builder.add_node("plan_gate", plan_gate_node)
    builder.add_node("scout", scout_node)
    builder.add_node("refiner", refiner_node)
    builder.add_node("verifier", verifier_node)
    builder.add_node("fact_checker", _critic_with_counter)
    builder.add_node("reflector", reflector_node)
    builder.add_node("consensus", consensus_node)
    builder.add_node("ghostwriter", writer_node)
    builder.add_node("citation_auditor", citation_auditor_node)

    builder.add_edge(START, "librarian")
    builder.add_edge("librarian", "plan_gate")
    builder.add_edge("plan_gate", "scout")
    builder.add_edge("scout", "refiner")
    builder.add_edge("refiner", "verifier")
    builder.add_edge("verifier", "fact_checker")

    # Reflector feeds back into Scout with targeted queries
    builder.add_edge("reflector", "scout")

    builder.add_conditional_edges(
        "fact_checker",
        route_after_critic,
        {
            "reflector": "reflector",
            "scout": "scout",
            "consensus": "consensus",
        },
    )

    # Consensus runs once on the final evidence, then the Writer synthesizes,
    # then the citation auditor checks the Writer's own sentences against
    # what they actually cite (Phase 10.A) before the run ends.
    builder.add_edge("consensus", "ghostwriter")
    builder.add_edge("ghostwriter", "citation_auditor")
    builder.add_edge("citation_auditor", END)

    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    graph = builder.compile(**compile_kwargs)

    logger.info("Graph compiled successfully.")
    return graph
