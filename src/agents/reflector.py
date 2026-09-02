"""
Reflector agent: generates highly specific follow-up search queries to fill
research gaps — rather than triggering a broad full re-search via the Librarian.

Two kinds of gap, prioritized in this order:
  1. coverage_gaps (from the Critic): original sub-questions with NO supporting
     evidence at all. These point at genuinely new search territory.
  2. knowledge_gaps (from the Verifier): specific claims that couldn't be
     verified. Searching for these often just re-finds the same low-credibility
     source that made the claim — useful, but lower priority than real coverage.
"""

import logging
from pydantic import BaseModel, Field
from typing import List

from src.schema.state import AgentState, SearchIntent
from src.config import CRITIC_MODEL, CRITIC_PROVIDER, CRITIC_FALLBACK_CHAIN
from src.utils.llm_factory import get_llm_with_fallbacks
from src.utils.embeddings import get_embeddings

logger = logging.getLogger(__name__)


class ReflectorOutput(BaseModel):
    search_queries: List[SearchIntent] = Field(
        description="2-4 highly specific, targeted search queries to fill identified knowledge gaps."
    )


def reflector_node(state: AgentState) -> dict:
    """
    Generate targeted sub-queries for the current research gaps.

    Priority order for gap source:
      1. coverage_gaps (set by the Critic this iteration) — new territory
      2. knowledge_gaps (set by the Verifier this iteration) — fills remaining slots
      3. KG find_gaps() — scans previously stored facts for weak support
      4. Critic's existing plan — pass-through if nothing above yields gaps
    """
    query = state["query"]
    session_id = state.get("session_id", "")
    coverage_gaps: List[str] = state.get("coverage_gaps", [])
    knowledge_gaps: List[str] = state.get("knowledge_gaps", [])
    gaps = coverage_gaps + [g for g in knowledge_gaps if g not in coverage_gaps]

    # Fallback: pull directly from KG if state has no gaps yet (scoped to this run).
    # Imported lazily, like every other agent module's KG access, to avoid a
    # module-load-time circular import with src.graph.builder (which imports
    # this module).
    if not gaps:
        try:
            from src.graph.kg import kg_store
            query_emb = get_embeddings([query])[0]
            gaps = kg_store.find_gaps(query_emb, session_id=session_id or None)
            logger.info(f"Reflector: Retrieved {len(gaps)} gap(s) from KG.")
        except Exception as e:
            logger.warning(f"Reflector: KG find_gaps fallback failed: {e}")

    if not gaps:
        logger.info("Reflector: No specific gaps found — passing through existing plan.")
        return {"active_node": "reflector"}

    # Focus on the most critical gaps to keep queries tight
    top_gaps = gaps[:5]
    gap_text = "\n".join(f"- {g}" for g in top_gaps)

    structured_llm = get_llm_with_fallbacks(
        CRITIC_MODEL, CRITIC_PROVIDER,
        fallback_chain=CRITIC_FALLBACK_CHAIN,
        temperature=0,
        structured_schema=ReflectorOutput,
    )

    prompt = (
        "You are a precision research assistant. The research so far has gaps — either "
        "sub-questions with no supporting evidence yet, or specific claims that could not "
        "be verified:\n\n"
        f"{gap_text}\n\n"
        f"Original research query: {query}\n\n"
        "Generate 2-4 highly specific search queries that would find credible, authoritative "
        "sources to fill these gaps. "
        "Queries must be narrow and targeted — not broad re-statements of the original query."
    )

    try:
        result: ReflectorOutput = structured_llm.invoke(prompt)
        gap_queries = [intent.model_dump() for intent in result.search_queries]
        logger.info(f"Reflector: Generated {len(gap_queries)} targeted gap-filling queries.")
    except Exception as e:
        logger.error(f"Reflector: LLM call failed ({e}). Falling back to critic's existing plan.")
        gap_queries = state.get("plan", [])

    # Merge with the Critic's own follow-up queries (state["plan"]) rather than
    # replacing them, so Scout's next pass covers both new territory (the
    # Critic's queries) and these targeted gap-fills — not just one or the other.
    critic_plan = state.get("plan", []) or []
    merged_plan = list(critic_plan)
    seen_queries = {intent.get("query", "") for intent in merged_plan}
    for intent in gap_queries:
        q = intent.get("query", "")
        if q and q not in seen_queries:
            merged_plan.append(intent)
            seen_queries.add(q)

    return {
        "plan": merged_plan,
        "gap_queries": gap_queries,
        "active_node": "reflector",
    }
