"""
Reflector agent: analyses knowledge_gaps (unsupported/uncertain claims from the
Verifier) and generates highly specific follow-up search queries to fill only
those gaps — rather than triggering a broad full re-search via the Librarian.
"""

import logging
from pydantic import BaseModel, Field
from typing import List

from src.schema.state import AgentState, SearchIntent
from src.config import CRITIC_MODEL, CRITIC_PROVIDER
from src.utils.llm_factory import get_llm
from src.utils.embeddings import get_embeddings
from src.graph.kg import kg_store

logger = logging.getLogger(__name__)


class ReflectorOutput(BaseModel):
    search_queries: List[SearchIntent] = Field(
        description="2-4 highly specific, targeted search queries to fill identified knowledge gaps."
    )


def reflector_node(state: AgentState) -> dict:
    """
    Generate targeted sub-queries for specific knowledge gaps.

    Priority order for gap source:
      1. knowledge_gaps already in state (set by Verifier this iteration)
      2. KG find_gaps() — scans previously stored facts for weak support
      3. Critic's existing plan — pass-through if neither source yields gaps
    """
    query = state["query"]
    session_id = state.get("session_id", "")
    knowledge_gaps: List[str] = state.get("knowledge_gaps", [])

    # Fallback: pull directly from KG if state has no gaps yet (scoped to this run)
    if not knowledge_gaps:
        try:
            query_emb = get_embeddings([query])[0]
            knowledge_gaps = kg_store.find_gaps(query_emb, session_id=session_id or None)
            logger.info(f"Reflector: Retrieved {len(knowledge_gaps)} gap(s) from KG.")
        except Exception as e:
            logger.warning(f"Reflector: KG find_gaps fallback failed: {e}")

    if not knowledge_gaps:
        logger.info("Reflector: No specific gaps found — passing through existing plan.")
        return {"active_node": "reflector"}

    # Focus on the most critical gaps to keep queries tight
    top_gaps = knowledge_gaps[:5]
    gap_text = "\n".join(f"- {g}" for g in top_gaps)

    llm = get_llm(CRITIC_MODEL, CRITIC_PROVIDER, temperature=0)
    structured_llm = llm.with_structured_output(ReflectorOutput)

    prompt = (
        "You are a precision research assistant. The following claims were flagged as "
        "unsupported or uncertain during fact-checking:\n\n"
        f"{gap_text}\n\n"
        f"Original research query: {query}\n\n"
        "Generate 2-4 highly specific search queries that would find credible, authoritative "
        "sources to either verify or disprove each of these claims. "
        "Queries must be narrow and targeted — not broad re-statements of the original query."
    )

    try:
        result: ReflectorOutput = structured_llm.invoke(prompt)
        gap_queries = [intent.model_dump() for intent in result.search_queries]
        logger.info(f"Reflector: Generated {len(gap_queries)} targeted gap-filling queries.")
    except Exception as e:
        logger.error(f"Reflector: LLM call failed ({e}). Falling back to critic's existing plan.")
        gap_queries = state.get("plan", [])

    return {
        "plan": gap_queries,
        "gap_queries": gap_queries,
        "active_node": "reflector",
    }
