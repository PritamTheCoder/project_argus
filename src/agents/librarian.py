"""
Project Argus - Librarian Agent

Role: Planner / Decomposer
Responsibility: Takes a high-level query and breaks it down into specific search terms.
"""

import logging
from langchain_core.prompts import ChatPromptTemplate
from src.schema.state import AgentState, ResearchPlan
from src.config import LIBRARIAN_MODEL, LIBRARIAN_PROVIDER
from src.utils.llm_factory import get_llm
from src.utils.retry import retry_on_rate_limit

logger = logging.getLogger(__name__)


def librarian_node(state: AgentState) -> dict:
    """
    Decompose the user's query into a precise research plan (list of search queries).
    
    Args:
        state: AgentState containing `query`.
        
    Returns:
        dict: Updates `plan` and `active_node`.
    """
    logger.info("Librarian: Analyzing query...")

    query = state["query"]
    iteration = state.get("iteration_count", 0)
    critique = state.get("critique", "")

    llm = get_llm(LIBRARIAN_MODEL, LIBRARIAN_PROVIDER, temperature=0)
    structured_llm = llm.with_structured_output(ResearchPlan)

    system_prompt = (
        "You are an expert technical researcher. Your goal is to plan a deep-dive "
        "investigation into the user's topic.\n\n"
        "Break the user's query into 3-5 highly specific, independent search queries "
        "that will cover different aspects of the topic (e.g., technical specs, "
        "market data, challenges, key players).\n\n"
        "For each query, assign the most appropriate `mode`:\n"
        "- `TRUSTED_ONLY`: For hard science, academic metrics, and technical specifications.\n"
        "- `TRUSTED_FIRST`: For broad analysis, industry reports, or general overviews.\n"
        "- `MIXED`: For market data, latest news, company announcements, or commercial pricing.\n\n"
        "Avoid generic queries. Be precise."
    )

    if iteration > 0 and critique:
        system_prompt += (
            f"\n\nPREVIOUS CRITIC ASSESSMENT:\n{critique}\n\n"
            "Generate queries that SPECIFICALLY address the identified gaps above. "
            "Do NOT repeat queries from the previous iteration."
        )
        logger.info(f"Librarian: Iteration {iteration} — injecting critique for targeted gap-fill queries.")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # Execute the chain with retry logic for rate-limit errors
    chain = prompt | structured_llm
    result: ResearchPlan = retry_on_rate_limit(chain.invoke, {"input": query})
    
    logger.info(f"Librarian: Generated {len(result.search_queries)} queries.")
    
    plan_dicts = [intent.model_dump() for intent in result.search_queries]
    
    return {
        "plan": plan_dicts,
        "active_node": "librarian"
    }

