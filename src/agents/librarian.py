"""
Project Argus - Librarian Agent

Role: Planner / Decomposer
Responsibility: Takes a high-level query and breaks it down into specific search terms.
"""

import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.schema.state import AgentState, ResearchPlan
from src.config import LIBRARIAN_MODEL

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
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model=LIBRARIAN_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(ResearchPlan)
    
    # System prompt to guide the planning
    system_prompt = (
        "You are an expert technical researcher. Your goal is to plan a deep-dive "
        "investigation into the user's topic.\n\n"
        "Break the user's query into 3-5 highly specific, independent search queries "
        "that will cover different aspects of the topic (e.g., technical specs, "
        "market data, challenges, key players).\n\n"
        "Avoid generic queries. Be precise."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # Execute the chain
    chain = prompt | structured_llm
    result: ResearchPlan = chain.invoke({"input": query})
    
    logger.info(f"Librarian: Generated {len(result.search_queries)} queries: {result.search_queries}")
    
    return {
        "plan": result.search_queries,
        "active_node": "librarian"
    }
