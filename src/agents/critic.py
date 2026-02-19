"""
Project Argus - Critic Agent

Role: Fact Checker / Validator
Responsibility: Reviews extracted evidence against the original query.
Decides if we have enough info or need to loop back.
"""

import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.schema.state import AgentState, FactCheckResult
from src.config import CRITIC_MODEL

logger = logging.getLogger(__name__)


def critic_node(state: AgentState) -> dict:
    """
    Review the gathered evidence and determine if more research is needed.
    
    Args:
        state: AgentState containing `query` and `structured_evidence`.
        
    Returns:
        dict: Updates `critique`, `re_search_required`, `active_node`.
    """
    logger.info("Critic: Reviewing evidence...")
    
    query = state["query"]
    evidence = state.get("structured_evidence", [])
    
    # Format evidence for the LLM
    evidence_text = ""
    for i, fact in enumerate(evidence[:50]): # Limit to top 50 facts to fit context
        evidence_text += f"- [{fact.get('source_id', '?')}] {fact['class']}: {fact['text']} ({fact.get('attributes')})\n"
        
    if not evidence_text:
        evidence_text = "(No evidence extracted independently.)"
        
    llm = ChatGoogleGenerativeAI(model=CRITIC_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(FactCheckResult)
    
    system_prompt = (
        "You are a rigorous Fact Checker. Your job is to verify if the extracted evidence "
        "sufficiently answers the user's query.\n\n"
        "Criteria:\n"
        "1. Coverage: Do we have facts for all parts of the query?\n"
        "2. Verification: Are there conflicting numbers?\n"
        "3. Specificity: Is the data specific enough (not vague)?\n\n"
        "Output a boolean `re_search_required` and a `critique` explaining what is missing."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Query: {query}\n\nEvidence:\n{evidence}")
    ])
    
    chain = prompt | structured_llm
    result: FactCheckResult = chain.invoke({"query": query, "evidence": evidence_text})
    
    logger.info(f"Critic: Re-search required? {result.re_search_required}. Critique: {result.critique}")
    
    return {
        "critique": result.critique,
        "re_search_required": result.re_search_required,
        "active_node": "critic"
    }
