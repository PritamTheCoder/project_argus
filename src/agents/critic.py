"""
Project Argus - Critic Agent

Role: Fact Checker / Validator / Reflector
Responsibility: Reviews extracted evidence and existing Knowledge Graph facts against the original query.
Performs Gap Analysis and generates dynamic follow-up queries if necessary.
"""

import logging
from langchain_core.prompts import ChatPromptTemplate
from src.schema.state import AgentState, FactCheckResult
from src.config import CRITIC_MODEL, CRITIC_PROVIDER
from src.utils.llm_factory import get_llm
from src.utils.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def critic_node(state: AgentState) -> dict:
    """
    Review the gathered evidence (and existing KG facts) to determine if more research is needed.
    
    Args:
        state: AgentState containing `query` and `knowledge_gap_detected`.
        
    Returns:
        dict: Updates `critique`, `re_search_required`, `plan` (with new queries), and `active_node`.
    """
    logger.info("Critic (Reflector): Reviewing evidence and analyzing Knowledge Graph for gaps...")

    from src.graph.kg import kg_store

    query = state["query"]
    gap_detected = state.get("knowledge_gap_detected", False)
    session_id = state.get("session_id", "")

    # 1. Fetch relevant existing facts from the KG (scoped to THIS run to avoid
    #    cross-session contamination of the gap analysis).
    kg_facts = ""
    source_stats = {}
    try:
        query_emb = get_embeddings([query])[0]
        retrieved_facts = kg_store.retrieve_relevant_facts(query_emb, k=20, session_id=session_id or None)
        
        for i, fact in enumerate(retrieved_facts):
            support = fact.get("support_level", "UNCERTAIN")
            claim = fact.get("claim", "Unknown Claim")
            cred = fact.get("credibility_score", 0.4)
            stype = fact.get("source_type", "Unknown")
            source_url = fact.get("source_url", "No URL")
            
            # Track source diversity
            if source_url not in source_stats:
                source_stats[source_url] = {"count": 1, "type": stype, "credibility": cred}
            else:
                source_stats[source_url]["count"] += 1
                
            kg_facts += f"- ({support}) [Cred: {cred}, Type: {stype}] {claim}\n"
    except Exception as e:
        logger.error(f"Critic failed to retrieve KG facts: {e}")
        kg_facts = "(No KG facts available.)"
        
    if not kg_facts.strip():
        kg_facts = "(No relevant facts found in the Knowledge Graph.)"
        
    # Build diversity summary
    diversity_str = "Diversity of Sources Available in KG:\n"
    for url, stats in source_stats.items():
        diversity_str += f"- {url} (Type: {stats['type']}, Credibility: {stats['credibility']}, Mentions: {stats['count']})\n"
        
    llm = get_llm(CRITIC_MODEL, CRITIC_PROVIDER, temperature=0)
    structured_llm = llm.with_structured_output(FactCheckResult)
    
    system_prompt = (
        "You are a rigorous Fact Checker, Reflector, and Source Cross-Referencing Engine. "
        "Your job is to review the currently verified facts from the Knowledge Graph against the user's original query.\n\n"
        "Criteria:\n"
        "1. Coverage: Do the existing valid facts fully answer the user's query? What is missing?\n"
        "2. Cross-Referencing: Are there CONTRADICTIONS or CONFLICTS between different facts?\n"
        "3. Specificity: Is the data specific enough (not vague)?\n"
        "4. Source Diversity & Credibility: Are the facts drawn from multiple varied sources? Are the sources credible? If relying purely on low-credibility sources, we need more research.\n\n"
        "If you find that information is missing, vague, or contradictory, OR if the system explicitly flagged a gap, "
        "you MUST set `status` to 'gaps_found' and generate 1-3 highly targeted follow-up search queries (`new_queries`) "
        "to fill those gaps.\n\n"
        "CRITICAL: If you find conflicting facts, explicitly list them and explain the contradiction in your critique."
    )
    
    user_prompt = f"Original Query: {query}\n\nSystem explicitly flagged an evidence gap due to dropped facts? {gap_detected}\n\n{diversity_str}\nKnowledge Graph Facts:\n{kg_facts}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    chain = prompt | structured_llm
    result: FactCheckResult = chain.invoke({"input": user_prompt})
    
    re_search_required = (result.status == "gaps_found")
    
    logger.info(f"Critic: Re-search required? {re_search_required}. Critique: {result.critique}")
    
    new_plan = []
    if re_search_required and result.new_queries:
        new_plan = [intent.model_dump() for intent in result.new_queries]
        logger.info(f"Critic generated {len(new_plan)} dynamic follow-up queries.")
    
    return {
        "critique": result.critique,
        "re_search_required": re_search_required,
        "plan": new_plan,
        "active_node": "critic"
    }
