"""
Project Argus - Writer Agent

Role: Ghostwriter
Responsibility: Synthesizes the final report using ONLY the provided evidence.
Ensures rigorous citation.
"""

import logging
from langchain_core.prompts import ChatPromptTemplate
from src.schema.state import AgentState
from src.config import WRITER_MODEL, WRITER_PROVIDER
from src.utils.llm_factory import get_llm

logger = logging.getLogger(__name__)


def writer_node(state: AgentState) -> dict:
    """
    Write the final report.
    
    Args:
        state: AgentState with `query`, `structured_evidence`, `source_map`.
        
    Returns:
        dict: Updates `report`.
    """
    logger.info("Writer: Synthesizing report...")
    
    query = state["query"]
    evidence = state.get("verified_facts", [])
    source_map = state.get("source_map", {})
    
    # --- Citation Deduplication & Mapping Layer ---
    unique_urls = {}       # url -> new_source_id (e.g. "[1]")
    new_source_map = {}    # new_source_id -> {"url": url, ...}
    old_to_new_id_map = {} # old_source_id (e.g. "[4]") -> new_source_id (e.g. "[1]")
    
    next_id = 1
    for old_id, data in source_map.items():
        url = data.get("url")
        if url not in unique_urls:
            new_id = f"[{next_id}]"
            unique_urls[url] = new_id
            new_source_map[new_id] = data
            next_id += 1
        
        old_to_new_id_map[old_id] = unique_urls[url]

    # Format evidence using the deduplicated IDs
    evidence_text = ""
    for fact in evidence:
        if fact.get("support_level") not in ["SUPPORTED", "PARTIALLY_SUPPORTED"]:
            continue
        old_sid = fact.get("source_id", "?")
        new_sid = old_to_new_id_map.get(old_sid, old_sid) # Use original if not in map somehow
        
        claim = fact.get("claim", fact.get("text", ""))
        evidence_text += f"- {claim} [Source: {new_sid}]\n"
        
    llm = get_llm(WRITER_MODEL, WRITER_PROVIDER, temperature=0.7)
    
    system_prompt = (
        "You are a technical Ghostwriter. Write a comprehensive answer to the user's query.\n\n"
        "RULES:\n"
        "1. Use ONLY the provided evidence. Do not use outside knowledge.\n"
        "2. Cite every claim using the source ID, e.g. 'The battery density is 500 Wh/kg [1].'\n"
        "3. If evidence is missing for a specific topic (e.g., Contradictions/Critiques), DO NOT mention the topic. DO NOT write 'No evidence found'. Simply omit the section entirely.\n"
        "4. Write in clean Markdown with headers.\n"
        "5. Do NOT generate a 'References' section manually; just use the [ID] markers in text.\n"
        "6. CRITICAL: Address any highlighted contradictions or caveats found in the Critique.\n"
        "7. Attempt to cite at least 6 unique sources to ensure a broad consensus, provided the sources meet the quality threshold."
    )
    
    critique = state.get("critique", "")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Query: {query}\n\nCritique/Contradictions:\n{critique}\n\nEvidence:\n{evidence}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"query": query, "critique": critique, "evidence": evidence_text})
    report_content = response.content
    
    # Append References Section from Source Map using new deduplicated mapping
    if new_source_map:
        report_content += "\n\n---\n### References\n"
        sorted_ids = sorted(new_source_map.keys(), key=lambda x: int(x.strip("[]")) if x.strip("[]").isdigit() else 0)
        
        for sid in sorted_ids:
            data = new_source_map[sid]
            url = data.get("url", "#")
            report_content += f"- **{sid}**: {url}\n"
            
    logger.info("Writer: Report generation complete.")
    
    return {
        "report": report_content,
        "active_node": "writer"
    }
