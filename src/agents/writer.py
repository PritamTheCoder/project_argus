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
    
    # Citation Deduplication & Mapping Layer
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

    # Build a url-to-newid fallback map
    url_to_new_id = {}
    for old_id, data in source_map.items():
        url = data.get("url")
        if url and url in unique_urls:
            url_to_new_id[url] = unique_urls[url]

    # Format evidence using the deduplicated IDs
    evidence_text = ""
    for fact in evidence:
        if fact.get("support_level") not in ["SUPPORTED", "PARTIALLY_SUPPORTED"]:
            continue
        old_sid = fact.get("source_id", "?")
        new_sid = old_to_new_id_map.get(old_sid)
        
        if new_sid is None:
            # Fallback: try to resolve by source_url
            fact_url = fact.get("source_url", "")
            new_sid = url_to_new_id.get(fact_url, None)
            
        if new_sid is None:
            # Last resort fallback if completely unmatched
            new_sid = old_sid
            logger.warning(f"Writer: Unresolvable source_id={old_sid}")
            
        claim = fact.get("claim", fact.get("text", ""))
        score = fact.get("credibility_score", 0.4)
        stype = fact.get("source_type", "Unknown")
        evidence_text += f"- {claim} [Source: {new_sid}] [Credibility: {score}, Type: {stype}]\n"
        
    sid_counts = {}
    for fact in evidence:
        sid = fact.get("source_id", "?")
        sid_counts[sid] = sid_counts.get(sid, 0) + 1
    logger.info(f"Writer: Citation distribution across source_ids: {sid_counts}")
    logger.info(f"Writer: Total unique sources in source_map: {len(source_map)}")
    logger.info(f"Writer: Total unique URLs after dedup: {len(new_source_map)}")
        
    llm = get_llm(WRITER_MODEL, WRITER_PROVIDER, temperature=0.7)
    
    system_prompt = (
        "You are an elite technical Ghostwriter. Your objective is to write a highly detailed, comprehensive, and exhaustive report answering the user's query.\n\n"
        "IMPORTANT GUIDELINES FOR LENGTH & STRUCTURE:\n"
        "- Write a LONG, in-depth report. Unpack all details thoroughly.\n"
        "- Use a clear structure: Introduction, well-reasoned Body sections with descriptive subheaders, and a strong Conclusion.\n"
        "- DO NOT summarize away important technical details. Expand upon them deeply based on the evidence.\n\n"
        "RULES:\n"
        "1. Use ONLY the provided evidence. Do not use outside knowledge.\n"
        "2. Cite every claim using the source ID (e.g., [1], [3]). Almost every factual sentence MUST be cited.\n"
        "3. **EVIDENCE-GATING & HEDGING (CRITICAL)**:\n"
        "   - For sources with credibility >= 0.7 (Academic/Government/Major News): Assert facts confidently.\n"
        "   - For sources with credibility 0.5-0.69 (Industry/Market Research): State as reported findings or projections.\n"
        "   - For sources with credibility < 0.5 (Unverified/Web): Lightly hedge (e.g., 'according to industry sources' or 'unverified reports suggest').\n"
        "   - IMPORTANT: Do NOT excessively prefix every sentence with 'unverified reports suggest'. Use hedging sparingly and naturally, only where required.\n"
        "   - In cases of contradiction across sources, explicitly prioritize the claim from the higher credibility source, noting the lower-credibility contention.\n"
        "4. **EXHAUSTIVE CITATION**: You must integrate and cite ALL provided evidence. Attempt to cite every unique source provided to ensure maximum coverage.\n"
        "5. Address any highlighted contradictions or caveats found in the Critique.\n"
        "6. Do NOT generate a 'References' section manually; just use the [ID] markers in text.\n"
        "7. If evidence is missing for a specific topic, simply omit that section entirely.\n"
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
            score = data.get("credibility_score", "N/A")
            stype = data.get("source_type", "Unknown")
            report_content += f"- **{sid}**: {url} *(Credibility: {score}, {stype})*\n"
            
    logger.info("Writer: Report generation complete.")
    
    return {
        "report": report_content,
        "active_node": "writer"
    }
