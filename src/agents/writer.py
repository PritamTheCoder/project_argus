"""
Project Argus - Writer Agent

Role: Ghostwriter
Responsibility: Synthesizes the final report using ONLY the provided evidence.
Ensures rigorous citation.
"""

import logging
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.schema.state import AgentState
from src.config import WRITER_MODEL

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
    evidence = state.get("structured_evidence", [])
    source_map = state.get("source_map", {})
    
    # Format evidence
    evidence_text = ""
    for fact in evidence:
        sid = fact.get("source_id", "?")
        evidence_text += f"- {fact['text']} [Source: {sid}]\n"
        
    llm = ChatGoogleGenerativeAI(model=WRITER_MODEL, temperature=0.7)
    
    system_prompt = (
        "You are a technical Ghostwriter. Write a comprehensive answer to the user's query.\n\n"
        "RULES:\n"
        "1. Use ONLY the provided evidence. Do not use outside knowledge.\n"
        "2. Cite every claim using the source ID, e.g. 'The battery density is 500 Wh/kg [1].'\n"
        "3. If evidence is missing, state 'Evidence not found for X'.\n"
        "4. Write in clean Markdown with headers.\n"
        "5. Do NOT generate a 'References' section manually; just use the [ID] markers in text."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Query: {query}\n\nEvidence:\n{evidence}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"query": query, "evidence": evidence_text})
    report_content = response.content
    
    # Append References Section from Source Map
    if source_map:
        report_content += "\n\n---\n### References\n"
        # Sort by ID if possible, roughly
        sorted_ids = sorted(source_map.keys(), key=lambda x: int(x.strip("[]")) if x.strip("[]").isdigit() else 0)
        
        for sid in sorted_ids:
            data = source_map[sid]
            url = data.get("url", "#")
            report_content += f"- **{sid}**: {url}\n"
            
    logger.info("Writer: Report generation complete.")
    
    return {
        "report": report_content,
        "active_node": "writer"
    }
