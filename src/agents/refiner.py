"""Refiner agent: extracts structured, source-linked facts from raw scraped text,
using an extraction schema generated dynamically from the research plan."""

import logging
from typing import Dict
from pydantic import BaseModel, Field
from src.schema.state import AgentState
from src.tools.refiner import extract_facts
from src.config import LIBRARIAN_MODEL, LIBRARIAN_PROVIDER
from src.utils.llm_factory import get_llm

logger = logging.getLogger(__name__)

class ExtractionSchema(BaseModel):
    schema_dict: Dict[str, str] = Field(
        description="A dictionary where keys are entity types (e.g., 'dates', 'companies', 'metrics') and values are short descriptions of what to extract."
    )


def _generate_dynamic_schema(plan: list[str]) -> dict:
    """Use an LLM to generate an extraction schema based on the research plan."""
    try:
        llm = get_llm(LIBRARIAN_MODEL, LIBRARIAN_PROVIDER, temperature=0)
        structured_llm = llm.with_structured_output(ExtractionSchema)
        
        system_prompt = (
            "You are a data schema expert. Given a research plan, define the precise data points we need to extract from web pages to answer the queries.\n"
            "Generate a dictionary where keys are entity shortnames and values are descriptions of the data to extract.\n"
            "CRITICAL: You MUST ALWAYS include keys named 'Notable_Outliers' and 'Lateral_Innovations' to capture unexpected, serendipitous, or edge-case discoveries that don't fit standard metrics."
        )
        
        user_msg = f"Research Plan target queries: {plan}"
        
        messages = [
            ("system", system_prompt),
            ("human", user_msg)
        ]
        
        result: ExtractionSchema = structured_llm.invoke(messages)
        return result.schema_dict
        
    except Exception as e:
        logger.warning(f"Refiner: Structured schema generation failed ({e}). Using fallback.")
        return {"facts": "important facts found in the text"}


def refiner_node(state: AgentState) -> dict:
    """Batch-extract facts from scraped data: packs scraped_data into a single
    string with source tags and invokes the refiner tool."""
    logger.info("Refiner: Starting batched fact extraction...")
    
    scraped_data = state.get("scraped_data", [])
    if not scraped_data:
        logger.warning("Refiner: No scraped data to process.")
        return {"structured_evidence": [], "active_node": "refiner"}
        
    plan = state.get("plan", [])
    schema = _generate_dynamic_schema(plan)
    logger.info(f"Refiner: Generated schema: {schema}")
    
    # Avoid re-extracting facts from documents already processed in a prior iteration.
    already_processed_sids = {
        f.get("source_id") for f in state.get("verified_facts", [])
    }
    new_docs = [
        doc for doc in scraped_data 
        if doc.get("source_id") not in already_processed_sids
    ]
    
    if not new_docs:
        logger.warning("Refiner: No new documents to process.")
        return {"structured_evidence": [], "active_node": "refiner"}
        
    batched_text = ""
    for doc in new_docs:
        text = doc.get("content", "")
        source_id = doc.get("source_id", "[?]")
        if text.strip():
            batched_text += f"\n<document source_id=\"{source_id}\">\n{text}\n</document>\n"
            
    if not batched_text.strip():
        logger.warning("Refiner: All scraped data was empty.")
        return {"structured_evidence": [], "active_node": "refiner"}
        
    source_map = state.get("source_map", {})

    all_facts = []
    try:
        extraction_result = extract_facts(batched_text, schema)
        all_facts = extraction_result.get("facts", [])

        for fact in all_facts:
            s_id = fact.get("source_id")
            if s_id and s_id in source_map:
                fact["source_url"] = source_map[s_id].get("url", "")
            else:
                fact["source_url"] = ""
                
        logger.info(f"Refiner: Extracted {len(all_facts)} total facts from batched payload.")
    except Exception as e:
        logger.error(f"Refiner: Batch extraction failed: {e}")
    
    return {
        "structured_evidence": all_facts,
        "active_node": "refiner"
    }
