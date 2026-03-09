"""
Project Argus - Refiner Agent

Role: Fact Extractor
Responsibility: Takes raw scraped text and extracts structured, source-linked facts.
Dynamically generates a schema based on the plan.
"""

import logging
from typing import Dict
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from src.schema.state import AgentState
from src.tools.refiner import extract_facts
from src.config import LIBRARIAN_MODEL

logger = logging.getLogger(__name__)

class ExtractionSchema(BaseModel):
    schema_dict: Dict[str, str] = Field(
        description="A dictionary where keys are entity types (e.g., 'dates', 'companies', 'metrics') and values are short descriptions of what to extract."
    )


def _generate_dynamic_schema(plan: list[str]) -> dict:
    """
    Use an LLM to generate an extraction schema based on the research plan.
    Uses Pydantic structured output to guarantee valid formatting.
    """
    try:
        llm = ChatGoogleGenerativeAI(model=LIBRARIAN_MODEL, temperature=0)
        structured_llm = llm.with_structured_output(ExtractionSchema)
        
        system_prompt = (
            "You are a data schema expert. Given a research plan, define the precise data points we need to extract from web pages to answer the queries."
            "Generate a dictionary where keys are entity shortnames and values are descriptions of the data to extract."
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
    """
    Batch extracts facts from scraped data.
    Takes all scraped_data, packs them into a single string with source tags, and invokes the Refiner Tool.
    """
    logger.info("Refiner: Starting batched fact extraction...")
    
    scraped_data = state.get("scraped_data", [])
    if not scraped_data:
        logger.warning("Refiner: No scraped data to process.")
        return {"structured_evidence": [], "active_node": "refiner"}
        
    plan = state.get("plan", [])
    
    # 1. Generate schema
    schema = _generate_dynamic_schema(plan)
    logger.info(f"Refiner: Generated schema: {schema}")
    
    # 2. Pack the context
    batched_text = ""
    for doc in scraped_data:
        text = doc.get("content", "")
        source_id = doc.get("source_id", "[?]")
        # Trim very long docs to prevent overflow if necessary, otherwise rely on 1M token limit
        if text.strip():
            batched_text += f"\n<document source_id=\"{source_id}\">\n{text}\n</document>\n"
            
    if not batched_text.strip():
        logger.warning("Refiner: All scraped data was empty.")
        return {"structured_evidence": [], "active_node": "refiner"}
        
    # 3. Call the Batch Tool
    all_facts = []
    try:
        logger.info(f"Refiner: Passing massive batched payload to natively extract facts...")
        extraction_result = extract_facts(batched_text, schema)
        all_facts = extraction_result.get("facts", [])
        logger.info(f"Refiner: Extracted {len(all_facts)} total facts from batched payload.")
    except Exception as e:
        logger.error(f"Refiner: Batch extraction failed: {e}")
    
    return {
        "structured_evidence": all_facts,
        "active_node": "refiner"
    }
