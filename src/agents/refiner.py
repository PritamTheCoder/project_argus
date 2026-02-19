"""
Project Argus - Refiner Agent

Role: Fact Extractor
Responsibility: Takes raw scraped text and extracts structured, source-linked facts.
Dynamically generates a schema based on the plan.
"""

import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.schema.state import AgentState
from src.tools.refiner import extract_facts
from src.config import LIBRARIAN_MODEL

logger = logging.getLogger(__name__)


def _generate_dynamic_schema(plan: list[str]) -> dict:
    """
    Use an LLM to generate an extraction schema based on the research plan.
    Ideally, this should be robust. For now, we use a simple prompt to get JSON.
    """
    try:
        llm = ChatGoogleGenerativeAI(model=LIBRARIAN_MODEL, temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
        
        system_prompt = (
            "You are a data schema expert. Given a research plan, generate a JSON object "
            "where keys are entity types (e.g., 'dates', 'metrics', 'costs', 'definitions') "
            "and values are short descriptions of what to extract."
            "\n\nExample Output:\n"
            "{\n  \"dates\": \"exact dates of events\",\n  \"metrics\": \"performance numbers like Wh/kg\"\n}"
        )
        
        user_msg = f"Plan: {plan}"
        
        messages = [
            ("system", system_prompt),
            ("human", user_msg)
        ]
        
        response = llm.invoke(messages)
        import json
        return json.loads(response.content)
        
    except Exception as e:
        logger.warning(f"Refiner: Schema generation failed ({e}). Using fallback.")
        return {"facts": "important facts found in the text"}


def refiner_node(state: AgentState) -> dict:
    """
    Extract facts from scraped data using a dynamically generated schema.
    
    Args:
        state: AgentState with `scraped_data` and `plan`.
        
    Returns:
        dict: Updates `structured_evidence`.
    """
    logger.info("Refiner: Starting fact extraction...")
    
    scraped_data = state.get("scraped_data", [])
    if not scraped_data:
        logger.warning("Refiner: No scraped data to process.")
        return {"structured_evidence": [], "active_node": "refiner"}
        
    plan = state.get("plan", [])
    
    # 1. Generate schema
    schema = _generate_dynamic_schema(plan)
    logger.info(f"Refiner: Generated schema: {schema}")
    
    # 2. Extract from each document
    all_facts = []
    
    import time
    
    for doc in scraped_data:
        text = doc.get("content", "")
        source_id = doc.get("source_id", "[?]")
        
        # Skip empty docs
        if not text.strip():
            continue
            
        try:
            # Rate limit protection
            time.sleep(2)
            
            # Call the tool
            extraction_result = extract_facts(text[:15000], schema) # Limit context win if needed
            
            # Tag facts with source ID
            for fact in extraction_result["facts"]:
                fact["source_id"] = source_id
                all_facts.append(fact)
                
        except Exception as e:
            logger.error(f"Refiner: Failed to extract from {source_id}: {e}")
            continue
            
    logger.info(f"Refiner: Extracted {len(all_facts)} total facts.")
    
    return {
        "structured_evidence": all_facts,
        "active_node": "refiner"
    }
