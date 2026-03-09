"""
Project Argus - Batch Refiner Tool

Extracts structured facts from concatenated raw Markdown using Gemini Guided JSON Output.
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import REFINER_MODEL

logger = logging.getLogger(__name__)

class ExtractedFact(BaseModel):
    extraction_class: str = Field(description="The category of the extracted fact (e.g., 'dates', 'metrics').")
    text: str = Field(description="The exact text extracted from the source document.")
    source_id: str = Field(description="The source_id of the document where this fact was found. MUST exactly match the source_id in the <document source_id=\"...\"> tag.")
    attributes: Dict[str, Any] = Field(description="Additional attributes summarizing the fact based on the schema.", default_factory=dict)

class FactExtractionResult(BaseModel):
    facts: List[ExtractedFact] = Field(description="List of all extracted facts from all provided documents.")

def extract_facts(batched_text: str, schema: dict) -> dict:
    """
    Extract structured facts from batched raw text using Gemini Structured Outputs.

    Args:
        batched_text: Raw Markdown containing multiple documents wrapped in <document source_id="..."> tags.
        schema: A dict mapping entity names to descriptions.
                Example: {"dates": "date", "metrics": "numeric value"}

    Returns:
        dict: {"facts": [...list of facts...], "raw_jsonl": ""}
    """
    logger.info(f"Refiner Tool: Extracting facts for schema keys {list(schema.keys())}")
    
    parts = [f"{key} ({desc})" for key, desc in schema.items()]
    joined_schema = ", ".join(parts)
    
    prompt = f"""You are a specialized Data Extraction Engine.
Your task is to extract highly specific facts from the provided documents.
Each document is wrapped in <document source_id="..."> tags. 

Target Schema: Extract the following entities: {joined_schema}.
For each fact:
1. Provide the exact text from the document.
2. Provide the extraction_class based on the Target Schema.
3. Critically: You must provide the exact `source_id` of the document where you found the fact.
4. Add meaningful `attributes` to provide context.

Documents to Analyze:
====================
{batched_text}
====================
"""

    try:
        # Context Window Meter
        estimated_tokens = len(batched_text) // 4
        limit = 1000000
        percent = (estimated_tokens / limit) * 100
        bars = int(percent / 5)
        meter = f"[{'|'*bars}{' '*(20-bars)}] {percent:.1f}% ({estimated_tokens:,} / 1M tokens)"
        logger.info(f"Refiner: Context Window Utilization:\n    {meter}")
        
        logger.info("Refiner: Passing massive batched payload to natively extract facts...")
        
        llm = ChatGoogleGenerativeAI(model=REFINER_MODEL, temperature=0, max_retries=2)
        structured_llm = llm.with_structured_output(FactExtractionResult)
        
        result: FactExtractionResult = structured_llm.invoke(prompt)
        
        # Convert Pydantic to old dict format for compatibility with the rest of the application
        facts = []
        if result and result.facts:
            for fact in result.facts:
                facts.append({
                    "class": fact.extraction_class,
                    "text": fact.text,
                    "source_id": fact.source_id,
                    "attributes": fact.attributes or {},
                    "source_span": {"start": None, "end": None} # Legacy compatibility
                })
        
        return {"facts": facts, "raw_jsonl": ""}
    except Exception as e:
        logger.error(f"Failed to extract facts natively: {e}")
        raise e
