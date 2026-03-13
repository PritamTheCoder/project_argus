"""
Project Argus - Batch Refiner Tool

Extracts structured facts from concatenated raw Markdown using Gemini Guided JSON Output.
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.config import REFINER_MODEL, REFINER_PROVIDER
from src.utils.llm_factory import get_llm

logger = logging.getLogger(__name__)

class ExtractedFact(BaseModel):
    extraction_class: str = Field(description="The category of the extracted fact (e.g., 'dates', 'metrics').")
    claim: str = Field(description="A clear, standalone factual claim extracted from the document.")
    source_excerpt: str = Field(description="The exact text quote from the document that explicitly supports the claim.")
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
                    1. Provide a clear, standalone factual `claim` based on the document.
                    2. Provide the `source_excerpt` containing the exact text quote that supports the claim.
                    3. Provide the `extraction_class` based on the Target Schema.
                    3. Critically: You must provide the exact `source_id` of the document where you found the fact.
                    4. Add meaningful `attributes` to provide context.

                    Documents to Analyze:
                    ====================
                    {batched_text}
                    ====================
            """

    def _invoke_extraction(model: str, provider: str) -> list[dict]:
        """Attempt fact extraction with the given model/provider."""
        llm = get_llm(model, provider, temperature=0)
        structured_llm = llm.with_structured_output(FactExtractionResult)
        result: FactExtractionResult = structured_llm.invoke(prompt)

        facts = []
        if result and result.facts:
            for fact in result.facts:
                facts.append({
                    "class": fact.extraction_class,
                    "claim": fact.claim,
                    "source_excerpt": fact.source_excerpt,
                    "source_id": fact.source_id,
                    "attributes": fact.attributes or {},
                    "source_span": {"start": None, "end": None}
                })
        return facts

    # --- Attempt primary model ---
    try:
        logger.info(f"Refiner: Attempting extraction with {REFINER_MODEL} via {REFINER_PROVIDER}...")
        facts = _invoke_extraction(REFINER_MODEL, REFINER_PROVIDER)
        return {"facts": facts, "raw_jsonl": ""}

    except Exception as primary_err:
        logger.warning(f"Refiner: Primary model failed ({primary_err}). Trying Gemini fallback...")

        # --- Fallback to Gemini (1M context, handles large payloads) ---
        try:
            facts = _invoke_extraction("gemini-2.5-flash", "gemini")
            logger.info(f"Refiner: Gemini fallback succeeded — extracted {len(facts)} facts.")
            return {"facts": facts, "raw_jsonl": ""}
        except Exception as fallback_err:
            logger.error(f"Refiner: Gemini fallback also failed: {fallback_err}")
            raise fallback_err

