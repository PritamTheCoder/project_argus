"""Extracts structured facts from concatenated raw Markdown via LLM structured output."""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, Field
from src.config import REFINER_MODEL, REFINER_PROVIDER, REFINER_FALLBACK_CHAIN
from src.utils.llm_factory import get_llm_with_fallbacks
from src.utils.dates import sanitize_as_of_date

logger = logging.getLogger(__name__)

class ExtractedFact(BaseModel):
    # Groq rejects a schema whose nested objects omit additionalProperties:false,
    # and a permissive schema also lets the model skip the wrapper and emit a bare
    # array, which fails the tool call. extra="forbid" emits it and prevents both.
    model_config = ConfigDict(extra="forbid")

    extraction_class: str = Field(default="", description="The category of the extracted fact (e.g., 'dates', 'metrics').")
    claim: str = Field(default="", description="A clear, standalone factual claim extracted from the document.")
    source_excerpt: str = Field(
        default="",
        description="The exact text quote from the document that explicitly supports the claim.",
    )
    source_id: str = Field(
        default="",
        description="The source_id of the document where this fact was found. MUST exactly match the source_id in the <document source_id=\"...\"> tag.",
    )
    as_of_date: str = Field(
        default="",
        description=(
            "The date this fact pertains to or was published, if explicitly stated in the "
            "source (ISO 'YYYY-MM-DD' or just a year like '2024'). Empty string if no date is stated. "
            "Do NOT guess or infer a date that is not present in the text."
        ),
    )
    attributes: Dict[str, Any] = Field(description="Additional attributes summarizing the fact based on the schema.", default_factory=dict)

class FactExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
                    4. Critically: You must provide the exact `source_id` of the document where you found the fact.
                    5. If the source explicitly states a date for this fact (publication date or the date it pertains to), put it in `as_of_date` (ISO format or year). Never invent a date.
                    6. Add meaningful `attributes` to provide context.

                    Documents to Analyze:
                    ====================
                    {batched_text}
                    ====================
            """

    def _shape_facts(result: FactExtractionResult) -> list[dict]:
        """Convert validated structured output into our internal fact dicts."""
        facts = []
        dropped = 0
        if result and result.facts:
            for fact in result.facts:
                # Drop records missing mandatory fields the LLM failed to populate.
                if not fact.claim or not fact.source_id:
                    dropped += 1
                    continue
                facts.append({
                    "class": fact.extraction_class,
                    "claim": fact.claim,
                    "source_excerpt": fact.source_excerpt,
                    "source_id": fact.source_id,
                    "as_of_date": sanitize_as_of_date(fact.as_of_date or ""),
                    "attributes": fact.attributes or {},
                    "source_span": {"start": None, "end": None}
                })
        if dropped:
            logger.warning(f"Refiner: dropped {dropped} fact(s) missing claim/source_id from LLM output.")
        return facts

    # Build the structured extractor once, armed with a cross-provider fallback
    # ladder. A runtime 503 on the primary (e.g. Gemini overloaded) now fails over
    # to an independent large-context provider instead of retrying the same one.
    structured_llm = get_llm_with_fallbacks(
        REFINER_MODEL,
        REFINER_PROVIDER,
        fallback_chain=REFINER_FALLBACK_CHAIN,
        temperature=0,
        structured_schema=FactExtractionResult,
    )

    try:
        logger.info(f"Refiner: Extracting with {REFINER_MODEL} via {REFINER_PROVIDER} (+ fallback ladder)...")
        result: FactExtractionResult = structured_llm.invoke(prompt)
        return {"facts": _shape_facts(result), "raw_jsonl": ""}
    except Exception as err:
        logger.error(f"Refiner: extraction failed across the entire fallback ladder: {err}")
        raise

