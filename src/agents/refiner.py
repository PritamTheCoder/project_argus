"""Refiner agent: extracts structured, source-linked facts from raw scraped text,
using an extraction schema generated dynamically from the research plan."""

import logging
from typing import Dict
from pydantic import BaseModel, Field
from src.schema.state import AgentState
from src.tools.refiner import extract_facts
from src.config import (
    LIBRARIAN_MODEL, LIBRARIAN_PROVIDER, LIBRARIAN_FALLBACK_CHAIN,
    REFINER_BATCH_MAX_CHARS, REFINER_MAX_DOC_CHARS,
)
from src.utils.llm_factory import get_llm_with_fallbacks

logger = logging.getLogger(__name__)

class ExtractionSchema(BaseModel):
    schema_dict: Dict[str, str] = Field(
        description="A dictionary where keys are entity types (e.g., 'dates', 'companies', 'metrics') and values are short descriptions of what to extract."
    )


def _generate_dynamic_schema(plan: list[str]) -> dict:
    """Use an LLM to generate an extraction schema based on the research plan."""
    try:
        structured_llm = get_llm_with_fallbacks(
            LIBRARIAN_MODEL, LIBRARIAN_PROVIDER,
            fallback_chain=LIBRARIAN_FALLBACK_CHAIN,
            temperature=0,
            structured_schema=ExtractionSchema,
        )

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


def _split_into_batches(
    docs: list[dict],
    max_chars: int = REFINER_BATCH_MAX_CHARS,
    max_doc_chars: int = REFINER_MAX_DOC_CHARS,
) -> list[str]:
    """
    Pack documents into source-tagged extraction payloads under ``max_chars``.

    Documents are kept whole (never split mid-page, so a fact never loses the
    context that supports it) and each payload stays inside the primary's
    per-minute token budget. A document larger than ``max_doc_chars`` is
    truncated and still gets its own batch rather than being dropped.

    Deterministic given input order. Returns [] when every document is empty.
    """
    batches: list[list[str]] = []
    current: list[str] = []
    current_len = 0

    for doc in docs:
        text = (doc.get("content") or "").strip()
        if not text:
            continue
        if len(text) > max_doc_chars:
            text = text[:max_doc_chars]
        source_id = doc.get("source_id", "[?]")
        block = f'\n<document source_id="{source_id}">\n{text}\n</document>\n'

        if current and current_len + len(block) > max_chars:
            batches.append(current)
            current = []
            current_len = 0

        current.append(block)
        current_len += len(block)

    if current:
        batches.append(current)

    return ["".join(blocks) for blocks in batches]


def refiner_node(state: AgentState) -> dict:
    """Extract facts from scraped data in source-tagged batches sized to the
    extraction model's token budget."""
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
        
    batches = _split_into_batches(new_docs)

    if not batches:
        logger.warning("Refiner: All scraped data was empty.")
        return {"structured_evidence": [], "active_node": "refiner"}

    source_map = state.get("source_map", {})

    logger.info(
        f"Refiner: Extracting from {len(new_docs)} document(s) across "
        f"{len(batches)} batch(es) (max {REFINER_BATCH_MAX_CHARS} chars/batch)..."
    )

    # Each batch is independent: one that fails across the whole fallback ladder
    # costs only its own documents, so a single 429 no longer discards the run's
    # entire evidence set.
    all_facts = []
    failed = 0
    for i, batch_text in enumerate(batches, start=1):
        try:
            extraction_result = extract_facts(batch_text, schema)
            batch_facts = extraction_result.get("facts", [])
            all_facts.extend(batch_facts)
            logger.info(f"Refiner: Batch {i}/{len(batches)} extracted {len(batch_facts)} fact(s).")
        except Exception as e:
            failed += 1
            logger.error(f"Refiner: Batch {i}/{len(batches)} failed: {e}")

    if failed:
        logger.warning(
            f"Refiner: {failed}/{len(batches)} batch(es) failed; "
            f"continuing with partial evidence ({len(all_facts)} fact(s))."
        )

    for fact in all_facts:
        s_id = fact.get("source_id")
        if s_id and s_id in source_map:
            fact["source_url"] = source_map[s_id].get("url", "")
        else:
            fact["source_url"] = ""

    logger.info(f"Refiner: Extracted {len(all_facts)} total facts across {len(batches)} batch(es).")

    return {
        "structured_evidence": all_facts,
        "active_node": "refiner"
    }
