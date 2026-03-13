import json
import logging
from src.schema.state import AgentState
from src.utils.llm_factory import get_llm
from pydantic import BaseModel, Field
from typing import Literal, List

class LLMVerificationResult(BaseModel):
    index: int = Field(description="The exact index matching the input fact.")
    reasoning: str = Field(description="Briefly explain why the excerpt does or does not support the claim.")
    support_level: Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED", "UNCERTAIN"] = Field(description="Level of support the excerpt provides for the claim.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")

class LLMBatchVerification(BaseModel):
    """Batch of verified facts with minimal output — returned by a single LLM call."""
    results: List[LLMVerificationResult] = Field(
        description="List of verification results IN THE EXACT SAME ORDER as the input facts."
    )


logger = logging.getLogger(__name__)

MINI_BATCH_SIZE = 10


def _build_batch_prompt(facts: list[dict]) -> str:
    """Build a single prompt containing ALL facts for batch verification."""
    facts_json = []
    for i, fact in enumerate(facts):
        facts_json.append({
            "index": i,
            "claim": fact.get("claim", ""),
            "source_url": fact.get("source_url", ""),
            "source_excerpt": fact.get("source_excerpt", ""),
        })

    return (
        "You are an expert fact verifier. You will receive a list of facts, each with an index, claim, source URL, and source excerpt.\n"
        "For EACH fact, you must:\n"
        "1. Write a `reasoning` string explaining how and why the excerpt proves or disproves the claim.\n"
        "2. Determine the `support_level`: SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED, or UNCERTAIN.\n"
        "3. Assign a `confidence` score (0.0 to 1.0) based on how well the excerpt supports the claim.\n"
        "4. Include the exact same `index` from the input to map the result back.\n\n"
        "Return ALL results.\n\n"
        f"Facts to verify:\n{json.dumps(facts_json, indent=2)}"
    )


def _verify_batch(structured_llm, facts: list[dict]) -> list[dict]:
    """Verify a batch of facts with a single LLM call. Returns list of verified fact dicts."""
    prompt = _build_batch_prompt(facts)
    result: LLMBatchVerification = structured_llm.invoke(prompt)

    verified = []
    
    # Create a mapping from index to verification result
    result_map = {res.index: res for res in result.results}
    
    for i, fact in enumerate(facts):
        v_dict = fact.copy()
        if i in result_map:
            v_dict["reasoning"] = result_map[i].reasoning
            v_dict["support_level"] = result_map[i].support_level
            v_dict["confidence"] = result_map[i].confidence
        else:
            # Fallback if LLM missed this index
            v_dict["reasoning"] = "LLM missed this index during batch processing."
            v_dict["support_level"] = "UNCERTAIN"
            v_dict["confidence"] = 0.0
        verified.append(v_dict)

    return verified


def verifier_node(state: AgentState) -> dict:
    """
    Verifies the extracted facts from the refiner node.
    Uses a single batched LLM call (with mini-batch fallback) to avoid 429 rate limits.
    Also stores verified and supported facts into the Knowledge Graph.
    """
    structured_evidence = state.get("structured_evidence", [])
    if not structured_evidence:
        return {"verified_facts": [], "active_node": "verifier"}

    # Filter out facts missing required fields
    valid_facts = [
        f for f in structured_evidence
        if f.get("claim") and f.get("source_excerpt")
    ]

    if not valid_facts:
        logger.warning("Verifier: All facts were missing claim or excerpt.")
        return {"verified_facts": [], "active_node": "verifier"}

    from src.config import VERIFIER_MODEL, VERIFIER_PROVIDER
    
    def _run_with_fallback(batch: list[dict], model: str, provider: str) -> list[dict]:
        """Attempt to verify the batch using the primary model. If it fails, fallback to Gemini."""
        try:
            llm = get_llm(model, provider, temperature=0)
            structured_llm = llm.with_structured_output(LLMBatchVerification)
            return _verify_batch(structured_llm, batch)
        except Exception as primary_err:
            logger.warning(f"Verifier: Primary model failed ({primary_err}). Falling back to Gemini...")
            try:
                fallback_llm = get_llm("gemini-2.5-flash", "gemini", temperature=0)
                fallback_structured = fallback_llm.with_structured_output(LLMBatchVerification)
                return _verify_batch(fallback_structured, batch)
            except Exception as fallback_err:
                logger.error(f"Verifier: Gemini fallback failed: {fallback_err}")
                raise fallback_err

    verified_facts = []

    # Try full-batch first
    try:
        logger.info(f"Verifier: Batch-verifying {len(valid_facts)} facts...")
        verified_facts = _run_with_fallback(valid_facts, VERIFIER_MODEL, VERIFIER_PROVIDER)
        logger.info(f"Verifier: Successfully verified {len(verified_facts)} facts in full batch.")
    except Exception as e:
        logger.warning(f"Verifier: Full-batch failed ({e}). Falling back to mini-batches of {MINI_BATCH_SIZE}...")

        # Mini-batch fallback: split into chunks of MINI_BATCH_SIZE
        verified_facts = []
        for start in range(0, len(valid_facts), MINI_BATCH_SIZE):
            batch = valid_facts[start:start + MINI_BATCH_SIZE]
            try:
                batch_results = _run_with_fallback(batch, VERIFIER_MODEL, VERIFIER_PROVIDER)
                verified_facts.extend(batch_results)
                logger.info(f"Verifier: Mini-batch {start // MINI_BATCH_SIZE + 1} verified {len(batch_results)} facts.")
            except Exception as batch_err:
                logger.error(f"Verifier: Mini-batch failed completely: {batch_err}")
                # Mark remaining facts as UNCERTAIN
                for fact in batch:
                    fact["support_level"] = "UNCERTAIN"
                    fact["confidence"] = 0.0
                    verified_facts.append(fact)

    logger.info(f"Verified {len(verified_facts)} facts total.")

    # Store SUPPORTED/PARTIALLY_SUPPORTED facts into the Knowledge Graph
    try:
        supported_facts = [
            f for f in verified_facts
            if f.get("support_level") in ("SUPPORTED", "PARTIALLY_SUPPORTED")
        ]
        if supported_facts:
            from src.utils.embeddings import get_embeddings
            from src.graph.kg import kg_store
            texts_to_embed = [f["claim"] for f in supported_facts]
            embeddings = get_embeddings(texts_to_embed)

            for i, fp in enumerate(supported_facts):
                fp["embedding"] = embeddings[i]

            kg_store.store_facts(supported_facts)
    except Exception as e:
        logger.error(f"Error embedding/storing facts into KG: {e}")

    return {"verified_facts": verified_facts, "active_node": "verifier"}
