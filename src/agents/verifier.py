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

# Facts are verified in source-grouped batches of this size. Verifying every
# fact in one giant call invites "lost in the middle" degradation and uniformly
# inflated confidence, so we batch by default rather than only on failure.
VERIFY_BATCH_SIZE = 15


def _group_facts_into_batches(facts: list[dict], batch_size: int = VERIFY_BATCH_SIZE) -> list[list[dict]]:
    """
    Partition facts into batches of at most ``batch_size``, keeping facts from
    the same source adjacent so the model has related context together.

    A single source with more facts than ``batch_size`` is split across batches.
    Order of facts within a source is preserved.
    """
    if batch_size < 1:
        batch_size = 1

    # Preserve first-seen source order for deterministic, readable batches.
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for fact in facts:
        sid = fact.get("source_id", "") or ""
        if sid not in groups:
            groups[sid] = []
            order.append(sid)
        groups[sid].append(fact)

    batches: list[list[dict]] = []
    current: list[dict] = []
    for sid in order:
        group = groups[sid]
        # Oversized single source: flush current, then chunk the group.
        if len(group) > batch_size:
            if current:
                batches.append(current)
                current = []
            for i in range(0, len(group), batch_size):
                batches.append(group[i:i + batch_size])
            continue
        if len(current) + len(group) > batch_size:
            batches.append(current)
            current = []
        current.extend(group)

    if current:
        batches.append(current)
    return batches


def _build_batch_prompt(facts: list[dict]) -> str:
    """Build a single prompt containing ALL facts for batch verification."""
    facts_json = []
    for i, fact in enumerate(facts):
        facts_json.append({
            "index": i,
            "claim": fact.get("claim", ""),
            "source_url": fact.get("source_url", ""),
            "source_excerpt": fact.get("source_excerpt", ""),
            "source_type": fact.get("source_type", "Unknown"),
            "credibility_score": fact.get("credibility_score", 0.5)
        })

    return (
        "You are an expert fact verifier. You will receive a list of facts, each with an index, claim, source URL, source excerpt, and explicitly calculated credibility scores.\n"
        "For EACH fact, you must:\n"
        "1. Write a `reasoning` string explaining how and why the excerpt proves or disproves the claim. You MUST mention the source credibility in your reasoning.\n"
        "2. Determine the `support_level`: SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED, or UNCERTAIN.\n"
        "3. Assign a `confidence` score (0.0 to 1.0) based on how well the excerpt supports the claim AND the `credibility_score` of the source.\n"
        "   - CRITICAL: If a claim comes from an 'Unverified/Web' source (credibility < 0.5), you MUST assign a severely lowered confidence, even if the text matches perfectly.\n"
        "   - If a claim comes from 'Academic/Scientific' or 'Government' sources (credibility > 0.8), assign a higher baseline confidence.\n"
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
    source_map = state.get("source_map", {})
    
    if not structured_evidence:
        return {"verified_facts": [], "knowledge_gap_detected": False, "knowledge_gaps": [], "active_node": "verifier"}
        
    # Inject credibility scores from the source_map into the facts before verification
    for fact in structured_evidence:
        source_id = fact.get("source_id", "")
        if source_id in source_map:
            s_map = source_map[source_id]
            fact["credibility_score"] = s_map.get("credibility_score", 0.4)
            fact["source_type"] = s_map.get("source_type", "Unverified/Web")
        else:
            fact["credibility_score"] = 0.4
            fact["source_type"] = "Unverified/Web"

    # Filter out facts missing required fields
    valid_facts = [
        f for f in structured_evidence
        if f.get("claim") and f.get("source_excerpt")
    ]

    if not valid_facts:
        logger.warning("Verifier: All facts were missing claim or excerpt.")
        return {"verified_facts": [], "knowledge_gap_detected": True, "knowledge_gaps": [], "active_node": "verifier"}

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

    # Verify in source-grouped batches. Each batch independently falls back to
    # Gemini on model failure, and only that batch is marked UNCERTAIN if it
    # fails outright — a single bad batch never discards the whole run.
    batches = _group_facts_into_batches(valid_facts, VERIFY_BATCH_SIZE)
    logger.info(
        f"Verifier: Verifying {len(valid_facts)} facts across {len(batches)} "
        f"source-grouped batch(es) (max {VERIFY_BATCH_SIZE}/batch)..."
    )

    verified_facts = []
    for i, batch in enumerate(batches, start=1):
        try:
            batch_results = _run_with_fallback(batch, VERIFIER_MODEL, VERIFIER_PROVIDER)
            verified_facts.extend(batch_results)
            logger.info(f"Verifier: Batch {i}/{len(batches)} verified {len(batch_results)} facts.")
        except Exception as batch_err:
            logger.error(f"Verifier: Batch {i}/{len(batches)} failed completely: {batch_err}")
            for fact in batch:
                fact["support_level"] = "UNCERTAIN"
                fact["confidence"] = 0.0
                verified_facts.append(fact)

    logger.info(f"Verified {len(verified_facts)} facts total.")

    # Store SUPPORTED/PARTIALLY_SUPPORTED facts into the Knowledge Graph,
    # scoped to this run so downstream gap analysis stays uncontaminated.
    session_id = state.get("session_id", "")
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

            kg_store.store_facts(supported_facts, session_id=session_id)
    except Exception as e:
        logger.error(f"Error embedding/storing facts into KG: {e}")

    # Extract unsupported/uncertain claims as explicit knowledge gaps
    knowledge_gaps = [
        f.get("claim", "")
        for f in verified_facts
        if f.get("support_level") in ("NOT_SUPPORTED", "UNCERTAIN") and f.get("claim")
    ]

    knowledge_gap_detected = len(knowledge_gaps) > 0

    if knowledge_gap_detected:
        logger.info(f"Verifier: {len(knowledge_gaps)} knowledge gap(s) detected.")

    return {
        "verified_facts": verified_facts,
        "knowledge_gap_detected": knowledge_gap_detected,
        "knowledge_gaps": knowledge_gaps,
        "active_node": "verifier"
    }
