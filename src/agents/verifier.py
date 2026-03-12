import json
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from src.schema.state import AgentState, VerifiedFact, VerifiedFactBatch

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
        "You are an expert fact verifier. You will receive a list of facts, each with a claim, source URL, and source excerpt.\n"
        "For EACH fact, you must:\n"
        "1. Copy the claim, source_url, and source_excerpt exactly as provided.\n"
        "2. Determine the support_level: SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED, or UNCERTAIN.\n"
        "3. Assign a confidence score (0.0 to 1.0) based on how well the excerpt supports the claim.\n\n"
        "Return ALL results in the same order as the input.\n\n"
        f"Facts to verify:\n{json.dumps(facts_json, indent=2)}"
    )


def _verify_batch(structured_llm, facts: list[dict]) -> list[dict]:
    """Verify a batch of facts with a single LLM call. Returns list of verified fact dicts."""
    prompt = _build_batch_prompt(facts)
    result: VerifiedFactBatch = structured_llm.invoke(prompt)

    verified = []
    for i, vf in enumerate(result.results):
        v_dict = vf.model_dump()
        # Carry over source_id from the original fact
        if i < len(facts):
            v_dict["source_id"] = facts[i].get("source_id", "?")
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

    from src.config import CRITIC_MODEL
    llm = ChatGoogleGenerativeAI(model=CRITIC_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(VerifiedFactBatch)

    verified_facts = []

    # Try full-batch first (1 API call for all facts)
    try:
        logger.info(f"Verifier: Batch-verifying {len(valid_facts)} facts in a single LLM call...")
        verified_facts = _verify_batch(structured_llm, valid_facts)
        logger.info(f"Verifier: Successfully verified {len(verified_facts)} facts in 1 call.")
    except Exception as e:
        logger.warning(f"Verifier: Full-batch call failed ({e}). Falling back to mini-batches of {MINI_BATCH_SIZE}...")

        # Mini-batch fallback: split into chunks of MINI_BATCH_SIZE
        verified_facts = []
        for start in range(0, len(valid_facts), MINI_BATCH_SIZE):
            batch = valid_facts[start:start + MINI_BATCH_SIZE]
            try:
                batch_results = _verify_batch(structured_llm, batch)
                verified_facts.extend(batch_results)
                logger.info(f"Verifier: Mini-batch {start // MINI_BATCH_SIZE + 1} verified {len(batch_results)} facts.")
            except Exception as batch_err:
                logger.error(f"Verifier: Mini-batch failed: {batch_err}")
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
