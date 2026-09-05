import json
import logging
import unicodedata
import re
from difflib import SequenceMatcher
from src.schema.state import AgentState
from src.utils.llm_factory import get_llm_with_fallbacks
from src.utils.grounding import _fact_source_key
from src.config import VERIFY_BATCH_SIZE, MAX_FACTS_TO_VERIFY
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, List

class LLMVerificationResult(BaseModel):
    # Groq rejects a schema whose nested objects omit additionalProperties:false,
    # and a permissive schema also lets the model skip the wrapper and emit a bare
    # array, which fails the tool call. extra="forbid" emits it and prevents both.
    model_config = ConfigDict(extra="forbid")

    index: int = Field(description="The exact index matching the input fact.")
    reasoning: str = Field(description="Briefly explain why the excerpt does or does not support the claim.")
    support_quote: str = Field(
        default="",
        description=(
            "The EXACT verbatim sentence/phrase copied character-for-character from the "
            "source excerpt that directly supports the claim. Copy it literally — do not "
            "paraphrase, summarize, or fix typos. If no span of the excerpt supports the "
            "claim, leave this empty and set support_level to NOT_SUPPORTED."
        ),
    )
    support_level: Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED", "UNCERTAIN"] = Field(description="Level of support the excerpt provides for the claim.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")

class LLMBatchVerification(BaseModel):
    """Batch of verified facts with minimal output — returned by a single LLM call."""
    model_config = ConfigDict(extra="forbid")

    results: List[LLMVerificationResult] = Field(
        description="List of verification results IN THE EXACT SAME ORDER as the input facts."
    )


logger = logging.getLogger(__name__)


def _store_sources(source_map: dict, session_id: str) -> None:
    """Persist source_map into the KG's evidence graph. Called on every
    Verifier pass, including the early-exit path, so a pass with no extracted
    facts still doesn't lose whatever sources Scout has already registered."""
    try:
        from src.graph.kg import kg_store
        sources = [{**v, "source_id": sid} for sid, v in (source_map or {}).items()]
        kg_store.store_sources(sources, session_id=session_id)
    except Exception as e:
        logger.error(f"Error storing sources into KG: {e}")


def _normalize_claim(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace for similarity comparison."""
    text = unicodedata.normalize("NFKD", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _merge_duplicate_facts(facts: list[dict], sim_threshold: float = 0.88) -> list[dict]:
    """
    Collapse near-duplicate claims into one canonical fact, recording every
    distinct source that stated it in ``merged_sources``.

    Two claims are duplicates when their normalised text similarity exceeds
    ``sim_threshold``. The highest-credibility copy is kept as canonical.

    A duplicate is MERGED, not dropped: when several sources restate the same
    measurement, that restatement *is* the cross-source corroboration signal
    ``annotate_corroboration`` looks for downstream. Merging keeps the token
    saving — one claim still goes to the verifier — without destroying it.

    O(n²) but fast enough for the typical 200-500 fact range.
    """
    # Sort so higher-credibility facts are seen first and kept as the canonical copy.
    sorted_facts = sorted(facts, key=lambda f: -f.get("credibility_score", 0.4))
    seen_norms: list[str] = []
    unique: list[dict] = []
    for fact in sorted_facts:
        norm = _normalize_claim(fact.get("claim", ""))
        if not norm:
            continue
        match_idx = None
        for i, seen in enumerate(seen_norms):
            if SequenceMatcher(None, norm, seen).ratio() >= sim_threshold:
                match_idx = i
                break

        if match_idx is None:
            seen_norms.append(norm)
            key = _fact_source_key(fact)
            fact["merged_sources"] = [key] if key else []
            unique.append(fact)
        else:
            canonical = unique[match_idx]
            key = _fact_source_key(fact)
            if key and key not in canonical["merged_sources"]:
                canonical["merged_sources"].append(key)
    return unique


def _cap_facts(facts: list[dict], max_facts: int) -> list[dict]:
    """
    Enforce a hard ceiling on the number of facts entering the verifier.

    Sorts by credibility descending so that if we must drop, we drop the
    lowest-quality sources first.
    """
    if len(facts) <= max_facts:
        return facts
    sorted_facts = sorted(facts, key=lambda f: -f.get("credibility_score", 0.4))
    return sorted_facts[:max_facts]


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
        "2. Extract a `support_quote`: the EXACT verbatim span copied character-for-character from the `source_excerpt` that supports the claim. Do NOT paraphrase. If the excerpt contains no span that directly supports the claim, leave `support_quote` empty.\n"
        "3. Determine the `support_level`: SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED, or UNCERTAIN. If you could not find a verbatim `support_quote`, you MUST NOT mark it SUPPORTED.\n"
        "4. Assign a `confidence` score (0.0 to 1.0) based on how well the excerpt supports the claim AND the `credibility_score` of the source.\n"
        "   - CRITICAL: If a claim comes from an 'Unverified/Web' source (credibility < 0.5), you MUST assign a severely lowered confidence, even if the text matches perfectly.\n"
        "   - If a claim comes from 'Academic/Scientific' or 'Government' sources (credibility > 0.8), assign a higher baseline confidence.\n"
        "5. Include the exact same `index` from the input to map the result back.\n\n"
        "Return ALL results.\n\n"
        f"Facts to verify:\n{json.dumps(facts_json, indent=2)}"
    )


def _verify_batch(structured_llm, facts: list[dict]) -> list[dict]:
    """Verify a batch of facts with a single LLM call. Returns list of verified fact dicts."""
    prompt = _build_batch_prompt(facts)
    result: LLMBatchVerification = structured_llm.invoke(prompt)

    verified = []
    result_map = {res.index: res for res in result.results}
    
    for i, fact in enumerate(facts):
        v_dict = fact.copy()
        if i in result_map:
            v_dict["reasoning"] = result_map[i].reasoning
            v_dict["support_quote"] = result_map[i].support_quote
            v_dict["support_level"] = result_map[i].support_level
            v_dict["confidence"] = result_map[i].confidence
        else:
            v_dict["reasoning"] = "LLM missed this index during batch processing."
            v_dict["support_quote"] = ""
            v_dict["support_level"] = "UNCERTAIN"
            v_dict["confidence"] = 0.0
        verified.append(v_dict)

    return verified


def verifier_node(state: AgentState) -> dict:
    """
    Verify the extracted facts from the refiner node using a single batched LLM
    call (with mini-batch fallback to stay under rate limits), then store
    verified/supported facts into the Knowledge Graph.
    """
    structured_evidence = state.get("structured_evidence", [])
    source_map = state.get("source_map", {})

    if not structured_evidence:
        _store_sources(source_map, state.get("session_id", ""))
        return {"verified_facts": [], "knowledge_gap_detected": False, "knowledge_gaps": [], "active_node": "verifier"}

    for fact in structured_evidence:
        source_id = fact.get("source_id", "")
        if source_id in source_map:
            s_map = source_map[source_id]
            fact["credibility_score"] = s_map.get("credibility_score", 0.4)
            fact["source_type"] = s_map.get("source_type", "Unverified/Web")
        else:
            fact["credibility_score"] = 0.4
            fact["source_type"] = "Unverified/Web"

    valid_facts = [
        f for f in structured_evidence
        if f.get("claim") and f.get("source_excerpt")
    ]

    if not valid_facts:
        logger.warning("Verifier: All facts were missing claim or excerpt.")
        return {"verified_facts": [], "knowledge_gap_detected": True, "knowledge_gaps": [], "active_node": "verifier"}

    # Merge near-identical claims before batching. A single scrape pass often
    # yields the same measurement stated across several pages; verifying each
    # copy burns RPM without adding information — but the copies themselves are
    # the corroboration evidence, so they are folded into the survivor's
    # `merged_sources` rather than discarded.
    before_dedup = len(valid_facts)
    valid_facts = _merge_duplicate_facts(valid_facts)
    dedup_dropped = before_dedup - len(valid_facts)
    if dedup_dropped:
        logger.info(f"Verifier: merged {dedup_dropped} near-duplicate fact(s) into canonical claims ({before_dedup} → {len(valid_facts)}).")

    # Hard cap: if still above limit, keep the highest-credibility facts.
    valid_facts = _cap_facts(valid_facts, MAX_FACTS_TO_VERIFY)
    if len(valid_facts) < before_dedup - dedup_dropped:
        logger.warning(
            f"Verifier: capped at {MAX_FACTS_TO_VERIFY} facts "
            f"(set MAX_FACTS_TO_VERIFY env var to raise the limit)."
        )

    from src.config import VERIFIER_MODEL, VERIFIER_PROVIDER, VERIFIER_FALLBACK_CHAIN

    # Build the structured verifier once, armed with a cross-provider fallback
    # ladder. A runtime 503 ("model overloaded") on the primary now fails over to
    # an independent provider instead of retrying the same saturated one.
    structured_llm = get_llm_with_fallbacks(
        VERIFIER_MODEL,
        VERIFIER_PROVIDER,
        fallback_chain=VERIFIER_FALLBACK_CHAIN,
        temperature=0,
        structured_schema=LLMBatchVerification,
    )

    def _run_with_fallback(batch: list[dict]) -> list[dict]:
        """Verify a batch; the LLM already carries its own cross-provider ladder."""
        return _verify_batch(structured_llm, batch)

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
            batch_results = _run_with_fallback(batch)
            verified_facts.extend(batch_results)
            logger.info(f"Verifier: Batch {i}/{len(batches)} verified {len(batch_results)} facts.")
        except Exception as batch_err:
            logger.error(f"Verifier: Batch {i}/{len(batches)} failed completely: {batch_err}")
            for fact in batch:
                fact["support_level"] = "UNCERTAIN"
                fact["confidence"] = 0.0
                verified_facts.append(fact)

    logger.info(f"Verified {len(verified_facts)} facts total.")

    # Chain-of-Verification grounding: drop any "supported" fact whose verbatim
    # quote is not actually present in the source excerpt.
    from src.utils.grounding import apply_quote_grounding, annotate_corroboration
    apply_quote_grounding(verified_facts)
    dropped = sum(1 for f in verified_facts if f.get("grounding_failed"))
    if dropped:
        logger.info(f"Verifier: Dropped {dropped} fact(s) lacking a grounded verbatim quote.")

    # Store SUPPORTED/PARTIALLY_SUPPORTED facts into the Knowledge Graph,
    # scoped to this run so downstream gap analysis stays uncontaminated.
    session_id = state.get("session_id", "")
    supported_facts = [
        f for f in verified_facts
        if f.get("support_level") in ("SUPPORTED", "PARTIALLY_SUPPORTED")
    ]
    try:
        from src.graph.kg import kg_store

        if supported_facts:
            from src.utils.embeddings import get_embeddings
            texts_to_embed = [f["claim"] for f in supported_facts]
            embeddings = get_embeddings(texts_to_embed)

            for i, fp in enumerate(supported_facts):
                fp["embedding"] = embeddings[i]

            # Cross-source corroboration: flag single-source claims and cap their
            # confidence so the Writer hedges them rather than asserting them.
            annotate_corroboration(supported_facts)
            singles = sum(1 for f in supported_facts if f.get("single_source_warning"))
            logger.info(
                f"Verifier: {len(supported_facts) - singles}/{len(supported_facts)} supported "
                f"claims corroborated by >=2 sources ({singles} single-source)."
            )

            kg_store.store_facts(supported_facts, session_id=session_id)
    except Exception as e:
        logger.error(f"Error embedding/corroborating/storing facts into KG: {e}")

    # Independent of whether this pass had any supported facts — source_map
    # accumulates across the whole run and store_sources is idempotent, so
    # this just picks up whatever Scout has registered so far.
    _store_sources(state.get("source_map", {}), session_id)

    knowledge_gaps = [
        f.get("claim", "")
        for f in verified_facts
        if f.get("support_level") in ("NOT_SUPPORTED", "UNCERTAIN") and f.get("claim")
    ]

    knowledge_gap_detected = len(knowledge_gaps) > 0

    if knowledge_gap_detected:
        logger.info(f"Verifier: {len(knowledge_gaps)} knowledge gap(s) detected.")

    # Interim quality score so the router can act on source quality mid-loop
    # (see route_after_critic). contradiction_count is unknown until Consensus
    # runs at the end of the loop, so it's 0 here — Consensus overwrites this
    # with the final version.
    from src.utils.quality import compute_quality_score
    quality_score = compute_quality_score(verified_facts, source_map, contradiction_count=0)
    logger.info(f"Verifier: interim quality score: {quality_score}")

    return {
        "verified_facts": verified_facts,
        "knowledge_gap_detected": knowledge_gap_detected,
        "knowledge_gaps": knowledge_gaps,
        "quality_score": quality_score,
        "active_node": "verifier"
    }
