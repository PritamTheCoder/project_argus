"""
Project Argus - Verifier Tests

Tests for the batched fact verification pipeline.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.agents.verifier import (
    verifier_node,
    _build_batch_prompt,
    _group_facts_into_batches,
    _merge_duplicate_facts,
    LLMBatchVerification,
    LLMVerificationResult,
)
from src.schema.state import AgentState


# ── Shared state builder ─────────────────────────────────────────────────────

def _make_state(**overrides):
    base = {
        "query": "test query",
        "plan": [],
        "scraped_data": [],
        "structured_evidence": [],
        "source_map": {},
        "critique": "",
        "report": "",
        "re_search_required": False,
        "iteration_count": 0,
        "verified_facts": [],
        "knowledge_gap_detected": False,
        "knowledge_gaps": [],
        "gap_queries": [],
        "active_node": "",
    }
    base.update(overrides)
    return base


# ── Empty Evidence ────────────────────────────────────────────────────────────

def test_verifier_no_evidence():
    result = verifier_node(_make_state())
    assert result["verified_facts"] == []
    assert result["active_node"] == "verifier"


# ── Batch Prompt Builder ─────────────────────────────────────────────────────

def test_build_batch_prompt():
    """Prompt should include all fact fields as JSON."""
    facts = [
        {"claim": "Fact A", "source_url": "http://a.com", "source_excerpt": "excerpt A"},
        {"claim": "Fact B", "source_url": "http://b.com", "source_excerpt": "excerpt B"},
    ]
    prompt = _build_batch_prompt(facts)
    assert "Fact A" in prompt
    assert "Fact B" in prompt
    assert "http://a.com" in prompt
    assert "excerpt A" in prompt


# ── Full-Batch Verification Path ─────────────────────────────────────────────

@patch("src.agents.verifier.get_llm_with_fallbacks")
@patch("src.utils.embeddings.get_embeddings")
@patch("src.graph.kg.kg_store")
def test_verifier_batch_path(mock_kg, mock_embeddings, mock_get_llm):
    """Two same-batch facts with grounded quotes from two sources are corroborated."""
    mock_get_llm.return_value.invoke.return_value = (
        LLMBatchVerification(results=[
            LLMVerificationResult(
                index=0, reasoning="source confirms",
                support_quote="reached 400 Wh/kg in 2024 lab trials",
                support_level="SUPPORTED", confidence=0.95,
            ),
            LLMVerificationResult(
                index=1, reasoning="source confirms",
                support_quote="reported 800 cycles at 80 percent capacity",
                support_level="SUPPORTED", confidence=0.80,
            ),
        ])
    )
    # Identical embeddings → the two claims cluster; distinct URLs → corroborated.
    mock_embeddings.return_value = [[0.1] * 384, [0.1] * 384]
    mock_kg.store_facts = MagicMock()

    state = _make_state(structured_evidence=[
        {"claim": "Reached 400 Wh/kg", "source_excerpt": "Solid-state batteries reached 400 Wh/kg in 2024 lab trials.", "source_url": "http://a.com", "source_id": "[1]"},
        {"claim": "800 cycles", "source_excerpt": "QuantumScape reported 800 cycles at 80 percent capacity.", "source_url": "http://b.com", "source_id": "[2]"},
    ])

    result = verifier_node(state)

    # Single LLM call for the single (combined) batch.
    assert mock_get_llm.return_value.invoke.call_count == 1
    assert len(result["verified_facts"]) == 2
    assert result["verified_facts"][0]["support_level"] == "SUPPORTED"
    # Corroborated by 2 sources → confidence not capped.
    assert result["verified_facts"][0]["confidence"] == 0.95
    assert result["verified_facts"][0]["corroboration_count"] == 2
    assert result["verified_facts"][0]["single_source_warning"] is False
    assert result["active_node"] == "verifier"


@patch("src.agents.verifier.get_llm_with_fallbacks")
@patch("src.utils.embeddings.get_embeddings")
@patch("src.graph.kg.kg_store")
def test_verifier_drops_ungrounded_supported_fact(mock_kg, mock_embeddings, mock_get_llm):
    """A 'SUPPORTED' verdict with a quote absent from the excerpt is dropped to NOT_SUPPORTED."""
    mock_get_llm.return_value.invoke.return_value = (
        LLMBatchVerification(results=[
            LLMVerificationResult(
                index=0, reasoning="claims support but quote fabricated",
                support_quote="achieved 900 Wh/kg world record breakthrough",
                support_level="SUPPORTED", confidence=0.95,
            ),
        ])
    )
    mock_embeddings.return_value = []  # nothing survives as supported → no embedding needed
    mock_kg.store_facts = MagicMock()

    state = _make_state(structured_evidence=[
        {"claim": "Reached 900 Wh/kg", "source_excerpt": "The cell reached 400 Wh/kg in lab testing.", "source_url": "http://a.com", "source_id": "[1]"},
    ])

    result = verifier_node(state)

    fact = result["verified_facts"][0]
    assert fact["support_level"] == "NOT_SUPPORTED"
    assert fact["grounding_failed"] is True
    # And it becomes a knowledge gap.
    assert "Reached 900 Wh/kg" in result["knowledge_gaps"]
    # Nothing ungrounded was stored in the KG.
    mock_kg.store_facts.assert_not_called()


# ── Cross-Provider Fallback Ladder (now lives in llm_factory) ────────────────

def test_get_llm_with_fallbacks_fails_over_to_next_provider():
    """A runtime error on the primary (e.g. 503) fails over to the next provider.

    Uses real RunnableLambdas so LangChain's ``.with_fallbacks()`` actually
    executes the failover rather than being stubbed away.
    """
    from src.utils import llm_factory
    from langchain_core.runnables import RunnableLambda

    success_result = LLMBatchVerification(results=[
        LLMVerificationResult(
            index=0, reasoning="ok",
            support_quote="reached 400 Wh/kg in lab testing",
            support_level="SUPPORTED", confidence=0.9,
        ),
    ])

    calls = {"primary": 0, "fallback": 0}

    def _boom(_):
        calls["primary"] += 1
        raise Exception("503 overloaded")

    def _ok(_):
        calls["fallback"] += 1
        return success_result

    primary = MagicMock()
    primary.with_structured_output.return_value = RunnableLambda(_boom)
    fallback = MagicMock()
    fallback.with_structured_output.return_value = RunnableLambda(_ok)

    # get_llm is called once per ladder rung (primary, then each fallback spec).
    with patch.object(llm_factory, "get_llm", side_effect=[primary, fallback]):
        chain = llm_factory.get_llm_with_fallbacks(
            "primary-model", "gemini",
            fallback_chain="llama-3.3-70b-versatile:groq",
            structured_schema=LLMBatchVerification,
        )
        out = chain.invoke("verify these")

    assert out == success_result
    assert calls["primary"] == 1
    assert calls["fallback"] == 1


def test_parse_fallback_chain_skips_malformed_entries():
    from src.utils.llm_factory import parse_fallback_chain
    specs = parse_fallback_chain("a:groq, , bad-entry, b:gemini")
    assert specs == [("a", "groq"), ("b", "gemini")]


# ── Batch Failure Isolation ──────────────────────────────────────────────────

@patch("src.agents.verifier.get_llm_with_fallbacks")
@patch("src.utils.embeddings.get_embeddings")
@patch("src.graph.kg.kg_store")
def test_verifier_failed_batch_marked_uncertain(mock_kg, mock_embeddings, mock_get_llm):
    """A batch whose entire fallback ladder fails is marked UNCERTAIN, not dropped."""
    invoke_mock = mock_get_llm.return_value.invoke
    invoke_mock.side_effect = Exception("everything is down")

    mock_embeddings.return_value = []
    mock_kg.store_facts = MagicMock()

    state = _make_state(structured_evidence=[
        {"claim": "Claim 1", "source_excerpt": "excerpt 1", "source_url": "http://a.com", "source_id": "[1]"},
    ])

    result = verifier_node(state)

    assert len(result["verified_facts"]) == 1
    assert result["verified_facts"][0]["support_level"] == "UNCERTAIN"
    assert result["verified_facts"][0]["confidence"] == 0.0


# ── Source-Grouped Batching Helper ───────────────────────────────────────────

def test_group_facts_keeps_sources_together_and_caps_size():
    facts = [{"source_id": "[1]"}] * 3 + [{"source_id": "[2]"}] * 3
    batches = _group_facts_into_batches(facts, batch_size=4)
    # src1 (3) fills a batch; adding src2 (3) would exceed 4, so it starts a new batch.
    assert len(batches) == 2
    assert all(len(b) <= 4 for b in batches)
    # No source is split across batches in this case.
    assert {f["source_id"] for f in batches[0]} == {"[1]"}
    assert {f["source_id"] for f in batches[1]} == {"[2]"}


def test_group_facts_splits_oversized_single_source():
    facts = [{"source_id": "[1]"}] * 25
    batches = _group_facts_into_batches(facts, batch_size=10)
    assert [len(b) for b in batches] == [10, 10, 5]


def test_group_facts_empty():
    assert _group_facts_into_batches([], batch_size=10) == []


# ── Skips Facts Missing Required Fields ──────────────────────────────────────

def test_verifier_skips_invalid_facts():
    """Facts missing claim or source_excerpt should be filtered out before the LLM call."""
    state = _make_state(structured_evidence=[
        {"claim": "", "source_excerpt": "something"},         # empty claim
        {"claim": "something", "source_excerpt": ""},         # empty excerpt
        {"source_excerpt": "missing claim field"},            # no claim key
    ])

    result = verifier_node(state)
    assert result["verified_facts"] == []
    assert result["active_node"] == "verifier"


# ── Near-duplicate merge preserves corroboration ─────────────────────────────

def test_merge_folds_duplicate_into_canonical_with_both_sources():
    """A claim restated by a second source is merged, not dropped, and both
    sources are recorded on the survivor."""
    facts = [
        {"claim": "SpaceX was valued at $350 billion in March 2026.",
         "source_url": "http://a.com", "source_id": "[1]", "credibility_score": 0.4},
        {"claim": "SpaceX was valued at $350 billion in March 2026",
         "source_url": "http://b.com", "source_id": "[2]", "credibility_score": 0.4},
    ]
    merged = _merge_duplicate_facts(facts)

    assert len(merged) == 1, "near-identical claims should collapse to one"
    assert sorted(merged[0]["merged_sources"]) == ["http://a.com", "http://b.com"]


def test_merge_keeps_highest_credibility_copy_as_canonical():
    facts = [
        {"claim": "Valuation reached $350 billion.", "source_url": "http://farm.com",
         "source_id": "[1]", "credibility_score": 0.4},
        {"claim": "Valuation reached $350 billion", "source_url": "http://reuters.com",
         "source_id": "[2]", "credibility_score": 0.7},
    ]
    merged = _merge_duplicate_facts(facts)

    assert len(merged) == 1
    assert merged[0]["source_url"] == "http://reuters.com"
    assert merged[0]["credibility_score"] == 0.7


def test_merge_leaves_distinct_claims_alone():
    facts = [
        {"claim": "Starlink reached 10 million customers.", "source_url": "http://a.com",
         "source_id": "[1]", "credibility_score": 0.4},
        {"claim": "Falcon 9 landed a booster in December 2015.", "source_url": "http://b.com",
         "source_id": "[2]", "credibility_score": 0.4},
    ]
    merged = _merge_duplicate_facts(facts)

    assert len(merged) == 2
    assert merged[0]["merged_sources"] != merged[1]["merged_sources"]


def test_merge_same_source_restating_itself_is_one_source():
    """Two pages of the same site are one source, not corroboration."""
    facts = [
        {"claim": "Valuation hit $350 billion.", "source_url": "http://a.com",
         "source_id": "[1]", "credibility_score": 0.4},
        {"claim": "Valuation hit $350 billion", "source_url": "http://a.com",
         "source_id": "[2]", "credibility_score": 0.4},
    ]
    merged = _merge_duplicate_facts(facts)

    assert len(merged) == 1
    assert merged[0]["merged_sources"] == ["http://a.com"]


@patch("src.agents.verifier.get_llm_with_fallbacks")
@patch("src.utils.embeddings.get_embeddings")
@patch("src.graph.kg.kg_store")
def test_verifier_corroborates_claim_restated_by_second_source(
    mock_kg, mock_embeddings, mock_get_llm
):
    """Regression: two sources restating one claim must yield corroboration_count 2.

    The merge step collapses them into a single verified fact (so only one claim
    is sent to the LLM), but the second source survives in `merged_sources` and
    must still count. Dropping the duplicate outright — the previous behaviour —
    pinned corroborated_fact_count at zero for every run.
    """
    mock_get_llm.return_value.invoke.return_value = (
        LLMBatchVerification(results=[
            LLMVerificationResult(
                index=0, reasoning="source confirms",
                support_quote="valued at $350 billion in March 2026",
                support_level="SUPPORTED", confidence=0.95,
            ),
        ])
    )
    mock_embeddings.return_value = [[0.1] * 384]
    mock_kg.store_facts = MagicMock()

    state = _make_state(structured_evidence=[
        {"claim": "SpaceX was valued at $350 billion in March 2026.",
         "source_excerpt": "SpaceX was valued at $350 billion in March 2026 per filings.",
         "source_url": "http://a.com", "source_id": "[1]"},
        {"claim": "SpaceX was valued at $350 billion in March 2026",
         "source_excerpt": "Reports say SpaceX was valued at $350 billion in March 2026.",
         "source_url": "http://b.com", "source_id": "[2]"},
    ])

    result = verifier_node(state)

    facts = result["verified_facts"]
    assert len(facts) == 1, "duplicates collapse to one verified claim"
    assert facts[0]["corroboration_count"] == 2
    assert facts[0]["single_source_warning"] is False
    # Corroborated → confidence must NOT be capped to the single-source ceiling.
    assert facts[0]["confidence"] == 0.95


# ── Interim quality_score (Phase 5A.4) ───────────────────────────────────────

@patch("src.agents.verifier.get_llm_with_fallbacks")
@patch("src.utils.embeddings.get_embeddings")
@patch("src.graph.kg.kg_store")
def test_verifier_returns_interim_quality_score(mock_kg, mock_embeddings, mock_get_llm):
    """The Verifier computes an interim quality_score so the router can act on
    source quality before the loop exits (contradiction_count is always 0 here —
    only Consensus, which hasn't run yet, can detect contradictions)."""
    mock_get_llm.return_value.invoke.return_value = LLMBatchVerification(results=[
        LLMVerificationResult(
            index=0, reasoning="ok", support_quote="reached 400 Wh/kg in 2024 lab trials",
            support_level="SUPPORTED", confidence=0.9,
        ),
    ])
    mock_embeddings.return_value = [[0.1] * 384]
    mock_kg.store_facts = MagicMock()

    state = _make_state(
        structured_evidence=[
            {"claim": "Reached 400 Wh/kg",
             "source_excerpt": "Solid-state batteries reached 400 Wh/kg in 2024 lab trials.",
             "source_url": "http://a.com", "source_id": "[1]"},
        ],
        source_map={"[1]": {"credibility_score": 0.4}},
    )

    result = verifier_node(state)

    qs = result["quality_score"]
    assert qs["verified_fact_count"] == 1
    assert qs["avg_source_credibility"] == 0.4
    assert qs["contradiction_count"] == 0


# ── Evidence graph: sources persisted alongside facts ────────────────────────

@patch("src.agents.verifier.get_llm_with_fallbacks")
@patch("src.utils.embeddings.get_embeddings")
@patch("src.graph.kg.kg_store")
def test_verifier_stores_sources_with_their_source_id(mock_kg, mock_embeddings, mock_get_llm):
    """source_map is keyed by source_id ("[n]") but its values don't carry that
    key inline — store_sources needs it attached for citation-linking."""
    mock_get_llm.return_value.invoke.return_value = LLMBatchVerification(results=[
        LLMVerificationResult(
            index=0, reasoning="ok", support_quote="reached 400 Wh/kg",
            support_level="SUPPORTED", confidence=0.9,
        ),
    ])
    mock_embeddings.return_value = [[0.1] * 384]
    mock_kg.store_facts = MagicMock()
    mock_kg.store_sources = MagicMock()

    state = _make_state(
        structured_evidence=[
            {"claim": "Reached 400 Wh/kg", "source_excerpt": "...reached 400 Wh/kg...",
             "source_url": "http://a.com", "source_id": "[1]"},
        ],
        source_map={"[1]": {"url": "http://a.com", "credibility_score": 0.9, "relevance_score": 0.7}},
    )

    verifier_node(state)

    mock_kg.store_sources.assert_called_once()
    stored = mock_kg.store_sources.call_args.args[0]
    assert stored == [{"url": "http://a.com", "credibility_score": 0.9,
                        "relevance_score": 0.7, "source_id": "[1]"}]


@patch("src.graph.kg.kg_store")
def test_verifier_stores_sources_even_with_no_extracted_facts(mock_kg):
    """A pass where the Refiner extracted nothing still hits the early-exit
    return — sources must be persisted before that return, not after, or a
    source Scout found on the run's last iteration is never recorded."""
    mock_kg.store_sources = MagicMock()
    state = _make_state(
        structured_evidence=[],
        source_map={"[1]": {"url": "http://a.com", "credibility_score": 0.9}},
    )

    verifier_node(state)

    mock_kg.store_sources.assert_called_once()


@patch("src.agents.verifier.get_llm_with_fallbacks")
@patch("src.utils.embeddings.get_embeddings")
@patch("src.graph.kg.kg_store")
def test_verifier_stores_sources_even_when_none_end_up_supported(mock_kg, mock_embeddings, mock_get_llm):
    """Facts were extracted and verified, but none were SUPPORTED — sources
    must still persist; only fact storage is conditional on supported_facts."""
    mock_get_llm.return_value.invoke.return_value = LLMBatchVerification(results=[
        LLMVerificationResult(
            index=0, reasoning="no support found", support_quote="",
            support_level="NOT_SUPPORTED", confidence=0.1,
        ),
    ])
    mock_kg.store_facts = MagicMock()
    mock_kg.store_sources = MagicMock()

    state = _make_state(
        structured_evidence=[
            {"claim": "unverifiable claim", "source_excerpt": "irrelevant text",
             "source_url": "http://a.com", "source_id": "[1]"},
        ],
        source_map={"[1]": {"url": "http://a.com", "credibility_score": 0.9}},
    )

    verifier_node(state)

    mock_kg.store_facts.assert_not_called()
    mock_kg.store_sources.assert_called_once()
