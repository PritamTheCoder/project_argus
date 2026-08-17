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
