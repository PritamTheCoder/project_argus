"""
Project Argus - Verifier Tests

Tests for the batched fact verification pipeline.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.agents.verifier import verifier_node, _build_batch_prompt, LLMBatchVerification, LLMVerificationResult
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

@patch("src.agents.verifier.get_llm")
@patch("src.utils.embeddings.get_embeddings")
@patch("src.graph.kg.kg_store")
def test_verifier_batch_path(mock_kg, mock_embeddings, mock_get_llm):
    """All facts should be verified in a single batched LLM call."""
    mock_get_llm.return_value.with_structured_output.return_value.invoke.return_value = (
        LLMBatchVerification(results=[
            LLMVerificationResult(index=0, reasoning="source confirms", support_level="SUPPORTED", confidence=0.95),
            LLMVerificationResult(index=1, reasoning="source contradicts", support_level="NOT_SUPPORTED", confidence=0.3),
        ])
    )
    mock_embeddings.return_value = [[0.1] * 384]
    mock_kg.store_facts = MagicMock()

    state = _make_state(structured_evidence=[
        {"claim": "Claim 1", "source_excerpt": "excerpt 1", "source_url": "http://a.com", "source_id": "[1]"},
        {"claim": "Claim 2", "source_excerpt": "excerpt 2", "source_url": "http://b.com", "source_id": "[2]"},
    ])

    result = verifier_node(state)

    # Single LLM call for the full batch
    assert mock_get_llm.return_value.with_structured_output.return_value.invoke.call_count == 1
    assert len(result["verified_facts"]) == 2
    assert result["verified_facts"][0]["support_level"] == "SUPPORTED"
    assert result["verified_facts"][0]["confidence"] == 0.95
    assert result["verified_facts"][1]["support_level"] == "NOT_SUPPORTED"
    assert result["active_node"] == "verifier"


# ── Mini-Batch Fallback Path ─────────────────────────────────────────────────

@patch("src.agents.verifier.get_llm")
@patch("src.utils.embeddings.get_embeddings")
@patch("src.graph.kg.kg_store")
def test_verifier_mini_batch_fallback(mock_kg, mock_embeddings, mock_get_llm):
    """If both primary and Gemini full-batch calls fail, verifier falls back to mini-batches."""
    success_result = LLMBatchVerification(results=[
        LLMVerificationResult(index=0, reasoning="ok", support_level="SUPPORTED", confidence=0.9),
    ])

    invoke_mock = mock_get_llm.return_value.with_structured_output.return_value.invoke
    invoke_mock.side_effect = [
        Exception("Context too large"),   # full-batch primary fails
        Exception("Gemini also fails"),   # full-batch Gemini fallback fails
        success_result,                   # mini-batch primary succeeds
    ]

    mock_embeddings.return_value = [[0.1] * 384]
    mock_kg.store_facts = MagicMock()

    state = _make_state(structured_evidence=[
        {"claim": "Claim 1", "source_excerpt": "excerpt 1", "source_url": "http://a.com", "source_id": "[1]"},
    ])

    result = verifier_node(state)

    # 2 failed full-batch calls + 1 successful mini-batch call
    assert invoke_mock.call_count == 3
    assert len(result["verified_facts"]) == 1
    assert result["verified_facts"][0]["support_level"] == "SUPPORTED"


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
