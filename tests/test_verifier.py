"""
Project Argus - Verifier Tests

Tests for the batched fact verification pipeline.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.agents.verifier import verifier_node, _build_batch_prompt, _verify_batch
from src.schema.state import AgentState, VerifiedFact, VerifiedFactBatch


# ── Empty Evidence (existing) ────────────────────────────────────────────────

def test_verifier_no_evidence():
    state = {
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
        "active_node": ""
    }
    
    result = verifier_node(state)
    assert result["verified_facts"] == []
    assert result["active_node"] == "verifier"


# ── Batch Prompt Builder ─────────────────────────────────────────────────────

def test_build_batch_prompt():
    """Prompt should include all facts as JSON."""
    facts = [
        {"claim": "Fact A", "source_url": "http://a.com", "source_excerpt": "excerpt A"},
        {"claim": "Fact B", "source_url": "http://b.com", "source_excerpt": "excerpt B"},
    ]
    prompt = _build_batch_prompt(facts)
    assert "Fact A" in prompt
    assert "Fact B" in prompt
    assert "http://a.com" in prompt
    assert "excerpt A" in prompt


# ── Batched Verification Path ────────────────────────────────────────────────

@patch("src.agents.verifier.ChatGoogleGenerativeAI")
@patch("src.utils.embeddings.get_embeddings")
@patch("src.graph.kg.kg_store")
def test_verifier_batch_path(mock_kg, mock_embeddings, mock_chat_cls):
    """All facts should be verified in a single batched call."""
    # Build mock LLM that returns VerifiedFactBatch
    mock_llm = MagicMock()
    mock_chat_cls.return_value = mock_llm
    
    mock_structured_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm
    
    # Simulate batch response with 2 verified facts
    mock_structured_llm.invoke.return_value = VerifiedFactBatch(results=[
        VerifiedFact(claim="Claim 1", source_url="http://a.com", source_excerpt="excerpt 1", support_level="SUPPORTED", confidence=0.95),
        VerifiedFact(claim="Claim 2", source_url="http://b.com", source_excerpt="excerpt 2", support_level="NOT_SUPPORTED", confidence=0.3),
    ])
    
    # Mock embeddings for KG storage
    mock_embeddings.return_value = [[0.1] * 384]
    mock_kg.store_facts = MagicMock()
    
    state = {
        "query": "test",
        "plan": [],
        "scraped_data": [],
        "structured_evidence": [
            {"claim": "Claim 1", "source_excerpt": "excerpt 1", "source_url": "http://a.com", "source_id": "[1]"},
            {"claim": "Claim 2", "source_excerpt": "excerpt 2", "source_url": "http://b.com", "source_id": "[2]"},
        ],
        "source_map": {},
        "critique": "",
        "report": "",
        "re_search_required": False,
        "iteration_count": 0,
        "verified_facts": [],
        "active_node": ""
    }
    
    result = verifier_node(state)
    
    # Should have called LLM only ONCE (batch)
    assert mock_structured_llm.invoke.call_count == 1
    
    assert len(result["verified_facts"]) == 2
    assert result["verified_facts"][0]["support_level"] == "SUPPORTED"
    assert result["verified_facts"][0]["confidence"] == 0.95
    assert result["verified_facts"][1]["support_level"] == "NOT_SUPPORTED"
    assert result["active_node"] == "verifier"


# ── Mini-Batch Fallback Path ─────────────────────────────────────────────────

@patch("src.agents.verifier.ChatGoogleGenerativeAI")
@patch("src.utils.embeddings.get_embeddings")
@patch("src.graph.kg.kg_store")
def test_verifier_mini_batch_fallback(mock_kg, mock_embeddings, mock_chat_cls):
    """If the full batch fails, the verifier should fall back to mini-batches."""
    mock_llm = MagicMock()
    mock_chat_cls.return_value = mock_llm
    
    mock_structured_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm
    
    # First call (full batch) fails, second call (mini-batch) succeeds
    mock_structured_llm.invoke.side_effect = [
        Exception("Context too large"),
        VerifiedFactBatch(results=[
            VerifiedFact(claim="Claim 1", source_url="http://a.com", source_excerpt="excerpt 1", support_level="SUPPORTED", confidence=0.9),
        ])
    ]
    
    mock_embeddings.return_value = [[0.1] * 384]
    mock_kg.store_facts = MagicMock()
    
    state = {
        "query": "test",
        "plan": [],
        "scraped_data": [],
        "structured_evidence": [
            {"claim": "Claim 1", "source_excerpt": "excerpt 1", "source_url": "http://a.com", "source_id": "[1]"},
        ],
        "source_map": {},
        "critique": "",
        "report": "",
        "re_search_required": False,
        "iteration_count": 0,
        "verified_facts": [],
        "active_node": ""
    }
    
    result = verifier_node(state)
    
    # Should have called LLM TWICE (1 failed full batch + 1 mini-batch)
    assert mock_structured_llm.invoke.call_count == 2
    assert len(result["verified_facts"]) == 1
    assert result["verified_facts"][0]["support_level"] == "SUPPORTED"


# ── Skips Facts Missing Required Fields ──────────────────────────────────────

def test_verifier_skips_invalid_facts():
    """Facts missing claim or source_excerpt should be filtered out."""
    state = {
        "query": "test",
        "plan": [],
        "scraped_data": [],
        "structured_evidence": [
            {"claim": "", "source_excerpt": "something"},  # empty claim
            {"claim": "something", "source_excerpt": ""},   # empty excerpt
            {"source_excerpt": "missing claim field"},       # no claim key
        ],
        "source_map": {},
        "critique": "",
        "report": "",
        "re_search_required": False,
        "iteration_count": 0,
        "verified_facts": [],
        "active_node": ""
    }
    
    result = verifier_node(state)
    assert result["verified_facts"] == []
    assert result["active_node"] == "verifier"
