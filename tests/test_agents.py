"""
Project Argus - Agent Logic Tests

Unit tests for the Phase 2 agent nodes.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.schema.state import AgentState, ResearchPlan, FactCheckResult
from src.agents.librarian import librarian_node
from src.agents.scout import scout_node
from src.agents.refiner import refiner_node
from src.agents.critic import critic_node
from src.agents.writer import writer_node

# ── Librarian Tests ──────────────────────────────────────────────────────────

@patch("src.agents.librarian.ChatGoogleGenerativeAI")
@patch("src.agents.librarian.ChatPromptTemplate")
def test_librarian_generates_plan(mock_prompt_cls, mock_chat_cls):
    """Librarian should return a list of queries and update active_node."""
    # Build the chain mock manually
    mock_prompt = MagicMock()
    mock_prompt_cls.from_messages.return_value = mock_prompt
    
    mock_chain = MagicMock()
    # prompt | structured_llm returns the chain or structured_llm depending on how it's mocked
    # Since structured_llm is also a mock, prompt | structured_llm -> MagicMock
    mock_prompt.__or__.return_value = mock_chain
    
    expected_plan = ResearchPlan(search_queries=["query 1", "query 2"])
    mock_chain.invoke.return_value = expected_plan
    
    state: AgentState = {"query": "test topic", "plan": [], "scraped_data": [], "structured_evidence": [], "source_map": {}, "critique": "", "report": "", "re_search_required": False, "active_node": ""}
    
    result = librarian_node(state)
    
    assert "plan" in result
    assert result["plan"] == ["query 1", "query 2"]
    assert result["active_node"] == "librarian"

# ── Scout Tests ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("src.agents.scout.run_scout")
async def test_scout_execution(mock_run_scout):
    """Scout should iterate plan, call tool, and build source_map."""
    # Mock scout tool output
    mock_run_scout.side_effect = [
        [{"url": "http://a.com", "content": "content A", "success": True}], # for query 1
        [{"url": "http://b.com", "content": "content B", "success": True}]  # for query 2
    ]
    
    state: AgentState = {
        "query": "test",
        "plan": ["q1", "q2"],
        "source_map": {}
    }
    
    result = await scout_node(state)
    
    assert "scraped_data" in result
    assert len(result["scraped_data"]) == 2
    assert result["active_node"] == "scout"
    
    # Check source map construction
    source_map = result["source_map"]
    assert "[1]" in source_map
    assert source_map["[1]"]["url"] == "http://a.com"
    assert "[2]" in source_map
    assert source_map["[2]"]["url"] == "http://b.com"

# ── Refiner Tests ────────────────────────────────────────────────────────────

@patch("src.agents.refiner.ChatGoogleGenerativeAI")
@patch("src.agents.refiner.extract_facts")
def test_refiner_extraction(mock_extract, mock_chat_cls):
    """Refiner should generate schema and extract facts per doc."""
    # Mock Schema Gen LLM
    mock_llm = MagicMock()
    mock_chat_cls.return_value = mock_llm
    mock_llm.invoke.return_value.content = '{"dates": "dates"}'
    
    # Mock Extraction Tool
    mock_extract.return_value = {
        "facts": [{"class": "dates", "text": "2026", "attributes": {}}]
    }
    
    state: AgentState = {
        "scraped_data": [{"content": "text", "source_id": "[1]"}],
        "plan": ["q1"]
    }
    
    result = refiner_node(state)
    
    assert "structured_evidence" in result
    evidence = result["structured_evidence"]
    assert len(evidence) == 1
    assert evidence[0]["text"] == "2026"
    assert evidence[0]["source_id"] == "[1]"
    assert result["active_node"] == "refiner"

# ── Critic Tests ─────────────────────────────────────────────────────────────

@patch("src.agents.critic.ChatGoogleGenerativeAI")
@patch("src.agents.critic.ChatPromptTemplate")
def test_critic_evaluation(mock_prompt_cls, mock_chat_cls):
    """Critic should return boolean re_search_required."""
    # Build the chain mock manually
    mock_prompt = MagicMock()
    mock_prompt_cls.from_messages.return_value = mock_prompt
    
    mock_chain = MagicMock()
    # prompt | structured_llm returns the chain
    mock_prompt.__or__.return_value = mock_chain
    
    # Configure chain output
    mock_chain.invoke.return_value = FactCheckResult(
        re_search_required=True,
        critique="Missing data."
    )
    
    state: AgentState = {"query": "test", "structured_evidence": []}
    
    result = critic_node(state)
    
    assert result["re_search_required"] is True
    assert result["critique"] == "Missing data."
    assert result["active_node"] == "critic"

# ── Writer Tests ─────────────────────────────────────────────────────────────

@patch("src.agents.writer.ChatGoogleGenerativeAI")
@patch("src.agents.writer.ChatPromptTemplate")
def test_writer_report_generation(mock_prompt_cls, mock_chat_cls):
    """Writer should produce report and append references from source_map."""
    # Build the chain mock
    mock_prompt = MagicMock()
    mock_prompt_cls.from_messages.return_value = mock_prompt
    
    mock_chain = MagicMock()
    mock_prompt.__or__.return_value = mock_chain
    
    # Configure chain output (AIMessage-like object)
    mock_response = MagicMock()
    mock_response.content = "This is the report."
    mock_chain.invoke.return_value = mock_response
    
    state: AgentState = {
        "query": "test",
        "structured_evidence": [],
        "source_map": {"[1]": {"url": "http://a.com"}}
    }
    
    result = writer_node(state)
    
    report = result["report"]
    assert "This is the report." in report
    assert "References" in report
    assert "http://a.com" in report # Check citation appending
    assert result["active_node"] == "writer"
