"""
Project Argus - Acquisition agent tests

Exercises the bounded tool-calling loop (with a mocked tool-calling LLM and
mocked providers), the deterministic mode-aware fallback, dedup, and the
mode-aware system prompt.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage

from src.agents.acquisition import (
    gather_sources_for_query,
    build_gatherer_system_prompt,
    _dedup_candidates,
)
from src.tools.providers import SearchResult


def _tool_call(name, args):
    return {"name": name, "args": args, "id": "call_1", "type": "tool_call"}


# ── Mode-aware prompt ────────────────────────────────────────────────────────

def test_prompt_trusted_only_forbids_web():
    p = build_gatherer_system_prompt("TRUSTED_ONLY")
    assert "ONLY academic tools" in p
    assert "Do NOT call web_search" in p


def test_prompt_mixed_prefers_web():
    p = build_gatherer_system_prompt("MIXED")
    assert "web_search primarily" in p


# ── Dedup ────────────────────────────────────────────────────────────────────

def test_dedup_drops_seen_and_duplicate_urls():
    cands = [
        {"url": "http://a"}, {"url": "http://a"}, {"url": "http://b"}, {"url": ""},
    ]
    out = _dedup_candidates(cands, seen_urls={"http://b"})
    assert [c["url"] for c in out] == ["http://a"]


# ── Tool-calling loop ────────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("src.agents.acquisition.get_llm")
@patch("src.tools.providers.web_search", new_callable=AsyncMock)
async def test_gather_executes_model_chosen_tool(mock_web, mock_get_llm):
    """The model emits a web_search tool call; its results become candidates."""
    mock_web.return_value = [SearchResult(title="T", url="http://x", snippet="s", source="web")]

    # Turn 1: call web_search. Turn 2: no tool calls → stop.
    ai1 = AIMessage(content="", tool_calls=[_tool_call("web_search", {"query": "q"})])
    ai2 = AIMessage(content="done")
    llm_tools = MagicMock()
    llm_tools.ainvoke = AsyncMock(side_effect=[ai1, ai2])
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = llm_tools
    mock_get_llm.return_value = mock_llm

    candidates = await gather_sources_for_query("q", "MIXED")

    assert any(c["url"] == "http://x" for c in candidates)
    mock_web.assert_awaited()
    # bind_tools was given a non-empty tool list.
    assert mock_llm.bind_tools.call_args[0][0]


@pytest.mark.asyncio
@patch("src.agents.acquisition.get_llm")
@patch("src.tools.providers.semantic_scholar_search", new_callable=AsyncMock)
async def test_gather_falls_back_when_no_tool_calls(mock_ss, mock_get_llm):
    """If the model returns no tool calls, the deterministic fallback runs."""
    mock_ss.return_value = [SearchResult(url="http://paper", snippet="abstract", source="semantic_scholar")]

    ai = AIMessage(content="I won't call tools")  # no tool_calls
    llm_tools = MagicMock()
    llm_tools.ainvoke = AsyncMock(return_value=ai)
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = llm_tools
    mock_get_llm.return_value = mock_llm

    with patch("src.tools.providers.web_search", new_callable=AsyncMock) as mock_web:
        mock_web.return_value = []
        candidates = await gather_sources_for_query("q", "TRUSTED_FIRST")

    assert any(c["url"] == "http://paper" for c in candidates)


@pytest.mark.asyncio
@patch("src.agents.acquisition.get_llm", side_effect=Exception("model has no tool calling"))
@patch("src.tools.providers.web_search", new_callable=AsyncMock)
@patch("src.tools.providers.semantic_scholar_search", new_callable=AsyncMock)
async def test_gather_fallback_on_llm_failure(mock_ss, mock_web, mock_get_llm):
    """If constructing/using the LLM fails entirely, fallback still gathers sources."""
    mock_web.return_value = [SearchResult(url="http://w", source="web")]
    mock_ss.return_value = [SearchResult(url="http://s", snippet="abstract", source="semantic_scholar")]

    candidates = await gather_sources_for_query("q", "MIXED")
    urls = {c["url"] for c in candidates}
    assert "http://w" in urls


@pytest.mark.asyncio
@patch("src.agents.acquisition.get_llm", side_effect=Exception("down"))
@patch("src.tools.providers.semantic_scholar_search", new_callable=AsyncMock)
@patch("src.tools.providers.arxiv_search", new_callable=AsyncMock)
@patch("src.tools.providers.web_search", new_callable=AsyncMock)
async def test_trusted_only_fallback_uses_academic_not_web(mock_web, mock_arxiv, mock_ss, mock_get_llm):
    """TRUSTED_ONLY fallback must not call web_search."""
    mock_ss.return_value = [SearchResult(url="http://s", snippet="a", source="semantic_scholar")]
    mock_arxiv.return_value = [SearchResult(url="http://a", snippet="a", source="arxiv")]
    mock_web.return_value = [SearchResult(url="http://w", source="web")]

    candidates = await gather_sources_for_query("q", "TRUSTED_ONLY")
    urls = {c["url"] for c in candidates}

    assert "http://s" in urls and "http://a" in urls
    assert "http://w" not in urls
    mock_web.assert_not_awaited()
