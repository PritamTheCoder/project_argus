"""
Project Argus - Tool search tests
"""

import pytest
from unittest.mock import patch
from langchain_core.tools import tool

from src.tools.registry import ToolRegistry, ToolSpec
from src.tools.tool_search import rank_tools, tool_document, select_tools, ToolSearchIndex


@tool
def _dummy(x: str) -> str:
    """dummy"""
    return x


def _spec(name, desc, tags):
    # Build a spec whose tool carries the description used for embedding.
    @tool
    def fn(x: str) -> str:
        """placeholder docstring (overridden below)"""
        return x
    fn.name = name
    fn.description = desc
    return ToolSpec(name=name, tool=fn, tags=tags, when_to_use=desc)


# ── Pure ranking ─────────────────────────────────────────────────────────────

def test_rank_tools_orders_by_cosine():
    q = [1.0, 0.0]
    embs = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.7, 0.7]}
    ranked = rank_tools(q, embs, k=2)
    assert ranked[0] == "a"
    assert ranked[1] == "c"


def test_rank_tools_skips_none():
    ranked = rank_tools([1.0, 0.0], {"a": None, "b": [1.0, 0.0]}, k=5)
    assert ranked == ["b"]


def test_tool_document_includes_name_and_tags():
    s = _spec("web_search", "search the web", ["retrieval", "web"])
    doc = tool_document(s)
    assert "web_search" in doc and "retrieval" in doc and "search the web" in doc


# ── select_tools small-catalog fast path (no embedding) ──────────────────────

def test_select_tools_returns_all_below_threshold_without_embedding():
    reg = ToolRegistry()
    reg.register(_spec("a", "alpha", ["retrieval"]))
    reg.register(_spec("b", "beta", ["retrieval"]))

    # If embeddings were touched, this patch would raise.
    with patch("src.utils.embeddings.get_embeddings", side_effect=AssertionError("should not embed")):
        tools = select_tools("anything", tags=["retrieval"], max_tools=6, registry=reg)
    assert len(tools) == 2


# ── select_tools large-catalog path (embedding-backed) ───────────────────────

def test_select_tools_picks_relevant_when_over_threshold():
    reg = ToolRegistry()
    reg.register(_spec("physics_search", "find physics papers", ["retrieval"]))
    for i in range(6):
        reg.register(_spec(f"misc_{i}", "general web stuff", ["retrieval"]))

    # Fake embeddings: 'physics' docs/query → [1,0], everything else → [0,1].
    def fake_embed(texts):
        return [[1.0, 0.0] if "physics" in t.lower() else [0.0, 1.0] for t in texts]

    # Fresh index over this registry so we don't touch the process index.
    idx = ToolSearchIndex(reg)
    with patch("src.utils.embeddings.get_embeddings", side_effect=fake_embed):
        specs = idx.search("latest physics research", k=2, tags=["retrieval"])

    names = [s.name for s in specs]
    assert "physics_search" in names
    assert len(names) == 2
