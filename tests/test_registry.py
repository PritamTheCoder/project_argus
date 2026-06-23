"""
Project Argus - Tool registry + research tool tests
"""

import pytest
from langchain_core.tools import tool

from src.tools.registry import ToolRegistry, ToolSpec
from src.tools.research_tools import registry as default_registry, evaluate_expression


@tool
def _dummy(x: str) -> str:
    """A dummy tool."""
    return x


def test_register_and_get():
    reg = ToolRegistry()
    spec = ToolSpec(name="dummy", tool=_dummy, tags=["t1"], yields_candidates=True)
    reg.register(spec)
    assert reg.has("dummy")
    assert reg.get("dummy").description == "A dummy tool."


def test_register_duplicate_raises_without_overwrite():
    reg = ToolRegistry()
    reg.register(ToolSpec(name="dummy", tool=_dummy))
    with pytest.raises(ValueError):
        reg.register(ToolSpec(name="dummy", tool=_dummy))
    # overwrite allowed
    reg.register(ToolSpec(name="dummy", tool=_dummy, tags=["new"]), overwrite=True)
    assert reg.get("dummy").tags == ["new"]


def test_by_tags_any_and_all():
    reg = ToolRegistry()
    reg.register(ToolSpec(name="a", tool=_dummy, tags=["x", "y"]))
    reg.register(ToolSpec(name="b", tool=_dummy, tags=["y"]))
    assert {s.name for s in reg.by_tags(["x"])} == {"a"}
    assert {s.name for s in reg.by_tags(["y"])} == {"a", "b"}
    assert {s.name for s in reg.by_tags(["x", "y"], match="all")} == {"a"}


def test_tools_resolution_by_names_and_tags():
    reg = ToolRegistry()
    reg.register(ToolSpec(name="a", tool=_dummy, tags=["retrieval"], yields_candidates=True))
    reg.register(ToolSpec(name="b", tool=_dummy, tags=["compute"]))
    assert len(reg.tools(tags=["retrieval"])) == 1
    assert len(reg.tools(names=["a", "b"])) == 2
    assert reg.candidate_tool_names() == ["a"]


# ── Default registry wiring ──────────────────────────────────────────────────

def test_default_registry_has_expected_tools():
    for name in ("web_search", "semantic_scholar_search", "arxiv_search",
                 "crossref_search", "kg_lookup", "calculator"):
        assert default_registry.has(name), f"missing tool {name}"


def test_default_registry_candidate_tools_are_search_backends():
    candidates = set(default_registry.candidate_tool_names())
    assert candidates == {"web_search", "semantic_scholar_search", "arxiv_search", "crossref_search"}


def test_retrieval_tag_excludes_utility_tools():
    names = {t.name for t in default_registry.tools(tags=["retrieval"])}
    assert "calculator" not in names
    assert "kg_lookup" not in names
    assert "web_search" in names


# ── Calculator pure logic ────────────────────────────────────────────────────

def test_calculator_basic():
    assert evaluate_expression("400 * 1.25")["result"] == 500.0
    assert evaluate_expression("(450-380)/380")["error"] is None


def test_calculator_rejects_unsafe():
    res = evaluate_expression("__import__('os').system('ls')")
    assert res["result"] is None
    assert res["error"] is not None
