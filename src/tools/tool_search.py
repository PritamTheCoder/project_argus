"""
Project Argus - Tool Search

Embedding-based retrieval over the tool registry. As the catalog grows (e.g.
once MCP servers contribute many tools), binding every tool schema to the model
bloats the prompt and degrades selection; instead each tool's description is
embedded and only the top-k most relevant tools are bound per query.

Below a threshold the whole (tag-filtered) set is returned unchanged with no
embedding work, so small catalogs are unaffected.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

from langchain_core.tools import BaseTool

from src.tools.registry import ToolRegistry, ToolSpec, registry as default_registry

logger = logging.getLogger(__name__)


def tool_document(spec: ToolSpec) -> str:
    """The text we embed to represent a tool for retrieval."""
    tags = ", ".join(spec.tags)
    return f"{spec.name}. {spec.description} Use when: {spec.when_to_use}. Tags: {tags}."


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def rank_tools(
    query_embedding: List[float],
    tool_embeddings: Dict[str, List[float]],
    k: int,
) -> List[str]:
    """Pure ranking: return up to ``k`` tool names ordered by cosine similarity."""
    scored = [
        (name, _cosine(query_embedding, emb))
        for name, emb in tool_embeddings.items()
        if emb is not None
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scored[:k]]


class ToolSearchIndex:
    """Lazily-built, incrementally-updated embedding index over registered tools."""

    def __init__(self, registry: ToolRegistry = default_registry):
        self.registry = registry
        # name -> (document_text, embedding)
        self._index: Dict[str, tuple[str, List[float]]] = {}

    def _ensure_embedded(self, specs: List[ToolSpec]) -> Dict[str, List[float]]:
        """Embed any tools whose document is new or changed; return name->embedding."""
        from src.utils.embeddings import get_embeddings

        stale = []
        for spec in specs:
            doc = tool_document(spec)
            cached = self._index.get(spec.name)
            if cached is None or cached[0] != doc:
                stale.append((spec.name, doc))

        if stale:
            embeddings = get_embeddings([doc for _, doc in stale])
            for (name, doc), emb in zip(stale, embeddings):
                self._index[name] = (doc, emb)

        return {spec.name: self._index[spec.name][1] for spec in specs}

    def search(self, query: str, k: int, tags: Optional[List[str]] = None) -> List[ToolSpec]:
        """Return up to ``k`` ToolSpecs most relevant to ``query``."""
        specs = self.registry.by_tags(tags) if tags else self.registry.all()
        if not specs:
            return []

        from src.utils.embeddings import get_embeddings
        tool_embs = self._ensure_embedded(specs)
        query_emb = get_embeddings([query])[0]
        ranked_names = rank_tools(query_emb, tool_embs, k)
        by_name = {s.name: s for s in specs}
        return [by_name[n] for n in ranked_names if n in by_name]


# Process-wide index over the default registry.
_index = ToolSearchIndex()


def select_tools(
    query: str,
    tags: Optional[List[str]] = None,
    max_tools: int = 6,
    registry: ToolRegistry = default_registry,
) -> List[BaseTool]:
    """
    Choose the tools to bind for a query.

    - If the (tag-filtered) catalog has <= ``max_tools`` tools, return them all
      with no embedding work (small-catalog fast path — behaviour unchanged).
    - Otherwise, embed tool descriptions and return the top-``max_tools`` by
      relevance to ``query`` (RAG-over-tools).
    """
    specs = registry.by_tags(tags) if tags else registry.all()
    if len(specs) <= max_tools:
        return [s.tool for s in specs]

    selected = _index.search(query, k=max_tools, tags=tags)
    logger.info(
        "Tool search: selected %d/%d tools for query: %s",
        len(selected), len(specs), [s.name for s in selected],
    )
    return [s.tool for s in selected]
