"""
Project Argus - Tool Registry (Phase 2)

A small, dynamic registry of agent tools. Tools register themselves with
metadata (tags, "when to use", whether they yield retrieval candidates), and
consumers fetch a subset — by name or tag — as LangChain tool objects ready to
`bind_tools(...)`.

This is the seam that makes the toolset pluggable: adding a new backend is a
matter of writing a tool and registering it (no graph or prompt edits), and it
is the foundation for Phase 3 tool-search (retrieving relevant tools by
embedding their descriptions).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    """Metadata + the bound LangChain tool for one registered capability."""
    name: str
    tool: BaseTool
    tags: List[str] = field(default_factory=list)
    when_to_use: str = ""
    # True if this tool returns a list of retrieval "candidate" dicts (each with
    # a 'url') that should be fed into the document pipeline.
    yields_candidates: bool = False

    @property
    def description(self) -> str:
        return self.tool.description


class ToolRegistry:
    """In-process registry of available tools, queryable by name or tag."""

    def __init__(self) -> None:
        self._specs: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec, *, overwrite: bool = False) -> ToolSpec:
        if spec.name in self._specs and not overwrite:
            raise ValueError(f"Tool '{spec.name}' is already registered.")
        self._specs[spec.name] = spec
        logger.debug("Registered tool '%s' (tags=%s)", spec.name, spec.tags)
        return spec

    def get(self, name: str) -> ToolSpec:
        return self._specs[name]

    def has(self, name: str) -> bool:
        return name in self._specs

    def all(self) -> List[ToolSpec]:
        return list(self._specs.values())

    def by_tags(self, tags: List[str], *, match: str = "any") -> List[ToolSpec]:
        """Specs matching the given tags ('any' = union, 'all' = intersection)."""
        tagset = set(tags)
        if match == "all":
            return [s for s in self._specs.values() if tagset.issubset(set(s.tags))]
        return [s for s in self._specs.values() if tagset & set(s.tags)]

    def tools(
        self,
        *,
        names: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> List[BaseTool]:
        """Resolve a subset of tools to LangChain tool objects for binding."""
        if names is not None:
            return [self._specs[n].tool for n in names if n in self._specs]
        if tags is not None:
            return [s.tool for s in self.by_tags(tags)]
        return [s.tool for s in self._specs.values()]

    def candidate_tool_names(self, *, tags: Optional[List[str]] = None) -> List[str]:
        """Names of tools that yield retrieval candidates (optionally tag-filtered)."""
        specs = self.by_tags(tags) if tags else self.all()
        return [s.name for s in specs if s.yields_candidates]


# Process-wide singleton used by the acquisition agent.
registry = ToolRegistry()
