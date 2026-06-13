"""
Project Argus - Acquisition Agent (Phase 2)

The model-driven source gatherer. Given a research sub-question and its search
`mode`, it lets a tool-calling LLM choose which backends to query (web vs.
academic vs. memory) and collects the resulting candidate documents.

This is the "tool utilization" core: tool *selection* is decided by the model
per query, not hard-coded. A deterministic, mode-aware fallback guarantees the
pipeline still gathers sources if tool-calling is unavailable or yields nothing.

Only the *acquisition* boundary is agentic — the downstream spine
(Refiner → Verifier → Consensus → Critic → Writer) stays deterministic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from src.config import GATHERER_MODEL, GATHERER_PROVIDER, GATHER_MAX_STEPS
from src.tools import providers
from src.tools.research_tools import registry  # importing also registers the tools
from src.utils.llm_factory import get_llm

logger = logging.getLogger(__name__)

# Tools the gatherer is allowed to call.
_GATHER_TOOL_TAGS = ["retrieval", "memory"]


def build_gatherer_system_prompt(mode: str) -> str:
    """Mode-aware instructions that make `DomainPolicy` actually drive routing."""
    base = (
        "You are a research source-gathering agent. Given a research sub-question, "
        "select and call the most appropriate search tools to find high-quality sources.\n"
        "You may call several tools. You may call kg_lookup first to avoid re-finding "
        "facts already known. Stop once you have gathered enough good sources.\n\n"
        "Routing guidance:\n"
        "- Scientific / technical / medical questions → semantic_scholar_search, arxiv_search, crossref_search.\n"
        "- News / market / commercial / company / general questions → web_search.\n"
    )
    mode = (mode or "MIXED").upper()
    if mode == "TRUSTED_ONLY":
        base += ("\nThis query is TRUSTED_ONLY: use ONLY academic tools "
                 "(semantic_scholar_search, arxiv_search, crossref_search). Do NOT call web_search.\n")
    elif mode == "TRUSTED_FIRST":
        base += "\nThis query is TRUSTED_FIRST: prioritise academic tools; use web_search only to fill gaps.\n"
    else:
        base += "\nThis query is MIXED: use web_search primarily; add an academic tool if the topic is technical.\n"
    return base


def _summarize_tool_result(name: str, result: Any) -> str:
    """Compact, model-facing summary of a tool result for the conversation."""
    if isinstance(result, list):
        lines = [f"{name} returned {len(result)} result(s):"]
        for r in result[:8]:
            if isinstance(r, dict):
                label = r.get("title") or r.get("claim") or r.get("url") or str(r)
                url = r.get("url", "")
                lines.append(f"- {label} {('(' + url + ')') if url else ''}".strip())
        return "\n".join(lines)
    return str(result)


def _dedup_candidates(candidates: List[Dict[str, Any]], seen_urls: Set[str]) -> List[Dict[str, Any]]:
    """Deduplicate by URL and drop anything already seen this run."""
    out: List[Dict[str, Any]] = []
    local_seen: Set[str] = set()
    for c in candidates:
        url = (c or {}).get("url")
        if not url or url in seen_urls or url in local_seen:
            continue
        local_seen.add(url)
        out.append(c)
    return out


async def _run_tool_loop(query: str, mode: str, llm_tools, tools, max_steps: int) -> List[Dict[str, Any]]:
    """Bounded bind_tools loop. Returns candidate dicts collected from tool calls."""
    candidate_names = set(registry.candidate_tool_names())
    tool_map = {t.name: t for t in tools}

    messages: List[Any] = [
        SystemMessage(content=build_gatherer_system_prompt(mode)),
        HumanMessage(content=f"Research sub-question: {query}\n\nGather the best sources to answer it."),
    ]
    candidates: List[Dict[str, Any]] = []

    for step in range(max_steps):
        ai = await llm_tools.ainvoke(messages)
        messages.append(ai)
        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            break

        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {}) or {}
            tcid = tc.get("id")
            tool = tool_map.get(name)
            if tool is None:
                messages.append(ToolMessage(content=f"Unknown tool '{name}'.", tool_call_id=tcid))
                continue
            try:
                result = await tool.ainvoke(args)
            except Exception as e:  # noqa: BLE001
                logger.warning("Acquisition: tool '%s' failed: %s", name, e)
                messages.append(ToolMessage(content=f"{name} error: {e}", tool_call_id=tcid))
                continue

            if name in candidate_names and isinstance(result, list):
                candidates.extend([r for r in result if isinstance(r, dict) and r.get("url")])
            messages.append(ToolMessage(content=_summarize_tool_result(name, result), tool_call_id=tcid))

    return candidates


async def _fallback_gather(query: str, mode: str) -> List[Dict[str, Any]]:
    """Deterministic, mode-aware gathering when tool-calling is unavailable."""
    mode = (mode or "MIXED").upper()
    out: List[Dict[str, Any]] = []

    if mode in ("TRUSTED_ONLY", "TRUSTED_FIRST"):
        for fn in (providers.semantic_scholar_search, providers.arxiv_search):
            out += [r.to_candidate() for r in await fn(query)]
        if mode == "TRUSTED_FIRST":
            out += [r.to_candidate() for r in await providers.web_search(query)]
    else:  # MIXED
        out += [r.to_candidate() for r in await providers.web_search(query)]
        out += [r.to_candidate() for r in await providers.semantic_scholar_search(query, limit=4)]

    return out


async def gather_sources_for_query(
    query: str,
    mode: str = "MIXED",
    seen_urls: Optional[Set[str]] = None,
    max_steps: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Gather candidate documents for one sub-query via model-driven tool selection,
    falling back to deterministic mode-aware search on any failure.

    Returns a list of candidate dicts (see ``SearchResult.to_candidate``),
    deduplicated by URL and excluding ``seen_urls``.
    """
    seen_urls = seen_urls or set()
    max_steps = max_steps or GATHER_MAX_STEPS

    candidates: List[Dict[str, Any]] = []
    try:
        tools = registry.tools(tags=_GATHER_TOOL_TAGS)
        llm = get_llm(GATHERER_MODEL, GATHERER_PROVIDER, temperature=0)
        llm_tools = llm.bind_tools(tools)
        candidates = await _run_tool_loop(query, mode, llm_tools, tools, max_steps)
        if not candidates:
            logger.info("Acquisition: tool loop yielded no candidates for '%s'; using fallback.", query)
            candidates = await _fallback_gather(query, mode)
    except Exception as e:  # noqa: BLE001
        logger.warning("Acquisition: tool-calling failed for '%s' (%s); using deterministic fallback.", query, e)
        candidates = await _fallback_gather(query, mode)

    deduped = _dedup_candidates(candidates, seen_urls)
    logger.info("Acquisition: gathered %d candidate source(s) for '%s' (mode=%s).", len(deduped), query, mode)
    return deduped
