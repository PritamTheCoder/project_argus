"""
Project Argus - Acquisition Agent

The model-driven source gatherer. Given a research sub-question and its search
``mode``, a tool-calling LLM chooses which backends to query (web vs. academic
vs. memory) and the resulting candidate documents are collected. Tool selection
is decided by the model per query; a deterministic, mode-aware fallback gathers
sources if tool-calling is unavailable or yields nothing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from src.config import (
    GATHERER_MODEL, GATHERER_PROVIDER, GATHERER_FALLBACK_CHAIN,
    GATHER_MAX_STEPS, TOOL_SEARCH_MAX_TOOLS,
)
from src.tools import providers
from src.tools.research_tools import (  # importing also registers the tools
    registry, set_kg_session, reset_kg_session, set_kg_owner, reset_kg_owner,
)
from src.tools.tool_search import select_tools
from src.tools.tool_telemetry import ToolCallTracer, timed_tool_call
from src.tools.mcp_loader import ensure_mcp_loaded
from src.utils.llm_factory import get_llm_with_fallbacks

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
        "- Clinical / medical / drug / life-sciences questions → europe_pmc_search FIRST "
        "(arXiv does not cover medicine), then semantic_scholar_search.\n"
        "- Physics / CS / maths / engineering questions → arxiv_search, semantic_scholar_search, crossref_search.\n"
        "- Company financials, funding rounds, ownership, regulatory risk → sec_edgar_search FIRST "
        "(official filings are primary sources), then web_search for context and reporting.\n"
        "- News / market / commercial / company / general questions → web_search.\n"
        "- With web_search, set `category` when the question calls for a specific kind of source "
        "(\"news\", \"company\", \"financial report\", \"research paper\", \"pdf\").\n"
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


async def _run_tool_loop(
    query: str, mode: str, llm_tools, tools, max_steps: int,
    tracer: Optional[ToolCallTracer] = None,
) -> List[Dict[str, Any]]:
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
                async with timed_tool_call(tracer, name, args) as call:
                    result = await tool.ainvoke(args)
                    if isinstance(result, list):
                        call.set_result(result_count=len(result))
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
    session_id: str = "",
    owner_key_hash: str = "",
    tracer: Optional[ToolCallTracer] = None,
) -> List[Dict[str, Any]]:
    """
    Gather candidate documents for one sub-query via model-driven tool selection,
    falling back to deterministic mode-aware search on any failure.

    ``session_id`` scopes the model's ``kg_lookup`` memory tool to the current run.
    Pass ``tracer`` to accumulate tool telemetry across every sub-query of a run;
    a backend that returns nothing all run is only visible at that scope.

    Returns a list of candidate dicts (see ``SearchResult.to_candidate``),
    deduplicated by URL and excluding ``seen_urls``.
    """
    seen_urls = seen_urls or set()
    max_steps = max_steps or GATHER_MAX_STEPS

    # Scope the model-invoked kg_lookup tool to this run for the duration of the call.
    session_token = set_kg_session(session_id)
    owner_token = set_kg_owner(owner_key_hash)
    await ensure_mcp_loaded()

    tracer = tracer if tracer is not None else ToolCallTracer()
    candidates: List[Dict[str, Any]] = []
    try:
        # Tool search: bind only the most relevant tools (RAG-over-tools). With a
        # small catalog this returns all of them unchanged; it scales selection
        # once MCP servers add many tools.
        tools = select_tools(query, tags=_GATHER_TOOL_TAGS, max_tools=TOOL_SEARCH_MAX_TOOLS)
        # Cross-provider ladder so a Groq rate-limit doesn't kill acquisition; each
        # rung is bound with the same tools. The deterministic fallback below still
        # covers a total tool-calling failure.
        llm_tools = get_llm_with_fallbacks(
            GATHERER_MODEL, GATHERER_PROVIDER,
            fallback_chain=GATHERER_FALLBACK_CHAIN,
            temperature=0,
            tools=tools,
        )
        candidates = await _run_tool_loop(query, mode, llm_tools, tools, max_steps, tracer=tracer)
        if not candidates:
            logger.info("Acquisition: tool loop yielded no candidates for '%s'; using fallback.", query)
            candidates = await _fallback_gather(query, mode)
    except Exception as e:  # noqa: BLE001
        logger.warning("Acquisition: tool-calling failed for '%s' (%s); using deterministic fallback.", query, e)
        candidates = await _fallback_gather(query, mode)
    finally:
        reset_kg_session(session_token)
        reset_kg_owner(owner_token)

    deduped = _dedup_candidates(candidates, seen_urls)
    if tracer.records:
        logger.info("Acquisition: tool-call telemetry for '%s': %s", query, tracer.summary())
    logger.info("Acquisition: gathered %d candidate source(s) for '%s' (mode=%s).", len(deduped), query, mode)
    return deduped
