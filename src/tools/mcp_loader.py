"""
MCP (Model Context Protocol) loader.

Discovers tools exposed by external MCP servers and registers them into the
tool registry, so they become selectable by the acquisition agent exactly like
built-in tools — no graph or prompt changes required. Combined with tool
search, many MCP tools can be registered while only the relevant few are bound
per query.

Everything here degrades gracefully:
  - no MCP servers configured  → no-op
  - `langchain-mcp-adapters` not installed → logged warning, no-op
  - a server fails to connect   → logged, the rest still load

Configure via the MCP_SERVERS_JSON env var (see config.py for the schema).
The pure helpers (`parse_servers`, `register_mcp_tools`) are unit-tested; the
live connection is exercised only when servers are actually configured.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langchain_core.tools import BaseTool

from src.config import MCP_SERVERS_JSON
from src.tools.registry import ToolRegistry, ToolSpec, registry as default_registry

logger = logging.getLogger(__name__)

# MCP tools are bound for the gatherer (retrieval pool) and tagged so they can be
# filtered/inspected. They don't yield our candidate-doc shape by default.
_MCP_TAGS = ["mcp", "retrieval"]

# Module-level guard so we only attempt to load once per process.
_loaded = False


def parse_servers(servers_json: str) -> Dict[str, Any]:
    """Parse the MCP_SERVERS_JSON config into a connections dict. Pure."""
    servers_json = (servers_json or "").strip()
    if not servers_json:
        return {}
    try:
        data = json.loads(servers_json)
        if not isinstance(data, dict):
            logger.warning("MCP_SERVERS_JSON must be a JSON object; got %s.", type(data).__name__)
            return {}
        return data
    except json.JSONDecodeError as e:
        logger.warning("MCP_SERVERS_JSON is not valid JSON: %s", e)
        return {}


def register_mcp_tools(tools: List[BaseTool], registry: ToolRegistry = default_registry) -> int:
    """Register a list of MCP-provided tools into the registry. Idempotent. Pure."""
    count = 0
    for tool in tools:
        name = getattr(tool, "name", None)
        if not name or registry.has(name):
            continue
        desc = (getattr(tool, "description", "") or "")[:160]
        registry.register(ToolSpec(
            name=name,
            tool=tool,
            tags=list(_MCP_TAGS),
            when_to_use=desc,
            yields_candidates=False,
        ))
        count += 1
    return count


async def load_mcp_tools(
    registry: ToolRegistry = default_registry,
    servers_json: str = MCP_SERVERS_JSON,
) -> int:
    """
    Connect to configured MCP servers and register their tools. Returns the
    number of tools registered. Never raises — degrades to 0 on any problem.
    """
    connections = parse_servers(servers_json)
    if not connections:
        return 0

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient  # lazy import
    except ImportError:
        logger.warning(
            "MCP servers are configured but 'langchain-mcp-adapters' is not installed. "
            "Run: pip install langchain-mcp-adapters. Skipping MCP tools."
        )
        return 0

    try:
        client = MultiServerMCPClient(connections)
        tools = await client.get_tools()
    except Exception as e:  # noqa: BLE001
        logger.warning("MCP: failed to load tools from servers %s: %s", list(connections), e)
        return 0

    n = register_mcp_tools(tools, registry)
    logger.info("MCP: registered %d tool(s) from %d server(s): %s", n, len(connections), list(connections))
    return n


async def ensure_mcp_loaded() -> int:
    """Load MCP tools once per process (no-op if already loaded or unconfigured)."""
    global _loaded
    if _loaded:
        return 0
    _loaded = True
    try:
        return await load_mcp_tools()
    except Exception as e:  # never block acquisition on MCP errors
        logger.warning("MCP: ensure_mcp_loaded failed: %s", e)
        return 0
