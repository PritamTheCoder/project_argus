"""
Project Argus - MCP loader tests

The live MCP connection needs a running server, so these tests cover the pure
config/registration wiring and the graceful-degradation paths.
"""

import pytest
from types import SimpleNamespace

from src.tools.registry import ToolRegistry
from src.tools.mcp_loader import parse_servers, register_mcp_tools, load_mcp_tools


def _fake_tool(name, description="a tool"):
    # register_mcp_tools only reads .name / .description at runtime.
    return SimpleNamespace(name=name, description=description)


# ── parse_servers ────────────────────────────────────────────────────────────

def test_parse_servers_empty():
    assert parse_servers("") == {}
    assert parse_servers("   ") == {}


def test_parse_servers_valid_json():
    cfg = '{"fs": {"transport": "stdio", "command": "npx", "args": ["-y", "server"]}}'
    parsed = parse_servers(cfg)
    assert "fs" in parsed
    assert parsed["fs"]["command"] == "npx"


def test_parse_servers_invalid_json_returns_empty():
    assert parse_servers("{not json}") == {}


def test_parse_servers_non_object_returns_empty():
    assert parse_servers("[1, 2, 3]") == {}


# ── register_mcp_tools ───────────────────────────────────────────────────────

def test_register_mcp_tools_adds_with_tags():
    reg = ToolRegistry()
    n = register_mcp_tools([_fake_tool("fs_read"), _fake_tool("fs_write")], reg)
    assert n == 2
    spec = reg.get("fs_read")
    assert "mcp" in spec.tags and "retrieval" in spec.tags
    assert spec.yields_candidates is False


def test_register_mcp_tools_is_idempotent():
    reg = ToolRegistry()
    register_mcp_tools([_fake_tool("fs_read")], reg)
    n = register_mcp_tools([_fake_tool("fs_read")], reg)  # already present
    assert n == 0


def test_register_mcp_tools_skips_unnamed():
    reg = ToolRegistry()
    n = register_mcp_tools([SimpleNamespace(name="", description="x")], reg)
    assert n == 0


# ── load_mcp_tools graceful paths ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_load_mcp_tools_noop_without_config():
    reg = ToolRegistry()
    n = await load_mcp_tools(reg, servers_json="")
    assert n == 0
    assert reg.all() == []
