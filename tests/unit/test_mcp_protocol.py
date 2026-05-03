"""MCP protocol-level tests using FastMCP's direct async API.

Verifies tool discovery, parameter schemas, call_tool round-trips,
prompt listing, and error propagation at the MCP protocol layer.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import mem0_mcp_selfhosted.server as server_mod

EXPECTED_TOOLS = {
    "add_memory",
    "search_memories",
    "get_memories",
    "get_memory",
    "update_memory",
    "delete_memory",
    "delete_all_memories",
    "list_entities",
    "delete_entities",
}

# Required parameters per tool (tool_name -> set of required param names)
REQUIRED_PARAMS = {
    "add_memory": {"text", "project"},
    "search_memories": {"query", "project"},
    "get_memories": {"project"},
    "get_memory": {"memory_id"},
    "update_memory": {"memory_id", "text"},
    "delete_memory": {"memory_id"},
    "delete_all_memories": {"project"},
    "list_entities": set(),
    "delete_entities": set(),
}


@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch):
    monkeypatch.setenv("MEM0_USER_ID", "test-user")


@pytest.fixture
def mock_memory():
    mem = MagicMock()
    mem.add.return_value = {"results": [{"id": "mem-1", "memory": "test fact"}]}
    mem.search.return_value = {"results": [{"id": "mem-1", "score": 0.95}]}
    mem.get_all.return_value = {"results": []}
    mem.get.return_value = {"id": "mem-1", "memory": "test fact"}
    mem.update.return_value = None
    mem.delete.return_value = None
    return mem


@pytest.fixture
def mcp_server(mock_memory):
    """Create a FastMCP server with mocked Memory for protocol testing."""
    original_memory = server_mod.memory
    server_mod.memory = mock_memory

    srv = server_mod._create_server()

    yield srv

    server_mod.memory = original_memory


# ============================================================
# Tool Discovery
# ============================================================


class TestToolDiscovery:
    @pytest.mark.asyncio
    async def test_list_tools_returns_all_9(self, mcp_server):
        """v3: exactly 9 tools (graph tools removed)."""
        tools = await mcp_server.list_tools()
        tool_names = {t.name for t in tools}
        assert tool_names == EXPECTED_TOOLS
        assert len(tools) == 9

    @pytest.mark.asyncio
    async def test_tool_schemas_have_required_params(self, mcp_server):
        tools = await mcp_server.list_tools()
        for tool in tools:
            schema = tool.inputSchema
            assert schema["type"] == "object"
            actual_required = set(schema.get("required", []))
            expected_required = REQUIRED_PARAMS[tool.name]
            assert actual_required == expected_required, f"Tool {tool.name!r}: expected required={expected_required}, got {actual_required}"


# ============================================================
# call_tool Round-trips
# ============================================================


class TestCallToolRoundTrip:
    @pytest.mark.asyncio
    async def test_add_memory(self, mcp_server, mock_memory):
        content_blocks, _ = await mcp_server.call_tool("add_memory", {"text": "I prefer Python", "project": "my-project"})
        assert len(content_blocks) > 0
        text = content_blocks[0].text
        parsed = json.loads(text)
        assert "results" in parsed
        mock_memory.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_memories(self, mcp_server, mock_memory):
        content_blocks, _ = await mcp_server.call_tool("search_memories", {"query": "Python preferences", "project": "my-project"})
        assert len(content_blocks) > 0
        text = content_blocks[0].text
        parsed = json.loads(text)
        # search_with_project returns a list, not {"results": [...]}
        assert isinstance(parsed, list)
        # Two searches: project-scoped + global
        assert mock_memory.search.call_count == 2


# ============================================================
# Prompt Discovery
# ============================================================


class TestPromptDiscovery:
    @pytest.mark.asyncio
    async def test_list_prompts_contains_memory_assistant(self, mcp_server):
        prompts = await mcp_server.list_prompts()
        prompt_names = {p.name for p in prompts}
        assert "memory_assistant" in prompt_names

    @pytest.mark.asyncio
    async def test_get_prompt_memory_assistant_content(self, mcp_server):
        result = await mcp_server.get_prompt("memory_assistant")
        assert len(result.messages) > 0
        text = result.messages[0].content.text
        assert len(text) > 0
        assert "memory" in text.lower()


# ============================================================
# Error Propagation
# ============================================================


class TestErrorPropagation:
    @pytest.mark.asyncio
    async def test_tool_exception_returns_json_error(self, mcp_server, mock_memory):
        mock_memory.get.side_effect = RuntimeError("connection lost")
        content_blocks, _ = await mcp_server.call_tool("get_memory", {"memory_id": "uuid-123"})
        assert len(content_blocks) > 0
        text = content_blocks[0].text
        parsed = json.loads(text)
        assert "error" in parsed
        assert "connection lost" in parsed.get("detail", "")
