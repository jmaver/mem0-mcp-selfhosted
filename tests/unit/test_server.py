"""Unit tests for server.py — MCP tool functions, _init_memory(), _create_server().

Tests tool orchestration logic with mocked Memory: kwargs assembly, user_id
defaults, scope validation, error handling, and delegation to helpers.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

import mem0_mcp_selfhosted.server as server_mod


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch):
    """Set default env vars for all tests."""
    monkeypatch.setenv("MEM0_USER_ID", "test-user")


@pytest.fixture
def mock_memory():
    """Create a mock Memory object and patch server globals."""
    mem = MagicMock()
    mem.add.return_value = {"results": [{"id": "mem-1", "memory": "test fact"}]}
    mem.search.return_value = {"results": [{"id": "mem-1", "score": 0.95}]}
    mem.get_all.return_value = {"results": [{"id": "mem-1"}]}
    mem.get.return_value = {"id": "mem-1", "memory": "test fact"}
    mem.update.return_value = None
    mem.delete.return_value = None
    return mem


@pytest.fixture
def server_with_mock(mock_memory):
    """Create a FastMCP server with mocked Memory and helpers."""
    original_memory = server_mod.memory
    server_mod.memory = mock_memory

    srv = server_mod._create_server()

    yield srv, mock_memory

    server_mod.memory = original_memory


def _get_tool_fn(srv, name: str):
    """Extract a tool function from the FastMCP server by name."""
    tool = srv._tool_manager._tools.get(name)
    assert tool is not None, f"Tool {name!r} not found in server"
    return tool.fn


# ============================================================
# 2. Tool Function Tests
# ============================================================


class TestAddMemory:
    def test_kwargs_assembly(self, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "add_memory")
        result = fn(
            text="I prefer Python",
            project="my-project",
            user_id="alice",
            metadata={"source": "chat"},
            infer=False,
        )
        mem.add.assert_called_once()
        args, kwargs = mem.add.call_args
        assert args[0] == [{"role": "user", "content": "I prefer Python"}]
        assert kwargs["user_id"] == "alice:my-project"
        assert kwargs["metadata"] == {"source": "chat"}
        assert kwargs["infer"] is False
        parsed = json.loads(result)
        assert "results" in parsed

    def test_messages_precedence(self, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "add_memory")
        custom_msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        fn(text="ignored text", project="my-project", messages=custom_msgs)
        args, _ = mem.add.call_args
        assert args[0] == custom_msgs

    def test_default_user_id(self, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "add_memory")
        fn(text="some fact", project="my-project")
        _, kwargs = mem.add.call_args
        assert kwargs["user_id"] == "test-user:my-project"

    def test_global_project_uses_bare_user_id(self, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "add_memory")
        fn(text="global fact", project="global")
        _, kwargs = mem.add.call_args
        assert kwargs["user_id"] == "test-user"


class TestSearchMemories:
    def test_all_kwargs(self, server_with_mock):
        """search_memories with project runs two searches (project + global)."""
        srv, mem = server_with_mock
        mem.search.return_value = {"results": [{"id": "mem-1", "score": 0.95}]}
        fn = _get_tool_fn(srv, "search_memories")
        fn(
            query="python preferences",
            project="my-project",
            user_id="bob",
            agent_id="agent-1",
            run_id="run-1",
            filters={"key": {"eq": "val"}},
            limit=5,
            threshold=0.8,
            rerank=True,
        )
        # Two searches: project-scoped + global
        assert mem.search.call_count == 2
        # Both calls use v3 filters dict
        for c in mem.search.call_args_list:
            kwargs = c.kwargs
            assert "filters" in kwargs
            assert "user_id" in kwargs["filters"]
            assert "user_id" not in kwargs, "v3: user_id must not be top-level kwarg"
            # agent_id / run_id must be in filters, not top-level
            assert kwargs["filters"].get("agent_id") == "agent-1"
            assert kwargs["filters"].get("run_id") == "run-1"
            assert "agent_id" not in kwargs
            assert "run_id" not in kwargs
            # limit is renamed to top_k
            assert kwargs.get("top_k") == 5
            assert "limit" not in kwargs

    def test_global_project_single_search(self, server_with_mock):
        """search_memories with project='global' runs one search."""
        srv, mem = server_with_mock
        mem.search.return_value = {"results": [{"id": "mem-1", "score": 0.95}]}
        fn = _get_tool_fn(srv, "search_memories")
        fn(query="python preferences", project="global")
        assert mem.search.call_count == 1
        kwargs = mem.search.call_args.kwargs
        assert kwargs["filters"]["user_id"] == "test-user"


class TestGetMemories:
    def test_scope_filters(self, server_with_mock):
        """get_memories passes filters dict to mem.get_all (v3 contract)."""
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "get_memories")
        fn(project="my-project", user_id="alice", agent_id="agent-1", run_id="run-1", limit=10)
        mem.get_all.assert_called_once_with(
            filters={"user_id": "alice:my-project", "agent_id": "agent-1", "run_id": "run-1"},
            top_k=10,
        )

    def test_global_project(self, server_with_mock):
        """get_memories with global project passes filters dict with bare user_id."""
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "get_memories")
        fn(project="global")
        mem.get_all.assert_called_once_with(filters={"user_id": "test-user"})

    def test_no_limit_omits_top_k(self, server_with_mock):
        """When limit is not provided, top_k is not passed to get_all."""
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "get_memories")
        fn(project="my-project")
        _, kwargs = mem.get_all.call_args
        assert "top_k" not in kwargs

    def test_no_top_level_user_id(self, server_with_mock):
        """v3: user_id must not appear as a top-level kwarg to mem.get_all."""
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "get_memories")
        fn(project="my-project", user_id="alice")
        _, kwargs = mem.get_all.call_args
        assert "user_id" not in kwargs, "v3: user_id must be inside filters dict, not top-level"
        assert kwargs["filters"]["user_id"] == "alice:my-project"


class TestGetMemory:
    def test_by_id(self, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "get_memory")
        fn(memory_id="uuid-123")
        mem.get.assert_called_once_with("uuid-123")


class TestUpdateMemory:
    def test_uses_data_param(self, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "update_memory")
        result = fn(memory_id="uuid-123", text="updated fact")
        mem.update.assert_called_once_with("uuid-123", data="updated fact")
        parsed = json.loads(result)
        assert parsed["message"] == "Memory updated successfully!"


class TestDeleteMemory:
    def test_delegation(self, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "delete_memory")
        result = fn(memory_id="uuid-123")
        mem.delete.assert_called_once_with("uuid-123")
        parsed = json.loads(result)
        assert parsed["message"] == "Memory deleted successfully!"


class TestDeleteAllMemories:
    @patch("mem0_mcp_selfhosted.server.safe_bulk_delete", return_value=0)
    def test_scope_defaults_to_user_id(self, mock_sbd, server_with_mock):
        """delete_all_memories with project scopes user_id."""
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "delete_all_memories")
        result = fn(project="my-project")
        mock_sbd.assert_called_once_with(mem, {"user_id": "test-user:my-project"})
        parsed = json.loads(result)
        assert parsed["count"] == 0

    @patch("mem0_mcp_selfhosted.server.safe_bulk_delete", return_value=3)
    def test_delegates_safe_bulk_delete(self, mock_sbd, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "delete_all_memories")
        result = fn(project="my-project", user_id="alice")
        mock_sbd.assert_called_once_with(mem, {"user_id": "alice:my-project"})
        parsed = json.loads(result)
        assert parsed["count"] == 3

    @patch("mem0_mcp_selfhosted.server.safe_bulk_delete", return_value=2)
    def test_global_project(self, mock_sbd, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "delete_all_memories")
        fn(project="global")
        mock_sbd.assert_called_once_with(mem, {"user_id": "test-user"})


class TestListEntities:
    @patch("mem0_mcp_selfhosted.server.list_entities_facet")
    def test_delegation(self, mock_facet, server_with_mock):
        mock_facet.return_value = {"users": [], "agents": [], "runs": []}
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "list_entities")
        result = fn()
        mock_facet.assert_called_once_with(mem)
        parsed = json.loads(result)
        assert "users" in parsed


class TestDeleteEntities:
    def test_scope_validation(self, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "delete_entities")
        result = fn()
        parsed = json.loads(result)
        assert "error" in parsed

    @patch("mem0_mcp_selfhosted.server.safe_bulk_delete", return_value=5)
    def test_delegates_safe_bulk_delete(self, mock_sbd, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "delete_entities")
        result = fn(user_id="alice")
        mock_sbd.assert_called_once_with(mem, {"user_id": "alice"})
        parsed = json.loads(result)
        assert parsed["count"] == 5


# ============================================================
# 3. Error Handling and Initialization Tests
# ============================================================


class TestToolErrorHandling:
    def test_exception_returns_json_error(self, server_with_mock):
        srv, mem = server_with_mock
        mem.get.side_effect = RuntimeError("connection lost")
        fn = _get_tool_fn(srv, "get_memory")
        result = fn(memory_id="uuid-123")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "connection lost" in parsed.get("detail", "")


class TestInitMemory:
    @patch("mem0.Memory.from_config")
    @patch("mem0.utils.factory.LlmFactory.register_provider")
    @patch("mem0_mcp_selfhosted.server.build_config")
    def test_registers_both_providers_anthropic(
        self, mock_bc, mock_reg, mock_from_config
    ):
        mock_memory = MagicMock()
        mock_from_config.return_value = mock_memory
        mock_bc.return_value = (
            {"llm": {"provider": "anthropic", "config": {}}},
            [
                {
                    "name": "ollama",
                    "class_path": "mem0_mcp_selfhosted.llm_ollama.OllamaToolLLM",
                },
                {
                    "name": "anthropic",
                    "class_path": "mem0_mcp_selfhosted.llm_anthropic.AnthropicOATLLM",
                },
            ],
        )
        server_mod._init_memory()
        assert mock_reg.call_count == 2  # Both Ollama and Anthropic registered

    @patch("mem0.Memory.from_config")
    @patch("mem0.utils.factory.LlmFactory.register_provider")
    @patch("mem0_mcp_selfhosted.server.build_config")
    def test_registers_ollama_only(self, mock_bc, mock_reg, mock_from_config):
        """When LLM is ollama, only ollama provider is registered (always included)."""
        mock_memory = MagicMock()
        mock_from_config.return_value = mock_memory
        mock_bc.return_value = (
            {"llm": {"provider": "ollama", "config": {}}},
            [
                {
                    "name": "ollama",
                    "class_path": "mem0_mcp_selfhosted.llm_ollama.OllamaToolLLM",
                },
            ],
        )
        server_mod._init_memory()
        mock_reg.assert_called_once()
        call_kwargs = mock_reg.call_args
        assert call_kwargs.kwargs["name"] == "ollama"


class TestResolveConfigClass:
    def test_anthropic_oat_returns_config(self):
        """anthropic_oat resolves to AnthropicOATConfig (same as anthropic)."""
        from mem0_mcp_selfhosted.llm_anthropic import AnthropicOATConfig

        result = server_mod._resolve_config_class("anthropic_oat")
        assert result is AnthropicOATConfig

    def test_anthropic_returns_config(self):
        """anthropic still resolves to AnthropicOATConfig."""
        from mem0_mcp_selfhosted.llm_anthropic import AnthropicOATConfig

        result = server_mod._resolve_config_class("anthropic")
        assert result is AnthropicOATConfig

    def test_openai_returns_config(self):
        """openai resolves to OpenAIConfig."""
        from mem0.configs.llms.openai import OpenAIConfig

        result = server_mod._resolve_config_class("openai")
        assert result is OpenAIConfig

    def test_unknown_returns_none(self):
        """Unknown provider returns None."""
        assert server_mod._resolve_config_class("unknown") is None


class TestRegisterProviders:
    @patch("mem0.utils.factory.LlmFactory.register_provider")
    def test_anthropic_oat_registers_without_error(self, mock_reg):
        """anthropic_oat provider registers successfully with LlmFactory."""
        server_mod.register_providers([
            {"name": "anthropic_oat", "class_path": "mem0_mcp_selfhosted.llm_anthropic.AnthropicOATLLM"},
        ])
        mock_reg.assert_called_once()
        assert mock_reg.call_args[1]["name"] == "anthropic_oat"

    @patch("mem0.utils.factory.LlmFactory.register_provider")
    def test_unknown_provider_logs_warning(self, mock_reg):
        """Unknown provider name logs a warning and is skipped."""
        with patch("mem0_mcp_selfhosted.server.logger") as mock_logger:
            server_mod.register_providers([
                {"name": "unknown_provider", "class_path": "some.module.SomeClass"},
            ])
        mock_logger.warning.assert_called_once()
        assert "unknown_provider" in mock_logger.warning.call_args[0][1]
        mock_reg.assert_not_called()


class TestCreateServer:
    def test_registers_9_tools(self):
        """v3: server registers exactly 9 tools (graph tools removed)."""
        srv = server_mod._create_server()
        tools = srv._tool_manager._tools
        assert len(tools) == 9, f"Expected 9 tools, got {len(tools)}: {list(tools.keys())}"

    def test_registers_prompt(self):
        srv = server_mod._create_server()
        prompts = srv._prompt_manager._prompts
        assert "memory_assistant" in prompts

    def test_tools_register_without_memory(self):
        """Server creates tools even when memory is None (lazy init)."""
        original = server_mod.memory
        server_mod.memory = None
        try:
            srv = server_mod._create_server()
            tools = srv._tool_manager._tools
            assert len(tools) == 9
        finally:
            server_mod.memory = original

    def test_no_graph_tools(self):
        """Graph tools mcp_search_graph and mcp_get_entity must not be registered."""
        srv = server_mod._create_server()
        tools = srv._tool_manager._tools
        assert "mcp_search_graph" not in tools
        assert "mcp_get_entity" not in tools


# ============================================================
# 4. Lazy Memory Initialization Tests
# ============================================================


class TestEnsureMemory:
    @pytest.fixture(autouse=True)
    def _reset_lazy_state(self):
        """Reset lazy init state before each test."""
        original_memory = server_mod.memory
        original_failure = server_mod._last_init_failure
        server_mod.memory = None
        server_mod._last_init_failure = 0.0
        yield
        server_mod.memory = original_memory
        server_mod._last_init_failure = original_failure

    @patch("mem0_mcp_selfhosted.server._init_memory")
    def test_lazy_init_on_first_call(self, mock_init):
        """_ensure_memory() triggers _init_memory() on first call."""
        mock_mem = MagicMock()

        def set_memory():
            server_mod.memory = mock_mem

        mock_init.side_effect = set_memory
        result = server_mod._ensure_memory()
        mock_init.assert_called_once()
        assert result is mock_mem

    def test_returns_cached_memory(self, mock_memory):
        """_ensure_memory() returns cached memory without re-init."""
        server_mod.memory = mock_memory
        result = server_mod._ensure_memory()
        assert result is mock_memory

    @patch("mem0_mcp_selfhosted.server._init_memory")
    def test_returns_none_on_failure(self, mock_init):
        """_ensure_memory() returns None when init fails."""
        mock_init.side_effect = ConnectionError("Qdrant unreachable")
        result = server_mod._ensure_memory()
        assert result is None
        mock_init.assert_called_once()

    @patch("mem0_mcp_selfhosted.server._init_memory")
    def test_no_retry_during_cooldown(self, mock_init):
        """_ensure_memory() skips retry within cooldown period."""
        mock_init.side_effect = ConnectionError("Qdrant unreachable")
        # First call fails
        server_mod._ensure_memory()
        mock_init.reset_mock()
        # Second call within cooldown — should NOT retry
        result = server_mod._ensure_memory()
        assert result is None
        mock_init.assert_not_called()

    @patch("mem0_mcp_selfhosted.server._init_memory")
    @patch("mem0_mcp_selfhosted.server.time")
    def test_retries_after_cooldown(self, mock_time, mock_init):
        """_ensure_memory() retries after cooldown expires."""
        mock_init.side_effect = ConnectionError("Qdrant unreachable")
        mock_time.monotonic.return_value = 100.0
        # First call fails at t=100
        server_mod._ensure_memory()
        mock_init.reset_mock()
        # Second call at t=131 (past 30s cooldown)
        mock_time.monotonic.return_value = 131.0
        mock_init.side_effect = ConnectionError("still down")
        server_mod._ensure_memory()
        mock_init.assert_called_once()


class TestToolsWithNoMemory:
    """Test that tools return structured errors when memory is unavailable."""

    @pytest.fixture(autouse=True)
    def _no_memory(self):
        """Ensure memory is None and lazy init fails."""
        original = server_mod.memory
        original_failure = server_mod._last_init_failure
        server_mod.memory = None
        server_mod._last_init_failure = 0.0
        yield
        server_mod.memory = original
        server_mod._last_init_failure = original_failure

    @patch("mem0_mcp_selfhosted.server._init_memory", side_effect=ConnectionError("no qdrant"))
    def test_add_memory_returns_error(self, mock_init):
        srv = server_mod._create_server()
        fn = _get_tool_fn(srv, "add_memory")
        result = fn(text="test", project="my-project")
        parsed = json.loads(result)
        assert "error" in parsed

    @patch("mem0_mcp_selfhosted.server._init_memory", side_effect=ConnectionError("no qdrant"))
    def test_search_memories_returns_error(self, mock_init):
        srv = server_mod._create_server()
        fn = _get_tool_fn(srv, "search_memories")
        result = fn(query="test", project="my-project")
        parsed = json.loads(result)
        assert "error" in parsed

    @patch("mem0_mcp_selfhosted.server._init_memory", side_effect=ConnectionError("no qdrant"))
    def test_get_memories_returns_error(self, mock_init):
        srv = server_mod._create_server()
        fn = _get_tool_fn(srv, "get_memories")
        result = fn(project="my-project")
        parsed = json.loads(result)
        assert parsed["error"] == "Memory not initialized"

    @patch("mem0_mcp_selfhosted.server._init_memory", side_effect=ConnectionError("no qdrant"))
    def test_get_memory_returns_error(self, mock_init):
        srv = server_mod._create_server()
        fn = _get_tool_fn(srv, "get_memory")
        result = fn(memory_id="uuid-123")
        parsed = json.loads(result)
        assert parsed["error"] == "Memory not initialized"
