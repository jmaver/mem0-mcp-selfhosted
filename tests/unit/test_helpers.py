"""Tests for helpers.py — error wrapper, safe_bulk_delete, user_id, Gemini patch, search_with_project."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


from mem0_mcp_selfhosted.helpers import (
    _mem0_call,
    get_default_user_id,
    make_project_user_id,
    patch_gemini_parse_response,
    safe_bulk_delete,
    search_with_project,
)


class TestMem0Call:
    def test_success_returns_json(self):
        result = _mem0_call(lambda: {"status": "ok"})
        parsed = json.loads(result)
        assert parsed == {"status": "ok"}

    def test_memory_error_caught(self):
        """MemoryError subclass returns structured error JSON."""

        # Create a mock MemoryError-like exception. Declare the metadata
        # attributes on the class so Pyright knows they exist.
        class FakeMemoryError(Exception):
            error_code: str = ""
            details: str = ""
            suggestion: str = ""

        FakeMemoryError.__name__ = "MemoryError"

        exc = FakeMemoryError("something failed")
        exc.error_code = "VALIDATION_ERROR"
        exc.details = "missing field"
        exc.suggestion = "add user_id"

        # Patch the MRO check
        def _raise():
            raise exc

        result = _mem0_call(_raise)
        parsed = json.loads(result)
        assert "error" in parsed

    def test_generic_exception_caught(self):
        """Generic Exception returns type name and detail."""

        def _raise():
            raise ValueError("bad input")

        result = _mem0_call(_raise)
        parsed = json.loads(result)
        assert parsed["error"] == "ValueError"
        assert parsed["detail"] == "bad input"

    def test_ensure_ascii_false(self):
        """Non-ASCII characters preserved in output."""
        result = _mem0_call(lambda: {"text": "Alice prefiere TypeScript"})
        assert "prefiere" in result


class TestSafeBulkDelete:
    def test_iterates_and_deletes(self):
        memory = MagicMock()

        # Mock vector_store.list returning items with .id
        item1 = MagicMock()
        item1.id = "id-1"
        item2 = MagicMock()
        item2.id = "id-2"
        memory.vector_store.list.return_value = [item1, item2]

        count = safe_bulk_delete(memory, {"user_id": "testuser"})

        assert count == 2
        assert memory.delete.call_count == 2
        memory.delete.assert_any_call("id-1")
        memory.delete.assert_any_call("id-2")

    def test_empty_memories_returns_zero(self):
        memory = MagicMock()
        memory.vector_store.list.return_value = []

        count = safe_bulk_delete(memory, {"user_id": "testuser"})

        assert count == 0
        memory.delete.assert_not_called()

    def test_no_graph_cleanup(self):
        """v3: no graph attribute on Memory — safe_bulk_delete never calls graph.delete_all."""
        memory = MagicMock(spec=["vector_store", "delete"])
        memory.vector_store.list.return_value = []

        safe_bulk_delete(memory, {"user_id": "testuser"})

        # The mock has no graph attribute — this should NOT raise AttributeError
        assert not hasattr(memory, "graph") or not memory.graph.delete_all.called

    def test_delete_error_is_swallowed(self):
        """Failed deletes are logged and counted as 0, not raised."""
        memory = MagicMock()
        item = MagicMock()
        item.id = "bad-id"
        memory.vector_store.list.return_value = [item]
        memory.delete.side_effect = RuntimeError("DB error")

        count = safe_bulk_delete(memory, {"user_id": "testuser"})

        assert count == 0


class TestGetDefaultUserId:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        assert get_default_user_id() == "user"

    def test_custom(self, monkeypatch):
        monkeypatch.setenv("MEM0_USER_ID", "bob")
        assert get_default_user_id() == "bob"


class TestMakeProjectUserId:
    def test_with_project(self):
        assert make_project_user_id("alice", "my-app") == "alice:my-app"

    def test_global_project(self):
        assert make_project_user_id("alice", "global") == "alice"

    def test_none_project(self):
        assert make_project_user_id("alice", None) == "alice"

    def test_empty_project(self):
        assert make_project_user_id("alice", "") == "alice"


class TestSearchWithProject:
    """Tests for search_with_project — v3 API contract."""

    def test_uses_filters_dict_not_top_level_user_id(self):
        """v3: mem.search must receive filters={'user_id': ...}, not user_id=... at top level."""
        mock_mem = MagicMock()
        mock_mem.search.return_value = {"results": []}

        search_with_project(mock_mem, "q", "alice", "myproj")

        # Both calls (project + global) must use filters dict
        calls = mock_mem.search.call_args_list
        assert len(calls) == 2
        for c in calls:
            kwargs = c.kwargs
            assert "filters" in kwargs, "v3 requires filters= kwarg on mem.search"
            assert "user_id" in kwargs["filters"], "user_id must be inside filters dict"
            assert "user_id" not in kwargs, "v3 forbids top-level user_id on search"

    def test_project_scoped_search_uses_composite_user_id(self):
        """Project search uses user_id:project composite."""
        mock_mem = MagicMock()
        mock_mem.search.return_value = {"results": []}

        search_with_project(mock_mem, "q", "alice", "myproj")

        first_call = mock_mem.search.call_args_list[0]
        assert first_call.kwargs["filters"]["user_id"] == "alice:myproj"

    def test_global_search_uses_bare_user_id(self):
        """Global scope uses bare user_id."""
        mock_mem = MagicMock()
        mock_mem.search.return_value = {"results": []}

        search_with_project(mock_mem, "q", "alice", "myproj")

        second_call = mock_mem.search.call_args_list[1]
        assert second_call.kwargs["filters"]["user_id"] == "alice"

    def test_global_project_runs_single_search(self):
        """project='global' runs only one search (no project-scoped call)."""
        mock_mem = MagicMock()
        mock_mem.search.return_value = {"results": []}

        search_with_project(mock_mem, "q", "alice", "global")

        assert mock_mem.search.call_count == 1
        kwargs = mock_mem.search.call_args.kwargs
        assert kwargs["filters"]["user_id"] == "alice"

    def test_none_project_runs_single_global_search(self):
        """project=None runs only global search."""
        mock_mem = MagicMock()
        mock_mem.search.return_value = {"results": []}

        search_with_project(mock_mem, "q", "alice", None)

        assert mock_mem.search.call_count == 1

    def test_agent_id_and_run_id_folded_into_filters(self):
        """v3 entity-ID folding: agent_id and run_id go into filters dict, NOT as top-level kwargs."""
        mock_mem = MagicMock()
        mock_mem.search.return_value = {"results": []}

        search_with_project(
            mock_mem,
            "q",
            "alice",
            "myproj",
            agent_id="agent-1",
            run_id="run-1",
        )

        for c in mock_mem.search.call_args_list:
            kwargs = c.kwargs
            # Must be inside filters
            assert kwargs["filters"].get("agent_id") == "agent-1"
            assert kwargs["filters"].get("run_id") == "run-1"
            # Must NOT be top-level kwargs
            assert "agent_id" not in kwargs, "agent_id must not be a top-level kwarg to mem.search"
            assert "run_id" not in kwargs, "run_id must not be a top-level kwarg to mem.search"

    def test_limit_renamed_to_top_k(self):
        """v3 rename: limit= passed by caller becomes top_k= in mem.search call."""
        mock_mem = MagicMock()
        mock_mem.search.return_value = {"results": []}

        search_with_project(mock_mem, "q", "alice", "myproj", limit=7)

        for c in mock_mem.search.call_args_list:
            kwargs = c.kwargs
            assert kwargs.get("top_k") == 7, "limit must be forwarded as top_k to mem.search"
            assert "limit" not in kwargs, "limit must not leak as a top-level kwarg to mem.search"

    def test_top_k_passed_directly(self):
        """top_k= passed directly by caller is forwarded as top_k= to mem.search."""
        mock_mem = MagicMock()
        mock_mem.search.return_value = {"results": []}

        search_with_project(mock_mem, "q", "alice", "myproj", top_k=10)

        for c in mock_mem.search.call_args_list:
            kwargs = c.kwargs
            assert kwargs.get("top_k") == 10

    def test_deduplication_by_id(self):
        """Results with the same ID are deduplicated; project results take precedence."""
        mock_mem = MagicMock()
        mock_mem.search.side_effect = [
            {"results": [{"id": "m1", "memory": "project fact"}, {"id": "m2", "memory": "project only"}]},
            {"results": [{"id": "m1", "memory": "global fact"}, {"id": "m3", "memory": "global only"}]},
        ]

        results = search_with_project(mock_mem, "q", "alice", "myproj")

        ids = [r["id"] for r in results]
        assert ids.count("m1") == 1, "m1 must appear only once (deduplication)"
        # Project result takes precedence for m1
        m1 = next(r for r in results if r["id"] == "m1")
        assert m1["memory"] == "project fact"

    def test_results_as_plain_list(self):
        """mem.search returning a plain list (not dict) is handled."""
        mock_mem = MagicMock()
        mock_mem.search.return_value = [{"id": "m1", "memory": "plain list fact"}]

        results = search_with_project(mock_mem, "q", "alice", "global")

        assert len(results) == 1
        assert results[0]["id"] == "m1"

    def test_extra_filters_merged(self):
        """Caller-supplied filters dict is merged with user_id filter."""
        mock_mem = MagicMock()
        mock_mem.search.return_value = {"results": []}

        search_with_project(
            mock_mem,
            "q",
            "alice",
            "myproj",
            filters={"source": "chat"},
        )

        for c in mock_mem.search.call_args_list:
            kwargs = c.kwargs
            assert kwargs["filters"].get("source") == "chat"
            assert "user_id" in kwargs["filters"]


class TestPatchGeminiParseResponse:
    """Tests for the Gemini null content guard monkey-patch."""

    def test_null_content_returns_empty_string(self):
        """When Gemini returns candidate with content=None, return empty string."""
        mock_module = MagicMock()
        mock_gemini_cls = MagicMock()
        original_parse = MagicMock(return_value="original result")
        mock_gemini_cls._parse_response = original_parse

        with patch.dict("sys.modules", {"mem0.llms.gemini": mock_module}):
            mock_module.GeminiLLM = mock_gemini_cls

            patch_gemini_parse_response()

            patched_method = mock_gemini_cls._parse_response
            assert patched_method is not original_parse

            response = MagicMock()
            candidate = MagicMock()
            candidate.content = None
            response.candidates = [candidate]

            result = patched_method(MagicMock(), response)
            assert result == ""

    def test_normal_response_delegates_to_original(self):
        """Normal responses with valid content delegate to original method."""
        mock_module = MagicMock()
        mock_gemini_cls = MagicMock()
        original_parse = MagicMock(return_value="parsed content")
        mock_gemini_cls._parse_response = original_parse

        with patch.dict("sys.modules", {"mem0.llms.gemini": mock_module}):
            mock_module.GeminiLLM = mock_gemini_cls

            patch_gemini_parse_response()
            patched_method = mock_gemini_cls._parse_response

            response = MagicMock()
            candidate = MagicMock()
            candidate.content = MagicMock()
            candidate.content.parts = [MagicMock()]
            response.candidates = [candidate]

            instance = MagicMock()
            patched_method(instance, response)
            original_parse.assert_called_once_with(instance, response)

    def test_empty_candidates_returns_empty_string(self):
        """Response with empty candidates list returns empty string."""
        mock_module = MagicMock()
        mock_gemini_cls = MagicMock()
        original_parse = MagicMock(return_value="original")
        mock_gemini_cls._parse_response = original_parse

        with patch.dict("sys.modules", {"mem0.llms.gemini": mock_module}):
            mock_module.GeminiLLM = mock_gemini_cls

            patch_gemini_parse_response()
            patched_method = mock_gemini_cls._parse_response

            response = MagicMock()
            response.candidates = []

            result = patched_method(MagicMock(), response)
            assert result == ""
