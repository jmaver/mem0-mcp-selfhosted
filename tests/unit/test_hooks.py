"""Tests for hooks.py — Claude Code session hooks."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mem0_mcp_selfhosted import hooks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_output(func, stdin_data: str = "{}") -> dict:
    """Run a hook entry point with mocked stdin and capture stdout JSON."""
    captured = StringIO()
    with patch("sys.stdin", StringIO(stdin_data)), patch("sys.stdout", captured):
        func()
    return json.loads(captured.getvalue())


# ---------------------------------------------------------------------------
# 6.1  get_default_user_id (previously _get_user_id — now from helpers)
# ---------------------------------------------------------------------------


class TestGetDefaultUserIdInHooks:
    def test_dotenv_loaded_at_module_level(self):
        """load_dotenv() runs at module init, so MEM0_USER_ID from .env is visible."""
        import inspect

        source = inspect.getsource(hooks)
        # load_dotenv() should be called at module level, not just inside a function
        lines = source.split("\n")
        found_module_level_call = False
        for line in lines:
            stripped = line.strip()
            # Skip comments and function/class definitions
            if stripped.startswith("#") or stripped.startswith("def ") or stripped.startswith("class "):
                continue
            if "load_dotenv()" in stripped and not line.startswith("    "):
                found_module_level_call = True
                break
        assert found_module_level_call, "load_dotenv() must be called at module level"

    def test_context_main_uses_get_default_user_id(self):
        """context_main imports get_default_user_id from helpers (not _get_user_id)."""
        import inspect

        source = inspect.getsource(hooks)
        assert "get_default_user_id" in source, "hooks must use helpers.get_default_user_id"
        assert "_get_user_id" not in source, "_get_user_id was removed; use helpers.get_default_user_id"


# ---------------------------------------------------------------------------
# 6.2  _get_memory
# ---------------------------------------------------------------------------


class TestGetMemory:
    def test_caching_returns_same_instance(self):
        """_get_memory() returns the cached instance on repeated calls."""
        sentinel = MagicMock(name="Memory")
        with patch.object(hooks, "_memory", sentinel):
            assert hooks._get_memory() is sentinel

    def test_initializes_and_caches(self, monkeypatch):
        """_get_memory() initializes Memory.from_config and caches result."""
        fake_mem = MagicMock(name="FreshMemory")

        # monkeypatch auto-restores _memory after the test
        monkeypatch.setattr(hooks, "_memory", None)
        with (
            patch("mem0_mcp_selfhosted.config.build_config", return_value=({}, [])),
            patch("mem0_mcp_selfhosted.server.register_providers"),
            patch("mem0.Memory.from_config", return_value=fake_mem),
        ):
            result = hooks._get_memory()

        assert result is fake_mem
        # Verify the result was cached in the module global
        assert hooks._memory is fake_mem


# ---------------------------------------------------------------------------
# 6.3  context_main
# ---------------------------------------------------------------------------


class TestContextMain:
    def _make_stdin(self, **overrides):
        data = {
            "session_id": "sess-1",
            "cwd": "/home/user/myproject",
            "hook_event_name": "startup",
        }
        data.update(overrides)
        return json.dumps(data)

    def _get_additional_context(self, result):
        """Extract additionalContext from hookSpecificOutput."""
        return result.get("hookSpecificOutput", {}).get("additionalContext")

    def test_memories_found(self):
        """When search returns memories, additionalContext is included."""
        fake_results = {
            "results": [
                {"id": "m1", "memory": "Uses TypeScript with strict mode"},
                {"id": "m2", "memory": "Prefers pytest for testing"},
            ]
        }

        mock_mem = MagicMock()
        mock_mem.search.return_value = fake_results

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(hooks.context_main, self._make_stdin())

        assert result["continue"] is True
        assert result["suppressOutput"] is True
        ctx = self._get_additional_context(result)
        assert ctx is not None
        assert "TypeScript" in ctx
        assert "pytest" in ctx
        assert "# mem0 Cross-Session Memory" in ctx

    def test_no_memories_omits_additional_context(self):
        """When no memories found, additionalContext is absent."""
        mock_mem = MagicMock()
        mock_mem.search.return_value = {"results": []}

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(hooks.context_main, self._make_stdin())

        assert result["continue"] is True
        assert result["suppressOutput"] is True
        assert "hookSpecificOutput" not in result

    def test_deduplication_across_queries(self):
        """Duplicate memory IDs across searches are deduplicated."""
        mock_mem = MagicMock()
        # search_with_project calls mem.search twice per query (project + global),
        # and context_main runs 2 queries = 4 total search calls
        mock_mem.search.side_effect = [
            # Query 1, project-scoped
            {"results": [{"id": "m1", "memory": "fact one"}]},
            # Query 1, global
            {"results": [{"id": "m2", "memory": "fact two"}]},
            # Query 2, project-scoped
            {"results": [{"id": "m2", "memory": "fact two"}]},  # duplicate
            # Query 2, global
            {"results": [{"id": "m3", "memory": "fact three"}]},
        ]

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(hooks.context_main, self._make_stdin())

        ctx = self._get_additional_context(result)
        # m2 should appear only once
        assert ctx.count("fact two") == 1
        assert "fact one" in ctx
        assert "fact three" in ctx

    def test_results_as_list_format(self):
        """Handle mem0 search returning a plain list (not dict with 'results')."""
        mock_mem = MagicMock()
        mock_mem.search.return_value = [
            {"id": "m1", "memory": "plain list result"},
        ]

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(hooks.context_main, self._make_stdin())

        ctx = self._get_additional_context(result)
        assert "plain list result" in ctx

    def test_exception_returns_nonfatal(self):
        """Any exception produces a non-fatal response."""
        with patch.object(hooks, "_get_memory", side_effect=RuntimeError("boom")):
            result = _capture_output(hooks.context_main, self._make_stdin())

        assert result == {"continue": True, "suppressOutput": True}

    def test_max_memories_cap(self):
        """Results are capped at _MAX_MEMORIES."""
        mock_mem = MagicMock()
        many = [{"id": f"m{i}", "memory": f"fact {i}"} for i in range(30)]
        mock_mem.search.return_value = {"results": many}

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(hooks.context_main, self._make_stdin())

        ctx = self._get_additional_context(result)
        lines = [line for line in ctx.split("\n") if line and line[0].isdigit()]
        assert len(lines) == hooks._MAX_MEMORIES

    def test_empty_cwd_uses_project_fallback(self):
        """Empty cwd falls back to 'project' in search queries."""
        mock_mem = MagicMock()
        mock_mem.search.return_value = {
            "results": [
                {"id": "m1", "memory": "some fact"},
            ]
        }

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(hooks.context_main, self._make_stdin(cwd=""))

        # v3: user_id is inside the filters dict, not a top-level kwarg
        first_filters = mock_mem.search.call_args_list[0].kwargs["filters"]
        first_uid = first_filters["user_id"]
        assert "project" in first_uid
        assert self._get_additional_context(result) is not None


# ---------------------------------------------------------------------------
# _extract_content edge cases
# ---------------------------------------------------------------------------


class TestExtractContent:
    def test_plain_string(self):
        assert hooks._extract_content("hello world") == "hello world"

    def test_content_blocks_text_only(self):
        content = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        assert hooks._extract_content(content) == "hello world"

    def test_mixed_block_types_filters_non_text(self):
        """Non-text blocks (tool_use, tool_result) are silently ignored."""
        content = [
            {"type": "tool_use", "id": "t1", "name": "Read"},
            {"type": "text", "text": "the actual response"},
            {"type": "tool_result", "tool_use_id": "t1", "content": "file data"},
        ]
        assert hooks._extract_content(content) == "the actual response"

    def test_text_block_missing_text_key(self):
        """A block with type=text but no 'text' key returns empty string for that part."""
        content = [{"type": "text"}, {"type": "text", "text": "ok"}]
        assert hooks._extract_content(content) == " ok"

    def test_none_content(self):
        assert hooks._extract_content(None) == ""

    def test_integer_content(self):
        assert hooks._extract_content(42) == ""

    def test_empty_list(self):
        assert hooks._extract_content([]) == ""


# ---------------------------------------------------------------------------
# _read_recent_messages edge cases
# ---------------------------------------------------------------------------


class TestReadRecentMessages:
    def test_malformed_jsonl_lines_skipped(self, tmp_path):
        """Corrupted lines in transcript are silently skipped."""
        p = tmp_path / "transcript.jsonl"
        # Content padded to clear the _MIN_EXCHANGE_LEN noise filter.
        valid_user = "first valid message that is long enough to clear the filter"
        valid_asst = "second valid response that is also long enough to clear the filter"
        p.write_text(
            f'{{"role": "user", "content": "{valid_user}"}}\n'
            "THIS IS NOT JSON\n"
            f'{{"role": "assistant", "content": "{valid_asst}"}}\n'
            "{ALSO BROKEN\n"
        )
        result = hooks._read_recent_messages(str(p))
        assert len(result) == 2
        assert result[0] == ("user", valid_user)
        assert result[1] == ("assistant", valid_asst)

    def test_empty_file_returns_empty(self, tmp_path):
        """Empty transcript returns empty list."""
        p = tmp_path / "transcript.jsonl"
        p.write_text("")
        assert hooks._read_recent_messages(str(p)) == []

    def test_returns_recent_window(self, tmp_path):
        """Returns the last _RECENT_WINDOW messages in chronological order."""
        p = tmp_path / "transcript.jsonl"
        # Content must clear _MIN_EXCHANGE_LEN (noise filter) to exercise window logic.
        lines = [
            json.dumps({"role": "user", "content": "first user message with enough length to clear the filter"}),
            json.dumps({"role": "assistant", "content": "first assistant response with sufficient durable content"}),
            json.dumps({"role": "user", "content": "second user message with enough length to clear the filter"}),
            json.dumps({"role": "assistant", "content": "second assistant response with sufficient durable content"}),
        ]
        p.write_text("\n".join(lines))
        result = hooks._read_recent_messages(str(p))
        assert len(result) == 4
        assert result[-1] == ("assistant", "second assistant response with sufficient durable content")
        assert result[-2] == ("user", "second user message with enough length to clear the filter")

    def test_window_truncates_old_messages(self, tmp_path):
        """Transcripts longer than _RECENT_WINDOW are truncated to recent end."""
        p = tmp_path / "transcript.jsonl"
        total = hooks._RECENT_WINDOW + 10  # always more than the window
        lines = []
        # Pad each message so the bytes-truncation filter has something to keep.
        pad = "x" * 40
        for i in range(total):
            role = "user" if i % 2 == 0 else "assistant"
            lines.append(json.dumps({"role": role, "content": f"msg {i:03d} {pad}"}))
        p.write_text("\n".join(lines))
        result = hooks._read_recent_messages(str(p))
        assert len(result) == hooks._RECENT_WINDOW
        first_kept = total - hooks._RECENT_WINDOW
        assert result[0][1].startswith(f"msg {first_kept:03d}")
        assert result[-1][1].startswith(f"msg {total - 1:03d}")

    def test_skips_non_user_assistant_roles(self, tmp_path):
        """tool_use, tool_result, system roles are excluded from the window."""
        p = tmp_path / "transcript.jsonl"
        lines = [
            json.dumps({"role": "user", "content": "user request that is long enough to pass the noise filter"}),
            json.dumps({"role": "tool_use", "content": "tool call that is also long enough to pass the filter"}),
            json.dumps({"role": "tool_result", "content": "tool output that is similarly long enough"}),
            json.dumps({"role": "system", "content": "system prompt that is also long enough to pass"}),
            json.dumps({"role": "assistant", "content": "assistant response that is long enough to pass"}),
        ]
        p.write_text("\n".join(lines))
        result = hooks._read_recent_messages(str(p))
        assert len(result) == 2
        assert result[0] == ("user", "user request that is long enough to pass the noise filter")
        assert result[1] == ("assistant", "assistant response that is long enough to pass")


# ---------------------------------------------------------------------------
# _is_noise — pre-filter that drops transient session content before extraction
# ---------------------------------------------------------------------------


class TestIsNoise:
    def test_user_pings_are_noise(self):
        """Pure ping-style user messages are the only short user content dropped."""
        assert hooks._is_noise("user", "ok thanks") is True
        assert hooks._is_noise("user", "thanks") is True
        assert hooks._is_noise("user", "OK!") is True
        assert hooks._is_noise("user", "yes") is True

    def test_short_user_directive_kept(self):
        """Short user directives carry durable signal and must not be filtered."""
        # Regression: prior 40-char cutoff dropped these. Codex flagged it as
        # high-severity recall loss for SessionEnd extraction.
        assert hooks._is_noise("user", "use postgres 17") is False
        assert hooks._is_noise("user", "switch to dark mode") is False
        assert hooks._is_noise("user", "we deploy on AWS") is False

    def test_empty_content_is_noise(self):
        assert hooks._is_noise("user", "") is True
        assert hooks._is_noise("assistant", "   ") is True

    def test_short_assistant_message_is_noise(self):
        """Sub-4-char assistant messages are pings ('ok', 'yes')."""
        assert hooks._is_noise("assistant", "ok") is True
        assert hooks._is_noise("assistant", "yes") is True

    def test_assistant_tool_narration_is_noise(self):
        """Short assistant messages starting with narration prefixes are dropped."""
        assert hooks._is_noise("assistant", "Let me check the file structure first.") is True
        assert hooks._is_noise("assistant", "Running the test suite now to verify.") is True
        assert hooks._is_noise("assistant", "I'll look at the auth module quickly.") is True

    def test_long_assistant_narration_kept(self):
        """A 200+ char assistant message is durable enough to keep, even if it starts with narration."""
        long_msg = "Let me explain the architecture: " + ("the auth layer wraps everything " * 8)
        assert hooks._is_noise("assistant", long_msg) is False

    def test_user_narration_prefix_not_filtered(self):
        """User messages aren't subject to narration-prefix filtering — only ping detection."""
        msg = "let me know when the deploy finishes please thank you very much"
        assert hooks._is_noise("user", msg) is False

    def test_assistant_code_block_kept(self):
        """A code-fence answer (config snippet, final command) is durable signal."""
        # Regression: prior heuristic dropped any short pure-code block.
        # Codex flagged this as recall loss — code-only answers often contain
        # the durable artifact the hook is supposed to save.
        code = "```python\nprint('hi')\nx = 1\n```"
        assert hooks._is_noise("assistant", code) is False

    def test_durable_user_message_kept(self):
        """Realistic user content with project info is preserved."""
        msg = "we use postgres with prisma in this project, tests run via pytest -v"
        assert hooks._is_noise("user", msg) is False


# ---------------------------------------------------------------------------
# _dedup_against_existing — post-add semantic dedup using v3 search contract
# ---------------------------------------------------------------------------


class TestDedupAgainstExisting:
    def test_search_uses_top_k_not_limit(self):
        """v3 contract: search uses top_k. limit was the v0.3 spelling."""
        # Regression for codex finding: passing limit=3 silently disables the
        # bound on v3, letting the search return many more hits than intended
        # and pushing the SessionEnd hook closer to its budget.
        mock_mem = MagicMock()
        mock_mem.get.return_value = {"memory": "test fact", "created_at": "2026-01-01"}
        mock_mem.search.return_value = {"results": []}

        hooks._dedup_against_existing(mock_mem, ["mid1"], "user:proj")

        assert mock_mem.search.called
        kwargs = mock_mem.search.call_args.kwargs
        assert "top_k" in kwargs, f"v3 contract requires top_k; got {list(kwargs)}"
        assert "limit" not in kwargs
        assert kwargs["top_k"] == 3
        assert kwargs["filters"] == {"user_id": "user:proj"}


# ---------------------------------------------------------------------------
# 6.4  session_end_main
# ---------------------------------------------------------------------------


class TestSessionEndMain:
    def _make_transcript(self, tmp_path, messages):
        """Write a JSONL transcript file and return its path."""
        p = tmp_path / "transcript.jsonl"
        lines = [json.dumps(m) for m in messages]
        p.write_text("\n".join(lines))
        return str(p)

    def _make_stdin(self, tmp_path=None, transcript_path="", **overrides):
        data = {
            "session_id": "sess-1",
            "cwd": "/home/user/myproject",
            "transcript_path": transcript_path,
        }
        data.update(overrides)
        return json.dumps(data)

    def test_normal_transcript_saves_to_mem0(self, tmp_path):
        """Normal session with meaningful messages saves to mem0."""
        transcript = self._make_transcript(
            tmp_path,
            [
                {"role": "user", "content": "Please refactor the authentication module to use JWT tokens instead of sessions"},
                {
                    "role": "assistant",
                    "content": "I've refactored the auth module. The key changes are: replaced express-session with jsonwebtoken, added token refresh endpoint, and updated all middleware to validate JWT headers.",
                },
            ],
        )

        mock_mem = MagicMock()

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(
                hooks.session_end_main,
                self._make_stdin(transcript_path=transcript),
            )

        assert result["continue"] is True
        mock_mem.add.assert_called_once()
        call_kwargs = mock_mem.add.call_args
        assert call_kwargs.kwargs["infer"] is True
        assert call_kwargs.kwargs["metadata"]["source"] == "session-end-hook"
        assert call_kwargs.kwargs["metadata"]["session_id"] == "sess-1"
        # Summary includes both user and assistant exchanges
        summary = call_kwargs.kwargs["messages"][0]["content"]
        assert "[User]:" in summary
        assert "[Assistant]:" in summary
        assert "refactor" in summary.lower()

    def test_content_blocks_format(self, tmp_path):
        """Handles Claude Code's content block format."""
        transcript = self._make_transcript(
            tmp_path,
            [
                {"role": "user", "content": [{"type": "text", "text": "Implement a caching layer for the database queries with TTL support"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Done. Added Redis-backed cache with configurable TTL per query type. Default is 5 minutes."}]},
            ],
        )

        mock_mem = MagicMock()

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            _capture_output(
                hooks.session_end_main,
                self._make_stdin(transcript_path=transcript),
            )

        mock_mem.add.assert_called_once()
        messages = mock_mem.add.call_args.kwargs["messages"]
        summary_text = messages[0]["content"]
        assert "[User]:" in summary_text
        assert "[Assistant]:" in summary_text
        assert "caching layer" in summary_text.lower() or "Implement" in summary_text

    def test_short_session_skipped(self, tmp_path):
        """Short sessions (both messages below threshold) are skipped."""
        transcript = self._make_transcript(
            tmp_path,
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello! How can I help?"},
            ],
        )

        mock_mem = MagicMock()

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(
                hooks.session_end_main,
                self._make_stdin(transcript_path=transcript),
            )

        assert result["continue"] is True
        mock_mem.add.assert_not_called()

    def test_missing_transcript_skipped(self):
        """Missing transcript_path produces non-fatal response."""
        mock_mem = MagicMock()

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(
                hooks.session_end_main,
                self._make_stdin(transcript_path=""),
            )

        assert result["continue"] is True
        mock_mem.add.assert_not_called()

    def test_nonexistent_transcript_file_skipped(self):
        """Transcript path pointing to non-existent file is handled."""
        mock_mem = MagicMock()

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(
                hooks.session_end_main,
                self._make_stdin(transcript_path="/tmp/nonexistent_transcript.jsonl"),
            )

        assert result["continue"] is True
        mock_mem.add.assert_not_called()

    def test_exception_returns_nonfatal(self):
        """Any exception during session end produces a non-fatal response."""
        with patch.object(hooks, "_get_memory", side_effect=RuntimeError("boom")):
            result = _capture_output(
                hooks.session_end_main,
                json.dumps(
                    {
                        "session_id": "s",
                        "cwd": "/x",
                        "transcript_path": "/nonexistent",
                    }
                ),
            )

        assert result == {"continue": True, "suppressOutput": True}

    def test_multi_exchange_captures_session_arc(self, tmp_path):
        """Multiple exchanges are included in the summary for richer context."""
        transcript = self._make_transcript(
            tmp_path,
            [
                {"role": "user", "content": "Let's add authentication to the API using JWT tokens"},
                {"role": "assistant", "content": "I'll set up JWT authentication. First, I'll install jsonwebtoken and create the middleware."},
                {"role": "user", "content": "Good. Now add refresh token rotation for security"},
                {"role": "assistant", "content": "Added refresh token rotation. Tokens are stored in Redis with a 7-day TTL and single-use enforcement."},
            ],
        )

        mock_mem = MagicMock()

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            _capture_output(
                hooks.session_end_main,
                self._make_stdin(transcript_path=transcript),
            )

        mock_mem.add.assert_called_once()
        summary = mock_mem.add.call_args.kwargs["messages"][0]["content"]
        assert "JWT" in summary
        assert "refresh token" in summary.lower()
        assert "Redis" in summary

    def test_mem_add_raises_returns_nonfatal(self, tmp_path):
        """Exception during mem.add() is caught and produces non-fatal response."""
        transcript = self._make_transcript(
            tmp_path,
            [
                {"role": "user", "content": "Please refactor the authentication module to use JWT tokens instead of sessions"},
                {"role": "assistant", "content": "I've refactored the auth module. Replaced express-session with jsonwebtoken and added refresh endpoint."},
            ],
        )

        mock_mem = MagicMock()
        mock_mem.add.side_effect = RuntimeError("LLM timeout")

        with patch.object(hooks, "_get_memory", return_value=mock_mem):
            result = _capture_output(
                hooks.session_end_main,
                self._make_stdin(transcript_path=transcript),
            )

        assert result == {"continue": True, "suppressOutput": True}
        mock_mem.add.assert_called_once()


# ---------------------------------------------------------------------------
# 6.5  install_main
# ---------------------------------------------------------------------------


class TestInstallMain:
    def test_fresh_install(self, tmp_path):
        """Fresh install creates settings.json with both hook entries in nested format."""
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(project_dir)]):
            hooks.install_main()

        settings_path = project_dir / ".claude" / "settings.json"
        assert settings_path.exists()

        settings = json.loads(settings_path.read_text())
        assert "hooks" in settings

        # SessionStart: matcher group with nested hooks array
        assert len(settings["hooks"]["SessionStart"]) == 1
        ss_group = settings["hooks"]["SessionStart"][0]
        assert ss_group["matcher"] == "startup|compact"
        assert len(ss_group["hooks"]) == 1
        assert ss_group["hooks"][0]["type"] == "command"
        assert ss_group["hooks"][0]["command"] == "mem0-hook-context"
        assert ss_group["hooks"][0]["timeout"] == 15000

        # SessionEnd: matcher group with nested hooks array
        assert len(settings["hooks"]["SessionEnd"]) == 1
        stop_group = settings["hooks"]["SessionEnd"][0]
        assert len(stop_group["hooks"]) == 1
        assert stop_group["hooks"][0]["type"] == "command"
        assert stop_group["hooks"][0]["command"] == "mem0-hook-session-end"
        assert stop_group["hooks"][0]["timeout"] == 30000

    def test_idempotent_reinstall(self, tmp_path, capsys):
        """Running install twice doesn't create duplicate entries."""
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(project_dir)]):
            hooks.install_main()

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(project_dir)]):
            hooks.install_main()

        settings = json.loads((project_dir / ".claude" / "settings.json").read_text())
        assert len(settings["hooks"]["SessionStart"]) == 1
        assert len(settings["hooks"]["SessionEnd"]) == 1

        captured = capsys.readouterr()
        assert "Already installed" in captured.out

    def test_preserves_existing_settings(self, tmp_path):
        """Existing settings (permissions, etc.) are preserved."""
        project_dir = tmp_path / "proj"
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir(parents=True)

        existing = {
            "permissions": {"allow": ["Read", "Write"]},
            "mcpServers": {"mem0": {"command": "mem0-mcp-selfhosted"}},
        }
        (claude_dir / "settings.json").write_text(json.dumps(existing))

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(project_dir)]):
            hooks.install_main()

        settings = json.loads((claude_dir / "settings.json").read_text())
        assert settings["permissions"] == {"allow": ["Read", "Write"]}
        assert settings["mcpServers"] == {"mem0": {"command": "mem0-mcp-selfhosted"}}
        assert "hooks" in settings

    def test_global_install(self, tmp_path, monkeypatch):
        """--global installs to ~/.claude/settings.json."""
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        with patch("sys.argv", ["mem0-install-hooks", "--global"]):
            hooks.install_main()

        settings_path = fake_home / ".claude" / "settings.json"
        assert settings_path.exists()

        settings = json.loads(settings_path.read_text())
        assert "SessionStart" in settings["hooks"]
        assert "SessionEnd" in settings["hooks"]

    def test_default_project_dir_uses_cwd(self, tmp_path, monkeypatch):
        """Without --project-dir, install uses CWD."""
        monkeypatch.chdir(tmp_path)

        with patch("sys.argv", ["mem0-install-hooks"]):
            hooks.install_main()

        settings_path = tmp_path / ".claude" / "settings.json"
        assert settings_path.exists()
        settings = json.loads(settings_path.read_text())
        assert "SessionStart" in settings["hooks"]

    def test_corrupt_settings_json_exits_with_error(self, tmp_path, capsys):
        """Invalid JSON in existing settings.json produces user-friendly error."""
        project_dir = tmp_path / "proj"
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir(parents=True)
        (claude_dir / "settings.json").write_text("{broken json")

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(project_dir)]):
            with pytest.raises(SystemExit) as exc_info:
                hooks.install_main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "invalid JSON" in captured.err

    def test_existing_hooks_with_different_commands_not_matched(self, tmp_path):
        """Hooks with different commands don't prevent mem0 hooks from being added."""
        project_dir = tmp_path / "proj"
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir(parents=True)

        existing = {
            "hooks": {
                "SessionStart": [{"hooks": [{"type": "command", "command": "other-hook", "timeout": 5000}]}],
                "SessionEnd": [{"hooks": [{"type": "command", "command": "another-stop-hook", "timeout": 10000}]}],
            }
        }
        (claude_dir / "settings.json").write_text(json.dumps(existing))

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(project_dir)]):
            hooks.install_main()

        settings = json.loads((claude_dir / "settings.json").read_text())
        # Both the original and new matcher groups should be present
        assert len(settings["hooks"]["SessionStart"]) == 2
        assert len(settings["hooks"]["SessionEnd"]) == 2
        # Extract commands from nested hooks arrays
        commands = [handler["command"] for group in settings["hooks"]["SessionStart"] for handler in group.get("hooks", [])]
        assert "other-hook" in commands
        assert "mem0-hook-context" in commands

    def test_fresh_install_output_messages(self, tmp_path, capsys):
        """Fresh install prints 'Installed:' for both hooks and the settings path."""
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(project_dir)]):
            hooks.install_main()

        captured = capsys.readouterr()
        assert "Installed: SessionStart (mem0-hook-context)" in captured.out
        assert "Installed: SessionEnd (mem0-hook-session-end)" in captured.out
        assert "Settings:" in captured.out

    def test_malformed_hooks_structure_is_repaired(self, tmp_path):
        """Handles settings where 'hooks' or event keys are wrong types."""
        project_dir = tmp_path / "proj"
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir(parents=True)

        # hooks is null, SessionStart is a string — both invalid types
        existing = {"hooks": None}
        (claude_dir / "settings.json").write_text(json.dumps(existing))

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(project_dir)]):
            hooks.install_main()

        settings = json.loads((claude_dir / "settings.json").read_text())
        assert isinstance(settings["hooks"], dict)
        assert len(settings["hooks"]["SessionStart"]) == 1
        assert len(settings["hooks"]["SessionEnd"]) == 1

    def test_nonexistent_project_dir_exits_with_error(self, tmp_path, capsys):
        """--project-dir pointing to nonexistent path exits with error."""
        fake_dir = tmp_path / "does_not_exist"

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(fake_dir)]):
            with pytest.raises(SystemExit) as exc_info:
                hooks.install_main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "does not exist" in captured.err

    def test_legacy_flat_format_migrated_on_reinstall(self, tmp_path, capsys):
        """Old flat-format hooks are migrated to nested format without duplicates."""
        project_dir = tmp_path / "proj"
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir(parents=True)

        # Old flat format from previous package version
        existing = {
            "hooks": {
                "SessionStart": [{"command": "mem0-hook-context", "matcher": "startup|compact", "timeout": 15000}],
                "SessionEnd": [{"command": "mem0-hook-session-end", "timeout": 30000}],
            }
        }
        (claude_dir / "settings.json").write_text(json.dumps(existing))

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(project_dir)]):
            hooks.install_main()

        settings = json.loads((claude_dir / "settings.json").read_text())

        # Should have exactly 1 entry per event (migrated, not duplicated)
        assert len(settings["hooks"]["SessionStart"]) == 1
        assert len(settings["hooks"]["SessionEnd"]) == 1

        # Migrated to nested format
        ss = settings["hooks"]["SessionStart"][0]
        assert ss["matcher"] == "startup|compact"
        assert ss["hooks"][0]["type"] == "command"
        assert ss["hooks"][0]["command"] == "mem0-hook-context"
        assert ss["hooks"][0]["timeout"] == 15000

        stop = settings["hooks"]["SessionEnd"][0]
        assert stop["hooks"][0]["type"] == "command"
        assert stop["hooks"][0]["command"] == "mem0-hook-session-end"
        assert stop["hooks"][0]["timeout"] == 30000

        captured = capsys.readouterr()
        assert "Already installed" in captured.out

    def test_legacy_mixed_with_other_hooks_preserved(self, tmp_path):
        """Migration preserves non-mem0 hooks alongside legacy mem0 hooks."""
        project_dir = tmp_path / "proj"
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir(parents=True)

        existing = {
            "hooks": {
                "SessionStart": [
                    {"command": "other-hook", "timeout": 5000},
                    {"command": "mem0-hook-context", "matcher": "startup|compact", "timeout": 15000},
                ],
            }
        }
        (claude_dir / "settings.json").write_text(json.dumps(existing))

        with patch("sys.argv", ["mem0-install-hooks", "--project-dir", str(project_dir)]):
            hooks.install_main()

        settings = json.loads((claude_dir / "settings.json").read_text())
        # Both hooks migrated, no duplicates for mem0-hook-context
        assert len(settings["hooks"]["SessionStart"]) == 2
        commands = [handler["command"] for group in settings["hooks"]["SessionStart"] for handler in group.get("hooks", [])]
        assert "other-hook" in commands
        assert "mem0-hook-context" in commands

        # SessionEnd hook auto-installed since it wasn't in the original settings
        assert len(settings["hooks"]["SessionEnd"]) == 1
        assert settings["hooks"]["SessionEnd"][0]["hooks"][0]["command"] == "mem0-hook-session-end"
