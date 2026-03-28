"""Claude Code session hooks for mem0-mcp-selfhosted.

Three entry points registered in pyproject.toml:
- mem0-hook-context  -> context_main()   (SessionStart)
- mem0-hook-stop     -> stop_main()      (Stop)
- mem0-install-hooks -> install_main()   (CLI installer)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from collections import deque
from pathlib import Path

from dotenv import load_dotenv

# Load .env early so _get_user_id() sees MEM0_USER_ID even when it's
# called before _get_memory().  load_dotenv(override=False) is the
# default — it never clobbers values already in os.environ.
load_dotenv()

# Hooks write JSON responses to stdout — logging must go to stderr
# so it never corrupts the hook response channel.
logging.basicConfig(stream=sys.stderr, format="%(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared initialization
# ---------------------------------------------------------------------------

_memory = None

_MAX_MEMORIES = 20
_MIN_USER_LEN = 20
_MIN_ASSISTANT_LEN = 50
_MAX_CONTENT_LEN = 4000
_RECENT_WINDOW = 20  # last ~10 exchanges (user+assistant pairs)


def _get_user_id() -> str:
    """Resolve user ID from MEM0_USER_ID env var, defaulting to ``'user'``."""
    return os.environ.get("MEM0_USER_ID", "user")


def _get_memory():
    """Lazy-initialize and cache a mem0 Memory instance with graph disabled.

    Graph is force-disabled for speed — hooks must complete within the
    Claude Code timeout (15s for context, 30s for stop).  The instance
    is cached in a module global; since each hook invocation is a
    separate process, this only initializes once.
    """
    global _memory
    if _memory is not None:
        return _memory

    # Force graph off — the hard os.environ set overrides any .env value
    # that load_dotenv() loaded at module init.
    os.environ["MEM0_ENABLE_GRAPH"] = "false"

    from mem0_mcp_selfhosted.config import build_config
    from mem0_mcp_selfhosted.server import register_providers

    config_dict, providers_info, _ = build_config()
    register_providers(providers_info)
    # patch_graph_sanitizer() skipped — graph is force-disabled in hooks,
    # so the relationship sanitizer modules are never invoked.

    from mem0 import Memory

    _memory = Memory.from_config(config_dict)
    return _memory


_HOOK_LOG = Path(tempfile.gettempdir()) / "mem0-hook-context.log"
_STOP_STATE_FILE = Path(tempfile.gettempdir()) / "mem0-stop-state.json"
_MIN_NEW_LINES = 6  # minimum new transcript lines required between saves for the same session


def _log_hook_event(hook: str, msg: str) -> None:
    """Append a timestamped line to the hook log file (best-effort)."""
    import datetime

    try:
        with open(_HOOK_LOG, "a", encoding="utf-8") as f:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts} [{hook}] {msg}\n")
    except OSError:
        pass


def _load_stop_state() -> dict:
    """Load per-session save-state from temp file (best-effort)."""
    try:
        if _STOP_STATE_FILE.exists():
            return json.loads(_STOP_STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save_stop_state(state: dict) -> None:
    """Persist per-session save-state to temp file (best-effort)."""
    try:
        _STOP_STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        pass


def _count_transcript_lines(transcript_path: str) -> int:
    """Count non-empty lines in transcript without full JSON parsing."""
    try:
        count = 0
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
    except OSError:
        return 0


def _output(data: dict) -> None:
    """Print JSON to stdout (the hook response channel)."""
    print(json.dumps(data))


def _nonfatal() -> dict:
    """Return the standard non-fatal / no-op hook response.

    Must return a **fresh** dict each time — callers may mutate it
    (e.g. adding ``additionalContext``).
    """
    return {"continue": True, "suppressOutput": True}


# ---------------------------------------------------------------------------
# Context Hook  (SessionStart)
# ---------------------------------------------------------------------------


def context_main() -> None:
    """SessionStart hook: inject cross-session memories as additionalContext."""
    _log_hook_event("context", "hook entry point reached")
    try:
        raw_stdin = sys.stdin.read()
        _log_hook_event("context", f"stdin length={len(raw_stdin)}")
        hook_input = json.loads(raw_stdin)
        _log_hook_event("context", f"parsed input keys={list(hook_input.keys())}")
        cwd = hook_input.get("cwd", "")
        project_name = Path(cwd).name if cwd else "project"
        if not project_name:
            project_name = "project"
        user_id = _get_user_id()
        _log_hook_event("context", f"project='{project_name}' user_id='{user_id}' cwd='{cwd}'")

        _log_hook_event("context", "initializing memory client...")
        mem = _get_memory()
        _log_hook_event("context", "memory client ready")

        # --- Multi-query search with deduplication ---
        from mem0_mcp_selfhosted.helpers import search_with_project

        seen_ids: set[str] = set()
        all_memories: list[dict] = []

        queries = [
            f"project context, architecture, conventions for {project_name}",
            f"recent session summary, decisions, key changes for {project_name}",
        ]

        for query in queries:
            _log_hook_event("context", f"searching: {query[:60]}...")
            results = search_with_project(mem, query, user_id, project_name, limit=15)
            _log_hook_event("context", f"  -> {len(results)} results")
            for r in results:
                mid = r.get("id")
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    all_memories.append(r)

        # Cap total injected memories
        all_memories = all_memories[:_MAX_MEMORIES]

        if not all_memories:
            _log_hook_event("context", f"no memories found for project '{project_name}'")
            _output(_nonfatal())
            return

        # Group by scope and format
        project_mems = [m for m in all_memories if m.get("scope") == "project"]
        global_mems = [m for m in all_memories if m.get("scope") == "global"]

        lines = ["# mem0 Cross-Session Memory\n"]
        i = 1
        if project_mems:
            lines.append(f"## Project: {project_name}")
            for m in project_mems:
                text = m.get("memory", m.get("text", ""))
                lines.append(f"{i}. {text}")
                i += 1
            lines.append("")
        if global_mems:
            lines.append("## Global")
            for m in global_mems:
                text = m.get("memory", m.get("text", ""))
                lines.append(f"{i}. {text}")
                i += 1

        _log_hook_event("context", f"injected {len(all_memories)} memories for project '{project_name}'")
        context_text = "\n".join(lines)
        response = _nonfatal()
        response["hookSpecificOutput"] = {
            "hookEventName": "SessionStart",
            "additionalContext": context_text,
        }
        _log_hook_event("context", f"outputting response with additionalContext ({len(context_text)} chars)")
        _output(response)

    except Exception as exc:
        import traceback
        _log_hook_event("context", f"FAILED: {exc}\n{traceback.format_exc()}")
        logger.debug("context_main failed", exc_info=True)
        _output(_nonfatal())


# ---------------------------------------------------------------------------
# Stop Hook
# ---------------------------------------------------------------------------


def _extract_content(content) -> str:
    """Extract plain text from a transcript content field.

    Claude Code transcripts use content blocks:
    ``[{"type": "text", "text": "..."}]``
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return " ".join(parts)
    return ""


def _read_recent_messages(transcript_path: str) -> list[tuple[str, str]]:
    """Read recent user/assistant messages from a JSONL transcript.

    Returns up to ``_RECENT_WINDOW`` ``(role, content)`` tuples in
    chronological order.  Uses a bounded deque so memory stays O(1)
    regardless of transcript length (which can reach ~900 KB).
    Content is truncated during parsing to avoid holding large
    assistant responses (tool results, file reads) in memory.
    """
    messages: deque[tuple[str, str]] = deque(maxlen=_RECENT_WINDOW)

    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Claude Code transcripts nest the message inside a "message" key:
            # {type: "user", message: {role: "user", content: [...]}}
            msg = entry.get("message", entry)
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue
            content = _extract_content(msg.get("content", ""))[:_MAX_CONTENT_LEN]
            if content:
                messages.append((role, content))

    return list(messages)


def stop_main() -> None:
    """Stop hook: save session summary to mem0."""
    _log_hook_event("stop", "hook entry point reached")
    try:
        raw_stdin = sys.stdin.read()
        _log_hook_event("stop", f"stdin length={len(raw_stdin)}")
        hook_input = json.loads(raw_stdin)
        _log_hook_event("stop", f"parsed input keys={list(hook_input.keys())}")

        # Infinite-loop guard: Claude Code sets this when re-entering
        if hook_input.get("stop_hook_active"):
            _log_hook_event("stop", "stop_hook_active guard — skipping")
            _output(_nonfatal())
            return

        session_id = hook_input.get("session_id", "")
        transcript_path = hook_input.get("transcript_path", "")
        cwd = hook_input.get("cwd", "")
        project_name = Path(cwd).name if cwd else "project"
        if not project_name:
            project_name = "project"

        _log_hook_event("stop", f"project='{project_name}' transcript='{transcript_path}' cwd='{cwd}'")

        # Missing / invalid transcript
        if not transcript_path or not Path(transcript_path).is_file():
            _log_hook_event("stop", f"no valid transcript — skipping")
            _output(_nonfatal())
            return

        recent = _read_recent_messages(transcript_path)
        _log_hook_event("stop", f"read {len(recent)} recent messages")

        # Skip short sessions — AND means we save when *either* side
        # contributed meaningful content (e.g. short question + long answer).
        user_total = sum(len(c) for r, c in recent if r == "user")
        asst_total = sum(len(c) for r, c in recent if r == "assistant")
        if user_total < _MIN_USER_LEN and asst_total < _MIN_ASSISTANT_LEN:
            _log_hook_event("stop", f"session too short (user={user_total}, asst={asst_total}) — skipping")
            _output(_nonfatal())
            return

        # Debounce: the Stop hook fires after *every* assistant turn, not only at
        # true session end.  For sessions we've already saved, require _MIN_NEW_LINES
        # of new transcript content before saving again.  First save is always allowed.
        line_count = _count_transcript_lines(transcript_path)
        stop_state = _load_stop_state()
        last_saved_lines = stop_state.get(session_id, 0)
        if last_saved_lines > 0 and (line_count - last_saved_lines) < _MIN_NEW_LINES:
            _log_hook_event(
                "stop",
                f"debounce: only {line_count - last_saved_lines} new lines since last save — skipping",
            )
            _output(_nonfatal())
            return

        # Build summary prompt with recent exchanges
        exchanges = []
        for role, content in recent:
            label = "User" if role == "user" else "Assistant"
            exchanges.append(f"[{label}]: {content}")

        summary = (
            f"Session summary for project '{project_name}':\n\n"
            + "\n\n".join(exchanges)
            + "\n\n"
            "Extract key decisions, solutions found, patterns discovered, "
            "configuration changes, and important context for future sessions."
        )
        _log_hook_event("stop", f"summary length={len(summary)} chars")

        _log_hook_event("stop", "initializing memory client...")
        mem = _get_memory()
        _log_hook_event("stop", "memory client ready")
        user_id = _get_user_id()

        from mem0_mcp_selfhosted.helpers import make_project_user_id

        project_uid = make_project_user_id(user_id, project_name)
        _log_hook_event("stop", f"calling mem.add (user_id={project_uid})...")

        mem.add(
            messages=[{"role": "user", "content": summary}],
            user_id=project_uid,
            infer=True,
            metadata={
                "source": "session-stop-hook",
                "session_id": session_id,
            },
        )

        # Update debounce state so the next turn doesn't re-save the same content
        stop_state[session_id] = line_count
        _save_stop_state(stop_state)

        _log_hook_event("stop", f"saved session for project '{project_name}' (user_id={project_uid})")
        _output(_nonfatal())

    except Exception as exc:
        logger.debug("stop_main failed", exc_info=True)
        _log_hook_event("stop", f"FAILED: {exc}")
        _output(_nonfatal())


# ---------------------------------------------------------------------------
# Install-Hooks CLI
# ---------------------------------------------------------------------------

_HOOK_CONTEXT_CMD = "mem0-hook-context"
_HOOK_STOP_CMD = "mem0-hook-stop"


def _has_hook(hooks_list: list, command: str) -> bool:
    """Check if a hook with the given command already exists.

    Searches both the current nested format and the legacy flat format::

        Nested:  [{"matcher": "...", "hooks": [{"type": "command", "command": "..."}]}]
        Legacy:  [{"matcher": "...", "command": "..."}]
    """
    for group in hooks_list:
        if not isinstance(group, dict):
            continue
        # Current nested format
        for handler in group.get("hooks") or []:
            if isinstance(handler, dict) and handler.get("command") == command:
                return True
        # Legacy flat format (pre-nested schema)
        if group.get("command") == command:
            return True
    return False


_HANDLER_KEYS = {"command", "timeout"}
_GROUP_KEYS = {"matcher"}


def _migrate_legacy_hooks(hooks_list: list) -> list:
    """Convert legacy flat-format hooks to the nested format.

    Flat entries (``{"command": "...", "timeout": ...}``) are converted to
    nested format (``{"hooks": [{"type": "command", ...}]}``).  Already-nested
    entries are kept as-is.  Non-dict entries are discarded.  Unknown keys are
    forwarded to preserve any extra properties the user may have set.
    """
    migrated = []
    for group in hooks_list:
        if not isinstance(group, dict):
            continue
        if "hooks" in group:
            # Already in nested format
            migrated.append(group)
        elif "command" in group:
            # Legacy flat format — convert, forwarding unknown keys to
            # group level so no user data is silently dropped.
            handler: dict = {"type": "command"}
            new_group: dict = {}
            for k, v in group.items():
                if k in _HANDLER_KEYS:
                    handler[k] = v
                elif k in _GROUP_KEYS:
                    new_group[k] = v
                else:
                    new_group[k] = v
            new_group["hooks"] = [handler]
            migrated.append(new_group)
        else:
            # Unknown format — preserve as-is
            migrated.append(group)
    return migrated


def install_main() -> None:
    """CLI: install mem0 hooks into .claude/settings.json."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="mem0-install-hooks",
        description="Install mem0 session hooks for Claude Code",
    )
    parser.add_argument(
        "--global",
        dest="global_install",
        action="store_true",
        help="Install to ~/.claude/settings.json instead of project directory",
    )
    parser.add_argument(
        "--project-dir",
        default=None,
        help="Project directory (defaults to CWD)",
    )
    args = parser.parse_args()

    if args.global_install:
        settings_dir = Path.home() / ".claude"
    else:
        project_dir = Path(args.project_dir) if args.project_dir else Path.cwd()
        if not project_dir.is_dir():
            print(f"Error: project directory does not exist: {project_dir}", file=sys.stderr)
            sys.exit(1)
        settings_dir = project_dir / ".claude"

    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_path = settings_dir / "settings.json"

    # Read existing settings (preserve everything)
    if settings_path.exists():
        try:
            with open(settings_path, encoding="utf-8") as f:
                settings = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: {settings_path} contains invalid JSON: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        settings = {}

    if not isinstance(settings.get("hooks"), dict):
        settings["hooks"] = {}

    hooks = settings["hooks"]

    # Migrate any legacy flat-format hooks to nested format
    for event_key in ("SessionStart", "Stop"):
        if isinstance(hooks.get(event_key), list):
            hooks[event_key] = _migrate_legacy_hooks(hooks[event_key])

    installed: list[str] = []
    skipped: list[str] = []

    # --- SessionStart hook ---
    if not isinstance(hooks.get("SessionStart"), list):
        hooks["SessionStart"] = []
    if _has_hook(hooks["SessionStart"], _HOOK_CONTEXT_CMD):
        skipped.append(f"SessionStart ({_HOOK_CONTEXT_CMD})")
    else:
        hooks["SessionStart"].append({
            "matcher": "startup|compact",
            "hooks": [{
                "type": "command",
                "command": _HOOK_CONTEXT_CMD,
                "timeout": 15000,
            }],
        })
        installed.append(f"SessionStart ({_HOOK_CONTEXT_CMD})")

    # --- Stop hook ---
    if not isinstance(hooks.get("Stop"), list):
        hooks["Stop"] = []
    if _has_hook(hooks["Stop"], _HOOK_STOP_CMD):
        skipped.append(f"Stop ({_HOOK_STOP_CMD})")
    else:
        hooks["Stop"].append({
            "hooks": [{
                "type": "command",
                "command": _HOOK_STOP_CMD,
                "timeout": 30000,
            }],
        })
        installed.append(f"Stop ({_HOOK_STOP_CMD})")

    # Atomic write: temp file + rename avoids truncated settings on crash
    fd, tmp_path = tempfile.mkstemp(dir=str(settings_dir), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, str(settings_path))
    except BaseException:
        os.unlink(tmp_path)
        raise

    # Report
    for hook in installed:
        print(f"Installed: {hook}")
    for hook in skipped:
        print(f"Already installed: {hook}")
    print(f"Settings: {settings_path}")
