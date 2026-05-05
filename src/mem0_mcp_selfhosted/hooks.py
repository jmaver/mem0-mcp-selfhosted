"""Claude Code session hooks for mem0-mcp-selfhosted.

Three entry points registered in pyproject.toml:
- mem0-hook-context       -> context_main()        (SessionStart)
- mem0-hook-session-end   -> session_end_main()    (SessionEnd)
- mem0-install-hooks      -> install_main()         (CLI installer)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

from dotenv import load_dotenv

# Load .env early so get_default_user_id() sees MEM0_USER_ID even when
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
_DEDUP_SIM_THRESHOLD = 0.88  # post-add: if a new memory matches an older one above this, drop the new one

# Pings that look like content but carry no extractable signal. Match is
# applied to user messages after lower-casing and stripping trailing punctuation.
_USER_PING_PHRASES = frozenset({
    "ok", "okay", "yes", "no", "y", "n", "thx", "thanks", "ty", "k",
    "sure", "got it", "cool", "nice", "great", "ok thanks", "thank you",
})

# Phrases that flag an assistant message as transient narration — not worth
# extracting from. Only applied when the message is also short (< 200 chars);
# longer narration usually has durable content past the prefix.
_NOISE_PREFIXES = (
    "let me ",
    "i'll ",
    "i will ",
    "checking ",
    "looking ",
    "running ",
    "let's ",
)


def _get_memory():
    """Lazy-initialize and cache a mem0 Memory instance.

    Hooks must complete within the Claude Code timeout budget (15s for
    context, 30s for session end).  The instance is cached in a module
    global; since each hook invocation is a separate process, this only
    initializes once.
    """
    global _memory
    if _memory is not None:
        return _memory

    from mem0_mcp_selfhosted.config import build_config
    from mem0_mcp_selfhosted.server import register_providers

    config_dict, providers_info = build_config()
    register_providers(providers_info)

    from mem0 import Memory

    _memory = Memory.from_config(config_dict)
    return _memory


_HOOK_LOG = Path(tempfile.gettempdir()) / "mem0-hook-context.log"


def _log_hook_event(hook: str, msg: str) -> None:
    """Append a timestamped line to the hook log file (best-effort)."""
    import datetime

    try:
        with open(_HOOK_LOG, "a", encoding="utf-8") as f:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts} [{hook}] {msg}\n")
    except OSError:
        pass


def _emit_profile(hook: str, phase: str, t0: float, **extras: object) -> None:
    """Emit a `profile.<phase>=<elapsed>s [k=v ...]` line for benchmark scrapers."""
    elapsed = time.perf_counter() - t0
    suffix = "".join(f" {k}={v}" for k, v in extras.items())
    _log_hook_event(hook, f"profile.{phase}={elapsed:.3f}s{suffix}")


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

        from mem0_mcp_selfhosted.helpers import get_default_user_id, search_with_project

        user_id = get_default_user_id()
        _log_hook_event("context", f"project='{project_name}' user_id='{user_id}' cwd='{cwd}'")

        _log_hook_event("context", "initializing memory client...")
        mem = _get_memory()
        _log_hook_event("context", "memory client ready")

        # search_with_project already deduplicates by ID across project + global,
        # so cross-query dedup happens here against the merged list.
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
# SessionEnd Hook
# ---------------------------------------------------------------------------


def _extract_content(content) -> str:
    """Extract plain text from a transcript content field.

    Claude Code transcripts use content blocks:
    ``[{"type": "text", "text": "..."}]``
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        return " ".join(parts)
    return ""


def _is_noise(role: str, content: str) -> bool:
    """Heuristic filter: drop transient narration before extraction.

    User messages are kept unless they're pure pings ("ok thanks") — short
    user directives ("use postgres 17") are durable signal. Assistant
    messages are dropped when they match a tool-call narration prefix and
    stay under 200 chars; code-fence blocks are kept (a code-only answer
    is often a config snippet or final command worth preserving).
    """
    stripped = content.strip()
    if not stripped:
        return True
    if role == "user":
        normalized = stripped.lower().rstrip(".!?")
        return normalized in _USER_PING_PHRASES
    if len(stripped) < 4:
        return True
    lower = stripped.lower()
    if any(lower.startswith(p) for p in _NOISE_PREFIXES) and len(stripped) < 200:
        return True
    return False


def _dedup_against_existing(mem, added_ids: list[str], project_uid: str) -> int:
    """Delete newly-added memories that semantically duplicate older ones.

    mem0's hash-based dedup catches verbatim duplicates only.  After an add,
    we re-search each new memory's text against the project scope; if an older
    memory matches above ``_DEDUP_SIM_THRESHOLD``, we drop the new one.
    Returns the count of memories deleted.
    """
    if not added_ids:
        return 0

    deleted = 0
    for mid in added_ids:
        try:
            new_mem = mem.get(mid)
        except Exception:
            continue
        if not new_mem:
            continue
        text = new_mem.get("memory") or new_mem.get("text") or ""
        if not text:
            continue
        try:
            results = mem.search(query=text, filters={"user_id": project_uid}, top_k=3)
        except Exception:
            continue
        hits = results.get("results", []) if isinstance(results, dict) else results
        for hit in hits or []:
            hit_id = hit.get("id")
            if hit_id == mid:
                continue
            score = hit.get("score") or 0.0
            hit_created = hit.get("created_at", "")
            new_created = new_mem.get("created_at", "")
            # Only drop the new one if the match is older (avoid mutual deletion)
            if score >= _DEDUP_SIM_THRESHOLD and hit_created and hit_created < new_created:
                try:
                    mem.delete(mid)
                    deleted += 1
                except Exception:
                    pass
                break
    return deleted


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
            if content and not _is_noise(role, content):
                messages.append((role, content))

    return list(messages)


def session_end_main() -> None:
    """SessionEnd hook: save session summary to mem0."""
    t_start = time.perf_counter()
    _log_hook_event("session_end", "hook entry point reached")
    try:
        t0 = time.perf_counter()
        raw_stdin = sys.stdin.read()
        hook_input = json.loads(raw_stdin)
        _log_hook_event("session_end", f"stdin length={len(raw_stdin)}")
        _log_hook_event("session_end", f"parsed input keys={list(hook_input.keys())}")
        _emit_profile("session_end", "stdin_parse", t0)

        session_id = hook_input.get("session_id", "")
        transcript_path = hook_input.get("transcript_path", "")
        cwd = hook_input.get("cwd", "")
        project_name = Path(cwd).name if cwd else "project"
        if not project_name:
            project_name = "project"

        _log_hook_event("session_end", f"project='{project_name}' transcript='{transcript_path}' cwd='{cwd}'")

        # Missing / invalid transcript
        if not transcript_path or not Path(transcript_path).is_file():
            _log_hook_event("session_end", "no valid transcript — skipping")
            _emit_profile("session_end", "total", t_start, arm="no_transcript")
            _output(_nonfatal())
            return

        t0 = time.perf_counter()
        recent = _read_recent_messages(transcript_path)
        _log_hook_event("session_end", f"read {len(recent)} recent messages")
        _emit_profile("session_end", "transcript_read", t0, n=len(recent))

        # Skip short sessions — AND means we save when *either* side
        # contributed meaningful content (e.g. short question + long answer).
        user_total = sum(len(c) for r, c in recent if r == "user")
        asst_total = sum(len(c) for r, c in recent if r == "assistant")
        if user_total < _MIN_USER_LEN and asst_total < _MIN_ASSISTANT_LEN:
            _log_hook_event("session_end", f"session too short (user={user_total}, asst={asst_total}) — skipping")
            _emit_profile("session_end", "total", t_start, arm="too_short")
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
            "Extract ONLY durable knowledge worth recalling in a future session, "
            "as one memory per fact. Each memory must fit one of these categories:\n"
            "  - USER: role, expertise, recurring preferences\n"
            "  - FEEDBACK: corrections or validated approaches the user gave, with the reason\n"
            "  - PROJECT: architecture decisions, conventions, constraints, deadlines (with motivation)\n"
            "  - REFERENCE: pointers to external systems, dashboards, repos, channels\n\n"
            "DO NOT extract:\n"
            "  - one-time choices (e.g. 'user chose Option 3')\n"
            "  - in-progress TODOs, action items, or 'next step' notes\n"
            "  - paths under /tmp, /var/folders, or other ephemeral locations\n"
            "  - debugging state, error messages being investigated, transient diagnostics\n"
            "  - session-procedural facts (e.g. 'user needs to relaunch the CLI')\n"
            "  - facts already obvious from reading the code or git history\n"
            "  - verbose narration of what was done; only the durable conclusion matters\n\n"
            "Prefer fewer, higher-signal memories over many shallow ones."
        )
        _log_hook_event("session_end", f"summary length={len(summary)} chars")

        _log_hook_event("session_end", "initializing memory client...")
        t0 = time.perf_counter()
        mem = _get_memory()
        _log_hook_event("session_end", "memory client ready")
        _emit_profile("session_end", "memory_init", t0)

        from mem0_mcp_selfhosted.helpers import get_default_user_id, make_project_user_id

        user_id = get_default_user_id()
        project_uid = make_project_user_id(user_id, project_name)
        _log_hook_event("session_end", f"calling mem.add (user_id={project_uid})...")

        t0 = time.perf_counter()
        add_result = mem.add(
            messages=[{"role": "user", "content": summary}],
            user_id=project_uid,
            infer=True,
            metadata={
                "source": "session-end-hook",
                "session_id": session_id,
            },
        )
        _emit_profile("session_end", "mem_add", t0)

        # Collect IDs of newly-added memories for the post-add dedup pass.
        events = add_result.get("results", []) if isinstance(add_result, dict) else (add_result or [])
        added_ids: list[str] = []
        for e in events:
            if not isinstance(e, dict) or e.get("event") != "ADD":
                continue
            mid = e.get("id")
            if isinstance(mid, str) and mid:
                added_ids.append(mid)
        _log_hook_event("session_end", f"mem.add returned {len(added_ids)} new memories")

        t0 = time.perf_counter()
        dropped = _dedup_against_existing(mem, added_ids, project_uid)
        _emit_profile("session_end", "dedup_post", t0)
        if dropped:
            _log_hook_event("session_end", f"dedup dropped {dropped}/{len(added_ids)} as semantic duplicates")

        _log_hook_event("session_end", f"saved session for project '{project_name}' (user_id={project_uid})")
        _emit_profile("session_end", "total", t_start)
        _output(_nonfatal())

    except Exception as exc:
        logger.debug("session_end_main failed", exc_info=True)
        _log_hook_event("session_end", f"FAILED: {exc}")
        _emit_profile("session_end", "total", t_start, arm="exception")
        _output(_nonfatal())


# ---------------------------------------------------------------------------
# Install-Hooks CLI
# ---------------------------------------------------------------------------

_HOOK_CONTEXT_CMD = "mem0-hook-context"
_HOOK_SESSION_END_CMD = "mem0-hook-session-end"


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
    for event_key in ("SessionStart", "Stop", "SessionEnd"):
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
        hooks["SessionStart"].append(
            {
                "matcher": "startup|compact",
                "hooks": [
                    {
                        "type": "command",
                        "command": _HOOK_CONTEXT_CMD,
                        "timeout": 15000,
                    }
                ],
            }
        )
        installed.append(f"SessionStart ({_HOOK_CONTEXT_CMD})")

    # --- SessionEnd hook ---
    if not isinstance(hooks.get("SessionEnd"), list):
        hooks["SessionEnd"] = []
    if _has_hook(hooks["SessionEnd"], _HOOK_SESSION_END_CMD):
        skipped.append(f"SessionEnd ({_HOOK_SESSION_END_CMD})")
    else:
        hooks["SessionEnd"].append(
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": _HOOK_SESSION_END_CMD,
                        "timeout": 30000,
                    }
                ],
            }
        )
        installed.append(f"SessionEnd ({_HOOK_SESSION_END_CMD})")

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
