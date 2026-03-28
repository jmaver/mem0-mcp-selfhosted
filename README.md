# mem0-mcp-selfhosted

***Note:*** This is a fork of [elvismdev/mem0-mcp-selfhosted](https://github.com/elvismdev/mem0-mcp-selfhosted).

**Changes from upstream:**
- **Claude Code session hooks** — `SessionStart` injects relevant memories as context on startup; `SessionEnd` saves a session summary to mem0 on exit (automatic, no manual tool calls needed)
- **Project-scoped memory isolation** — memories are namespaced per working directory via `user_id` encoding; hooks auto-scope by CWD
- **OAT token self-refresh** — proactively refreshes Claude Code's OAuth token before expiry so long-running sessions survive token rotation
- **Custom Anthropic LLM provider** — handles OAT authentication headers, structured JSON outputs via `output_config`, and tool-call parsing inside `mem0ai`
- **Split-model graph router** (`gemini_split`) — routes entity extraction to Gemini (85.4% accuracy, cheaper) and contradiction detection to Claude (100% accuracy)
- **Neo4j relationship sanitizer fix** — monkey-patches `mem0ai`'s `sanitize_relationship_for_cypher()` to handle hyphens and leading-digit edge cases
- **`safe_bulk_delete()` workaround** — explicitly cleans Neo4j nodes after `memory.delete()`, working around mem0ai bug #3245
- **LM Studio / local Ollama support** — fully local setup with no cloud dependencies

<a href="https://glama.ai/mcp/servers/elvismdev/mem0-mcp-selfhosted"><img width="380" height="200" src="https://glama.ai/mcp/servers/elvismdev/mem0-mcp-selfhosted/badge?v=1" alt="mem0-mcp-selfhosted MCP server" /></a>

Self-hosted [mem0](https://github.com/mem0ai/mem0) MCP server for Claude Code. Run a complete memory server against self-hosted Qdrant + Neo4j + Ollama, with your choice of Anthropic (Claude) or Ollama as the main LLM.

Uses the `mem0ai` package directly as a library, supports both Claude's OAT token and fully local Ollama setups, and exposes 11 MCP tools for full memory management.

## Prerequisites

| Service | Required | Purpose |
|---------|----------|---------|
| **Qdrant** | Yes | Vector memory storage and search |
| **Ollama** | Yes | Embedding generation (`bge-m3`) and optionally local LLM |
| **Neo4j 5+** | Optional | Knowledge graph (entity relationships) |
| **Google API Key** | Optional | Required only for `gemini`/`gemini_split` graph providers |

Python >= 3.10 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

> **Authentication:** The default setup uses Claude (Anthropic) as the LLM for fact extraction. No API key needed, the server automatically uses your Claude Code session token. For fully local setups, set `MEM0_PROVIDER=ollama`. See [Authentication](#authentication) for advanced options.

## Quick Start

### Default (Anthropic)

Add the MCP server globally (available across all projects):

```bash
claude mcp add --scope user --transport stdio mem0 \
  --env MEM0_USER_ID=your-user-id \
  -- uvx --from git+https://github.com/elvismdev/mem0-mcp-selfhosted.git mem0-mcp-selfhosted
```

All defaults work out of the box: Qdrant on `localhost:6333`, Ollama embeddings on `localhost:11434` with `bge-m3` (1024 dims). Override any default via `--env` (see [Configuration](#configuration)).

`uvx` automatically downloads, installs, and runs the server in an isolated environment, no manual installation needed. Claude Code launches it on demand when the MCP connection starts.

The server auto-reads your OAT token from `~/.claude/.credentials.json`, no manual token configuration needed.

### Fully Local (Ollama)

For a fully local setup with no cloud dependencies, use Ollama for both the main LLM and embeddings:

```bash
claude mcp add --scope user --transport stdio mem0 \
  --env MEM0_PROVIDER=ollama \
  --env MEM0_LLM_MODEL=qwen3:14b \
  --env MEM0_USER_ID=your-user-id \
  -- uvx --from git+https://github.com/elvismdev/mem0-mcp-selfhosted.git mem0-mcp-selfhosted
```

`MEM0_PROVIDER=ollama` cascades to both the main LLM and graph LLM providers. Same infrastructure defaults apply (Qdrant on `localhost:6333`, `bge-m3` embeddings). Per-service overrides (e.g. `MEM0_LLM_URL`, `MEM0_EMBED_URL`) still work when needed.

Or add it to a single project by creating `.mcp.json` in the project root:

```json
{
  "mcpServers": {
    "mem0": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/elvismdev/mem0-mcp-selfhosted.git", "mem0-mcp-selfhosted"],
      "env": {
        "MEM0_PROVIDER": "ollama",
        "MEM0_LLM_MODEL": "qwen3:14b",
        "MEM0_USER_ID": "your-user-id"
      }
    }
  }
}
```

### Try It

Restart Claude Code, then:

```
> Search my memories for TypeScript preferences
> Remember that I prefer Hatch for Python packaging
> Show me all entities in my knowledge graph
```

## CLAUDE.md Integration

Add these rules to your project's `CLAUDE.md` (or `~/.claude/CLAUDE.md` for global use) so Claude Code proactively uses memory tools throughout the session:

```markdown
# MCP Servers

- **mem0**: Persistent memory across sessions. Always pass `project` parameter with the current project directory name (e.g. `project="my-project"`). Use `project="global"` only for cross-project memories. At the start of each session, `search_memories` for relevant context before asking the user to re-explain anything. Use `add_memory` whenever you discover project architecture, coding conventions, debugging insights, key decisions, or user preferences. Use `update_memory` when prior context changes. Save information like: "This project uses PostgreSQL with Prisma", "Tests run with pytest -v", "Auth uses JWT validated in middleware". When in doubt, save it, future sessions benefit from over-remembering.
```

This gives Claude Code behavioral instructions to actively search and save memories during the session. For best results, combine with [Claude Code Hooks](#claude-code-hooks), the CLAUDE.md rules tell Claude *how to use* memory tools mid-session, while hooks handle the *automatic* injection and saving at session boundaries.

## Claude Code Hooks

Session hooks automate memory at session boundaries, injecting memories on startup and saving summaries on exit. This happens automatically without manual tool calls.

| Hook | Event | What it does |
|------|-------|--------------|
| `mem0-hook-context` | SessionStart (`startup`, `compact`) | Searches mem0 for project-relevant memories and injects them as `additionalContext` |
| `mem0-hook-session-end` | SessionEnd | Reads the last ~10 user/assistant exchanges from the transcript and saves a summary to mem0 via `infer=True` |

Both hooks are non-fatal, if mem0 is unreachable or any error occurs, Claude Code continues normally.

### Install

Install hooks into your project:

```bash
mem0-install-hooks
```

Or install globally (all projects):

```bash
mem0-install-hooks --global
```

This adds the hook entries to `.claude/settings.json`. The installer is idempotent, running it twice won't create duplicates.

### How it works

**On session start**, the context hook searches mem0 with two queries (project architecture + recent session summaries). Each query runs two searches — project-scoped and global — and results are deduplicated by memory ID and grouped under `## Project: <name>` and `## Global` headings. These are injected via the hook's `additionalContext` response field.

**On session end**, the SessionEnd hook fires once when the session terminates. It reads the JSONL transcript, extracts the last ~10 user/assistant messages (a sliding window via bounded deque), builds a summary prompt, and calls `memory.add(infer=True)` to extract atomic facts. Session summaries are saved as project-scoped memories. Graph is force-disabled in hooks to stay within the 60s timeout budget.

### Entry points

| Command | Function | Registered in `pyproject.toml` |
|---------|----------|-------------------------------|
| `mem0-hook-context` | `hooks:context_main` | SessionStart hook |
| `mem0-hook-session-end` | `hooks:session_end_main` | SessionEnd hook |
| `mem0-install-hooks` | `hooks:install_main` | CLI installer |

### Hooks + CLAUDE.md

Hooks and CLAUDE.md are complementary layers that work best together:

| Layer | Role | When |
|-------|------|------|
| **Hooks** | Automated data flow, injects stored memories on startup, saves session summaries on exit | Session boundaries (SessionStart/SessionEnd) |
| **CLAUDE.md** | Behavioral instructions, tells Claude to actively search and save memories during the session | Throughout the session |

Hooks alone give you passive recall (memories appear at startup) and passive saving (summaries saved at exit). CLAUDE.md instructions add active mid-session behavior, Claude searches for relevant memories when encountering new topics, and saves important discoveries immediately rather than waiting for session end.

For the best experience, use both. Hooks ensure memories flow in and out automatically at session boundaries, while CLAUDE.md ensures Claude actively engages with memory tools during the session.

## Project-Scoped Memory

Memories are isolated per project using `user_id` encoding. When you pass `project="my-app"`, the memory is stored under `user_id:my-app` (e.g. `jmaver:my-app`). Global memories use the bare `user_id` (e.g. `jmaver`).

| Scope | user_id | When to use |
|-------|---------|-------------|
| **Project** | `jmaver:my-app` | Default. Architecture, conventions, decisions specific to this project |
| **Global** | `jmaver` | Cross-project preferences, tools, environment setup |

### How it works

- **Saving** defaults to project-scoped. Pass `project="global"` to save globally.
- **Searching** with a project returns both project-scoped and global results, but excludes other projects. Results are tagged with `scope: "project"` or `scope: "global"`.
- **Hooks** automatically scope by the current working directory — no manual `project` param needed.
- **Existing memories** (created before project scoping) live under the bare `user_id` and are treated as global.

### MCP tool behavior

| Tool | project param | Behavior |
|------|--------------|----------|
| `add_memory` | Required | Stores under `user_id:project` (or bare `user_id` if `"global"`) |
| `search_memories` | Required | Two searches (project + global), merged and deduplicated |
| `get_memories` | Required | Lists memories for one scope |
| `delete_all_memories` | Required | Deletes within the specified scope |
| `update_memory` | Not needed | Operates by memory ID |
| `delete_memory` | Not needed | Operates by memory ID |
| `get_memory` | Not needed | Operates by memory ID |

## Authentication

The server resolves an Anthropic token using a prioritized fallback chain:

| Priority | Source | Details |
|----------|--------|---------|
| 1 | `MEM0_ANTHROPIC_TOKEN` env var | Explicit, user-controlled |
| 2 | `~/.claude/.credentials.json` | Auto-reads Claude Code's OAT token (zero-config) |
| 3 | `ANTHROPIC_API_KEY` env var | Standard pay-per-use API key |
| 4 | Disabled | Warns and disables Anthropic LLM features |

**In Claude Code, priority 2 always wins**, the credentials file exists as long as you're logged in. This means `ANTHROPIC_API_KEY` (priority 3) is never reached. To override the OAT token in Claude Code, use `MEM0_ANTHROPIC_TOKEN` (priority 1). `ANTHROPIC_API_KEY` is only useful for non-Claude-Code deployments (Docker, CI, standalone).

**OAT tokens** (`sk-ant-oat...`) use your Claude subscription. The server automatically detects the token type and configures the SDK accordingly. OAT tokens are automatically refreshed before expiry: the server proactively checks the token lifetime and refreshes via the Anthropic OAuth endpoint when nearing expiry (default: 30 minutes). On authentication failures, a 3-step defensive strategy kicks in, piggybacking on Claude Code's credentials file, self-refreshing via OAuth, and wait-and-retry, so long-running sessions survive token rotation seamlessly.

**API keys** (`sk-ant-api...`) use standard pay-per-use billing.

## Tools

### Memory Tools (9 core)

| Tool | Description |
|------|-------------|
| `add_memory` | Store text or conversation history as memories. Requires `project`. Supports `enable_graph`, `infer`, `metadata`. |
| `search_memories` | Semantic search across project + global memories. Requires `project`. Supports `filters`, `threshold`, `rerank`, `enable_graph`. |
| `get_memories` | List memories for a project scope. Requires `project`. Supports `limit` and scope filters. |
| `get_memory` | Fetch a single memory by UUID. |
| `update_memory` | Replace memory text. Re-embeds and re-indexes in Qdrant. |
| `delete_memory` | Delete a single memory by UUID. |
| `delete_all_memories` | Bulk-delete all memories in a scope. Requires `project`. |
| `list_entities` | List users/agents/runs with memory counts. Uses Qdrant Facet API. |
| `delete_entities` | Cascade-delete an entity and all its memories. |

### Graph Tools

| Tool | Description |
|------|-------------|
| `search_graph` | Search Neo4j entities by name substring. Returns entities + outgoing relationships. |
| `get_entity` | Get all relationships for an entity (bidirectional: incoming + outgoing). |

### Prompt

The server registers a `memory_assistant` MCP prompt that provides Claude with a quick-start guide for using the memory tools effectively.

### Parameters

All tools use Pydantic `Annotated[type, Field(description=...)]` for self-documenting parameter schemas. Common patterns:

- **`project`** (required on most tools) scopes memories per project. Use the directory name (e.g. `"my-app"`) or `"global"` for cross-project memories
- **`user_id`** defaults to `MEM0_USER_ID` env var when not provided
- **`enable_graph`** overrides the default `MEM0_ENABLE_GRAPH` per-call
- **`filters`** supports structured operators: `{"key": {"eq": "value"}}`, `{"AND": [...]}`
- All responses are JSON strings via `json.dumps(result, ensure_ascii=False)`

## Configuration

All configuration is via environment variables. Create a `.env` file or set them in your MCP config.

### Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `MEM0_ANTHROPIC_TOKEN` | -- | Anthropic OAT or API token (priority 1) |
| `ANTHROPIC_API_KEY` | -- | Standard Anthropic API key (priority 3) |
| `MEM0_OAT_HEADERS` | `auto` | OAT identity headers: `auto` or `none` |
| `MEM0_OAT_REFRESH_THRESHOLD_SECONDS` | `1800` | Seconds before expiry to trigger proactive OAT token refresh |

### LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `MEM0_PROVIDER` | `anthropic` | Top-level provider (`anthropic` or `ollama`). Cascades to `MEM0_LLM_PROVIDER` and `MEM0_GRAPH_LLM_PROVIDER` when those are not set. Does **not** affect `MEM0_EMBED_PROVIDER`. |
| `MEM0_LLM_PROVIDER` | _(MEM0_PROVIDER)_ | Main LLM provider: `anthropic` or `ollama`. Inherits from `MEM0_PROVIDER` when not set. |
| `MEM0_OLLAMA_URL` | `http://localhost:11434` | Shared Ollama base URL. Cascades to `MEM0_LLM_URL`, `MEM0_EMBED_URL`, and `MEM0_GRAPH_LLM_URL` when those are not set. |
| `MEM0_LLM_MODEL` | _(per-provider)_ | Model for the selected LLM provider. Defaults to `claude-opus-4-6` for Anthropic, `qwen3:14b` for Ollama |
| `MEM0_LLM_URL` | _(cascades)_ | Ollama base URL for the main LLM. Cascades: `MEM0_LLM_URL` → `MEM0_OLLAMA_URL` → `http://localhost:11434`. Only used when `MEM0_LLM_PROVIDER=ollama` |
| `MEM0_LLM_MAX_TOKENS` | `16384` | Max tokens for LLM responses (Anthropic only) |
| `MEM0_GRAPH_LLM_PROVIDER` | _(MEM0_PROVIDER)_ | Graph LLM provider (`anthropic`, `anthropic_oat`, `ollama`, `gemini`, `gemini_split`). Inherits from `MEM0_PROVIDER` when not set. |
| `MEM0_GRAPH_LLM_URL` | _(cascades)_ | Ollama base URL for graph LLM. Cascades: `MEM0_GRAPH_LLM_URL` → `MEM0_LLM_URL` → `MEM0_OLLAMA_URL` → `http://localhost:11434` |
| `MEM0_GRAPH_LLM_MODEL` | _(varies)_ | Graph model. Inherits `MEM0_LLM_MODEL` for anthropic/ollama; defaults to `gemini-2.5-flash-lite` for gemini/gemini_split |
| `GOOGLE_API_KEY` | -- | Google API key (required for `gemini`/`gemini_split` graph providers) |
| `MEM0_GRAPH_CONTRADICTION_LLM_PROVIDER` | `anthropic` | Contradiction LLM provider in `gemini_split` mode (`anthropic`, `anthropic_oat`, `ollama`) |
| `MEM0_GRAPH_CONTRADICTION_LLM_MODEL` | _(provider-aware)_ | Contradiction model in `gemini_split` mode. Defaults to `claude-opus-4-6` for `anthropic`/`anthropic_oat` providers; inherits `MEM0_LLM_MODEL` for others. |
| `MEM0_OLLAMA_KEEP_ALIVE` | `30m` | How long Ollama keeps the model in VRAM between calls (e.g., `1h`, `5m`). Prevents model unload during multi-call graph pipelines |
| `MEM0_OLLAMA_THINK` | `false` | Set to `true` to re-enable qwen3 thinking mode (disabled by default to prevent `<think>` + `format:"json"` collision) |

### Embedder

| Variable | Default | Description |
|----------|---------|-------------|
| `MEM0_EMBED_PROVIDER` | `ollama` | Embedding provider (`ollama` or `openai`) |
| `MEM0_EMBED_MODEL` | `bge-m3` | Embedding model name |
| `MEM0_EMBED_URL` | _(cascades)_ | Ollama URL for embeddings. Cascades: `MEM0_EMBED_URL` → `MEM0_OLLAMA_URL` → `http://localhost:11434` |
| `MEM0_EMBED_DIMS` | `1024` | Embedding vector dimensions |

### Vector Store (Qdrant)

| Variable | Default | Description |
|----------|---------|-------------|
| `MEM0_QDRANT_URL` | `http://localhost:6333` | Qdrant REST API URL |
| `MEM0_QDRANT_API_KEY` | -- | Qdrant API key (for Qdrant Cloud) |
| `MEM0_QDRANT_ON_DISK` | `false` | Store vectors on disk (reduces RAM, slower search) |
| `MEM0_QDRANT_TIMEOUT` | _(client default)_ | Qdrant REST API timeout in seconds (e.g., `30`). Only set if you hit `ReadTimeout` during collection operations |
| `MEM0_COLLECTION` | `mem0_mcp_selfhosted` | Qdrant collection name |

### Graph Store (Neo4j)

| Variable | Default | Description |
|----------|---------|-------------|
| `MEM0_ENABLE_GRAPH` | `false` | Enable graph memory (entity extraction to Neo4j) |
| `MEM0_NEO4J_URL` | `bolt://127.0.0.1:7687` | Neo4j Bolt endpoint |
| `MEM0_NEO4J_USER` | `neo4j` | Neo4j username |
| `MEM0_NEO4J_PASSWORD` | `mem0graph` | Neo4j password |
| `MEM0_NEO4J_DATABASE` | -- | Neo4j database name (multi-database setups) |
| `MEM0_NEO4J_BASE_LABEL` | -- | Custom Neo4j base label for node type grouping |
| `MEM0_GRAPH_THRESHOLD` | `0.7` | Embedding similarity threshold for node matching |

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `MEM0_TRANSPORT` | `stdio` | Transport: `stdio`, `sse`, or `streamable-http` |
| `MEM0_HOST` | `0.0.0.0` | Host for SSE/HTTP transports |
| `MEM0_PORT` | `8081` | Port for SSE/HTTP transports |
| `MEM0_USER_ID` | `user` | Default user ID for memory scoping |
| `MEM0_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `MEM0_HISTORY_DB_PATH` | -- | SQLite path for memory change history |

## Architecture

```
Claude Code
  |
  ├── MCP stdio/SSE/streamable-http
  │     |
  │     ├── env.py               ← Centralized env var readers (whitespace-safe)
  │     ├── auth.py              ← Hybrid token fallback chain + OAT self-refresh
  │     ├── llm_anthropic.py     ← Custom Anthropic LLM provider (OAT + structured outputs)
  │     ├── llm_ollama.py        ← Custom Ollama LLM provider (restored tool-calling)
  │     ├── config.py            ← Env vars → MemoryConfig dict (provider + URL cascades)
  │     ├── helpers.py           ← Error wrapper, concurrency lock, safe bulk-delete, monkey-patches
  │     ├── graph_tools.py       ← Direct Neo4j Cypher queries (lazy driver)
  │     ├── llm_router.py        ← Split-model graph LLM router (gemini_split)
  │     ├── __init__.py          ← Telemetry suppression (before any mem0 import)
  │     └── server.py            ← FastMCP orchestrator (11 tools + prompt)
  │           |
  │           ├── mem0ai Memory class
  │           │     ├── Vector: LLM fact extraction → Ollama embed → Qdrant
  │           │     └── Graph: LLM entity extraction (tool calls) → Neo4j
  │           |
  │           └── Infrastructure
  │                 ├── Qdrant          ← Vector store
  │                 ├── Ollama          ← Embeddings
  │                 ├── Neo4j           ← Knowledge graph (optional)
  │                 └── Anthropic/Ollama ← Main LLM (configurable)
  |
  └── Session Hooks (subprocess, not MCP)
        |
        └── hooks.py             ← Cross-session memory (SessionStart + SessionEnd hooks)
              ├── context_main()       → Injects memories as additionalContext on startup/compact
              ├── session_end_main()   → Saves session summary to mem0 on exit
              └── install_main()       → CLI to patch .claude/settings.json
```

## Graph Memory & Quota

Graph memory is **disabled by default** (`MEM0_ENABLE_GRAPH=false`) to protect your Claude quota. Each `add_memory` with graph enabled triggers 3 additional LLM calls for entity extraction, relationship generation, and conflict resolution.

### Using Ollama for Graph Operations

To eliminate Claude quota usage for graph ops, use a local Ollama model:

```env
MEM0_ENABLE_GRAPH=true
MEM0_GRAPH_LLM_PROVIDER=ollama
MEM0_GRAPH_LLM_MODEL=qwen3:14b
```

Qwen3:14b has 0.971 tool-calling F1 (nearly matching GPT-4's 0.974) and runs in ~7-8GB VRAM with Q4_K_M quantization.

### Using Gemini for Graph Operations

Google's Gemini 2.5 Flash Lite is the cheapest option for graph ops while maintaining strong entity extraction accuracy:

```env
MEM0_ENABLE_GRAPH=true
MEM0_GRAPH_LLM_PROVIDER=gemini
MEM0_GRAPH_LLM_MODEL=gemini-2.5-flash-lite
GOOGLE_API_KEY=your-google-api-key
```

### Using Split-Model for Best Accuracy

The `gemini_split` provider routes graph pipeline calls to different LLMs based on the operation. Entity extraction (Calls 1 & 2) goes to Gemini for speed and cost; contradiction detection (Call 3) goes to Claude for accuracy.

```env
MEM0_ENABLE_GRAPH=true
MEM0_GRAPH_LLM_PROVIDER=gemini_split
GOOGLE_API_KEY=your-google-api-key
MEM0_GRAPH_CONTRADICTION_LLM_PROVIDER=anthropic
MEM0_GRAPH_CONTRADICTION_LLM_MODEL=claude-opus-4-6
```

Benchmark results across 248 test cases: Gemini scores 85.4% on entity extraction (vs Claude's 79.1%), while Claude scores 100% on contradiction detection (vs Gemini's 80%). The split-model combines the best of both.

## Transport Modes

| Mode | Use Case | Config |
|------|----------|--------|
| `stdio` (default) | Claude Code integration | `MEM0_TRANSPORT=stdio` |
| `sse` | Legacy remote clients | `MEM0_TRANSPORT=sse` |
| `streamable-http` | Modern remote clients | `MEM0_TRANSPORT=streamable-http` |

For remote deployments, MCP SDK >= 1.23.0 enables DNS rebinding protection by default.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run unit tests
python3 -m pytest tests/unit/ -v

# Run contract tests (validates mem0ai internal API assumptions)
python3 -m pytest tests/contract/ -v

# Run integration tests (requires live Qdrant + Neo4j + Ollama)
python3 -m pytest tests/integration/ -v

# Run all tests
python3 -m pytest tests/ -v
```

### Test Structure

- **`tests/unit/`** -- Pure unit tests with mocked dependencies (env, auth, config, config matrix, concurrency, MCP protocol, helpers, hooks, LLM providers, graph tools, LLM router, server)
- **`tests/contract/`** -- Validates assumptions about mem0ai internals (schema detection invariant, `vector_store.client` access path, `LlmFactory` registration idempotency)
- **`tests/integration/`** -- Live infrastructure tests (memory lifecycle, graph ops, bulk operations, hooks) against real Qdrant + Neo4j + Ollama. Marked with `@pytest.mark.integration`.

Contract tests catch breaking changes in `mem0ai` upgrades before they reach production.

## Telemetry

All mem0ai telemetry is suppressed. `os.environ["MEM0_TELEMETRY"] = "false"` is set at package import time, before any `mem0` module is loaded. No PostHog events are sent.

## License

MIT
