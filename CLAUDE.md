# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.


## Build & Test Commands

```bash
uv sync --extra dev                  # Install with dev dependencies
uv run pytest tests/ -m "not integration" -v  # Unit + contract tests
uv run pytest tests/ -v              # All tests (incl. integration)
```

## Lint & Type Check

```bash
uv run ruff check src/               # Lint
uv run ruff check src/ --fix         # Lint + auto-fix
uv run pyright src/                  # Type check
```

## Running & Hooks

```bash
uv run mem0-mcp-selfhosted           # Start MCP server
uv run mem0-install-hooks            # Install Claude Code SessionStart/SessionEnd hooks
```

Hook entry points (registered in `pyproject.toml`):
- `mem0-hook-context` — provides memory context on SessionStart
- `mem0-hook-session-end` — saves session summary to memory

**Hook architecture:** `settings.json` → shell wrapper (`~/.claude/hooks/mem0-hook-*`) → `mem0-env.sh` (sources env from `~/.claude.json` MCP config via `jq`) → `.venv/bin/mem0-hook-*`. Hooks log to `/tmp/mem0-hooks.log`. SessionStart fires on `startup|compact` events; SessionEnd fires unconditionally.

## Architecture

Self-hosted MCP server using `mem0ai` (>=2.0.2, "v3") as a library. 9 memory tools, FastMCP orchestrator. Hybrid retrieval (semantic + BM25 + entity matching) — no graph database.

**Module roles:**
- `server.py` — FastMCP orchestrator, registers all tools + `memory_assistant` prompt, lazy Memory init
- `config.py` — Env vars → mem0ai `MemoryConfig` dict
- `auth.py` — 3-tier token fallback: `MEM0_ANTHROPIC_TOKEN` → `~/.claude/.credentials.json` → `ANTHROPIC_API_KEY`
- `llm_anthropic.py` — Custom Anthropic provider registered with mem0ai's `LlmFactory`; handles OAT headers, structured outputs (JSON schema via `output_config`), and tool-call parsing
- `llm_ollama.py` — Custom Ollama provider with restored tool-calling and defense-in-depth against `<think>` + JSON-mode collisions
- `llm_openai_compat.py` — OpenAI-compat provider that strips `json_object` response_format (LM Studio / vLLM / llama.cpp only accept `json_schema` or text)
- `helpers.py` — `_mem0_call()` error wrapper, `safe_bulk_delete()` iterates+deletes individually (never calls `memory.delete_all()`), `make_project_user_id()` + `search_with_project()` for project-scoped memory isolation, `patch_gemini_parse_response()` null-content guard
- `__init__.py` — Suppresses mem0ai telemetry before any imports

## Environment

All config flows through env vars (no config file). The MCP server reads these from `~/.claude.json` via the `mem0-env.sh` wrapper script.

| Var | Default | Purpose |
|-----|---------|---------|
| `MEM0_PROVIDER` | `anthropic` | LLM: `anthropic`, `ollama`, or `openai` |
| `MEM0_LLM_PROVIDER` | = `MEM0_PROVIDER` | Override LLM provider independently |
| `MEM0_LLM_MODEL` | provider-specific | `claude-opus-4-6`, `qwen3.5:4b`, or model name |
| `MEM0_LLM_URL` | — | Override base URL (LM Studio `:1234`, Ollama `:11434`, etc.) |
| `MEM0_LLM_MAX_TOKENS` | `16384` | Max completion tokens |
| `MEM0_EMBED_PROVIDER` | `ollama` | `ollama` or `openai` |
| `MEM0_EMBED_MODEL` | provider-specific | `bge-m3` or `text-embedding-3-small` |
| `MEM0_EMBED_URL` | = `MEM0_LLM_URL` | Embedder base URL override |
| `MEM0_EMBED_DIMS` | provider-specific | `1024` (ollama) or `1536` (openai) |
| `MEM0_QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `MEM0_QDRANT_API_KEY` | — | Qdrant API key (optional) |
| `MEM0_QDRANT_TIMEOUT` | — | Qdrant request timeout in seconds |
| `MEM0_COLLECTION` | `mem0_mcp_selfhosted` | Qdrant collection name |
| `MEM0_HOST` / `MEM0_PORT` | `0.0.0.0` / `8081` | Server bind address |
| `MEM0_TRANSPORT` | `stdio` | `stdio`, `sse`, or `streamable-http` |
| `MEM0_OPENAI_API_KEY` | — | Required for openai provider |

**Model selection flow:** `MEM0_PROVIDER` sets the default → `MEM0_LLM_PROVIDER` can override → `MEM0_LLM_MODEL` picks the model (defaults: Anthropic=`claude-opus-4-6`, Ollama=`qwen3.5:4b`, OpenAI=required). `MEM0_LLM_URL` routes to any compatible endpoint (LM Studio, Ollama, vLLM, etc.).

**Auth:** `MEM0_ANTHROPIC_TOKEN` → `~/.claude/.credentials.json` → `ANTHROPIC_API_KEY` (first available wins).

**Legacy:** `MEM0_*_GRAPH*` / `MEM0_NEO4J_*` vars are set but silently ignored (graph removed in v0.4/mem0 v3).

## Critical implementation details

- `Memory.search()` v3 contract: entity IDs go inside `filters={"user_id": ...}`, not as top-level kwargs. `search_with_project()` already handles this.
- `Memory.update()` uses `data=` parameter, not `text=`
- `Memory.add()` v3 returns ADD events only (no UPDATE/DELETE/NONE) — single-pass extraction collapsed the old two-call diff pipeline
- Structured output support requires claude-opus-4/sonnet-4/haiku-4 models; older models fall back to JSON extraction
- Qdrant must be v1.12+ (sparse vectors for BM25 alongside dense vectors in the same collection)
- spaCy `en_core_web_sm` is required for v3's entity-linking pipeline; bundled as a wheel in `pyproject.toml` so `uv sync` installs it automatically
- Contract tests (`tests/contract/`) validate mem0ai internal API assumptions — if these fail after a mem0ai upgrade, the code needs updating
- Avoid MLX 4-bit quants for production fact extraction: the 6-way bench showed ~24 pp F1 regression vs GGUF Q4_K_M on the same model (see `benchmarks/v3/results/2026-05-04-six-way-provider-battle.md`)
