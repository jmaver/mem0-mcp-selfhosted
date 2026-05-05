# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.


## Build & Test Commands

```bash
uv sync --extra dev                  # Install with dev dependencies
uv run pytest tests/unit/ -v         # Unit tests (mocked, no infra needed)
uv run pytest tests/contract/ -v     # Contract tests (validates mem0ai internals)
uv run pytest tests/integration/ -v  # Integration tests (requires live Qdrant + Ollama)
uv run pytest tests/ -v              # All tests
uv run pytest tests/ -m "not integration" -v  # Skip integration
uv run pytest tests/unit/test_auth.py::TestIsOatToken -v  # Single test class
uv run pytest tests/unit/test_auth.py::TestIsOatToken::test_oat_token_detected -v  # Single test
```

One-time post-install: `uv run python -m spacy download en_core_web_sm` (mem0 v3 entity linking).

## Architecture

Self-hosted MCP server using `mem0ai` (>=2.0.1, "v3") as a library. 9 memory tools, FastMCP orchestrator. Hybrid retrieval (semantic + BM25 + entity matching) — no graph database.

**Module roles:**
- `server.py` — FastMCP orchestrator, registers all tools + `memory_assistant` prompt
- `config.py` — Env vars → mem0ai `MemoryConfig` dict
- `auth.py` — 3-tier token fallback: `MEM0_ANTHROPIC_TOKEN` → `~/.claude/.credentials.json` → `ANTHROPIC_API_KEY`
- `llm_anthropic.py` — Custom Anthropic provider registered with mem0ai's `LlmFactory`; handles OAT headers, structured outputs (JSON schema via `output_config`), and tool-call parsing
- `llm_ollama.py` — Custom Ollama provider with restored tool-calling and defense-in-depth against `<think>` + JSON-mode collisions
- `llm_openai_compat.py` — OpenAI-compat provider that strips `json_object` response_format (LM Studio / vLLM / llama.cpp only accept `json_schema` or text)
- `helpers.py` — `_mem0_call()` error wrapper, `safe_bulk_delete()` iterates+deletes individually (never calls `memory.delete_all()`), `make_project_user_id()` + `search_with_project()` for project-scoped memory isolation, `patch_gemini_parse_response()` null-content guard
- `__init__.py` — Suppresses mem0ai telemetry before any imports

**Critical implementation details:**
- `Memory.search()` v3 contract: entity IDs go inside `filters={"user_id": ...}`, not as top-level kwargs. `search_with_project()` already handles this.
- `Memory.update()` uses `data=` parameter, not `text=`
- `Memory.add()` v3 returns ADD events only (no UPDATE/DELETE/NONE) — single-pass extraction collapsed the old two-call diff pipeline
- Structured output support requires claude-opus-4/sonnet-4/haiku-4 models; older models fall back to JSON extraction
- Qdrant must be v1.12+ (sparse vectors for BM25 alongside dense vectors in the same collection)
- spaCy `en_core_web_sm` is required for v3's entity-linking pipeline; install via `uv run python -m spacy download en_core_web_sm`
- Contract tests (`tests/contract/`) validate mem0ai internal API assumptions — if these fail after a mem0ai upgrade, the code needs updating
- Avoid MLX 4-bit quants for production fact extraction: the 6-way bench showed ~24 pp F1 regression vs GGUF Q4_K_M on the same model (see `benchmarks/v3/results/2026-05-04-six-way-provider-battle.md`)
