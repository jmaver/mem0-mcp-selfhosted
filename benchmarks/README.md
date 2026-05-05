# benchmarks/v3

Quantitative benchmarks for the v3 (mem0 OSS v3, hybrid retrieval, no graph)
stack. The four runners cover the dimensions that matter post-v3:

| Runner | What it measures |
| --- | --- |
| `retrieval_quality.py` | hybrid (semantic + BM25 + entity boost) vs semantic-only recall@1, recall@5, MRR |
| `provider_battle.py` | F1 / recall / hallucination + add latency for Anthropic OAT, OpenAI-compat, Ollama |
| `hook_and_scoping.py` | end-to-end SessionStart / SessionEnd hook P50/P95 vs the 15 s / 30 s budgets, plus 3-way project-scoping isolation |
| `dedup_and_entity.py` | confusion matrix for `_dedup_against_existing` at threshold 0.88, plus per-add overhead of spaCy entity linking |

`framework.py` holds the shared dataclasses, scoring helpers, infra checks,
`OperationTimer`, and the `make_memory()` factory (mirrors
`tests/integration/conftest.py`). `_corpora.py` holds the labeled fact /
search / update / robustness cases salvaged from the v0.3-era
`mcp_e2e_battle.py`.

## Required infrastructure

- **Qdrant ≥ 1.12** at `MEM0_QDRANT_URL` — sparse-vector slot is required for
  hybrid retrieval.
- **Ollama** (or any embedding endpoint compatible with the configured
  embedder) at `MEM0_EMBED_URL`.
- For `provider_battle.py`, at least one of:
  - Anthropic token resolvable via `auth.resolve_token()` (env vars or the
    Claude Code credentials file).
  - `MEM0_OPENAI_API_KEY` and/or `MEM0_LLM_URL` for OpenAI-compatible
    endpoints (LM Studio, vLLM, llama.cpp).
  - Ollama with an instruct-tuned model pulled (default `qwen3.5:4b`).

Providers without prereqs are skipped, not failed.

### Bench-time tuning

Set `MEM0_QDRANT_TIMEOUT=60` before any bench run. Without it, every run under
load hits a timeout on `create_collection` and the runner aborts. The default
Qdrant client timeout (5 s) is too short when Qdrant is busy creating sparse +
dense vector slots; 60 s is the tested safe value.

## Running

```bash
# Smoke runs (small case counts, < 1 min on warm infra)
uv run python -m benchmarks.v3.retrieval_quality --limit 4
uv run python -m benchmarks.v3.provider_battle --limit 3 --providers ollama
uv run python -m benchmarks.v3.hook_and_scoping --iterations 3
uv run python -m benchmarks.v3.dedup_and_entity --limit 4 --entity-iters 5

# Full runs
uv run python -m benchmarks.v3.retrieval_quality
uv run python -m benchmarks.v3.provider_battle
uv run python -m benchmarks.v3.hook_and_scoping --iterations 10
uv run python -m benchmarks.v3.dedup_and_entity
```

Each runner uses a unique `bench-<runner>-<ts>` user_id and calls
`safe_bulk_delete` in `finally`, so concurrent runs and aborted runs do not
leak memories. The default `MEM0_COLLECTION` override (`mem0_bench_v3`) keeps
benchmark data out of your normal collection — pass `--collection` to change.

## Output reading guide

- **retrieval_quality.py**: per-category and per-query side-by-side. Hybrid
  should beat semantic on the keyword and entity categories; semantic-heavy
  ones may be roughly equal.
- **provider_battle.py**: `mean_f1` with 95% CI, hallucination total, and add
  latency P50/P95 per provider.
- **hook_and_scoping.py**: hook timing table with `over_budget` count flags
  any iteration that exceeded the configured timeout. Scoping section asserts
  zero canary leaks across `alice/projA`, `alice/projB`, `bob/projA`, plus
  the global scope.
- **dedup_and_entity.py**: TP / FN / FP / TN confusion at 0.88 threshold;
  entity arm prints mean delta in ms and as a percentage.