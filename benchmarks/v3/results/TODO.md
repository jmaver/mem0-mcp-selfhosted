# Outstanding work — v3 benchmarks + provider matrix

Snapshot as of `feat/mem0-v3-migration` HEAD = `98778c8`.

## Critical follow-ups (validated bugs / known broken paths)

### 1. Re-run the provider battle now that the P1 schema bug is fixed
The 6-way results in `2026-05-04-six-way-provider-battle.md` were captured
**before** Codex's P1 schema fix landed. The Anthropic legs may have been
running with the broken `{"facts": [...]}` schema and silently extracting
nothing on some cases. Re-run with:

```bash
source ~/.claude/hooks/mem0-env.sh && set -a && source .env.local && set +a
MEM0_QDRANT_TIMEOUT=60 uv run python -m benchmarks.v3.provider_battle \
  --leg anthropic-api,openai-cloud-nano,lmstudio-qwen35-mlx,lmstudio-gemma,ollama-qwen35-4b,ollama-qwen3-14b \
  --limit 5 --sleep 1
```

Save as `benchmarks/v3/results/<DATE>-six-way-rerun-after-p1-fix.md`.
If `anthropic-api` numbers change meaningfully, that's confirmation the
schema bug was masking real signal.

### 2. `anthropic-qwen36` leg still doesn't work
Endpoint accepts requests (no auth error after the base_url fix) but
returns content in a shape `_extract_text_block` doesn't recognize, so
every case scores F1=0. Two paths forward:

- **Inspect the response shape**: write a tiny one-shot script that calls
  the endpoint directly via the `anthropic` SDK with `base_url` overridden
  and prints `response.content`. Likely it's returning text in a format
  this code expects to find at `response.content[0].text` but actually
  lives elsewhere (e.g., the whole response is a single string, or
  content is wrapped differently).
- **If the endpoint is OpenAI-compat-shaped under the hood**: skip the
  Anthropic provider entirely and add an `openai-qwen36` leg pointing at
  the OpenAI-compat path. Many "Anthropic-compatible" endpoints are
  actually OpenAI-compat with an Anthropic auth header.

### 3. `anthropic-oat` rate-limit on subscription tier
The retry path is working — 429s are caught with `Retry-After`-aware
backoff — but Claude Pro / Max subscription tiers don't have enough
budget for mem0's 6-10 LLM calls per `add()`. Two options:

- **Document as expected**: the bench leg exists for completeness but
  shouldn't be in regular runs. Add a comment in `legs.toml` saying so.
- **Use the paid API leg**: `anthropic-api` is the right path for
  benchmarks. The OAT leg is mainly useful for testing the OAT auth
  refresh flow, not for fact extraction at scale.

## Production code follow-ups (flagged in simplify review)

### 4. `_run_leg` in `provider_battle.py` doesn't gate on availability
`_run_provider` calls `_provider_available()` to print friendly skip
messages; `_run_leg` doesn't. If a leg's `anthropic_token_env` resolves
to None or its `llm_url` is unreachable, you only learn at the first
`add()` call. Add a `_leg_available(leg)` helper that:

- For anthropic: checks the resolved token is non-empty
- For openai: checks the resolved api_key OR llm_url is set
- For ollama: pings `<llm_url>/api/tags`

Then call it from `_run_leg` and emit `[skip] <name>: <reason>`
consistent with the legacy path.

### 5. Redundant `_get_common_params` override in OpenAICompatLLM
The literal model name `"gpt-5"` is in both mem0's reasoning-model set
AND our `_GPT5_REJECTS_MAX_TOKENS_PREFIXES`. Harmless (mem0
short-circuits us), but the override comment should note the overlap or
the prefix list could exclude exact-match reasoning models for clarity.

### 6. 429 retry consumes the global attempt counter
A 429 followed by a 5xx tightens the 5xx retry budget. By design (total
cap is 3) but worth knowing if you ever want independent budgets. Not
broken, just a behavior to flag if someone debugs an unusual error
sequence.

## Bench follow-ups (lower priority)

### 7. Profile the `session_end` cold-start
Baseline run had `session_end` exceed the 30s budget in 2/3 cold-start
iterations. Hasn't been investigated. Add `cProfile` or
`time.perf_counter()` checkpoints to the SessionEnd hook to break down:

- mem0 import (~?ms)
- spaCy `en_core_web_sm` load (~?ms)
- Memory init + Qdrant client + first embed (~?ms)
- Extraction LLM call (~?ms)

Expected dominant cost is the extraction LLM call given what we now
know about provider latency, but the baseline was on LM Studio Qwen
which has 5-7s P50 — that doesn't account for the full 30s. Either
init is heavier than expected or the extraction prompt produces a
much longer response than the bench cases.

### 8. Expand the retrieval corpus to 100+ facts
`retrieval_quality.py` saturated at recall@5=1.0 on both arms with the
current 18-fact corpus. The plan flagged this. Build a corpus with:

- ~100 facts across 5 thematic clusters with semantic confusables
  within each cluster
- Mix of keyword-match queries (where BM25 should win), semantic-match
  queries (where embeddings should win), and entity-match queries
  (where the entity-boost path should win)
- ~30 queries total with hand-labeled relevant fact IDs

Without this, the hybrid-vs-semantic comparison can't produce a real
verdict.

### 9. Switch v3's default Ollama model
`config.py:89` has `"ollama": "qwen3:14b"` but the 6-way bench shows
`qwen3.5:4b` beats it on extraction quality (1.0 vs 0.9 F1) at the same
backend, while being 3.5x smaller. Worth changing the default —
single-line change, but should validate with a re-run on a larger corpus
first since N=5 is small.

### 10. Avoid MLX 4-bit for extraction
The 6-way bench shows MLX 4-bit costs ~24 percentage points of F1 vs
the same model in GGUF Q4_K_M. If anyone is using LM Studio with MLX
quants for production extraction, they're paying significant accuracy.
Worth flagging in the README or a CLAUDE.md note. Not a code change,
just documentation.

## Hygiene / nice-to-haves

### 11. Document `MEM0_QDRANT_TIMEOUT=60` in benchmarks/README.md
Every bench run needed it set to survive `create_collection` under
load. Currently undocumented — future-you will hit the same wall.

### 12. Single-entry-point `bench_all.sh`
Running 4 separate `python -m` commands is fine but tedious. A wrapper
that sources the env, sets the timeout, runs each suite in sequence,
and writes a combined results file would simplify the routine.

### 13. Memory.search filter contract test for project scoping
The hook_and_scoping bench passes (3-way isolation holds), but there's
no contract test in `tests/contract/` that locks down "user_id encoded
as `user:project` is the source of truth for cross-project isolation."
A small contract test would catch a regression where someone adds a
top-level `user_id=` kwarg that bypasses the filter.

### 14. Pre-existing dead code in `framework.py`
`_fuzzy_match_any` is defined but unused (pyright surfaces this every
run). Either delete or keep with a note in the docstring saying it's
exposed for runner imports. Lowest priority.

## What's been done already (for context if you re-orient)

- ✅ v3 benchmark suite scaffolding (4 runners + framework + corpora)
- ✅ Pre-v3 graph benchmark scripts deleted
- ✅ Hook tightening (extraction prompt + dedup + entity-link timing)
- ✅ Named-leg system in `provider_battle.py` with hermetic env
- ✅ 5 production LLM bug fixes from the bench expansion (config base_url,
  AnthropicOATConfig rewrap, structured-output OAT gate, 429 retry,
  gpt-5.x max_tokens)
- ✅ All 3 Codex review findings (P1 schema, P2 Qdrant scroll, P3 version)
- ✅ Simplify pass on framework.py + retry loop
- ✅ Two real bench result files captured + committed

Branch is at `98778c8`, 8 commits ahead of `fork/feat/mem0-v3-migration`.
All commits already pushed. 300 unit tests passing. Pyright clean.
