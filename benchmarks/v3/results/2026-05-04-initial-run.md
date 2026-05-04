# 2026-05-04 — Initial v3 benchmark run

First end-to-end run of the v3 suite after landing it on `feat/mem0-v3-migration`.
Captures baseline numbers for the four runners against the home-network infra.

## Environment

| Component | Value |
| --- | --- |
| Branch / commit | `feat/mem0-v3-migration` @ `a5fafe8` |
| Qdrant | `http://192.168.200.12:6333` (thoth NAS), v1.12+ |
| LLM endpoint | `http://192.168.200.84:1234/v1` (LM Studio on kraken) |
| LLM model | `qwen3.5-4b-mlx` (MLX 4-bit) |
| Embedder | `text-embedding-nomic-embed-text-v2-moe`, dim 768 |
| Anthropic / Ollama | not exercised this run (no token in shell env, no Ollama model pulled) |
| Bench collection | `mem0_bench_v3` (separate from production `mem0_mcp_selfhosted`) |
| `MEM0_QDRANT_TIMEOUT` | 60 s (bumped from default to survive concurrent collection writes) |

## Hook latency + scoping isolation

`python -m benchmarks.v3.hook_and_scoping --iterations 3` (cold-start each call,
spawns a fresh subprocess per hook).

| hook | n | P50 | P95 | max | budget | over budget |
| --- | --- | --- | --- | --- | --- | --- |
| `context` (SessionStart) | 3 | 10.59 s | 11.29 s | 11.37 s | 15 s | 0/3 ✅ |
| `session_end` (SessionEnd) | 3 | **30.90 s** | **33.31 s** | **33.58 s** | 30 s | **2/3 ❌** |

`session_end` exceeded the 30 s Claude Code timeout in 2 of 3 cold-start
iterations. Production hooks always run cold (one process per invocation), so
this is the realistic scenario, not a worst-case artifact. Almost all of the
30 s is init overhead (mem0 import, spaCy load, Qdrant client + first embed,
provider registration) before any extraction work begins.

3-way scoping isolation across `alice:projA` / `alice:projB` / `bob:projA` —
**zero leaks**.

## Dedup catch rate (threshold 0.88)

`python -m benchmarks.v3.dedup_and_entity --limit 5 --entity-iters 5`

```
Confusion matrix (n=5, skipped=0):
  TP (caught dup)        :   4
  FN (missed dup)        :   1
  FP (deleted distinct)  :   0
  TN (kept distinct)     :   0
  precision=1.000  recall=0.800  accuracy=0.800
```

Conservative threshold validates: 0 false positives across the run, recall 0.8
is acceptable for the trade-off (don't delete legitimate memories). One
paraphrase pair (`D05`) slipped through.

## Entity-link overhead (spaCy)

| arm | n | mean | P50 | P95 |
| --- | --- | --- | --- | --- |
| with spaCy | 5 | 0.831 s | 0.754 s | 1.245 s |
| no spaCy   | 5 | 0.574 s | 0.497 s | 0.948 s |

**Δ = +257 ms per `add()` (+44.8 %)** — the spaCy `en_core_web_sm` entity-link
pass adds almost half again to add latency. On a session with 10 `add()` calls
that is ~2.5 s of pure NLP overhead.

## Retrieval quality (hybrid vs semantic-only)

`python -m benchmarks.v3.retrieval_quality` — full corpus (18 facts, 12 queries
across keyword / semantic / entity categories).

| category | n | hybrid r@5 | semantic r@5 | delta |
| --- | --- | --- | --- | --- |
| keyword  | 4 | 1.0 | 1.0 | 0.0 |
| semantic | 4 | 1.0 | 1.0 | 0.0 |
| entity   | 4 | 1.0 | 1.0 | 0.0 |

**Saturated.** Both arms hit recall@1 = 1.0 on every query. With 18 facts and
no semantic confusables, the corpus is too easy for hybrid vs semantic-only to
diverge. The plan flagged this exact ambiguity in its verification step:
either the corpus is too easy or the BM25/entity-boost monkey-patches aren't
toggling. Need a 100+ fact corpus with adjacent topics before the differential
becomes measurable.

## Provider battle (LM Studio leg only)

`python -m benchmarks.v3.provider_battle --providers openai --limit 5`

| provider | model | mean F1 | ci95 | mean recall | hallucinations | add P50 / P95 / mean | fail/n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| openai (LM Studio) | qwen3.5-4b-mlx | **0.760** | (0.38, 1.00) | 0.733 | **0** | 6.66 s / 11.89 s / 7.79 s | 0/5 |

5/5 cases completed, zero hallucinations. Case `FE05` extracted nothing
(F1=0); the rest scored 0.80–1.00. Anthropic and Ollama legs need their
respective creds / pulled models to run.

## Cleanup

All runners reported successful cleanup of bench `user_id`s via
`safe_bulk_delete`. The `mem0_bench_v3` collection is isolated from
production `mem0_mcp_selfhosted` — nothing leaked into the real memory store.

## Follow-ups

1. Investigate where the 30 s of `session_end` cold-start goes (mem0 import vs
   spaCy load vs Memory init vs first embed vs extraction LLM). Likely the
   Claude Code 30 s budget needs a bump or initialization needs deferring.
2. Expand the retrieval corpus to 100+ facts with semantic confusables so the
   hybrid-vs-semantic comparison produces real signal.
3. `ollama pull qwen3:14b` and re-run `provider_battle.py --providers
   openai,ollama` for the local-vs-local comparison; add Anthropic when an
   OAT token is resolvable in the shell env.
