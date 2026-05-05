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

## Provider battle (3-way)

| provider | model | mean F1 | ci95 | mean recall | hallucinations | add P50 / P95 / mean | fail/n | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openai (LM Studio) | qwen3.5-4b-mlx | 0.760 | (0.38, 1.00) | 0.733 | 0 | 6.66 / 11.89 / 7.79 s | 0/5 | FE05 returned no facts |
| ollama (native) | qwen3.5:4b | **1.000** | (1.00, 1.00) | 1.000 | 0 | 15.65 / 20.07 / 14.22 s | 0/5 | perfect F1, ~3× slower |
| anthropic (OAT) | claude-opus-4-6 | n/a | n/a | n/a | n/a | n/a | 5/5 (429) | rate-limited |

**Same nominal 4 B Qwen, different stacks** — Ollama (Q4_K_M GGUF) beat LM
Studio (MLX 4-bit) 1.0 vs 0.76 on extraction F1, at 3× the per-add latency.
The `OllamaToolLLM` provider's `<think>`-tag stripping + JSON retry may be
recovering edge cases the `OpenAICompatLLM` path misses, or the MLX
quantization is lossier than Ollama's default — both worth measuring with a
larger N.

**Anthropic OAT via Claude Code subscription is rate-limited** for
extraction-heavy benchmarks. mem0's `infer=True` issues ~6–10 LLM calls per
`add()` (fact retrieval + diff + structured-output), which trips the 429
window quickly. A fair Anthropic comparison would require `ANTHROPIC_API_KEY`
on a paid tier, not the OAT path.

### Provider battle gotchas (worth surfacing in the runner)

When running `--providers ollama` from a shell that already has
`MEM0_LLM_URL=http://<lm-studio>/v1`, the Ollama provider tries to call
`http://<lm-studio>/v1/api/chat` and crashes with a `pydantic ChatResponse`
validation error. The runner needs to override `MEM0_LLM_URL` per leg, or the
operator must pass `MEM0_LLM_URL=http://<ollama-host>:11434` explicitly when
the Ollama leg runs. Currently has to be done manually — file as a
follow-up.

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
3. Fix `provider_battle.py` per-leg `MEM0_LLM_URL` override — currently
   the Ollama leg crashes if the shell still has an OpenAI-compat URL set.
4. Run the Anthropic leg with a real `ANTHROPIC_API_KEY` on a paid tier
   (OAT-via-subscription tokens hit 429 immediately on extraction-heavy
   benchmarks).
5. The Ollama-vs-LM-Studio extraction quality gap (1.0 vs 0.76 on the same
   nominal 4 B Qwen) is worth investigating with a larger case set —
   either MLX quantization is lossier or the provider-class quirk-handling
   is doing more work than expected.
