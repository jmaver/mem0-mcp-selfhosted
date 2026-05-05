# Outstanding work — v3 benchmarks + provider matrix

Updated 2026-05-05. Most of the original list landed in this session; what
remains is corpus expansion (lower priority) and one default-config flip
that's deliberately gated on a larger N.

## Still open

### Corpus expansion (TODO #8 — original)
`retrieval_quality.py` saturated at recall@5=1.0 on both arms with the
current 18-fact corpus. The same ceiling problem applies to the extraction
corpus (5 cases, mostly easy; cloud legs hit F1=1.0). Build a corpus with:

- ~100 facts across 5 thematic clusters with semantic confusables within
  each cluster
- Mix of keyword-match queries (where BM25 should win), semantic-match
  queries (where embeddings should win), and entity-match queries (where
  the entity-boost path should win)
- ~30 queries total with hand-labeled relevant fact IDs
- For extraction: ~30 cases skewed harder (multi-fact prompts,
  contradictions, entities with overlapping aliases) so the schema-vs-no-
  schema effect is detectable.

Without this, the hybrid-vs-semantic comparison and the schema-shape
verdict can't move beyond "ceiling reached."

### Profile the `session_end` cold-start (TODO #7 — original)
Baseline run had `session_end` exceed the 30 s budget in 2/3 cold-start
iterations. Hasn't been investigated. Add `cProfile` or
`time.perf_counter()` checkpoints to the SessionEnd hook to break down:

- mem0 import (~?ms)
- spaCy `en_core_web_sm` load (~?ms)
- Memory init + Qdrant client + first embed (~?ms)
- Extraction LLM call (~?ms)

Now that we know cloud extraction is 2-5 s and local is 15-30 s, init cost
must dominate the remaining ~25 s of the 30 s budget. Likely candidates:
spaCy load, Qdrant collection check, first embedder warm-up.

### Switch v3's default Ollama model (TODO #9 — original)
`config.py:89` has `"ollama": "qwen3:14b"` but two N=5 runs agree that
`qwen3.5:4b` beats it on extraction quality (1.0 vs 0.9 F1) at the same
backend, while being 3.5× smaller. Single-line change. Deliberately
deferred until the larger corpus (above) lands so the default flip rests
on a stronger statistical footing.

## What landed in this session

- ✅ TODO #1 — Six-way provider battle re-run with P1 fix in place. Result:
  identical F1 numbers to the 2026-05-04 run; the schema bug existed but
  didn't bite on this corpus. See
  `2026-05-05-six-way-rerun-after-p1-fix.md`.
- ✅ TODO #2 — `anthropic-qwen36` leg fixed. Root cause: DashScope's
  `qwen3.6-plus` always emits `[ThinkingBlock, TextBlock]`;
  `_extract_text_block` was reading `content[0].text` (None on a
  `ThinkingBlock`). Fixed to scan for first text block. Same fix protects
  any future Claude config with extended thinking enabled. Probe at
  `benchmarks/v3/_probes/probe_anthropic_qwen36.py`. Smoke F1=0.833.
- ✅ TODO #3 — `anthropic-oat` documented as auth-flow-test-only in
  `legs.toml`.
- ✅ TODO #4 — `_leg_available()` gate added to `_run_leg`; legs now skip
  with a friendly reason instead of failing at first `add()`.
- ✅ TODO #5 — gpt-5/reasoning-model overlap noted in `OpenAICompatLLM`.
- ✅ TODO #6 — 429-retry shared-budget behavior documented in
  `_call_with_transient_retry`.
- ✅ TODO #10 — MLX-4-bit accuracy warning added to `CLAUDE.md`.
- ✅ TODO #11 — `MEM0_QDRANT_TIMEOUT=60` documented in `benchmarks/README.md`.
- ✅ TODO #12 — `benchmarks/bench_all.sh` wrapper added.
- ✅ TODO #13 — Project-scoping contract test added at
  `tests/contract/test_project_scoping_contract.py`.
- ✅ TODO #14 — Dead `_fuzzy_match_any` removed from `framework.py`.

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
