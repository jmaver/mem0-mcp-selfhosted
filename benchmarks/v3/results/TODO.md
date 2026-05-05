# Outstanding work — v3 benchmarks + provider matrix

Updated 2026-05-05 (second pass). All previously deferred items now
landed. Remaining work is verification: re-run the bench suite with the
expanded corpora to see if the ceiling-saturation problem is gone.

## Still open

### Re-run benches with expanded corpora
Corpus expansion landed (100 facts / 30 queries / 30 extraction cases) but
the bench has not been re-run against it. Until that runs, we don't know
whether the hybrid-vs-semantic gap is now detectable, or whether
`anthropic-api` extraction F1 falls below 1.0 on the harder cases. The run
itself is mechanical — `benchmarks/bench_all.sh` is the wrapper.

### Sub-checkpoint `_get_memory()` for tighter init breakdown
The session_end profile localized 4–5 s to `memory_init` but couldn't
isolate which sub-phase (mem0 import / spaCy load / Qdrant init / first
embed). Add the same `time.perf_counter()` pattern inside `_get_memory()`
in `hooks.py` if init cost ever needs targeted optimization. Low priority
while `mem_add` dominates at 13 s mean.

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
- ✅ TODO #7 (original) — `session_end` cold-start profiled. `mem_add`
  (Ollama LLM call) dominates at 13.4 s mean (73 % of total); `memory_init`
  is 4.7 s (26 %); everything else is &lt;1 s. See
  `2026-05-05-session-end-profile.md`. The ~30.9 s prior p50 was Ollama
  queue load, not init cost.
- ✅ TODO #8 (original) — Corpora expanded. `_corpora.py`
  `FACT_EXTRACTION_CASES` 20 → 30 (10 hard / adversarial: contradictions,
  negation traps, alias resolution, hedged statements, density traps).
  `retrieval_quality.py` 18 facts → 100 across 5 thematic clusters; 12
  queries → 30 with multi-relevant labels.
- ✅ TODO #9 (original) — Default Ollama model flipped from `qwen3:14b` to
  `qwen3.5:4b` in `config.py:89`. Cascading updates in 3 unit tests,
  README.md, benchmarks/README.md, and `provider_battle.py` benchmark
  fallback default. Historical bench results files left untouched.

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
