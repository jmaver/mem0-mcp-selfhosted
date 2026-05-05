# Outstanding work — v3 benchmarks + provider matrix

Updated 2026-05-05 (second pass). All previously deferred items now
landed. Remaining work is verification: re-run the bench suite with the
expanded corpora to see if the ceiling-saturation problem is gone.

## Still open

### FE29: qwen3.5:4b multi-contradiction ceiling at F1=0.50
The bench F1=0.00 was a transient Ollama JSON failure (handled by the Layer 6 retry).
The underlying persistent limit is F1=0.50: qwen3.5:4b retains superseded database
versions in its extraction ("MySQL 5.7, MySQL 8.0, and PostgreSQL 16" all appear in
the stored memory) rather than suppressing the old values. This is a model-reasoning
gap, not a parser bug. Options: (a) accept ≤0.50 for qwen3.5:4b on
multi-contradiction cases and document it, or (b) add a post-extraction
contradiction-resolution pass to the extraction prompt system message. See
`2026-05-05-fe-failure-probes.md`.

## What landed in this session

- ✅ FE05 / FE27 / FE29 failure probes — 5-iteration isolated probes run for
  all three zero-F1 cases from the 2026-05-05 30-case bench. Findings:
  - FE05 (lmstudio-qwen35-mlx): one-off transient; 5/5 F1=1.00 today.
  - FE27 (lmstudio-qwen35-mlx): one-off transient; 5/5 F1=1.00 today. Markdown-fenced
    JSON appears 2/5 in direct LLM calls but is already handled by `_clean_response()`
    and mem0's `remove_code_blocks()`. No code fix needed.
  - FE29 (ollama-qwen35-4b): bench F1=0.00 was a transient Ollama empty-JSON event
    (one-off). Persistent baseline is F1=0.50 — model-capability limit on multi-
    contradiction (retains superseded DB versions). No parser fix available.
  See `2026-05-05-fe-failure-probes.md`. Probes at `benchmarks/v3/_probes/`.

- ✅ Sub-checkpoint `_get_memory()` — four `profile.init.*` checkpoints added
  to `hooks.py`. `Memory.from_config()` (Qdrant client init + collection-ensure
  + spaCy pipeline load) is the dominant sub-phase at 83 % of `memory_init`
  (mean 5.5 s of 6.7 s). Imports cost ~0.5 s; first warm-up search ~0.6 s;
  `build_config()` + `register_providers()` ~0.01 s. See
  `2026-05-05-session-end-profile.md` "Sub-checkpoint breakdown" section.

- ✅ Dedup threshold tuning — swept 0.82, 0.85, 0.88; all three produce
  identical confusion matrices (TP=6 FN=2 FP=0 TN=8). D05 and D08 score
  below 0.82 in embedding space; no threshold in the safe range can catch
  them. `_DEDUP_SIM_THRESHOLD` stays at 0.88. See
  `2026-05-05-dedup-threshold-tuning.md`.

- ✅ Re-run benches with expanded corpora — 30-case extraction corpus + 30-query
  / 100-fact retrieval corpus. Ceiling saturation confirmed broken; all legs now
  sub-1.0 F1; hard-case tier (FE21–FE30) shows 0.694–0.778 range. See
  `2026-05-05-bench-all-30case.md`.

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
