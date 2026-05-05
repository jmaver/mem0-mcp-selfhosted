# 2026-05-05 — Six-way provider battle, rerun after P1 schema fix

Same `FACT_EXTRACTION_CASES` corpus (5 cases, easy + medium), same setup as
`2026-05-04-six-way-provider-battle.md`. The TODO suspected the P1 schema
bug (Anthropic structured-output `{"facts": [...]}` schema) might have been
masking signal on the `anthropic-api` leg — this rerun answers that.

Run command:

```bash
source ~/.claude/hooks/mem0-env.sh && set -a && source .env.local && set +a
MEM0_QDRANT_TIMEOUT=60 PYTHONUNBUFFERED=1 uv run python -u \
  -m benchmarks.v3.provider_battle \
  --leg anthropic-api,openai-cloud-nano,lmstudio-qwen35-mlx,\
lmstudio-gemma,ollama-qwen35-4b,ollama-qwen3-14b \
  --limit 5 --sleep 1
```

## Results

| Leg | Provider class | Model | F1 | hallucinations | add P50 | add P95 |
| --- | --- | --- | --- | --- | --- | --- |
| anthropic-api | AnthropicOATLLM | claude-sonnet-4-6 | **1.000** | 0 | **2.44 s** | 4.89 s |
| openai-cloud-nano | OpenAICompatLLM | gpt-5.4-nano | **1.000** | 0 | 2.90 s | 6.73 s |
| ollama-qwen35-4b | OllamaToolLLM | qwen3.5:4b (Q4_K_M GGUF) | **1.000** | 0 | 30.70 s | 38.81 s |
| lmstudio-gemma | OpenAICompatLLM | gemma-4-26b-a4b-it (MLX) | 0.900 | 0 | 9.48 s | 23.95 s |
| ollama-qwen3-14b | OllamaToolLLM | qwen3:14b (GGUF) | 0.900 | 0 | 31.26 s | 78.04 s |
| lmstudio-qwen35-mlx | OpenAICompatLLM | qwen3.5-4b-mlx (MLX 4-bit) | 0.760 | 0 | **4.93 s** | 8.89 s |

## Verdict on the P1 schema bug

**The F1 numbers are identical to the 2026-05-04 run** (1.000 / 1.000 / 0.760 /
0.900 / 1.000 / 0.900 across the same six legs in the same order). The
Anthropic legs were *not* silently extracting nothing on the prior run — Sonnet
4.6 was returning the right facts, even with the broken `{"facts": [...]}`
schema. The schema bug existed but didn't bite on this corpus because:

- The schema was used as a structured-output hint. Sonnet 4.x is permissive
  about list-of-strings vs object-with-`facts`-key and either way the JSON
  extraction in `_parse_response` recovers the strings.
- The 5-case corpus skews easy (4 easy, 1 medium); fact extraction is over-
  determined and the model has plenty of redundancy.

Translation: the P1 fix is still correct (the schema *was* malformed against
the documented contract) but it was not the cause of any visible quality
regression in this corpus. **A larger / harder corpus is needed to detect
schema-induced silent extraction loss.** Tracked under TODO #8 (expand
retrieval corpus to 100+ facts; same logic applies to the extraction corpus).

## Latency changed; quality didn't

The Ollama legs are notably slower this run (qwen3.5:4b mean 27.80 s vs prior
~17 s, qwen3:14b mean 48.08 s vs prior ~32 s). Suspect machine load —
nothing in the code path changed that would affect ollama latency by 2×.
Running on the same kraken host the day before showed faster numbers. Worth
re-checking under controlled conditions before drawing conclusions about
local-extraction throughput, but doesn't affect the quality verdict.

The cloud legs (anthropic-api, openai-cloud-nano) shifted only modestly
(< 1 s on both p50 and p95) — within the run-to-run noise band.

## Confirmed findings (carried forward from 2026-05-04)

1. **MLX 4-bit costs ~24 pp F1** vs GGUF Q4_K_M for the same Qwen3.5-4B model
   (0.760 vs 1.000). Two independent runs now agree.
2. **qwen3.5:4b > qwen3:14b** on the same Ollama backend (1.000 vs 0.900). The
   smaller, newer-generation model wins on this corpus. Two runs agree.
3. **Cloud APIs are 5-10× faster than local extraction** even before counting
   model-load overhead (anthropic-api 2.44 s p50 vs ollama-qwen35-4b 30.70 s).

## Side benefits captured during this rerun

- **TODO #2 fix landed** (anthropic-qwen36 leg now scores). Probe (`benchmarks/
  v3/_probes/probe_anthropic_qwen36.py`) revealed DashScope's `qwen3.6-plus`
  always emits `[ThinkingBlock, TextBlock]`, and `_extract_text_block` was
  reading `content[0].text` — which on a `ThinkingBlock` is `None`. Fixed by
  scanning for the first text block. Smoke verification:
  `anthropic-qwen36` now F1=0.833 on a 3-case smoke (was F1=0 before).
  Same fix protects against any future Claude config that turns on extended
  thinking.
- **`anthropic-glm` leg added** (Anthropic-shaped third-party endpoint, same
  pattern as qwen36). Smoke F1=0.833 on the 3-case corpus, add P50 5.79 s —
  much faster than qwen36 (P50 22.65 s) on the same prompt. Benefits from
  the same `_extract_text_block` fix.

## Open follow-ups (forwarded to TODO.md)

- TODO #8: expand the extraction corpus to ≥30 cases with hand-graded edge-
  case prompts. Until then, F1=1.0 on cloud legs is "ceiling reached," not
  "high-confidence parity."
- TODO #9: switch the v3 default Ollama model to `qwen3.5:4b`. Two runs at
  N=5 now agree on the gap (1.000 vs 0.900). Single-line change in
  `config.py:89`. Low risk given the agreement, but TODO author wanted
  larger N first — leave until corpus expansion lands.
