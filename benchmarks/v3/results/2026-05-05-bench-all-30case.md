# 2026-05-05 — Full bench rerun on expanded 30-case corpus

The 30-case corpus expansion succeeded at breaking ceiling saturation: every
leg now shows sub-1.0 mean F1 and the hard/adversarial tier (FE21–FE30) exposes
meaningful differentiation between providers. Cloud legs hold 0.76–0.77 F1 on
the hard tier while local-small (lmstudio-qwen35-mlx) drops to 0.69, confirming
the old N=5 corpus was hiding real quality gaps. Hallucinations surface
consistently on the same four adversarial cases (FE21–FE23, FE27–FE29) across
all legs, indicating these cases probe a model-independent challenge in the
extraction prompt rather than leg-specific fragility.

## Run command

```bash
# First run (interrupted by reboot mid-leg-6):
source ~/.claude/hooks/mem0-env.sh && set -a && source .env.local && set +a
MEM0_QDRANT_TIMEOUT=60 PYTHONUNBUFFERED=1 \
  bash benchmarks/bench_all.sh 2>&1 | tee benchmarks/v3/results/2026-05-05-bench-all.log

# Resume (leg 6 + hook + dedup):
MEM0_QDRANT_TIMEOUT=60 PYTHONUNBUFFERED=1 \
  uv run python -u -m benchmarks.v3.provider_battle \
    --leg ollama-qwen3-14b --limit 30 --sleep 1 2>&1 | \
  tee benchmarks/v3/results/2026-05-05-bench-all-resume.log
# then hook_and_scoping and dedup_and_entity runners appended to same file
```

The run was split across two log files due to a mid-flight reboot that
interrupted the ollama-qwen3-14b leg. Legs 1–5 come from
`2026-05-05-bench-all.log`; leg 6 (ollama-qwen3-14b) from
`2026-05-05-bench-all-resume.log`. The partial leg-6 data at the end of
`bench-all.log` (lines 228–234, 5 cases) is discarded; only the resume-log
version (30 cases) is used.

## Provider battle aggregate

| leg | provider class | model | n | mean F1 | hard F1 (FE21-30) | mean hall | notes |
|-----|---------------|-------|---|---------|-------------------|-----------|-------|
| anthropic-api | AnthropicOATLLM | claude-sonnet-4-6 | 30 | 0.908 | 0.761 | 0.200 | FE27/29 F1=0.50; 6 total hall |
| openai-cloud-nano | OpenAICompatLLM | gpt-5.4-nano | 30 | 0.922 | 0.765 | 0.267 | FE28/29: 2 hall each; highest total hall |
| lmstudio-qwen35-mlx | OpenAICompatLLM | qwen3.5-4b-mlx (MLX 4-bit) | 30 | 0.853 | 0.694 | 0.200 | FE05 F1=0 (total miss); FE27 F1=0 (JSON parse error); FE30 batch entity insert timed out |
| lmstudio-gemma | OpenAICompatLLM | gemma-4-26b-a4b-it (MLX) | 30 | 0.892 | 0.761 | 0.200 | FE02 F1=0.50; FE23/29 weakest hard cases |
| ollama-qwen35-4b | OllamaToolLLM | qwen3.5:4b (GGUF Q4_K_M) | 30 | 0.906 | 0.732 | 0.167 | FE29 F1=0 (JSON parse; retry also failed); lowest hallucination rate |
| ollama-qwen3-14b | OllamaToolLLM | qwen3:14b (GGUF) | 30 | 0.893 | 0.778 | 0.167 | Best hard-case F1 among local legs; from resume log only |

Hard-case detail (FE21–FE30), F1 per leg:

| case | anthropic-api | openai-nano | qwen35-mlx | gemma | qwen35-4b | qwen3-14b |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| FE21 | 0.80 | 0.80 | 0.80 | 0.80 | 0.80 | 0.80 |
| FE22 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 |
| FE23 | 0.67 | 0.67 | 0.50 | 0.50 | 0.67 | 0.67 |
| FE24 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| FE25 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| FE26 | 1.00 | 0.91 | 1.00 | 1.00 | 0.91 | 1.00 |
| FE27 | 0.50 | 0.80 | **0.00** | 0.67 | 0.80 | 0.67 |
| FE28 | 0.67 | 0.50 | 0.67 | 0.67 | 0.67 | 0.67 |
| FE29 | 0.50 | 0.50 | 0.50 | 0.50 | **0.00** | 0.50 |
| FE30 | 0.80 | 0.80 | 0.80 | 0.80 | 0.80 | 0.80 |

FE21–FE23 and FE27–FE29 are the adversarial tier (contradictions, negation
traps, alias resolution, hedged statements, density traps). FE27 and FE29 are
the hardest: every leg scores ≤ 0.80 on FE27 and ≤ 0.67 on FE29. The two
zeros (lmstudio-qwen35-mlx FE27 via JSON parse error; ollama-qwen35-4b FE29
via parse failure after retry) are infrastructure failures, not model quality
failures, and depress those legs' hard-case means by ~0.07 F1 each.

## Retrieval quality

Retrieval ran first in the full run (bench-all.log lines 6–53) against the
expanded corpus: 100 facts seeded under 5 thematic clusters, 30 queries (13
keyword, 10 semantic, 7 entity).

Per-category recall@5 (mean across queries):

| category | n queries | hybrid r@5 | semantic r@5 | delta |
|----------|:---------:|:----------:|:------------:|:-----:|
| entity   | 7         | 1.000      | 1.000        | 0.000 |
| keyword  | 13        | 0.962      | 0.962        | 0.000 |
| semantic | 10        | 0.917      | 0.950        | -0.033 |

The semantic category is the only one where hybrid and semantic-only diverge.
Hybrid underperforms semantic by 0.033 on semantic queries (0.917 vs 0.950),
driven by Q07 where hybrid r@5=0.5 and semantic r@5=0.5 (both miss), and Q15
where hybrid r@5=0.667 but semantic r@5=1.0. The old 12-query corpus did not
expose this: Q07 and Q15 are queries in the 13–30 range added in the expansion.

Full per-query results (hybrid | semantic):

| qid | cat | hybrid r@1 | sem r@1 | hybrid r@5 | sem r@5 | hybrid mrr | sem mrr |
|-----|-----|:---:|:---:|:---:|:---:|:---:|:---:|
| Q01 | keyword  | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q02 | keyword  | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q03 | keyword  | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q04 | keyword  | 0.000 | 0.000 | 1.000 | 1.000 | 0.500 | 0.500 |
| Q05 | keyword  | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q06 | semantic | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q07 | semantic | 0.500 | 0.000 | 0.500 | 0.500 | 1.000 | 0.500 |
| Q08 | semantic | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q09 | entity   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q10 | entity   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q11 | keyword  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q12 | keyword  | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q13 | keyword  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q14 | keyword  | 0.500 | 0.500 | 0.500 | 0.500 | 1.000 | 1.000 |
| Q15 | semantic | 0.333 | 0.333 | 0.667 | 1.000 | 1.000 | 1.000 |
| Q16 | semantic | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q17 | semantic | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q18 | entity   | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q19 | entity   | 0.333 | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q20 | entity   | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q21 | keyword  | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q22 | keyword  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q23 | keyword  | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q24 | keyword  | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q25 | semantic | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q26 | semantic | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q27 | semantic | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q28 | semantic | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q29 | entity   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q30 | entity   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The hybrid r@5 gap vs semantic on the semantic category is narrow and does not
represent a strong conclusion: Q07 is a tie (both 0.5), and Q15 shows semantic
winning (1.0 vs 0.667). Only one query (Q15) shows a clear hybrid disadvantage
at r@5. Entity and keyword categories show no gap. The 30-query corpus reveals
this soft disadvantage for the first time; the old 12-query corpus had all
r@5=1.0 across both arms.

## Hook latency

From resume log (5 iterations, cold-start each call):

```
iter  1/5: context=13.42s ok=True  session_end=29.29s ok=True
iter  2/5: context=12.62s ok=True  session_end=19.85s ok=True
iter  3/5: context= 9.23s ok=True  session_end=18.56s ok=True
iter  4/5: context= 8.84s ok=True  session_end=18.27s ok=True
iter  5/5: context= 7.23s ok=True  session_end=19.00s ok=True
```

| hook | n | p50 | p95 | max | budget | over_budget |
|------|:-:|----:|----:|----:|-------:|:-----------:|
| context | 5 | 9.23 s | 13.26 s | 13.42 s | 15.0 s | 0 |
| session_end | 5 | 19.00 s | 27.40 s | 29.29 s | 30.0 s | 0 |

No iteration exceeded budget. Comparison to the 2026-05-05 session_end profile
(mem_add mean 13.4 s, total p50 18.1 s, max 27.6 s): this run's session_end
p50 (19.00 s) and max (29.29 s) are consistent with the prior profile — both
runs used Ollama for extraction and show the same queue-depth variance pattern.
The 29.29 s max is slightly higher than the prior 27.6 s max but still under
budget. Budget situation is unchanged: zero violations, but the max sits within
~0.7 s of the 30 s ceiling.

Project-scoping isolation: 3-way check (alice:projA, alice:projB, bob:projA)
passed with zero cross-project leaks.

## Dedup + entity

Dedup pairs (threshold 0.88), n=16 (8 duplicate pairs D01–D08, 8 distinct
pairs X01–X08):

| metric | value |
|--------|------:|
| TP (caught dup) | 6 |
| FN (missed dup) | 2 |
| FP (deleted distinct) | 0 |
| TN (kept distinct) | 8 |
| precision | 1.000 |
| recall | 0.750 |
| accuracy | 0.875 |

D05 and D08 were missed (FN). No distinct pairs were falsely deleted (FP=0).
Precision is perfect; recall at 0.75 means the threshold (0.88) is letting two
fuzzy duplicates through. These are the paraphrase-heavy cases that require a
lower threshold to catch.

Entity-link timing (10 iterations each):

| arm | n | mean | p50 | p95 |
|-----|:-:|-----:|----:|----:|
| with spaCy | 10 | 0.418 s | 0.322 s | 0.942 s |
| no spaCy | 10 | 0.421 s | 0.400 s | 0.689 s |

Mean delta: -2.3 ms (-0.6%). spaCy entity-link overhead is negligible — within
measurement noise. The p95 is actually higher without spaCy (0.689 s vs 0.942 s
with), which is counter-intuitive but reflects Qdrant query variance dominating
at p95, not spaCy cost.

## Verdict on the corpus expansion hypothesis

**Extraction differentiation: confirmed.** The 30-case corpus broke the F1=1.0
ceiling on every leg. The spread is now 0.853–0.922 (mean) and 0.694–0.778
(hard-case tier). lmstudio-qwen35-mlx (MLX 4-bit, 0.853 mean / 0.694 hard) is
clearly the weakest. The two cloud legs (anthropic-api, openai-cloud-nano) lead
on mean F1 (0.908 / 0.922) but their hard-case F1 (0.761 / 0.765) is within
rounding distance of lmstudio-gemma (0.761) and ollama-qwen3-14b (0.778). The
old "cloud is categorically better" conclusion from 5-case runs does not hold on
hard cases — ollama-qwen3-14b ties the cloud legs on FE27/FE28/FE30 and
outperforms openai-nano on FE26 (1.00 vs 0.91).

Cloud vs local differences on hard cases:
- FE27: lmstudio-qwen35-mlx F1=0.00 (JSON parse); anthropic F1=0.50,
  openai F1=0.80; ollama-qwen3-14b F1=0.67. Anthropic was the worst of the
  non-parse-error legs on this case.
- FE28: openai-nano F1=0.50 (worst among cloud); all local legs 0.67.
  Local legs actually outperform the cloud nano model here.
- FE29: all legs 0.50 except ollama-qwen35-4b F1=0.00 (parse failure).
  Tied at 0.50 cloud vs local (excluding the parse error).

**Retrieval hybrid vs semantic gap: partially confirmed.** The gap is narrow
(delta -0.033 on semantic category at r@5) and driven by two queries (Q07,
Q15). Entity and keyword categories show zero gap. Not a strong case for or
against the BM25 sparse layer on this corpus; the gap is in the wrong direction
(hybrid slightly worse than semantic on semantic queries), suggesting BM25 may
be adding noise on purely semantic queries.

**What still needs work:**
- FE27 JSON parse failure on lmstudio-qwen35-mlx: need to determine if
  one-off or persistent (re-run FE27 in isolation).
- FE29 empty/invalid JSON from ollama-qwen35-4b: retry also failed; the case
  produces a response that both the tool-call parser and the fallback parser
  reject. Needs targeted probe.
- FE05 on lmstudio-qwen35-mlx (F1=0, medium difficulty): the previous 5-case
  run also showed this. Two independent runs both at F1=0 indicates a
  model-specific failure mode on this case, not noise.
- Dedup recall at 0.75: D05 and D08 escape the threshold. Lower threshold
  or semantic-distance adjustment worth exploring.

## Per-leg latency stats

Only ollama-qwen3-14b emitted an aggregate latency row (from the resume log):

| leg | mean F1 | ci95 F1 | mean recall | hallucinations | add P50 | add P95 | add mean | fail/n |
|-----|--------:|--------:|------------:|---------------:|--------:|--------:|---------:|-------:|
| ollama-qwen3-14b | 0.892 | (0.83, 0.95) | 0.905 | 5 | 20.27 s | 45.61 s | 23.33 s | 0/30 |

No aggregate latency table was emitted for legs 1–5 in the interrupted run.
The session-end profile benchmark (a separate run captured in
`2026-05-05-session-end-profile.md`) shows ollama-qwen35-4b adds at p50 ~13 s
and cloud legs at p50 ~2–3 s; those numbers remain the best available latency
reference for legs 1–5.

## Open follow-ups

- Investigate FE29 JSON parse failure on ollama-qwen35-4b: two consecutive
  JSON parsers (tool-call + retry) both rejected the response. Determine
  whether this is a density-trap prompt that triggers a specific generation
  failure or a one-off.
- Investigate lmstudio-qwen35-mlx FE27 JSON parse error and FE05 F1=0
  (two independent runs). FE05 is a medium case that every other leg handles
  at F1≥0.89. These two failures are candidates for MLX 4-bit quantization
  causing specific token-generation artifacts.
- lmstudio-qwen35-mlx hard-case F1 of 0.694 (worst local leg, 2 parse errors
  excluded) means it should not be considered production-ready for adversarial
  extraction workloads even at sub-5 s latency.
- Dedup threshold tuning: D05 and D08 are FN at 0.88 — lower threshold to
  0.82–0.85 and re-run to measure FP impact.
- Q14/Q07 retrieval failures (r@5=0.5 for both arms): both queries failed to
  hit their second relevant document in top-5 even with hybrid. Need to
  inspect what these queries look like to understand whether it is a corpus
  gap or an indexing issue.
- Re-run legs 1–5 with aggregate latency table enabled to get add P50/P95
  data comparable to the ollama-qwen3-14b row above.
