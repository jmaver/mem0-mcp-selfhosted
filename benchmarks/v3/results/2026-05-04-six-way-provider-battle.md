# 2026-05-04 — Six-way provider extraction battle

Same `FACT_EXTRACTION_CASES` corpus (5 cases, easy + medium), same 768-dim
nomic embedder on LM Studio, fresh `mem0_bench_v3` Qdrant collection per leg.
All legs run via the new `benchmarks/v3/legs.toml` named-leg system with
hermetic env (each leg fully owns its `MEM0_LLM_*` config).

## Results

| Leg | Provider class | Model | F1 | hallucinations | add P50 | add P95 |
| --- | --- | --- | --- | --- | --- | --- |
| anthropic-api | AnthropicOATLLM | claude-sonnet-4-6 | **1.000** | 0 | **2.17 s** | 7.33 s |
| openai-cloud-nano | OpenAICompatLLM | gpt-5.4-nano | **1.000** | 0 | 3.93 s | 4.49 s |
| ollama-qwen35-4b | OllamaToolLLM | qwen3.5:4b (Q4_K_M GGUF) | **1.000** | 0 | 15.35 s | 17.40 s |
| lmstudio-gemma | OpenAICompatLLM | gemma-4-26b-a4b-it (MLX) | 0.900 | 0 | 4.54 s | 5.82 s |
| ollama-qwen3-14b | OllamaToolLLM | qwen3:14b (GGUF) | 0.900 | 0 | 15.98 s | 46.56 s |
| lmstudio-qwen35-mlx | OpenAICompatLLM | qwen3.5-4b-mlx (MLX 4-bit) | 0.760 | 0 | **1.90 s** | 4.84 s |

## Surprising findings

**1. Quantization format matters more than model size.**
qwen3.5:4b on Ollama (Q4_K_M GGUF) hit **F1 = 1.0**, while the same nominal
4 B model on LM Studio (`qwen3.5-4b-mlx`, MLX 4-bit) hit **F1 = 0.76**. Same
training, same parameter count, different compression — the MLX 4-bit quant
is measurably lossier for fact-extraction tasks. Worth knowing if you've
been picking MLX for performance reasons; you're paying ~25 % accuracy.

**2. Smaller Qwen beat larger Qwen on the same backend.**
Ollama qwen3.5:4b (1.0) > Ollama qwen3:14b (0.9). Counter-intuitive but
consistent: qwen3.5 is a newer-generation tune. The 14 B model loses one
case (FE02) where the 4 B model gets it right. Suggests the v3 default
(`qwen3:14b` from `config.py:89`) should be revisited.

**3. Cloud beats local on speed, even for tiny models.**
Anthropic Sonnet 4.6 P50 = 2.17 s; OpenAI gpt-5.4-nano P50 = 3.93 s; LM
Studio's MLX 4-bit Qwen3.5-4B P50 = 1.90 s. The local MLX wins by
~270 ms but loses ~24 percentage points of F1. Cloud APIs are
network-bound on a fast home network; local extraction is compute-bound on
a 4-bit quant struggling with the task.

**4. The hermetic-legs bug was a real foot-gun.**
Initial runs of `anthropic-api` returned in 0.04 s with empty content.
Cause: shell had `MEM0_LLM_URL=http://lm-studio:1234/v1` from
`mem0-env.sh`, and `config.py:104` reads `MEM0_LLM_URL` for the Anthropic
provider as `anthropic_base_url`. Requests went to LM Studio (which
doesn't host Sonnet) and got fast empty responses. Fixed by making the
leg system explicitly clear unset env vars rather than inherit them from
the shell.

## Production recommendations

- **For hooks (where the 30 s session-end budget was hit in baseline)**:
  switch to `anthropic-api` if cost permits — fastest at 2.17 s P50, perfect
  quality. Or `openai-cloud-nano` for the cost/quality sweet spot.
- **For local-only deployments**: `ollama-qwen35-4b` is the right pick.
  Perfect F1, ~15 s latency. Stop using the v0.3 default `qwen3:14b` — it's
  slower *and* less accurate.
- **Avoid MLX 4-bit for extraction tasks** — the 25-point F1 gap vs GGUF
  is too high a price for ~250 ms of latency.

## Bugs found and fixed during this run

1. **`config.py` didn't plumb `MEM0_LLM_URL` to the Anthropic provider.**
   Fixed in `config.py:96-103`: now sets `anthropic_base_url` when
   `MEM0_LLM_URL` is set with `MEM0_LLM_PROVIDER=anthropic`.

2. **`AnthropicOATConfig` rewrap dropped custom fields.**
   When mem0ai passed a plain `BaseLlmConfig`, the rewrap at
   `llm_anthropic.py:174` reconstructed it without `auth_token` /
   `anthropic_base_url`. Fixed via `getattr` with default `None`.

3. **`_supports_structured_output()` didn't check token type.**
   `output_config` requires the `anthropic-beta` header that's only sent
   for OAT tokens. Real API keys silently got empty responses. Fixed:
   the check now requires both the model prefix AND OAT auth.

4. **No 429 retry on Anthropic.**
   Existing retry list was `(500, 502, 503, 529)`. Added 429 with
   conservative one-shot retry honoring `Retry-After` header. Doesn't
   help subscription-tier OAT (which 429s on every burst) but reduces
   nuisance failures on paid tier under steady-state load.

5. **`OpenAICompatLLM` sent `max_tokens` to gpt-5.x.**
   mem0's upstream `_is_reasoning_model` excludes gpt-5.x explicitly. We
   patched the gap by overriding `_get_common_params` to drop
   `max_tokens` for `gpt-5*` and `chatgpt-5*` model names.

6. **Leg system inherited shell env (the hermetic-legs bug above).**
   Fixed `_run_leg` + `make_memory` to use a `_UNSET` sentinel. Omitted
   leg fields → kwarg omitted → env left alone; explicit `None` /
   `""` → env var is unset.

## Outstanding

- `anthropic-oat` leg still 429s on subscription tier. The retry path is
  working; it's a tier limitation.
- `anthropic-qwen36` leg returns content in a shape `_extract_text_block`
  doesn't recognize. Needs per-endpoint investigation — may need provider
  class extension or response-shape detection.
