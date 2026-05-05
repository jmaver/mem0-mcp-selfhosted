# 2026-05-05 — SessionEnd hook cold-start profile

`mem_add` (extraction LLM + mem0 internal dedup) is the dominant cost at mean
13.4 s (p95 15.4 s), followed by `memory_init` at mean 4.7 s (p95 5.1 s).
`stdin_parse`, `transcript_read`, and `dedup_post` are negligible. The two
phases together account for 97–98 % of wall time. The prior benchmark's p50
~30.9 s reflected heavier Ollama load; today's runs landed in the 12–28 s
range. No iteration exceeded the 30 s budget, but iter 1 (27.6 s wall, 21.1 s
mem_add) came close and would exceed budget under any additional queue delay.

Run command:

```bash
source ~/.claude/hooks/mem0-env.sh
set -a; source .env.local; set +a
export MEM0_QDRANT_TIMEOUT=60 PYTHONUNBUFFERED=1
uv run python -u -m benchmarks.v3.hook_and_scoping --iterations 5 2>&1 | tee /tmp/session_end_profile.log
```

Profile lines read from:
`/var/folders/hx/2jkq2lp12z7gwfk1h17cy5k80000gn/T/mem0-hook-context.log`
(via `_log_hook_event("session_end", "profile.*=...")`)

## Per-iteration breakdown

| iter | stdin_parse | transcript_read | memory_init | mem_add | dedup_post | total |
| ---: | ----------: | --------------: | ----------: | ------: | ---------: | ----: |
| 1 | 0.000 s | 0.000 s | 5.147 s | 21.128 s | 1.300 s | 27.577 s |
| 2 | 0.000 s | 0.000 s | 4.878 s | 15.392 s | 0.315 s | 20.587 s |
| 3 | 0.000 s | 0.000 s | 4.875 s | 13.203 s | 0.000 s | 18.080 s |
| 4 | 0.000 s | 0.000 s | 5.549 s | 8.346 s | 0.000 s | 13.897 s |
| 5 | 0.000 s | 0.000 s | 3.183 s | 9.063 s | 0.000 s | 12.247 s |

Wall-clock totals from `_time_hook`: 27.81, 20.88, 18.30, 14.11, 12.48 s
(~0.2–0.4 s overhead: subprocess fork + Python startup before `t_start`).

## Aggregate stats

| phase | mean | p50 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| stdin_parse | 0.000 s | 0.000 s | 0.000 s | 0.000 s |
| transcript_read | 0.000 s | 0.000 s | 0.000 s | 0.000 s |
| memory_init | 4.726 s | 4.878 s | 5.147 s | 5.549 s |
| mem_add | 13.426 s | 13.203 s | 15.392 s | 21.128 s |
| dedup_post | 0.323 s | 0.000 s | 0.315 s | 1.300 s |
| **total** | **18.478 s** | **18.080 s** | **20.587 s** | **27.577 s** |

## Bottleneck analysis

**`mem_add` dominates (73 % of mean total, 77 % of max total).** This covers
the Ollama LLM call for fact extraction plus mem0's internal hash-dedup pass
against the existing collection. The high variance (8.3–21.1 s) tracks Ollama
queue depth — the local model handles exactly one request at a time; anything
ahead in the queue adds directly to latency.

**`memory_init` is the second cost (26 % of mean total, 20 % of max total).**
It is stable (3.2–5.5 s across 5 cold starts). It covers: Python interpreter
startup (already counted in subprocess wall time), `import mem0` + transitive
heavy imports (torch-adjacent, spaCy), `Memory.from_config()` (Qdrant client
init + collection-ensure), and first embed call triggered by collection setup.
The `memory_init` window was tighter here than the prior benchmark's suspected
~22 s estimate — that estimate was likely wrong or reflected a different model
load path. spaCy `en_core_web_sm` contributes to this window but cannot be
isolated without sub-checkpoints inside `_get_memory()`.

**`dedup_post` is small but non-zero.** When memories are added (iters 1–2),
it fires per-memory search + possible delete (1.3 s, 0.3 s). When nothing is
added (iters 3–5, due to hash-dedup on identical synthetic transcript), it
short-circuits at 0.

**Mitigation options (not implemented):**
1. Switch from Ollama (local queue) to a cloud LLM for extraction — reduces
   `mem_add` p50 from ~13 s to ~2–3 s (see six-way battle results).
2. Sub-checkpoint `_get_memory()` to isolate import vs Qdrant-init vs first
   embed; at 4–5 s stable it is a secondary concern but worth quantifying.
3. The 0.2–0.4 s subprocess overhead (Python fork + early imports before
   `t_start`) is not instrumented — if `memory_init` were cut, this becomes
   visible. It is addressable only by running in a persistent process (daemon
   mode, not a subprocess per invocation).
4. Lazy spaCy load: if spaCy is loaded eagerly during `Memory.from_config()`,
   moving it to first-use only could shave 1–2 s from `memory_init`.

## Open follow-ups

- Add sub-checkpoints inside `_get_memory()`: (a) after `import mem0`, (b)
  after `build_config()` + `register_providers()`, (c) after
  `Memory.from_config()`. This will isolate import vs Qdrant-init vs embed.
- Re-run under controlled load (Ollama idle, no concurrent requests) to
  establish a true baseline for `mem_add` without queue variance.
- Re-run the original 2026-05-04 hook bench with profiling enabled to confirm
  whether the prior ~30.9 s p50 was load-driven (would show up as higher
  `mem_add`) or a different code path.
- If cloud LLM is adopted for extraction, re-profile to confirm `memory_init`
  becomes the new bottleneck (at ~5 s it would still fit in the 30 s budget
  but leaves little headroom for a cold Qdrant connection).
