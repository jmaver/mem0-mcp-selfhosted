"""Probe: FE05 (personal/medium) × lmstudio-qwen35-mlx — 5 iterations.

FE05 input: "I have two cats named Luna and Mochi, both are 3 years old."
Expected:   ["luna", "mochi", "cats"]
Observed:   F1=0.00, recall=0.00 in both the 2026-05-04 and 2026-05-05 bench runs.

This probe:
  1. Calls mem.add() 5× with the same input via the full mem0 pipeline.
  2. Captures the list of extracted memories after each call.
  3. Also calls the LLM directly with the exact extraction prompt to capture
     what the model returns before any mem0 post-processing.
  4. Computes per-iteration F1 using the benchmark framework grader.
  5. Prints a summary verdict.

Run:
    uv run python -u -m benchmarks.v3._probes.probe_fe05_mlx
"""

from __future__ import annotations

import os
import time

# Suppress mem0 telemetry
os.environ.setdefault("MEM0_TELEMETRY", "false")

from openai import OpenAI

from benchmarks.v3._corpora import FACT_EXTRACTION_CASES
from benchmarks.v3.framework import (
    bench_user_id,
    make_memory,
    score_fact_extraction,
)
from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT, generate_additive_extraction_prompt
from mem0_mcp_selfhosted.llm_openai_compat import _clean_response
from mem0_mcp_selfhosted.helpers import safe_bulk_delete

# --- Config matching legs.toml [legs.lmstudio-qwen35-mlx] ---
LLM_URL = "http://192.168.200.84:1234/v1"
MODEL = "qwen3.5-4b-mlx"
CASE = next(c for c in FACT_EXTRACTION_CASES if c.id == "FE05")
N_ITERS = 5
COLLECTION = "mem0_probe_fe05_mlx"


def _raw_llm_response() -> str:
    """Call the LLM directly with the extraction prompt and return raw text."""
    client = OpenAI(base_url=LLM_URL, api_key="not-needed")
    system_prompt = ADDITIVE_EXTRACTION_PROMPT.strip()
    user_prompt = generate_additive_extraction_prompt(CASE.messages, [])
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "text"},  # LM Studio doesn't support json_object
        max_tokens=512,
        timeout=60,
    )
    return resp.choices[0].message.content or ""


def main() -> int:
    print(f"=== probe_fe05_mlx: FE05 × lmstudio-qwen35-mlx (N={N_ITERS}) ===")
    print(f"model:   {MODEL}")
    print(f"url:     {LLM_URL}")
    print(f"input:   {CASE.messages[0]['content']!r}")
    print(f"expected:{CASE.expected_facts}")
    print()

    # --- Part 1: direct LLM probe (3 samples) ---
    print("--- Direct LLM responses (no mem0 pipeline) ---")
    for i in range(3):
        try:
            raw = _raw_llm_response()
            cleaned = _clean_response(raw)
            print(f"[direct {i+1}] raw ({len(raw)} chars):")
            # Show full response — key data for diagnosis
            print(f"  RAW: {raw!r}")
            print(f"  CLEANED: {cleaned!r}")
        except Exception as exc:
            print(f"[direct {i+1}] ERROR: {exc}")
        print()

    # --- Part 2: full mem0 pipeline × N_ITERS ---
    print("--- Full mem0 pipeline (mem.add → mem.search → score) ---")
    try:
        mem, cleanup_env = make_memory(
            provider="openai",
            model=MODEL,
            llm_url=LLM_URL,
            llm_api_key=None,
            collection=COLLECTION,
        )
    except Exception as exc:
        print(f"make_memory failed: {exc}")
        return 1

    f1_scores: list[float] = []
    try:
        for i in range(N_ITERS):
            uid = bench_user_id(f"fe05-probe-{i}")
            try:
                mem.add(messages=CASE.messages, user_id=uid, infer=True)
            except Exception as exc:
                print(f"[iter {i+1}] ADD failed: {exc}")
                f1_scores.append(0.0)
                continue

            try:
                raw = mem.search(
                    query=" ".join(CASE.expected_facts),
                    filters={"user_id": uid},
                    top_k=10,
                )
                memories = raw.get("results", []) if isinstance(raw, dict) else raw
            except Exception as exc:
                print(f"[iter {i+1}] SEARCH failed: {exc}")
                f1_scores.append(0.0)
                safe_bulk_delete(mem, {"user_id": uid})
                continue

            score = score_fact_extraction(memories, CASE)
            f1_scores.append(score["f1"])

            mem_texts = [m.get("memory", "") for m in memories]
            print(f"[iter {i+1}] F1={score['f1']:.2f} recall={score['recall']:.2f} "
                  f"hallucinated={score['hallucinated']} extracted={score['extracted_count']}")
            for j, t in enumerate(mem_texts):
                print(f"           mem[{j}]: {t!r}")

            safe_bulk_delete(mem, {"user_id": uid})
            time.sleep(1)
    finally:
        cleanup_env()

    # --- Summary ---
    print()
    print("=== SUMMARY ===")
    print(f"F1 scores across {N_ITERS} iterations: {[round(s, 2) for s in f1_scores]}")
    fail_count = sum(1 for s in f1_scores if s == 0.0)
    print(f"Zero-F1 count: {fail_count}/{N_ITERS}")
    if fail_count == N_ITERS:
        print("VERDICT: PERSISTENT failure — 5/5 zero-F1. Model limitation, not one-off noise.")
    elif fail_count >= 3:
        print("VERDICT: MOSTLY PERSISTENT failure — likely a model limitation.")
    elif fail_count >= 1:
        print("VERDICT: INTERMITTENT — could be transient or model instability.")
    else:
        print("VERDICT: NO FAILURE — original bench result was likely a one-off transient.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
