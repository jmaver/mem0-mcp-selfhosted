"""Probe: FE27 (time-relative/hard) × lmstudio-qwen35-mlx — 5 iterations.

FE27 input: "I joined the team three months ago and am still ramping up on the codebase."
Expected:   ["joined", "three months"]
Observed:   F1=0.00 in 2026-05-05 bench. Error log shows:
  "Error parsing extraction response: Expecting ',' delimiter: line 1 column 122 (char 121)"

This suggests the model returned malformed JSON (not merely markdown-fenced, since
mem0's main.py already calls remove_code_blocks() before parsing).  This probe:
  1. Calls the LLM directly 5× to capture the raw model output.
  2. Inspects the output: is it malformed JSON? Missing commas? Trailing garbage?
  3. Runs the full mem0 pipeline 5× to get F1 scores.
  4. Checks whether our _clean_response() in llm_openai_compat.py could have
     saved a recovery that mem0's own parser missed.

Run:
    uv run python -u -m benchmarks.v3._probes.probe_fe27_mlx
"""

from __future__ import annotations

import json
import os
import re
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
CASE = next(c for c in FACT_EXTRACTION_CASES if c.id == "FE27")
N_ITERS = 5
COLLECTION = "mem0_probe_fe27_mlx"

def _raw_llm_response() -> str:
    """Call LLM directly with the extraction prompt and return raw text."""
    client = OpenAI(base_url=LLM_URL, api_key="not-needed")
    system_prompt = ADDITIVE_EXTRACTION_PROMPT.strip()
    user_prompt = generate_additive_extraction_prompt(CASE.messages, [])
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "text"},
        max_tokens=512,
        timeout=60,
    )
    return resp.choices[0].message.content or ""


def _try_parse(text: str) -> tuple[bool, str]:
    """Try json.loads; return (success, error_description)."""
    try:
        json.loads(text, strict=False)
        return True, "ok"
    except json.JSONDecodeError as e:
        return False, str(e)


def _try_recover(text: str) -> tuple[bool, str]:
    """Try common JSON recovery: strip think tags, code fences, trailing commas."""
    cleaned = _clean_response(text)

    # Remove trailing commas before closing brackets/braces
    trailing_comma_re = re.compile(r',\s*([}\]])')
    cleaned_no_tc = trailing_comma_re.sub(r'\1', cleaned)

    for candidate in [cleaned, cleaned_no_tc]:
        ok, _ = _try_parse(candidate)
        if ok:
            return True, candidate
    return False, cleaned


def main() -> int:
    print(f"=== probe_fe27_mlx: FE27 × lmstudio-qwen35-mlx (N={N_ITERS}) ===")
    print(f"model:   {MODEL}")
    print(f"url:     {LLM_URL}")
    print(f"input:   {CASE.messages[0]['content']!r}")
    print(f"expected:{CASE.expected_facts}")
    print()

    # --- Part 1: direct LLM probe (all 5 samples) ---
    print("--- Direct LLM responses (5 samples) ---")
    parse_failures = 0
    recover_ok = 0
    raw_responses: list[str] = []

    for i in range(N_ITERS):
        try:
            raw = _raw_llm_response()
            raw_responses.append(raw)
            cleaned = _clean_response(raw)

            ok_raw, err_raw = _try_parse(raw)
            ok_cleaned, err_cleaned = _try_parse(cleaned)
            recovered, recovered_text = _try_recover(raw)

            print(f"[sample {i+1}]")
            print(f"  raw ({len(raw)} chars): {raw!r}")
            print(f"  cleaned ({len(cleaned)} chars): {cleaned!r}")
            print(f"  parse raw:     {'OK' if ok_raw else 'FAIL — ' + err_raw}")
            print(f"  parse cleaned: {'OK' if ok_cleaned else 'FAIL — ' + err_cleaned}")
            print(f"  recovery:      {'OK → ' + repr(recovered_text[:120]) if recovered else 'FAIL'}")

            if not ok_raw:
                parse_failures += 1
            if recovered and not ok_raw:
                recover_ok += 1

        except Exception as exc:
            print(f"[sample {i+1}] LLM ERROR: {exc}")
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
            uid = bench_user_id(f"fe27-probe-{i}")
            try:
                mem.add(messages=CASE.messages, user_id=uid, infer=True)
            except Exception as exc:
                print(f"[iter {i+1}] ADD failed: {exc}")
                f1_scores.append(0.0)
                continue

            try:
                raw_result = mem.search(
                    query=" ".join(CASE.expected_facts),
                    filters={"user_id": uid},
                    top_k=10,
                )
                memories = raw_result.get("results", []) if isinstance(raw_result, dict) else raw_result
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
    print(f"Direct LLM parse failures: {parse_failures}/{N_ITERS}")
    print(f"Recoverable with trailing-comma fix: {recover_ok}/{parse_failures} of the failures")

    print()
    if parse_failures >= 3:
        if recover_ok > 0:
            print("VERDICT: PERSISTENT JSON parse failure. Recovery via trailing-comma strip is")
            print("  possible. Consider adding trailing-comma recovery to llm_openai_compat.py")
            print("  _clean_response() OR patching mem0's extraction parser.")
        else:
            print("VERDICT: PERSISTENT JSON parse failure. The malformed JSON is not recoverable")
            print("  by simple cleanup (not a fence/trailing-comma issue). Model limitation.")
    elif parse_failures >= 1:
        if recover_ok > 0:
            print("VERDICT: INTERMITTENT JSON parse failure, recoverable. Add trailing-comma fix.")
        else:
            print("VERDICT: INTERMITTENT parse failure, not easily recoverable. One-off or model instability.")
    else:
        print("VERDICT: NO PARSE FAILURE in direct samples. Original bench error was likely a one-off.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
