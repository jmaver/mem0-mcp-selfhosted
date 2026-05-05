"""Probe: FE29 (multi-contradiction/hard) × ollama-qwen35-4b — 5 iterations.

FE29 input: "The service originally ran on MySQL 5.7, then we migrated to MySQL 8.0,
             and last week we completed the move to PostgreSQL 16."
Expected:   ["postgresql 16"]
Anti-facts: ["mysql 5.7", "mysql 8.0"]
Observed:   F1=0.00 in 2026-05-05 bench. Log shows:
  "Empty or invalid JSON from Ollama, retrying once"
  "Retry also returned empty/invalid JSON — returning as-is"

Both the initial call and the retry failed.  This probe:
  1. Calls the Ollama LLM directly 5× to capture the raw model output.
  2. Examines: is the response empty? Non-JSON text? Unclosed think block?
     Truncated JSON?  Something else?
  3. Runs the full mem0 pipeline 5× to get F1 scores.
  4. Tests whether the OllamaToolLLM's extract_json() could recover the output
     even though _is_json_valid() rejected it (e.g., {} empty dict edge case).

Run:
    uv run python -u -m benchmarks.v3._probes.probe_fe29_ollama
"""

from __future__ import annotations

import json
import os
import time

# Suppress mem0 telemetry
os.environ.setdefault("MEM0_TELEMETRY", "false")

import ollama as _ollama

from benchmarks.v3._corpora import FACT_EXTRACTION_CASES
from benchmarks.v3.framework import (
    bench_user_id,
    make_memory,
    score_fact_extraction,
)
from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT, generate_additive_extraction_prompt
from mem0_mcp_selfhosted.llm_ollama import extract_json, _strip_think_tags
from mem0_mcp_selfhosted.helpers import safe_bulk_delete

# --- Config matching legs.toml [legs.ollama-qwen35-4b] ---
OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen3.5:4b"
CASE = next(c for c in FACT_EXTRACTION_CASES if c.id == "FE29")
N_ITERS = 5
COLLECTION = "mem0_probe_fe29_ollama"


def _raw_ollama_response() -> str:
    """Call Ollama directly with the extraction prompt (JSON mode) and return raw content."""
    client = _ollama.Client(host=OLLAMA_URL)
    system_prompt = ADDITIVE_EXTRACTION_PROMPT.strip()
    user_prompt = generate_additive_extraction_prompt(CASE.messages, [])

    # Match OllamaToolLLM's generate_response() for JSON mode:
    #   - format="json"
    #   - /no_think appended
    #   - "Please respond with valid JSON only." appended
    #   - temperature=0, repeat_penalty=1.0
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt + " /no_think\n\nPlease respond with valid JSON only."},
    ]

    resp = client.chat(
        model=MODEL,
        messages=messages,
        format="json",
        options={
            "num_predict": 512,
            "temperature": 0,
            "repeat_penalty": 1.0,
        },
        keep_alive="30m",
    )
    content = resp.message.content or ""
    return content


def _analyze_response(raw: str) -> dict:
    """Analyze a raw Ollama response for common failure modes."""
    stripped = _strip_think_tags(raw)
    extracted = extract_json(stripped)

    result = {
        "raw_len": len(raw),
        "stripped_len": len(stripped),
        "extracted_len": len(extracted),
        "raw": raw,
        "stripped": stripped,
        "extracted": extracted,
        "is_empty": not raw.strip(),
        "has_think_tag": "<think>" in raw,
        "has_unclosed_think": "<think>" in raw and "</think>" not in raw,
        "parse_raw": None,
        "parse_stripped": None,
        "parse_extracted": None,
        "extracted_is_valid_memory": False,
    }

    for key, text in [("parse_raw", raw), ("parse_stripped", stripped), ("parse_extracted", extracted)]:
        try:
            parsed = json.loads(text, strict=False)
            if parsed == {}:
                result[key] = "EMPTY_DICT"
            elif isinstance(parsed, dict) and "memory" in parsed:
                result[key] = f"OK — {len(parsed.get('memory', []))} memories"
                if key == "parse_extracted":
                    result["extracted_is_valid_memory"] = True
            else:
                result[key] = f"OK — keys={list(parsed.keys())[:5]}"
        except json.JSONDecodeError as e:
            result[key] = f"FAIL — {e}"
        except Exception as e:
            result[key] = f"ERROR — {e}"

    return result


def main() -> int:
    print(f"=== probe_fe29_ollama: FE29 × ollama-qwen35-4b (N={N_ITERS}) ===")
    print(f"model:   {MODEL}")
    print(f"url:     {OLLAMA_URL}")
    print(f"input:   {CASE.messages[0]['content']!r}")
    print(f"expected:{CASE.expected_facts}")
    print(f"anti:    {CASE.hallucination_traps}")
    print()

    # --- Part 1: direct Ollama probe ---
    print("--- Direct Ollama responses (5 samples in JSON mode) ---")
    parse_failures = 0
    empty_responses = 0
    unclosed_think = 0
    empty_dict_count = 0

    for i in range(N_ITERS):
        try:
            raw = _raw_ollama_response()
            analysis = _analyze_response(raw)

            print(f"[sample {i+1}]")
            print(f"  raw ({analysis['raw_len']} chars): {raw!r}")
            print(f"  stripped ({analysis['stripped_len']} chars): {analysis['stripped']!r}")
            print(f"  extracted ({analysis['extracted_len']} chars): {analysis['extracted']!r}")
            print(f"  parse_raw:       {analysis['parse_raw']}")
            print(f"  parse_stripped:  {analysis['parse_stripped']}")
            print(f"  parse_extracted: {analysis['parse_extracted']}")

            flags = []
            if analysis["is_empty"]:
                empty_responses += 1
                flags.append("EMPTY_RESPONSE")
            if analysis["has_unclosed_think"]:
                unclosed_think += 1
                flags.append("UNCLOSED_THINK")
            if analysis["parse_extracted"] == "EMPTY_DICT":
                empty_dict_count += 1
                flags.append("EMPTY_DICT_AFTER_EXTRACT")
            if "FAIL" in str(analysis["parse_extracted"]):
                parse_failures += 1
                flags.append("PARSE_FAIL")

            print(f"  flags: {flags if flags else ['none — clean response']}")

        except Exception as exc:
            print(f"[sample {i+1}] LLM ERROR: {exc}")
        print()

    # --- Part 2: full mem0 pipeline × N_ITERS ---
    print("--- Full mem0 pipeline (mem.add → mem.search → score) ---")
    try:
        mem, cleanup_env = make_memory(
            provider="ollama",
            model=MODEL,
            llm_url=OLLAMA_URL,
            collection=COLLECTION,
        )
    except Exception as exc:
        print(f"make_memory failed: {exc}")
        return 1

    f1_scores: list[float] = []
    try:
        for i in range(N_ITERS):
            uid = bench_user_id(f"fe29-probe-{i}")
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
    print(f"Direct LLM diagnostics ({N_ITERS} samples):")
    print(f"  empty responses:   {empty_responses}")
    print(f"  unclosed <think>:  {unclosed_think}")
    print(f"  empty dict ({{}}):  {empty_dict_count}")
    print(f"  JSON parse fails:  {parse_failures}")

    print()
    if fail_count == N_ITERS:
        print("VERDICT: PERSISTENT failure — 5/5 zero-F1.")
        if unclosed_think > 0:
            print("  Root cause: unclosed <think> block truncates JSON output.")
            print("  Fix: _strip_think_tags() should strip content from <think> to EOF.")
            print("  Note: OllamaToolLLM._strip_think_tags() already does this for")
            print("  unclosed tags — check if /no_think injection is working.")
        elif empty_responses > 0:
            print("  Root cause: model returns empty string (no output).")
            print("  This is a model-capability limitation for this specific case.")
        elif empty_dict_count > 0:
            print("  Root cause: model returns {} (empty dict) — no memories extracted.")
            print("  The prompt triggers 'no memorable facts' judgment in this model.")
            print("  This is a model-capability limitation.")
        elif parse_failures > 0:
            print("  Root cause: malformed JSON not recoverable by current pipeline.")
        else:
            print("  Root cause unclear — no direct failures but pipeline F1=0. Check mem0 internals.")
    elif fail_count >= 3:
        print("VERDICT: MOSTLY PERSISTENT — likely a model limitation on this prompt.")
    else:
        print("VERDICT: INTERMITTENT — original bench result may have been stochastic.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
