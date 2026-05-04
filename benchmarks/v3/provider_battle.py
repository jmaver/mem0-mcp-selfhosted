"""Anthropic OAT vs OpenAI-compat vs Ollama on `infer=True` extraction quality.

For each available provider:
1. Build a fresh ``mem0.Memory`` via ``framework.make_memory(provider=...)``.
2. Run ``FACT_EXTRACTION_CASES`` through ``mem.add(infer=True)`` and time
   each call.
3. Search for what landed and score with ``score_fact_extraction``.

Providers are skipped (not failed) when prerequisites are missing:
- anthropic: ``auth.resolve_token()`` returns None
- openai: ``MEM0_OPENAI_API_KEY`` and ``MEM0_LLM_URL`` both unset (no LM Studio
  / OpenAI key in env)
- ollama: ``check_ollama()`` returns None

Usage::

    uv run python -m benchmarks.v3.provider_battle [--limit N] \
        [--providers anthropic,ollama,openai]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

from benchmarks.v3._corpora import FACT_EXTRACTION_CASES
from benchmarks.v3.framework import (
    OperationTimer,
    bench_user_id,
    check_ollama,
    check_qdrant,
    ci95,
    make_memory,
    score_fact_extraction,
)
from mem0_mcp_selfhosted.helpers import safe_bulk_delete


def _provider_available(provider: str) -> tuple[bool, str]:
    """Return (available, reason-if-not)."""
    if provider == "anthropic":
        try:
            from mem0_mcp_selfhosted.auth import resolve_token
        except Exception as exc:
            return False, f"auth import failed: {exc}"
        return (True, "") if resolve_token() else (False, "no Anthropic token")
    if provider == "openai":
        if os.environ.get("MEM0_OPENAI_API_KEY") or os.environ.get("MEM0_LLM_URL"):
            return True, ""
        return False, "neither MEM0_OPENAI_API_KEY nor MEM0_LLM_URL set"
    if provider == "ollama":
        return (True, "") if check_ollama() else (False, "ollama unreachable")
    return False, f"unknown provider {provider!r}"


def _default_model(provider: str) -> str | None:
    """Fall back to the same defaults config.py uses, but allow env override."""
    env_model = os.environ.get(f"MEM0_BENCH_{provider.upper()}_MODEL")
    if env_model:
        return env_model
    if provider == "anthropic":
        return "claude-opus-4-6"
    if provider == "ollama":
        return "qwen3:14b"
    if provider == "openai":
        # OpenAI-compat needs a model name explicitly; fall back to a common LM Studio default.
        return os.environ.get("MEM0_LLM_MODEL") or "qwen3-14b"
    return None


def _extract_results(raw: Any) -> list[dict]:
    if isinstance(raw, dict):
        return raw.get("results", []) or []
    if isinstance(raw, list):
        return raw
    return []


def _run_provider(provider: str, cases: list, collection: str) -> dict[str, Any] | None:
    available, reason = _provider_available(provider)
    if not available:
        print(f"  [skip] {provider}: {reason}")
        return None

    model = _default_model(provider)
    print(f"\n=== {provider} ({model}) ===")
    mem, cleanup_env = make_memory(provider=provider, model=model, collection=collection)
    uid = bench_user_id(f"provider-{provider}")
    timer = OperationTimer()
    f1_scores: list[float] = []
    recalls: list[float] = []
    hallucinated_total = 0
    failures = 0

    try:
        for case in cases:
            try:
                t0 = time.perf_counter()
                mem.add(messages=case.messages, user_id=uid, infer=True,
                        metadata={"bench_case_id": case.id})
                timer.record("add", time.perf_counter() - t0)
            except Exception as exc:
                failures += 1
                print(f"  {case.id}: ADD failed: {type(exc).__name__}: {exc}")
                continue

            # Search for everything under this user; score against expected facts.
            try:
                t0 = time.perf_counter()
                raw = mem.search(query=" ".join(case.expected_facts) or case.id,
                                 filters={"user_id": uid}, top_k=10)
                timer.record("search", time.perf_counter() - t0)
            except Exception as exc:
                failures += 1
                print(f"  {case.id}: SEARCH failed: {type(exc).__name__}: {exc}")
                continue

            memories = _extract_results(raw)
            score = score_fact_extraction(memories, case)
            f1_scores.append(score["f1"])
            recalls.append(score["recall"])
            hallucinated_total += score["hallucinated"]
            print(f"  {case.id} ({case.difficulty}): F1={score['f1']:.2f} "
                  f"recall={score['recall']:.2f} hallucinated={score['hallucinated']}")

            # Per-case cleanup so search results don't bleed across cases.
            safe_bulk_delete(mem, {"user_id": uid})

        mean_f1, lo_f1, hi_f1 = ci95(f1_scores)
        latency = timer.summary()
        return {
            "provider": provider,
            "model": model,
            "n_cases": len(cases),
            "n_failures": failures,
            "mean_f1": round(mean_f1, 4),
            "ci95_f1": (round(lo_f1, 4), round(hi_f1, 4)),
            "mean_recall": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
            "hallucinations": hallucinated_total,
            "add_p50": latency.get("add", {}).get("p50", 0.0),
            "add_p95": latency.get("add", {}).get("p95", 0.0),
            "add_mean": latency.get("add", {}).get("mean", 0.0),
        }
    finally:
        try:
            safe_bulk_delete(mem, {"user_id": uid})
        except Exception as exc:
            print(f"  Cleanup failed: {exc}", file=sys.stderr)
        cleanup_env()


def _print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No providers ran.")
        return
    headers = ["provider", "model", "mean_f1", "ci95_f1", "mean_recall",
               "hallucinations", "add_p50", "add_p95", "add_mean", "fail/n"]
    widths = [len(h) for h in headers]
    table_rows = []
    for r in rows:
        row = [
            r["provider"], str(r["model"]), f"{r['mean_f1']:.3f}",
            f"({r['ci95_f1'][0]:.2f},{r['ci95_f1'][1]:.2f})",
            f"{r['mean_recall']:.3f}", str(r["hallucinations"]),
            f"{r['add_p50']:.2f}s", f"{r['add_p95']:.2f}s",
            f"{r['add_mean']:.2f}s", f"{r['n_failures']}/{r['n_cases']}",
        ]
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))
        table_rows.append(row)
    print()
    print(" | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in table_rows:
        print(" | ".join(f"{v:<{w}}" for v, w in zip(row, widths)))


def main() -> int:
    parser = argparse.ArgumentParser(description="provider extraction battle")
    parser.add_argument("--limit", type=int, default=0, help="truncate case list")
    parser.add_argument("--providers", default="anthropic,ollama,openai",
                        help="comma-separated subset to run")
    parser.add_argument("--collection", default="mem0_bench_v3")
    args = parser.parse_args()

    if not check_qdrant():
        print("Qdrant unreachable — set MEM0_QDRANT_URL to a live instance", file=sys.stderr)
        return 1
    if not check_ollama():
        print("Ollama unreachable (needed for embedder bge-m3)", file=sys.stderr)
        return 1

    cases = FACT_EXTRACTION_CASES[: args.limit] if args.limit else FACT_EXTRACTION_CASES
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    print(f"Running {len(cases)} cases × {len(providers)} providers …")

    rows = []
    for p in providers:
        result = _run_provider(p, cases, args.collection)
        if result:
            rows.append(result)

    _print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())