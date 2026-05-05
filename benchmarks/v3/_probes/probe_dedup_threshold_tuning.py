"""Threshold-tuning probe: run the dedup bench at 0.82, 0.85, and 0.88.

For each threshold, monkey-patches ``hooks._DEDUP_SIM_THRESHOLD`` before
running the corpus, then restores it afterwards.  Only runs Part 1 (dedup
confusion matrix); entity-link overhead is skipped so the probe finishes
quickly.

Usage::

    uv run python -m benchmarks.v3._probes.probe_dedup_threshold_tuning
"""

from __future__ import annotations

import sys

import mem0_mcp_selfhosted.hooks as hooks
from benchmarks.v3.dedup_and_entity import PAIRS, _bench_dedup, _print_confusion
from benchmarks.v3.framework import check_ollama, check_qdrant, make_memory

THRESHOLDS = [0.82, 0.85, 0.88]


def run_at_threshold(mem, threshold: float) -> dict[str, int]:
    original = hooks._DEDUP_SIM_THRESHOLD
    hooks._DEDUP_SIM_THRESHOLD = threshold
    try:
        print(f"\n{'=' * 60}")
        print(f"Threshold = {threshold}")
        print("=" * 60)
        counts = _bench_dedup(mem, PAIRS)
        _print_confusion(counts)
        return counts
    finally:
        hooks._DEDUP_SIM_THRESHOLD = original


def main() -> int:
    if not check_qdrant():
        print("Qdrant unreachable", file=sys.stderr)
        return 1
    if not check_ollama():
        print("Ollama unreachable", file=sys.stderr)
        return 1

    mem, cleanup_env = make_memory(collection="mem0_bench_v3")
    results: dict[float, dict[str, int]] = {}
    try:
        for thresh in THRESHOLDS:
            results[thresh] = run_at_threshold(mem, thresh)
    finally:
        cleanup_env()

    # Summary table
    print()
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'thresh':>6} | {'TP':>3} | {'FN':>3} | {'FP':>3} | {'TN':>3} | {'precision':>9} | {'recall':>6} | {'accuracy':>8} | total_err")
    print("-" * 80)
    for thresh in THRESHOLDS:
        c = results[thresh]
        tp, fn, fp, tn = c["TP"], c["FN"], c["FP"], c["TN"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        acc = (tp + tn) / (tp + fn + fp + tn) if (tp + fn + fp + tn) else 0.0
        total_err = fn + fp
        print(f"{thresh:>6.2f} | {tp:>3} | {fn:>3} | {fp:>3} | {tn:>3} | {prec:>9.3f} | {rec:>6.3f} | {acc:>8.3f} | {total_err}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
