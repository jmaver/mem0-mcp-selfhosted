"""SessionEnd dedup catch rate + spaCy entity-link overhead.

Part 1 — Dedup confusion matrix.
  Drives ``_dedup_against_existing`` (``hooks.py:244``, threshold 0.88) on
  hand-labeled pairs. Each pair is added (``infer=False`` — keep raw text), then
  the second add's id is passed to the dedup helper. The helper deletes the new
  memory when it matches an older one above 0.88. Pairs are labeled either
  ``duplicate`` (we want it deleted) or ``distinct`` (we want it kept).

Part 2 — Entity-link overhead.
  Times ``mem.add(infer=False, ...)`` with and without spaCy by patching
  ``mem0.memory.main.extract_entities`` and ``extract_entities_batch`` to
  return empty lists. Mean delta is the per-add entity-linking cost.

Usage::

    uv run python -m benchmarks.v3.dedup_and_entity [--limit N] [--entity-iters N]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Any

from benchmarks.v3.framework import (
    bench_user_id,
    check_ollama,
    check_qdrant,
    make_memory,
    percentile,
)
from mem0_mcp_selfhosted.helpers import make_project_user_id, safe_bulk_delete
from mem0_mcp_selfhosted.hooks import _dedup_against_existing


@dataclass
class Pair:
    pid: str
    label: str  # "duplicate" or "distinct"
    first: str
    second: str


PAIRS: list[Pair] = [
    # --- Duplicates: paraphrases / mild rewordings (should be caught) ---
    Pair("D01", "duplicate",
         "User prefers dark mode in their terminal and editor.",
         "The user likes dark mode for their terminal and editor."),
    Pair("D02", "duplicate",
         "Project deploys via blue-green to AWS EKS.",
         "Project uses blue-green deployments on AWS EKS."),
    Pair("D03", "duplicate",
         "On-call rotation is capped at 24 hours of pager duty per week.",
         "Engineers do at most 24 hours of pager duty per week on rotation."),
    Pair("D04", "duplicate",
         "The team uses pyright for type checking and ruff for linting.",
         "Type checking is pyright, linting is ruff in this project."),
    Pair("D05", "duplicate",
         "Documentation lives next to the code it describes.",
         "Docs are colocated with the modules they document."),
    Pair("D06", "duplicate",
         "The dedup similarity threshold is 0.88 for paraphrases.",
         "Paraphrase dedup uses a 0.88 cosine threshold."),
    Pair("D07", "duplicate",
         "Releases soak in staging for one week before promotion.",
         "There is a one-week staging soak prior to promoting a release."),
    Pair("D08", "duplicate",
         "Code reviews require at least one engineer outside the author's squad.",
         "Each PR needs a reviewer from a different squad than the author."),

    # --- Distinct: same domain, different fact (should be kept) ---
    Pair("X01", "distinct",
         "User prefers dark mode in their terminal and editor.",
         "User prefers Vim keybindings in their editor."),
    Pair("X02", "distinct",
         "Project deploys via blue-green to AWS EKS.",
         "Project uses GitHub Actions for the CI pipeline."),
    Pair("X03", "distinct",
         "On-call rotation is capped at 24 hours of pager duty per week.",
         "On-call escalation goes through PagerDuty to the lead first."),
    Pair("X04", "distinct",
         "The team uses pyright for type checking and ruff for linting.",
         "The team uses pytest for unit tests and tox for matrix runs."),
    Pair("X05", "distinct",
         "Documentation lives next to the code it describes.",
         "Documentation is published to a Mintlify site on every merge."),
    Pair("X06", "distinct",
         "The dedup similarity threshold is 0.88 for paraphrases.",
         "The search recall target is 0.85 at top-5 across categories."),
    Pair("X07", "distinct",
         "Releases soak in staging for one week before promotion.",
         "Releases are tagged with semver and signed by maintainers."),
    Pair("X08", "distinct",
         "Code reviews require at least one engineer outside the author's squad.",
         "Code reviews must include automated test coverage above 80 percent."),
]


def _add_one(mem: Any, text: str, uid: str) -> str | None:
    result = mem.add(messages=[{"role": "user", "content": text}], user_id=uid, infer=False)
    events = result.get("results", []) if isinstance(result, dict) else result or []
    for e in events:
        if isinstance(e, dict) and e.get("event") == "ADD" and isinstance(e.get("id"), str):
            return e["id"]
    return None


def _bench_dedup(mem: Any, pairs: list[Pair]) -> dict[str, int]:
    """Run each pair through _dedup_against_existing, return confusion counts."""
    counts = {"TP": 0, "FN": 0, "FP": 0, "TN": 0, "skipped": 0}

    base_uid = bench_user_id("dedup")
    print(f"\nDedup pairs (threshold 0.88) under {base_uid}/<pid> …")

    for pair in pairs:
        # Use a unique sub-uid per pair so dedup search doesn't see cross-pair matches.
        uid = make_project_user_id(base_uid, pair.pid)
        try:
            first_id = _add_one(mem, pair.first, uid)
            if not first_id:
                counts["skipped"] += 1
                print(f"  {pair.pid} ({pair.label}): could not seed first; skipping")
                continue

            # Ensure created_at strictly orders first < second; mem0 stores
            # iso8601 strings with microsecond precision so a small sleep
            # is enough.
            time.sleep(0.05)

            second_id = _add_one(mem, pair.second, uid)
            if not second_id:
                # mem0's hash-dedup may have rejected the second add as
                # exact-duplicate. Treat that as TP for paraphrase pairs.
                if pair.label == "duplicate":
                    counts["TP"] += 1
                    print(f"  {pair.pid} ({pair.label}): mem0 hash-dedup rejected second add (TP)")
                else:
                    counts["skipped"] += 1
                    print(f"  {pair.pid} ({pair.label}): could not seed second; skipping")
                continue

            dropped = _dedup_against_existing(mem, [second_id], uid)
            survived = mem.get(second_id) is not None

            if pair.label == "duplicate":
                if dropped >= 1 or not survived:
                    counts["TP"] += 1
                    print(f"  {pair.pid} (dup): caught (TP)")
                else:
                    counts["FN"] += 1
                    print(f"  {pair.pid} (dup): MISSED (FN)")
            else:
                if dropped >= 1 or not survived:
                    counts["FP"] += 1
                    print(f"  {pair.pid} (distinct): wrongly dropped (FP)")
                else:
                    counts["TN"] += 1
                    print(f"  {pair.pid} (distinct): kept (TN)")
        finally:
            try:
                safe_bulk_delete(mem, {"user_id": uid})
            except Exception as exc:
                print(f"  Cleanup failed for {uid}: {exc}", file=sys.stderr)

    return counts


def _print_confusion(counts: dict[str, int]) -> None:
    tp, fn, fp, tn = counts["TP"], counts["FN"], counts["FP"], counts["TN"]
    pos = tp + fn
    neg = tn + fp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / pos if pos else 0.0
    accuracy = (tp + tn) / (pos + neg) if (pos + neg) else 0.0

    print()
    print(f"Confusion matrix (n={tp + fn + fp + tn}, skipped={counts['skipped']}):")
    print(f"  TP (caught dup)        : {tp:>3}")
    print(f"  FN (missed dup)        : {fn:>3}")
    print(f"  FP (deleted distinct)  : {fp:>3}")
    print(f"  TN (kept distinct)     : {tn:>3}")
    print(f"  precision={precision:.3f}  recall={recall:.3f}  accuracy={accuracy:.3f}")


def _bench_entity_overhead(mem: Any, iterations: int) -> dict[str, dict[str, float]]:
    """Time mem.add(infer=False) with and without spaCy entity extraction."""
    import mem0.memory.main as mem0_main

    samples = [
        "Dr. Yuki Tanaka leads the Tokyo research office for the platform team.",
        "Priya Iyer reviews all changes touching the Bangalore data plane.",
        "The Berlin office hosts the SRE rotation for European business hours.",
        "Carlos Mendes is the technical lead for the Sao Paulo edge POP.",
        "Elena Petrova owns the Moscow disaster recovery runbook.",
    ]

    uid = bench_user_id("entity-on")
    on_durations: list[float] = []
    print(f"\nEntity-link timing (with spaCy) × {iterations} under {uid} …")
    try:
        for i in range(iterations):
            text = samples[i % len(samples)] + f" (iter {i})"
            t0 = time.perf_counter()
            _add_one(mem, text, uid)
            on_durations.append(time.perf_counter() - t0)
    finally:
        safe_bulk_delete(mem, {"user_id": uid})

    # Patch extract_entities + extract_entities_batch to no-op
    orig_extract = mem0_main.extract_entities
    orig_batch = mem0_main.extract_entities_batch
    mem0_main.extract_entities = lambda *a, **kw: []
    mem0_main.extract_entities_batch = lambda texts, *a, **kw: [[] for _ in texts]

    uid_off = bench_user_id("entity-off")
    off_durations: list[float] = []
    print(f"Entity-link timing (spaCy patched out) × {iterations} under {uid_off} …")
    try:
        for i in range(iterations):
            text = samples[i % len(samples)] + f" (iter {i})"
            t0 = time.perf_counter()
            _add_one(mem, text, uid_off)
            off_durations.append(time.perf_counter() - t0)
    finally:
        mem0_main.extract_entities = orig_extract
        mem0_main.extract_entities_batch = orig_batch
        safe_bulk_delete(mem, {"user_id": uid_off})

    def stats(vals: list[float]) -> dict[str, float]:
        return {
            "n": float(len(vals)),
            "mean": sum(vals) / len(vals) if vals else 0.0,
            "p50": percentile(vals, 50),
            "p95": percentile(vals, 95),
        }

    return {"with_spacy": stats(on_durations), "without_spacy": stats(off_durations)}


def _print_entity_summary(s: dict[str, dict[str, float]]) -> None:
    on = s["with_spacy"]
    off = s["without_spacy"]
    delta = on["mean"] - off["mean"]
    pct = (delta / off["mean"] * 100.0) if off["mean"] else 0.0
    print()
    print(f"{'arm':<14} | {'n':>3} | {'mean':>7} | {'p50':>7} | {'p95':>7}")
    print("-" * 52)
    print(f"{'with spaCy':<14} | {int(on['n']):>3} | {on['mean']:>6.3f}s | {on['p50']:>6.3f}s | {on['p95']:>6.3f}s")
    print(f"{'no spaCy':<14} | {int(off['n']):>3} | {off['mean']:>6.3f}s | {off['p50']:>6.3f}s | {off['p95']:>6.3f}s")
    print(f"\nMean Δ = {delta * 1000:+.1f} ms ({pct:+.1f}%) — entity-link overhead per add")


def main() -> int:
    parser = argparse.ArgumentParser(description="dedup catch rate + entity overhead")
    parser.add_argument("--limit", type=int, default=0, help="truncate dedup pairs")
    parser.add_argument("--entity-iters", type=int, default=20, help="entity-overhead iterations per arm")
    parser.add_argument("--collection", default="mem0_bench_v3")
    parser.add_argument("--skip-dedup", action="store_true")
    parser.add_argument("--skip-entity", action="store_true")
    args = parser.parse_args()

    if not check_qdrant():
        print("Qdrant unreachable", file=sys.stderr)
        return 1
    if not check_ollama():
        print("Ollama unreachable", file=sys.stderr)
        return 1

    mem, cleanup_env = make_memory(collection=args.collection)
    try:
        if not args.skip_dedup:
            pairs = PAIRS[: args.limit] if args.limit else PAIRS
            counts = _bench_dedup(mem, pairs)
            _print_confusion(counts)

        if not args.skip_entity:
            entity_stats = _bench_entity_overhead(mem, args.entity_iters)
            _print_entity_summary(entity_stats)
    finally:
        cleanup_env()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())