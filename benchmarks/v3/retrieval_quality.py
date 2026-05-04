"""Hybrid (semantic + BM25 + entity boost) vs semantic-only retrieval.

mem0 v3's hybrid retrieval is hard-coded inside ``Memory._search_vector_store``
(see ``mem0/memory/main.py`` step 4 ``vector_store.keyword_search`` and step 6
``_compute_entity_boosts``). There is no env knob to disable it, so this runner
monkey-patches both at the instance level for the semantic-only arm and
restores them after.

Corpus has three query categories:
- keyword-heavy: rare/unique tokens that BM25 should win on
- semantic-heavy: paraphrases the dense embedder should match
- entity-heavy: queries naming a person/place that entity boost should lift

For each query, recall@1 / recall@5 / MRR are computed against a hand-labeled
list of expected-relevant fact IDs.

Usage::

    uv run python -m benchmarks.v3.retrieval_quality [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any

from benchmarks.v3.framework import (
    bench_user_id,
    check_ollama,
    check_qdrant,
    make_memory,
)
from mem0_mcp_selfhosted.helpers import safe_bulk_delete


@dataclass
class Fact:
    fid: str
    text: str


@dataclass
class Query:
    qid: str
    category: str  # keyword | semantic | entity
    query: str
    relevant: list[str]  # fact ids that should be retrieved


# 18 facts × 12 queries — small enough to run in <2 min on a warm instance.
FACTS: list[Fact] = [
    # Rare-token / acronym-heavy facts (BM25 should help)
    Fact("F01", "The deployment uses Argo Rollouts with a canary strategy and Kayenta analysis."),
    Fact("F02", "Our observability stack is OpenTelemetry collector feeding Tempo, Loki, and Mimir."),
    Fact("F03", "We pin pyright==1.1.405 and ruff==0.13.0 in pyproject.toml dev extras."),
    Fact("F04", "The fastembed Qdrant/bm25 sparse encoder needs roughly 350 MB of RAM resident."),
    Fact("F05", "QUIC over UDP/443 is enabled on the L7 load balancer with HTTP/3 advertised."),
    Fact("F06", "Postgres logical replication slot is named 'cdc_slot_main' with wal2json plugin."),

    # Conceptual facts (semantic embedding wins)
    Fact("F07", "The team prefers measured, evidence-based decisions over moving fast."),
    Fact("F08", "Releases are gated on a one-week soak period in staging before promotion."),
    Fact("F09", "Code reviews must include at least one engineer outside the author's squad."),
    Fact("F10", "We avoid premature abstractions; duplicate code three times before extracting."),
    Fact("F11", "On-call rotations cap at 24 hours of pager duty per engineer per week."),
    Fact("F12", "Documentation is treated as code and lives next to the modules it describes."),

    # Entity-heavy facts
    Fact("F13", "Dr. Yuki Tanaka leads the Tokyo research office for the platform team."),
    Fact("F14", "Priya Iyer reviews all changes touching the Bangalore data plane."),
    Fact("F15", "The Berlin office hosts the SRE rotation for European business hours."),
    Fact("F16", "Carlos Mendes is the technical lead for the Sao Paulo edge POP."),
    Fact("F17", "Elena Petrova owns the Moscow disaster recovery runbook."),
    Fact("F18", "Aiko Sato manages the Osaka observability stack migration."),
]


QUERIES: list[Query] = [
    # keyword-heavy: terms a paraphrase wouldn't preserve
    Query("Q01", "keyword", "argo rollouts canary kayenta", ["F01"]),
    Query("Q02", "keyword", "opentelemetry tempo loki mimir", ["F02"]),
    Query("Q03", "keyword", "fastembed bm25 sparse encoder memory", ["F04"]),
    Query("Q04", "keyword", "wal2json logical replication slot", ["F06"]),

    # semantic-heavy: paraphrases without shared keywords
    Query("Q05", "semantic", "what is the team's decision-making philosophy?", ["F07"]),
    Query("Q06", "semantic", "how long do we burn in releases before shipping?", ["F08"]),
    Query("Q07", "semantic", "rule for when to refactor duplicated logic", ["F10"]),
    Query("Q08", "semantic", "policy on after-hours support burden", ["F11"]),

    # entity-heavy: named person/place
    Query("Q09", "entity", "who runs the Tokyo office?", ["F13"]),
    Query("Q10", "entity", "Bangalore code reviewer", ["F14"]),
    Query("Q11", "entity", "Sao Paulo edge POP lead", ["F16"]),
    Query("Q12", "entity", "Moscow DR runbook owner", ["F17"]),
]


def _extract_results(raw: Any) -> list[dict]:
    if isinstance(raw, dict):
        return raw.get("results", []) or []
    if isinstance(raw, list):
        return raw
    return []


def _seed_facts(mem: Any, uid: str, facts: list[Fact]) -> dict[str, str]:
    """Add facts with infer=False to avoid LLM variance.

    Returns a map from fact id -> memory id (for recall scoring).
    """
    fid_to_mid: dict[str, str] = {}
    for fact in facts:
        result = mem.add(
            messages=[{"role": "user", "content": fact.text}],
            user_id=uid,
            infer=False,
            metadata={"bench_fact_id": fact.fid},
        )
        events = result.get("results", []) if isinstance(result, dict) else result or []
        for e in events:
            if isinstance(e, dict) and e.get("event") == "ADD" and isinstance(e.get("id"), str):
                fid_to_mid[fact.fid] = e["id"]
                break
    return fid_to_mid


def _run_query(mem: Any, q: Query, uid: str, top_k: int) -> list[str]:
    """Run a single query, return memory IDs in score order."""
    raw = mem.search(query=q.query, filters={"user_id": uid}, top_k=top_k)
    results = _extract_results(raw)
    return [r.get("id", "") for r in results if r.get("id")]


def _recall_and_mrr(retrieved_mids: list[str], relevant_mids: set[str], k: int) -> tuple[float, float]:
    top_k = retrieved_mids[:k]
    found = sum(1 for mid in top_k if mid in relevant_mids)
    recall_k = found / len(relevant_mids) if relevant_mids else 0.0
    rr = 0.0
    for i, mid in enumerate(retrieved_mids, start=1):
        if mid in relevant_mids:
            rr = 1.0 / i
            break
    return recall_k, rr


def _disable_hybrid(mem: Any) -> tuple[Any, Any]:
    """Patch keyword_search & entity boost to disable BM25 and entity layers.

    Returns the originals so the caller can restore them.
    """
    orig_keyword = mem.vector_store.keyword_search
    orig_entity = mem._compute_entity_boosts
    mem.vector_store.keyword_search = lambda *a, **kw: None
    mem._compute_entity_boosts = lambda *a, **kw: {}
    return orig_keyword, orig_entity


def _restore_hybrid(mem: Any, orig_keyword: Any, orig_entity: Any) -> None:
    mem.vector_store.keyword_search = orig_keyword
    mem._compute_entity_boosts = orig_entity


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    widths = [max(len(h), max((len(str(r[h])) for r in rows), default=0)) for h in headers]
    line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print(line)
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(f"{str(r[h]):<{w}}" for h, w in zip(headers, widths)))


def main() -> int:
    parser = argparse.ArgumentParser(description="hybrid vs semantic-only retrieval quality")
    parser.add_argument("--limit", type=int, default=0, help="truncate query set (smoke run)")
    parser.add_argument("--collection", default="mem0_bench_v3", help="qdrant collection name")
    args = parser.parse_args()

    if not check_qdrant():
        print("Qdrant unreachable — set MEM0_QDRANT_URL to a live instance", file=sys.stderr)
        return 1
    if not check_ollama():
        print("Ollama unreachable (needed for embedder bge-m3)", file=sys.stderr)
        return 1

    queries = QUERIES[: args.limit] if args.limit else QUERIES

    mem, cleanup_env = make_memory(collection=args.collection)
    uid = bench_user_id("retrieval")
    try:
        print(f"Seeding {len(FACTS)} facts under user_id={uid} …")
        fid_to_mid = _seed_facts(mem, uid, FACTS)
        if len(fid_to_mid) != len(FACTS):
            print(
                f"WARN: only {len(fid_to_mid)}/{len(FACTS)} facts got memory IDs back — "
                "scores below may be biased",
                file=sys.stderr,
            )

        per_arm: dict[str, list[dict[str, Any]]] = {"hybrid": [], "semantic": []}

        # Arm A: hybrid (default)
        for q in queries:
            mids = _run_query(mem, q, uid, top_k=5)
            relevant = {fid_to_mid[fid] for fid in q.relevant if fid in fid_to_mid}
            r1, _ = _recall_and_mrr(mids, relevant, k=1)
            r5, mrr = _recall_and_mrr(mids, relevant, k=5)
            per_arm["hybrid"].append({"qid": q.qid, "cat": q.category, "r@1": round(r1, 3),
                                       "r@5": round(r5, 3), "mrr": round(mrr, 3)})

        # Arm B: semantic-only via instance monkey-patch
        orig_kw, orig_ent = _disable_hybrid(mem)
        try:
            for q in queries:
                mids = _run_query(mem, q, uid, top_k=5)
                relevant = {fid_to_mid[fid] for fid in q.relevant if fid in fid_to_mid}
                r1, _ = _recall_and_mrr(mids, relevant, k=1)
                r5, mrr = _recall_and_mrr(mids, relevant, k=5)
                per_arm["semantic"].append({"qid": q.qid, "cat": q.category, "r@1": round(r1, 3),
                                             "r@5": round(r5, 3), "mrr": round(mrr, 3)})
        finally:
            _restore_hybrid(mem, orig_kw, orig_ent)

        # Per-category aggregate
        print()
        print("Per-category recall@5 (mean across queries):")
        cats = sorted({q.category for q in queries})
        agg_rows = []
        for cat in cats:
            hybrid_r5 = [row["r@5"] for row in per_arm["hybrid"] if row["cat"] == cat]
            sem_r5 = [row["r@5"] for row in per_arm["semantic"] if row["cat"] == cat]
            agg_rows.append({
                "category": cat,
                "n": len(hybrid_r5),
                "hybrid r@5": round(sum(hybrid_r5) / len(hybrid_r5), 3) if hybrid_r5 else 0.0,
                "semantic r@5": round(sum(sem_r5) / len(sem_r5), 3) if sem_r5 else 0.0,
                "delta": round((sum(hybrid_r5) - sum(sem_r5)) / max(len(hybrid_r5), 1), 3),
            })
        _print_table(agg_rows)

        # Per-query side-by-side
        print()
        print("Per-query (hybrid | semantic):")
        side_rows = []
        for h, s in zip(per_arm["hybrid"], per_arm["semantic"]):
            side_rows.append({
                "qid": h["qid"],
                "cat": h["cat"],
                "hybrid r@1": h["r@1"],
                "sem r@1": s["r@1"],
                "hybrid r@5": h["r@5"],
                "sem r@5": s["r@5"],
                "hybrid mrr": h["mrr"],
                "sem mrr": s["mrr"],
            })
        _print_table(side_rows)
        return 0
    finally:
        try:
            n = safe_bulk_delete(mem, {"user_id": uid})
            print(f"\nCleaned up {n} memories under {uid}")
        except Exception as exc:
            print(f"Cleanup failed: {exc}", file=sys.stderr)
        cleanup_env()


if __name__ == "__main__":
    raise SystemExit(main())