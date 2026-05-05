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


# 100 facts × 30 queries across 5 thematic clusters — expect ~8-12 min on a warm instance.
FACTS: list[Fact] = [
    # --- Cluster 1: Cloud / k8s / observability (F01-F20) ---
    Fact("F01", "The deployment uses Argo Rollouts with a canary strategy and Kayenta analysis."),
    Fact("F02", "Our observability stack is OpenTelemetry collector feeding Tempo, Loki, and Mimir."),
    Fact("F03", "We pin pyright==1.1.405 and ruff==0.13.0 in pyproject.toml dev extras."),
    Fact("F04", "The fastembed Qdrant/bm25 sparse encoder needs roughly 350 MB of RAM resident."),
    Fact("F05", "QUIC over UDP/443 is enabled on the L7 load balancer with HTTP/3 advertised."),
    Fact("F06", "Postgres logical replication slot is named 'cdc_slot_main' with wal2json plugin."),
    Fact("F07", "Istio service mesh runs in ambient mode with ztunnel replacing per-pod sidecars."),
    Fact("F08", "Cilium CNI replaces kube-proxy with eBPF-based load balancing and network policy."),
    Fact("F09", "Jaeger distributed traces are sampled at 5% and stored for 14 days in object store."),
    Fact("F10", "Linkerd data plane proxies are compiled from Rust and inject as init containers."),
    Fact("F11", "The Argo CD sync wave ordering is: CRDs (wave 0), namespaces (wave 1), apps (wave 2)."),
    Fact("F12", "Karpenter node provisioner uses spot instances for batch workloads and on-demand for serving."),
    Fact("F13", "Prometheus scrape interval is 15 s for application metrics and 60 s for node-exporter."),
    Fact("F14", "VictoriaMetrics long-term storage retains raw samples for 365 days with 2-of-3 replication."),
    Fact("F15", "The Argo Rollouts analysis run for the canary validates the p99 latency SLO via Kayenta."),
    Fact("F16", "Grafana Tempo ingestion endpoint uses OTLP/gRPC on port 4317 with TLS mutual auth."),
    Fact("F17", "Loki chunk storage is GCS with a 72-hour in-memory ring buffer before compaction."),
    Fact("F18", "Mimir ruler evaluates recording rules every 30 s and alerts every 60 s."),
    Fact("F19", "OpenTelemetry collector tail-sampling processor drops traces with no error spans and duration < 200 ms."),
    Fact("F20", "Cilium Hubble relay exposes a gRPC API for real-time network flow observability."),

    # --- Cluster 2: Engineering culture / process / policies (F21-F40) ---
    Fact("F21", "The team prefers measured, evidence-based decisions over moving fast."),
    Fact("F22", "Releases are gated on a one-week soak period in staging before promotion."),
    Fact("F23", "Code reviews must include at least one engineer outside the author's squad."),
    Fact("F24", "We avoid premature abstractions; duplicate code three times before extracting."),
    Fact("F25", "On-call rotations cap at 24 hours of pager duty per engineer per week."),
    Fact("F26", "Documentation is treated as code and lives next to the modules it describes."),
    Fact("F27", "Every incident requires a blameless post-mortem published within five business days."),
    Fact("F28", "Feature flags must be cleaned up within two sprints of reaching 100% rollout."),
    Fact("F29", "All API changes require an ADR approved by at least two senior engineers before merging."),
    Fact("F30", "The on-call engineer is exempt from sprint commitments during their rotation week."),
    Fact("F31", "PR descriptions must include a testing plan and a rollback procedure section."),
    Fact("F32", "Security patches classified P0 must be deployed to production within 24 hours of disclosure."),
    Fact("F33", "Sprint planning caps team capacity at 80% to leave headroom for unplanned work."),
    Fact("F34", "We require at least two reviewers — one domain expert and one outside the squad."),
    Fact("F35", "Staging promotion is blocked until smoke tests and a 7-day error-rate baseline pass."),
    Fact("F36", "Engineers rotate the on-call pager weekly; no engineer pages more than once per month."),
    Fact("F37", "Technical debt items must be tracked in the backlog with a severity label before closing."),
    Fact("F38", "Major architectural decisions require sign-off from the platform lead and CTO."),
    Fact("F39", "Code freeze begins 48 hours before a scheduled release and lifts after the deploy completes."),
    Fact("F40", "All external dependencies must be pinned to an exact version and reviewed in a security scan."),

    # --- Cluster 3: People + locations (F41-F60) ---
    Fact("F41", "Dr. Yuki Tanaka leads the Tokyo research office for the platform team."),
    Fact("F42", "Priya Iyer reviews all changes touching the Bangalore data plane."),
    Fact("F43", "The Berlin office hosts the SRE rotation for European business hours."),
    Fact("F44", "Carlos Mendes is the technical lead for the Sao Paulo edge POP."),
    Fact("F45", "Elena Petrova owns the Moscow disaster recovery runbook."),
    Fact("F46", "Aiko Sato manages the Osaka observability stack migration."),
    Fact("F47", "Dr. Yuki Tanaka also holds the on-call pager for the Tokyo region every third week."),
    Fact("F48", "Ravi Shankar is the release manager for the Bangalore region and approves all PTO cutoffs."),
    Fact("F49", "The Berlin office holds the European platform team's weekly architecture review on Thursdays."),
    Fact("F50", "Luisa Carvalho is the SRE lead for the Sao Paulo edge and manages capacity planning."),
    Fact("F51", "Dmitri Volkov joined Elena Petrova's Moscow team as a senior SRE in Q1 2025."),
    Fact("F52", "Aiko Sato collaborates with the Tokyo research office on unified metrics pipelines."),
    Fact("F53", "Fatima Al-Rashidi leads the Dubai engineering hub and oversees MENA compliance."),
    Fact("F54", "James Okafor manages the Lagos office and coordinates sub-Saharan data residency."),
    Fact("F55", "Sofia Lindqvist is the Stockholm platform lead responsible for the Nordic k8s clusters."),
    Fact("F56", "Ben Adler is the Berlin SRE who maintains the European on-call runbooks."),
    Fact("F57", "Haruto Mori is a senior engineer in the Osaka office and owns the GPU node pool."),
    Fact("F58", "Dr. Wei Chen leads the Singapore AI research team and co-authors the quarterly eval report."),
    Fact("F59", "Nadia Kowalski is the Warsaw backend lead and co-owns the EU GDPR compliance checklist."),
    Fact("F60", "Takeshi Yamamoto in the Tokyo office is the primary contact for vendor SLA negotiations."),

    # --- Cluster 4: Database / storage / replication (F61-F80) ---
    Fact("F61", "Postgres logical replication slot is named 'cdc_slot_main' with wal2json plugin."),
    Fact("F62", "The primary PostgreSQL cluster uses Patroni for HA with a 3-node setup and etcd quorum."),
    Fact("F63", "MySQL binlog replication runs in GTID mode with semi-sync replication to one replica."),
    Fact("F64", "Redis Sentinel manages failover for the session cache cluster with a 3-replica topology."),
    Fact("F65", "Qdrant vector store collection uses named vectors: 'dense' (1024-dim) and 'sparse' (BM25)."),
    Fact("F66", "WAL archiving ships segments to S3 every 60 s with a 30-day retention policy."),
    Fact("F67", "The backup schedule runs full snapshots on Sunday 02:00 UTC and incrementals nightly."),
    Fact("F68", "Point-in-time recovery target is 5 minutes RPO and 15 minutes RTO for the primary cluster."),
    Fact("F69", "Pgbouncer pools connections in transaction mode with pool_size=50 per database."),
    Fact("F70", "The read-replica lag SLO is < 100 ms p95; alerts fire when lag exceeds 1 s for 2 min."),
    Fact("F71", "Qdrant shard count is set to 4 with replication_factor=2 across two availability zones."),
    Fact("F72", "Redis cluster mode shards across 6 primaries with one replica each for the product catalog."),
    Fact("F73", "The Cassandra ring has 12 nodes with RF=3 and LOCAL_QUORUM consistency for writes."),
    Fact("F74", "DynamoDB tables use on-demand billing except the hot events table which has provisioned 5k RCU."),
    Fact("F75", "Postgres vacuum is tuned with autovacuum_vacuum_scale_factor=0.01 for large tables."),
    Fact("F76", "The wal2json CDC stream is consumed by Debezium and published to a Kafka topic per table."),
    Fact("F77", "Clickhouse columnar store holds 18 months of analytics data with a TTL merge policy."),
    Fact("F78", "S3 bucket versioning is enabled with a lifecycle rule to transition versions to Glacier after 90 days."),
    Fact("F79", "The MySQL replica is promoted to primary via Orchestrator with a max failover time of 30 s."),
    Fact("F80", "Patroni leader election requires a quorum of at least 2 of 3 nodes; DCS is etcd v3."),

    # --- Cluster 5: AI/ML/LLM stack (F81-F100) ---
    Fact("F81", "The embedder is bge-m3 served by Ollama and produces 1024-dimensional dense vectors."),
    Fact("F82", "Hybrid retrieval combines bge-m3 dense vectors with fastembed BM25 sparse vectors."),
    Fact("F83", "The fact-extraction LLM is claude-sonnet-4-5 with structured-output JSON schema mode."),
    Fact("F84", "Reranking uses a cross-encoder model (ms-marco-MiniLM-L-12-v2) hosted on TorchServe."),
    Fact("F85", "The eval pipeline runs on a nightly cron and publishes F1/recall metrics to Grafana."),
    Fact("F86", "LM Studio serves llama-3.3-70b-instruct with a 4096-token context window on the local bench."),
    Fact("F87", "HyDE query expansion generates five hypothetical passages before dense retrieval."),
    Fact("F88", "The vector index uses HNSW with ef_construction=200 and m=16 for the dense collection."),
    Fact("F89", "Ollama pull policy is 'always' on CI and 'if-not-present' on local dev machines."),
    Fact("F90", "The MLflow experiment tracker stores run artifacts in S3 under the 'mlflow-artifacts' bucket."),
    Fact("F91", "Batch inference jobs use vLLM with continuous batching and tensor parallelism across 4 GPUs."),
    Fact("F92", "The RAG pipeline seeds the vector store with infer=False to bypass LLM extraction overhead."),
    Fact("F93", "Sentence-transformers all-MiniLM-L6-v2 is used for the lightweight re-ranking pre-filter."),
    Fact("F94", "The RLHF labeling queue uses Label Studio with a 3-annotator consensus threshold."),
    Fact("F95", "mem0 v3 uses spaCy en_core_web_sm for entity extraction before vector indexing."),
    Fact("F96", "The model registry on Hugging Face Hub gates production promotion behind a manual approval step."),
    Fact("F97", "vLLM's OpenAI-compat server strips json_object response_format for LM Studio compatibility."),
    Fact("F98", "The hybrid search scoring formula weights dense recall at 0.6 and BM25 at 0.4."),
    Fact("F99", "Gemini Pro is used as the fallback LLM when Anthropic token refresh fails."),
    Fact("F100", "The nightly eval cron seeds 50 benchmark queries and writes results to benchmarks/v3/results/."),
]


QUERIES: list[Query] = [
    # --- Easy (10): one clearly best fact ---
    # keyword
    Query("Q01", "keyword", "argo rollouts canary kayenta", ["F01", "F15"]),
    Query("Q02", "keyword", "opentelemetry collector tempo loki mimir", ["F02", "F19"]),
    Query("Q03", "keyword", "fastembed bm25 sparse encoder ram", ["F04", "F82"]),
    Query("Q04", "keyword", "wal2json logical replication slot cdc", ["F61", "F76"]),
    Query("Q05", "keyword", "patroni etcd quorum HA", ["F62", "F80"]),
    # semantic
    Query("Q06", "semantic", "what is the team's decision-making philosophy?", ["F21"]),
    Query("Q07", "semantic", "how long do we burn in releases before shipping to prod?", ["F22", "F35"]),
    Query("Q08", "semantic", "rule for when to refactor or extract duplicated logic", ["F24"]),
    # entity
    Query("Q09", "entity", "who leads the Tokyo research office?", ["F41"]),
    Query("Q10", "entity", "Moscow disaster recovery runbook owner", ["F45"]),

    # --- Medium (10): two relevant facts, distractors present ---
    # keyword
    Query("Q11", "keyword", "Istio ambient ztunnel sidecar", ["F07"]),
    Query("Q12", "keyword", "Cilium eBPF kube-proxy CNI", ["F08", "F20"]),
    Query("Q13", "keyword", "Prometheus scrape interval node-exporter", ["F13"]),
    Query("Q14", "keyword", "Qdrant shard replication factor availability zone", ["F65", "F71"]),
    # semantic
    Query("Q15", "semantic", "policy on after-hours pager duty burden for engineers", ["F25", "F30", "F36"]),
    Query("Q16", "semantic", "what happens at code freeze before a release?", ["F39"]),
    Query("Q17", "semantic", "how are security patches prioritized?", ["F32"]),
    # entity
    Query("Q18", "entity", "Bangalore data plane reviewer and release manager", ["F42", "F48"]),
    Query("Q19", "entity", "Berlin SRE office and on-call runbooks", ["F43", "F49", "F56"]),
    Query("Q20", "entity", "Sao Paulo edge POP technical lead", ["F44", "F50"]),

    # --- Hard (10): paraphrase + adjacent-fact confusables ---
    # keyword (rare tokens mixed with confusables in same cluster)
    Query("Q21", "keyword", "WAL archiving S3 retention backup schedule", ["F66", "F67"]),
    Query("Q22", "keyword", "HyDE hypothetical passage query expansion retrieval", ["F87"]),
    Query("Q23", "keyword", "HNSW ef_construction dense index parameters", ["F88"]),
    Query("Q24", "keyword", "vLLM continuous batching tensor parallelism GPU", ["F91", "F97"]),
    # semantic (paraphrase — must not pick near-duplicate wrong fact)
    Query("Q25", "semantic", "who is responsible for Tokyo on-call pager coverage?", ["F47"]),
    Query("Q26", "semantic", "how does Osaka engineering collaborate with Tokyo?", ["F46", "F52"]),
    Query("Q27", "semantic", "process for getting an API change approved before merging", ["F29"]),
    Query("Q28", "semantic", "hybrid search scoring weights between dense and sparse", ["F82", "F98"]),
    # entity (same office, different people — must pick the right one)
    Query("Q29", "entity", "who manages the Osaka GPU node pool?", ["F57"]),
    Query("Q30", "entity", "Singapore AI research lead and quarterly eval authorship", ["F58"]),
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