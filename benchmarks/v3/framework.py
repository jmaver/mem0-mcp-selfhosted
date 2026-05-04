"""Shared framework for v3 benchmark runners.

Salvaged from the v0.3-era ``benchmarks/mcp_e2e_battle.py`` — keeps the
v3-agnostic dataclasses, scoring helpers, infra health checks, operation
timer, and Memory factory. Drops everything that referenced the deleted
graph pipeline (GraphEntityCase, ContradictionCase, score_graph_*,
check_neo4j, query_graph_*, SplitModelGraphLLM, patch_graph_sanitizer).
"""

from __future__ import annotations

import math
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Suppress mem0 telemetry before any mem0 import — same guard as src/__init__.py
os.environ.setdefault("MEM0_TELEMETRY", "false")


# ============================================================
# Test Case Dataclasses
# ============================================================

@dataclass
class FactExtractionCase:
    id: str
    category: str
    difficulty: str  # easy, medium, hard
    messages: list[dict[str, str]]
    expected_facts: list[str]
    hallucination_traps: list[str] = field(default_factory=list)


@dataclass
class SearchCase:
    id: str
    category: str
    difficulty: str
    seed_messages: list[list[dict[str, str]]]  # multiple add() calls
    query: str
    expected_relevant: list[str]
    expected_absent: list[str] = field(default_factory=list)


@dataclass
class UpdateCase:
    id: str
    category: str
    difficulty: str
    initial_message: list[dict[str, str]]
    update_text: str
    verify_present: list[str]
    verify_absent: list[str]


@dataclass
class ComplexScenarioCase:
    id: str
    category: str
    difficulty: str
    messages_sequence: list[list[dict[str, str]]]
    search_queries: list[str]
    expected_search_hits: list[list[str]]
    expected_entities: list[str]
    expected_relationships: list[tuple[str, str, str]]


@dataclass
class RobustnessCase:
    id: str
    category: str
    difficulty: str
    messages: list[dict[str, str]]
    expect_success: bool
    min_facts: int


# ============================================================
# Scoring Helpers
# ============================================================

def _norm(s: str) -> str:
    return s.lower().strip().replace("_", " ").replace("-", " ")


def _fuzzy_match(needle: str, haystack: str) -> bool:
    return _norm(needle) in _norm(haystack)


def _fuzzy_match_any(needle: str, texts: list[str]) -> bool:
    nn = _norm(needle)
    return any(nn in _norm(t) for t in texts)


def score_fact_extraction(memories: list[dict], case: FactExtractionCase) -> dict:
    """F1 of expected facts + hallucination penalty."""
    all_text = " ".join(m.get("memory", "") for m in memories).lower()
    found = sum(1 for f in case.expected_facts if _fuzzy_match(f, all_text))
    total_expected = len(case.expected_facts)
    hallucinated = sum(1 for t in case.hallucination_traps if _fuzzy_match(t, all_text))
    recall = found / total_expected if total_expected > 0 else 1.0
    precision = found / (found + hallucinated) if (found + hallucinated) > 0 else (1.0 if total_expected == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "f1": round(f1, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "found": found,
        "expected": total_expected,
        "hallucinated": hallucinated,
        "extracted_count": len(memories),
    }


def score_search(results: list[dict], case: SearchCase) -> dict:
    """Recall of expected results + false-positive penalty."""
    all_text = " ".join(r.get("memory", "") for r in results).lower()
    found_relevant = sum(1 for e in case.expected_relevant if _fuzzy_match(e, all_text))
    total_relevant = len(case.expected_relevant)
    false_positives = sum(1 for a in case.expected_absent if _fuzzy_match(a, all_text))
    recall = found_relevant / total_relevant if total_relevant > 0 else 1.0
    fp_penalty = false_positives * 0.2
    score = max(0.0, recall - fp_penalty)
    return {
        "score": round(score, 4),
        "recall": round(recall, 4),
        "found": found_relevant,
        "expected": total_relevant,
        "false_positives": false_positives,
    }


def score_update(search_results: list[dict], case: UpdateCase) -> dict:
    """New content present + old content absent."""
    all_text = " ".join(r.get("memory", "") for r in search_results).lower()
    present_found = sum(1 for p in case.verify_present if _fuzzy_match(p, all_text))
    absent_found = sum(1 for a in case.verify_absent if _fuzzy_match(a, all_text))
    presence_score = present_found / len(case.verify_present) if case.verify_present else 1.0
    absence_score = 1.0 - (absent_found / len(case.verify_absent)) if case.verify_absent else 1.0
    score = (presence_score + absence_score) / 2.0
    return {
        "score": round(score, 4),
        "presence": round(presence_score, 4),
        "absence": round(absence_score, 4),
        "present_found": present_found,
        "absent_found": absent_found,
    }


def score_robustness(success: bool, memories: list[dict], case: RobustnessCase) -> dict:
    num_facts = len(memories)
    meets_threshold = num_facts >= case.min_facts
    score = (0.5 if success else 0.0) + (0.5 if meets_threshold else 0.0)
    return {
        "score": round(score, 4),
        "success": success,
        "num_facts": num_facts,
        "min_required": case.min_facts,
        "meets_threshold": meets_threshold,
    }


def ci95(scores: list[float]) -> tuple[float, float, float]:
    """(mean, ci_low, ci_high) for a 95% confidence interval."""
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(scores) / n
    if n == 1:
        return mean, mean, mean
    var = sum((x - mean) ** 2 for x in scores) / (n - 1)
    se = math.sqrt(var / n)
    margin = 1.96 * se
    return mean, max(0.0, mean - margin), min(1.0, mean + margin)


def bar(val: float, width: int = 30) -> str:
    filled = int(val * width)
    return f"[{'#' * filled}{'.' * (width - filled)}]"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[-1]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


# ============================================================
# Infrastructure Health Checks
# ============================================================

def check_qdrant() -> str | None:
    url = os.environ.get("MEM0_QDRANT_URL", "http://localhost:6333")
    try:
        urllib.request.urlopen(f"{url}/healthz", timeout=5)
        return url
    except (urllib.error.URLError, OSError, socket.timeout):
        return None


def check_ollama() -> str | None:
    url = os.environ.get("MEM0_EMBED_URL") or os.environ.get("MEM0_OLLAMA_URL") or "http://localhost:11434"
    try:
        urllib.request.urlopen(f"{url}/api/tags", timeout=5)
        return url
    except (urllib.error.URLError, OSError, socket.timeout):
        return None


def check_cloud_creds() -> bool:
    """Return True iff an Anthropic token is resolvable (any source).

    Prefer ANTHROPIC_API_KEY for long benchmark runs (no expiry risk vs OAT).
    If ANTHROPIC_API_KEY is set, mirrors it into MEM0_ANTHROPIC_TOKEN so
    build_config()'s resolve_token() picks it up first.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        os.environ["MEM0_ANTHROPIC_TOKEN"] = api_key
        return True
    try:
        from mem0_mcp_selfhosted.auth import resolve_token
        return bool(resolve_token())
    except Exception:
        return False


# ============================================================
# Operation Timer
# ============================================================

class OperationTimer:
    """Track latencies bucketed by operation name."""

    def __init__(self) -> None:
        self.latencies: dict[str, list[float]] = {}

    def record(self, op_type: str, duration: float) -> None:
        self.latencies.setdefault(op_type, []).append(duration)

    def summary(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for op, vals in self.latencies.items():
            result[op] = {
                "count": len(vals),
                "mean": round(sum(vals) / len(vals), 3) if vals else 0.0,
                "p50": round(percentile(vals, 50), 3),
                "p95": round(percentile(vals, 95), 3),
                "total": round(sum(vals), 3),
            }
        return result


# ============================================================
# Memory Instance Factory
# ============================================================

# Make src/ importable when running as `python -m benchmarks.v3.<runner>` from
# a checkout. uv run / installed-package usage doesn't need this but it's cheap.
_SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if _SRC_DIR.is_dir():
    import sys
    sys_path_str = str(_SRC_DIR)
    if sys_path_str not in sys.path:
        sys.path.insert(0, sys_path_str)


def make_memory(
    provider: str | None = None,
    model: str | None = None,
    collection: str = "mem0_bench_v3",
) -> tuple[Any, Callable[[], None]]:
    """Build a fresh ``mem0.Memory`` for benchmarking.

    Mirrors ``tests/integration/conftest.py`` exactly: build_config →
    register_providers → patch_gemini_parse_response → Memory.from_config.

    Env vars (``MEM0_LLM_PROVIDER``, ``MEM0_LLM_MODEL``, ``MEM0_COLLECTION``)
    are temporarily overridden when *provider* / *model* / *collection* are
    passed and restored by the returned cleanup callable. Cleanup does NOT
    drop the collection — runners are responsible for ``safe_bulk_delete``
    on their own ``user_id`` so concurrent benchmarks don't wipe each other.
    """
    overrides: dict[str, str | None] = {}
    if provider is not None:
        overrides["MEM0_LLM_PROVIDER"] = provider
    if model is not None:
        overrides["MEM0_LLM_MODEL"] = model
    overrides["MEM0_COLLECTION"] = collection

    originals: dict[str, str | None] = {k: os.environ.get(k) for k in overrides}
    for k, v in overrides.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

    from mem0_mcp_selfhosted.config import build_config
    from mem0_mcp_selfhosted.helpers import patch_gemini_parse_response
    from mem0_mcp_selfhosted.server import register_providers

    config_dict, providers_info = build_config()
    register_providers(providers_info)
    patch_gemini_parse_response()

    from mem0 import Memory

    memory = Memory.from_config(config_dict)

    def _cleanup() -> None:
        for k, v in originals.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    return memory, _cleanup


def bench_user_id(runner: str) -> str:
    """Unique-per-run user_id. Lets concurrent runners coexist & cleanup safely."""
    return f"bench-{runner}-{int(time.time())}"