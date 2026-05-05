"""End-to-end hook latency + project-scoping isolation.

The Claude Code SessionStart hook has a 15 s timeout and SessionEnd has 30 s
(see ``hooks.py:569,588``). This runner subprocess-invokes both hooks against
real Qdrant + LLM infrastructure and reports P50/P95/max wall-clock.

Each invocation is a fresh process, so ``_get_memory()`` cold-start (mem0
import + Qdrant collection ensure + first embed) is included — that's the
dominant cost in production.

The scoping pass plants three distinguishable memories under different
``user_id:project`` keys, queries via ``search_with_project``, and asserts no
cross-project / cross-user leakage.

Usage::

    uv run python -m benchmarks.v3.hook_and_scoping [--iterations N] [--skip-hooks]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from benchmarks.v3.framework import (
    bench_user_id,
    check_ollama,
    check_qdrant,
    percentile,
)
from mem0_mcp_selfhosted.helpers import (
    make_project_user_id,
    safe_bulk_delete,
    search_with_project,
)

# Synthetic transcript exchanges — 12 messages in the JSONL shape expected by
# ``_read_recent_messages`` at hooks.py:289.
_FAKE_TRANSCRIPT = [
    ("user", "Help me plan a refactor of the deduplication code in the SessionEnd hook."),
    ("assistant", "Looking at the current implementation, the dedup uses a 0.88 cosine threshold "
                  "post-add. The main risk is mutual deletion when two new memories match each "
                  "other above threshold. Let me trace the order-of-operations to confirm."),
    ("user", "What invariant does the created_at comparison give us?"),
    ("assistant", "It guarantees the *older* memory survives. The new memory is only deleted when "
                  "an older memory matches it above 0.88. Without this, two near-duplicate adds in "
                  "the same batch would both be deleted, leaving zero memories."),
    ("user", "Should we widen the threshold or tighten it?"),
    ("assistant", "Probably tighten to 0.92 for paraphrases that rephrase the same fact, and "
                  "leave 0.88 for the literal-rewording bucket. We'd need a benchmark to validate."),
    ("user", "Architecture decision: we should treat documentation as code, living next to the "
              "modules it describes, and require code review on doc changes too."),
    ("assistant", "Agreed. The convention will be: every public function gets a one-line "
                  "docstring with the WHY, not the WHAT, and any cross-cutting concerns get a "
                  "module-level comment block."),
    ("user", "User preference: I want all benchmark output as side-by-side tables with "
              "consistent column widths."),
    ("assistant", "Got it. I'll add a small `_print_table` helper that computes column widths "
                  "from the data and reuses it across runners."),
    ("user", "One more — the on-call rotation should be capped at 24 hours of pager duty per "
              "engineer per week. Document this as a hard policy."),
    ("assistant", "Documented. Memory-worthy as a project policy, not a one-time decision."),
]


def _write_transcript(path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for role, text in _FAKE_TRANSCRIPT:
            entry = {
                "type": role,
                "message": {"role": role, "content": [{"type": "text", "text": text}]},
            }
            f.write(json.dumps(entry) + "\n")


def _hook_command(entry_point: str) -> list[str]:
    """Invoke a hook entry point through the current Python — no PATH dependency."""
    module = "mem0_mcp_selfhosted.hooks"
    return [sys.executable, "-c", f"from {module} import {entry_point}; {entry_point}()"]


def _time_hook(name: str, command: list[str], stdin_json: str, timeout: float) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            command, input=stdin_json, capture_output=True, text=True, timeout=timeout,
        )
        dt = time.perf_counter() - t0
        ok = proc.returncode == 0
        # Hook stdout is the response channel; failures may dump tracebacks to stderr.
        if not ok:
            tail = proc.stderr.strip().splitlines()[-3:]
            print(f"  [{name}] non-zero exit {proc.returncode}: {' / '.join(tail)}")
        return {"duration": dt, "ok": ok}
    except subprocess.TimeoutExpired:
        dt = time.perf_counter() - t0
        print(f"  [{name}] TIMED OUT after {dt:.1f}s")
        return {"duration": dt, "ok": False, "timed_out": True}


def _bench_hooks(iterations: int, project_dir: Path) -> dict[str, list[float]]:
    transcript = project_dir / "transcript.jsonl"
    _write_transcript(transcript)

    context_in = json.dumps({"cwd": str(project_dir), "session_id": "bench-session"})
    session_in = json.dumps({
        "cwd": str(project_dir),
        "session_id": "bench-session",
        "transcript_path": str(transcript),
    })

    results: dict[str, list[float]] = {"context": [], "session_end": []}

    print(f"\nRunning {iterations} iterations of each hook (cold-start each call) …")
    for i in range(iterations):
        r = _time_hook("context", _hook_command("context_main"), context_in, timeout=20.0)
        results["context"].append(r["duration"])
        print(f"  iter {i + 1:>2}/{iterations}: context={r['duration']:.2f}s ok={r['ok']}")

        r = _time_hook("session_end", _hook_command("session_end_main"), session_in, timeout=35.0)
        results["session_end"].append(r["duration"])
        print(f"            session_end={r['duration']:.2f}s ok={r['ok']}")

    return results


def _hook_summary(timings: dict[str, list[float]]) -> None:
    budgets = {"context": 15.0, "session_end": 30.0}
    print()
    print(f"{'hook':<13} | {'n':>3} | {'p50':>6} | {'p95':>6} | {'max':>6} | {'budget':>6} | over_budget")
    print("-" * 74)
    for name, vals in timings.items():
        if not vals:
            continue
        p50, p95, mx = percentile(vals, 50), percentile(vals, 95), max(vals)
        budget = budgets[name]
        over = sum(1 for v in vals if v > budget)
        flag = " <-- " + (f"{over} over budget!" if over else "ok")
        print(f"{name:<13} | {len(vals):>3} | {p50:>6.2f} | {p95:>6.2f} | {mx:>6.2f} | "
              f"{budget:>6.1f} | {over}{flag}")


def _bench_scoping(collection: str) -> int:
    """Return the number of isolation failures (0 = pass)."""
    from benchmarks.v3.framework import make_memory

    print("\nProject-scoping isolation check …")
    mem, cleanup_env = make_memory(collection=collection)

    user_a = bench_user_id("scope-alice")
    user_b = bench_user_id("scope-bob")
    proj_a, proj_b = "projA", "projB"

    # Three planted memories with distinguishable canary tokens.
    plants = [
        (user_a, proj_a, "alice-projA-CANARY-AAAA workflow note"),
        (user_a, proj_b, "alice-projB-CANARY-BBBB workflow note"),
        (user_b, proj_a, "bob-projA-CANARY-CCCC workflow note"),
    ]
    planted_uids: list[str] = []
    for user, proj, text in plants:
        uid = make_project_user_id(user, proj)
        planted_uids.append(uid)
        mem.add(messages=[{"role": "user", "content": text}], user_id=uid, infer=False)

    failures = 0
    try:
        # alice@projA must never see CANARY-BBBB or CANARY-CCCC
        results = search_with_project(mem, "workflow note", user_a, proj_a, top_k=10)
        text_blob = " ".join(r.get("memory", "") for r in results)
        if "CANARY-BBBB" in text_blob:
            print("  FAIL: alice/projA query leaked CANARY-BBBB (alice/projB)")
            failures += 1
        if "CANARY-CCCC" in text_blob:
            print("  FAIL: alice/projA query leaked CANARY-CCCC (bob/projA)")
            failures += 1
        if "CANARY-AAAA" not in text_blob:
            print("  FAIL: alice/projA query missed its own CANARY-AAAA")
            failures += 1

        # alice@global must only see bare-alice (none in this set) — should be empty
        results_global = search_with_project(mem, "workflow note", user_a, project=None, top_k=10)
        text_global = " ".join(r.get("memory", "") for r in results_global)
        for canary in ("CANARY-BBBB", "CANARY-CCCC"):
            if canary in text_global:
                print(f"  FAIL: alice/global leaked {canary}")
                failures += 1

        if failures == 0:
            print("  OK: 3-way scoping isolation holds for project + global")
    finally:
        for uid in planted_uids:
            try:
                safe_bulk_delete(mem, {"user_id": uid})
            except Exception as exc:
                print(f"  Cleanup failed for {uid}: {exc}", file=sys.stderr)
        cleanup_env()

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="hook latency + scoping isolation")
    parser.add_argument("--iterations", type=int, default=10, help="hook iterations per side")
    parser.add_argument("--skip-hooks", action="store_true",
                        help="run only the scoping isolation check (faster)")
    parser.add_argument("--collection", default="mem0_bench_v3")
    args = parser.parse_args()

    if not check_qdrant():
        print("Qdrant unreachable — set MEM0_QDRANT_URL", file=sys.stderr)
        return 1
    if not check_ollama():
        print("Ollama unreachable", file=sys.stderr)
        return 1

    if not args.skip_hooks:
        with tempfile.TemporaryDirectory(prefix="mem0-bench-") as tmpdir:
            timings = _bench_hooks(args.iterations, Path(tmpdir))
        _hook_summary(timings)

    failures = _bench_scoping(args.collection)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())