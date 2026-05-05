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
import tomllib
from pathlib import Path
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

_LEGS_FILE = Path(__file__).parent / "legs.toml"


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
        return "qwen3.5:4b"
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


def _load_legs() -> dict[str, dict[str, Any]]:
    """Read benchmarks/v3/legs.toml and return a name -> spec mapping."""
    if not _LEGS_FILE.is_file():
        return {}
    with _LEGS_FILE.open("rb") as f:
        data = tomllib.load(f)
    return data.get("legs", {})


def _resolve_leg_value(spec: dict[str, Any], static_key: str, env_key: str) -> str | None:
    """Resolve a leg value: prefer the static key, fall back to ``env_key`` env lookup.

    e.g. _resolve_leg_value(spec, "model", "model_env") returns spec["model"]
    if set, otherwise os.environ[spec["model_env"]] when that key exists.
    Returns None if neither is configured.
    """
    if static_key in spec and spec[static_key]:
        return str(spec[static_key])
    env_var = spec.get(env_key)
    if env_var:
        val = os.environ.get(str(env_var))
        if val:
            return val
        print(f"  [warn] leg references {env_var} but it is not set in the environment")
    return None


def _resolve_leg(name: str, spec: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve all env-references in a leg spec into concrete values."""
    provider = spec.get("provider")
    if not provider:
        print(f"  [skip] {name}: no provider in spec")
        return None
    return {
        "name": name,
        "provider": str(provider),
        "model": _resolve_leg_value(spec, "model", "model_env"),
        "llm_url": _resolve_leg_value(spec, "llm_url", "llm_url_env"),
        "llm_api_key": _resolve_leg_value(spec, "llm_api_key", "llm_api_key_env"),
        "anthropic_token": _resolve_leg_value(spec, "anthropic_token", "anthropic_token_env"),
        "description": spec.get("description", ""),
    }


def _run_with_overrides(
    label: str,
    provider: str,
    cases: list,
    collection: str,
    sleep_between_cases: float,
    **make_memory_overrides: Any,
) -> dict[str, Any] | None:
    """Shared inner loop. ``label`` is the human-readable run identifier;
    ``provider`` is the mem0 provider class to register; ``make_memory_overrides``
    accepts any kwargs ``framework.make_memory`` understands (model, llm_url,
    llm_api_key, anthropic_token).
    """
    model = make_memory_overrides.get("model")
    print(f"\n=== {label} ({provider} / {model or '<default>'}) ===")
    mem, cleanup_env = make_memory(
        provider=provider,
        collection=collection,
        **make_memory_overrides,
    )
    uid = bench_user_id(f"leg-{label}")
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

            # Inter-case throttle for rate-limited providers (Anthropic OAT,
            # OpenAI cloud, etc.). mem0's infer=True issues 6-10 LLM calls per
            # add(); spacing cases out keeps short rate-limit windows from
            # compounding across the run.
            if sleep_between_cases > 0:
                time.sleep(sleep_between_cases)

        mean_f1, lo_f1, hi_f1 = ci95(f1_scores)
        latency = timer.summary()
        return {
            "provider": provider,
            "label": label,
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
    headers = ["leg", "provider", "model", "mean_f1", "ci95_f1", "mean_recall",
               "hallucinations", "add_p50", "add_p95", "add_mean", "fail/n"]
    widths = [len(h) for h in headers]
    table_rows = []
    for r in rows:
        row = [
            r.get("label") or r["provider"], r["provider"], str(r["model"]),
            f"{r['mean_f1']:.3f}",
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


def _run_provider(provider: str, cases: list, collection: str, sleep_between_cases: float = 0.0) -> dict[str, Any] | None:
    """Legacy ``--providers`` path: gate on availability, then defer to
    ``_run_with_overrides`` with the provider's default model."""
    available, reason = _provider_available(provider)
    if not available:
        print(f"  [skip] {provider}: {reason}")
        return None
    return _run_with_overrides(
        label=provider,
        provider=provider,
        cases=cases,
        collection=collection,
        sleep_between_cases=sleep_between_cases,
        model=_default_model(provider),
    )


# OpenAI-compat URLs that always require an API key (cloud, not local).
_OPENAI_CLOUD_HOSTS = ("api.openai.com",)


def _openai_url_needs_key(url: str) -> bool:
    return any(host in url.lower() for host in _OPENAI_CLOUD_HOSTS)


def _leg_available(leg: dict[str, Any]) -> tuple[bool, str]:
    provider = leg.get("provider", "")
    if provider == "anthropic":
        if leg.get("anthropic_token"):
            return True, ""
        # Mirror the legacy --providers path: fall through to resolve_token()
        # so legs without an explicit token can still authenticate via
        # MEM0_ANTHROPIC_TOKEN, ~/.claude/.credentials.json, or ANTHROPIC_API_KEY.
        try:
            from mem0_mcp_selfhosted.auth import resolve_token
        except Exception as exc:
            return False, f"auth import failed: {exc}"
        return (True, "") if resolve_token() else (False, "no Anthropic token")
    if provider == "openai":
        if leg.get("llm_api_key"):
            return True, ""
        url = leg.get("llm_url")
        if not url:
            return False, "neither llm_api_key nor llm_url set"
        if _openai_url_needs_key(url):
            return False, f"{url} requires llm_api_key"
        return True, ""
    if provider == "ollama":
        url = leg.get("llm_url")
        return (True, "") if check_ollama(url) else (False, f"ollama unreachable{f' at {url}' if url else ''}")
    return False, f"unknown provider {provider!r}"


def _run_leg(leg: dict[str, Any], cases: list, collection: str, sleep_between_cases: float) -> dict[str, Any] | None:
    """Named-leg path: run a leg dict produced by ``_resolve_leg``.

    Legs are intentionally hermetic — fields the leg doesn't set are passed
    through as None to ``make_memory`` so the corresponding env var is *unset*
    rather than inherited from the shell. Otherwise an unrelated
    ``MEM0_LLM_URL`` (e.g. LM Studio's URL from ``mem0-env.sh``) leaks into
    the Anthropic ``anthropic_base_url`` and silently breaks the run.
    """
    available, reason = _leg_available(leg)
    if not available:
        print(f"  [skip] {leg['name']}: {reason}")
        return None
    overrides: dict[str, str | None] = {
        "model": leg.get("model"),
        "llm_url": leg.get("llm_url"),
        "llm_api_key": leg.get("llm_api_key"),
        "anthropic_token": leg.get("anthropic_token"),
    }
    return _run_with_overrides(
        label=leg["name"],
        provider=leg["provider"],
        cases=cases,
        collection=collection,
        sleep_between_cases=sleep_between_cases,
        **overrides,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="provider extraction battle")
    parser.add_argument("--limit", type=int, default=0, help="truncate case list")
    parser.add_argument("--providers", default="",
                        help="comma-separated provider list (legacy path: anthropic,ollama,openai)")
    parser.add_argument("--leg", default="",
                        help="comma-separated leg names from legs.toml, or 'all'")
    parser.add_argument("--collection", default="mem0_bench_v3")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="seconds to sleep between cases (rate-limit throttle)")
    args = parser.parse_args()

    if not check_qdrant():
        print("Qdrant unreachable — set MEM0_QDRANT_URL to a live instance", file=sys.stderr)
        return 1
    if not check_ollama():
        print("Ollama unreachable (needed for embedder bge-m3)", file=sys.stderr)
        return 1

    cases = FACT_EXTRACTION_CASES[: args.limit] if args.limit else FACT_EXTRACTION_CASES
    rows: list[dict[str, Any]] = []

    if args.leg:
        legs_table = _load_legs()
        if not legs_table:
            print(f"No legs.toml at {_LEGS_FILE} or it had no [legs.*] tables", file=sys.stderr)
            return 1
        names = list(legs_table.keys()) if args.leg.strip() == "all" else [n.strip() for n in args.leg.split(",") if n.strip()]
        unknown = [n for n in names if n not in legs_table]
        if unknown:
            print(f"Unknown leg(s) {unknown}. Defined: {list(legs_table.keys())}", file=sys.stderr)
            return 1
        print(f"Running {len(cases)} cases × {len(names)} legs (sleep={args.sleep}s) …")
        for name in names:
            leg = _resolve_leg(name, legs_table[name])
            if leg is None:
                continue
            result = _run_leg(leg, cases, args.collection, args.sleep)
            if result:
                rows.append(result)
    else:
        providers_str = args.providers or "anthropic,ollama,openai"
        providers = [p.strip() for p in providers_str.split(",") if p.strip()]
        print(f"Running {len(cases)} cases × {len(providers)} providers (sleep={args.sleep}s) …")
        for p in providers:
            result = _run_provider(p, cases, args.collection, sleep_between_cases=args.sleep)
            if result:
                rows.append(result)

    _print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())