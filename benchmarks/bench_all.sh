#!/usr/bin/env bash
# Run every v3 benchmark in sequence and write a combined results file.
#
# Sources the same env layers every individual run needs:
#   - ~/.claude/hooks/mem0-env.sh (MCP-server env: Qdrant URL, embedder, etc.)
#   - .env.local (per-user secrets: API tokens, provider keys)
# and pins MEM0_QDRANT_TIMEOUT=60 so create_collection survives sparse+dense
# vector provisioning under load.
#
# Usage:
#   benchmarks/bench_all.sh                                      # full sweep
#   benchmarks/bench_all.sh --provider-legs anthropic-api,ollama-qwen35-4b  # subset
#   benchmarks/bench_all.sh --hook-iters 3 --limit 4             # smoke run
#
# Output: benchmarks/v3/results/<UTC-DATE>-bench-all.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- Args ---
PROVIDER_LEGS="anthropic-api,openai-cloud-nano,lmstudio-qwen35-mlx,lmstudio-gemma,ollama-qwen35-4b,ollama-qwen3-14b"
LIMIT=5
HOOK_ITERS=10
ENTITY_ITERS=20
SLEEP=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --provider-legs) PROVIDER_LEGS="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --hook-iters) HOOK_ITERS="$2"; shift 2 ;;
    --entity-iters) ENTITY_ITERS="$2"; shift 2 ;;
    --sleep) SLEEP="$2"; shift 2 ;;
    -h|--help)
      # Print the leading comment block (everything from the first `#` line
      # to the first blank line). Robust to header growth.
      awk '/^#/{print; next} {exit}' "${BASH_SOURCE[0]}"
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# --- Env ---
# shellcheck disable=SC1090
source ~/.claude/hooks/mem0-env.sh
if [[ -f .env.local ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
fi
export MEM0_QDRANT_TIMEOUT=60
export PYTHONUNBUFFERED=1

DATE="$(date -u +%Y-%m-%d)"
OUT="benchmarks/v3/results/${DATE}-bench-all.log"
mkdir -p "$(dirname "$OUT")"

run() {
  local label="$1"; shift
  echo
  echo "=========================================================="
  echo "  $label"
  echo "=========================================================="
  uv run python -u -m "$@"
}

{
  echo "# bench_all.sh — ${DATE}"
  echo "# provider_legs: ${PROVIDER_LEGS}"
  echo "# limit: ${LIMIT}  hook_iters: ${HOOK_ITERS}  entity_iters: ${ENTITY_ITERS}  sleep: ${SLEEP}s"
  echo

  run "retrieval_quality" benchmarks.v3.retrieval_quality --limit "$LIMIT"
  run "provider_battle"   benchmarks.v3.provider_battle --leg "$PROVIDER_LEGS" --limit "$LIMIT" --sleep "$SLEEP"
  run "hook_and_scoping"  benchmarks.v3.hook_and_scoping --iterations "$HOOK_ITERS"
  run "dedup_and_entity"  benchmarks.v3.dedup_and_entity --limit "$LIMIT" --entity-iters "$ENTITY_ITERS"
} 2>&1 | tee "$OUT"

echo
echo "Combined results written to: $OUT"
