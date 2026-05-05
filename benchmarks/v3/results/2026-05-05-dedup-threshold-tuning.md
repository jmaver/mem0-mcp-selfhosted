# Dedup Threshold Tuning — 2026-05-05

**Recommended threshold: keep 0.88 (no change).** Three-way sweep at 0.82,
0.85, and 0.88 produced identical confusion matrices across all 16 corpus
pairs. D05 and D08 score below 0.82 against their first memory, meaning they
cannot be caught by lowering the threshold within any precision-safe range.
Going below 0.88 buys nothing (recall unchanged at 0.750) while permanently
narrowing the headroom between real duplicates and near-miss distinct pairs.

---

## Methodology

Probe script: `benchmarks/v3/_probes/probe_dedup_threshold_tuning.py`

The probe monkey-patches `hooks._DEDUP_SIM_THRESHOLD` before each run via
`hooks._DEDUP_SIM_THRESHOLD = <thresh>` and restores the original value
afterwards; `hooks.py` is not permanently modified. Each threshold run
exercises all 16 pairs from `dedup_and_entity.py` (8 duplicate + 8 distinct)
using isolated per-pair user IDs, clean teardown after each pair. Entity
timing was skipped; only Part 1 (dedup confusion matrix) ran.

Infrastructure: Qdrant local + Ollama local (nomic-embed-text embeddings).

---

## Results

| threshold | TP | FN | FP | TN | precision | recall | accuracy | D05 | D08 |
|-----------|----|----|----|----|-----------|--------|----------|-----|-----|
| 0.82      |  6 |  2 |  0 |  8 | 1.000     | 0.750  | 0.875    | FN  | FN  |
| 0.85      |  6 |  2 |  0 |  8 | 1.000     | 0.750  | 0.875    | FN  | FN  |
| 0.88      |  6 |  2 |  0 |  8 | 1.000     | 0.750  | 0.875    | FN  | FN  |

All three thresholds are identical. No FPs at any setting; no improvement in
recall at lower thresholds.

---

## Analysis

D05 and D08 use conceptually equivalent phrasing that relies on domain
knowledge to recognize as duplicates ("docs colocated with modules" ↔
"documentation lives next to the code"; "PR needs a reviewer from a different
squad" ↔ "code review requires one engineer outside the author's squad").
These semantically overlap strongly to a human reader but map to embedding
vectors with cosine similarity **below 0.82**. A threshold low enough to
catch them would likely pull in unrelated pairs from the same topical domain,
creating FPs on the X-pairs.

---

## Decision

**Leave `_DEDUP_SIM_THRESHOLD` at 0.88.**

The 0.88 floor is already at the outer edge of what the embedding model
considers high-confidence paraphrase. Pushing lower gains nothing in recall
and erodes the safety margin. D05 and D08 are correctly classified as
"hard negatives for this embedding model" rather than threshold failures.

A follow-up option: add D05 and D08 to a curated "hard pair" corpus and
evaluate whether a cross-encoder re-ranker or a shorter prompt reformulation
(stripping the domain hint before embedding) can bridge the gap. This is out
of scope for the current self-hosted configuration.
