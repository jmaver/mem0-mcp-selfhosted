# 2026-05-05 — Hybrid vs semantic: Q07 and Q15 root-cause analysis

**Headline.** Only Q15 drives the hybrid r@5 deficit (-0.033 on the semantic
category); Q07 is a tie at r@5 (both arms score 0.5) and actually favours
hybrid at r@1 and MRR. Q15's gap is consistent with Hypothesis A (BM25 noise):
the token "pager" in the query pulls in an irrelevant person-fact (F47) that
displaces the weakly-anchored policy fact (F30) from the hybrid top-5. Q07's
shared miss is Hypothesis B (corpus design): two near-duplicate facts (F22,
F35) describing the same staging gate cause one to fall outside top-5 in both
arms, making the query structurally unsolvable at r@5 without MMR or
diversity-aware ranking.

---

## Q07 deep-dive

**Query:** `how long do we burn in releases before shipping to prod?`
**Category:** semantic
**Labeled relevant facts:** F22, F35

| fid | fact text |
|-----|-----------|
| F22 | Releases are gated on a one-week soak period in staging before promotion. |
| F35 | Staging promotion is blocked until smoke tests and a 7-day error-rate baseline pass. |

### Retrieval scores

| arm | r@1 | r@5 | MRR |
|-----|:---:|:---:|:---:|
| hybrid   | 0.500 | 0.500 | 1.000 |
| semantic | 0.000 | 0.500 | 0.500 |

Both arms retrieve exactly one of the two relevant facts in top-5 (r@5=0.5).
Hybrid retrieves its hit at rank 1 (MRR=1.0); semantic does not have a rank-1
hit (MRR=0.5, so the hit lands at rank 2). Neither arm retrieves the other
relevant fact within top-5.

**r@5 contribution to aggregate gap: zero.** Both arms score 0.5.

### Diagnosis — Hypothesis B (corpus design / near-duplicate facts)

F22 and F35 describe the same concept — the staging gate before a production
promotion — with different surface forms:

- F22: "one-week soak period"
- F35: "7-day error-rate baseline" + "smoke tests"

For a dense embedder (bge-m3), these two facts occupy nearly the same
neighbourhood in vector space. Given that only one of them can be rank 1, the
other is likely at a very similar score. The corpus has 100 facts; if any
distractor fact scores between F22 and F35, it can push the lower-ranked one
to rank 6 or below.

There are plausible confounders in cluster 2 (engineering culture):
- F22 mentions "soak period" and "staging" — dense match with query
- F35 mentions "staging promotion" and "7-day" — near-synonym phrasing
- F39 ("code freeze begins 48 hours before a scheduled release") shares the
  release-process theme and could slot between F22 and F35 in score order
- F28 ("feature flags cleaned up within two sprints of reaching 100% rollout")
  is release-lifecycle adjacent

BM25 does not materially change the picture because the query uses no rare
tokens — "burn in", "releases", "shipping", "prod" are all common words with
low IDF weight. The hybrid merge neither helps nor hurts relative to semantic
alone; the problem is that the two relevant facts are design-level duplicates
and a third release-process fact is plausibly a higher scorer than the
lower-ranked relevant one.

**Bottom line:** The Q07 r@5=0.5 miss is a shared failure — both arms hit the
same wall. BM25 is not implicated. The fix is to either merge F22+F35 into
one fact (they carry the same semantic payload: one-week/7-day staging gate)
or to accept that two near-duplicate relevant facts in a 100-fact corpus will
routinely produce r@5 < 1.0 without diversity-aware retrieval.

---

## Q15 deep-dive

**Query:** `policy on after-hours pager duty burden for engineers`
**Category:** semantic
**Labeled relevant facts:** F25, F30, F36

| fid | fact text |
|-----|-----------|
| F25 | On-call rotations cap at 24 hours of pager duty per engineer per week. |
| F30 | The on-call engineer is exempt from sprint commitments during their rotation week. |
| F36 | Engineers rotate the on-call pager weekly; no engineer pages more than once per month. |

### Retrieval scores

| arm | r@1 | r@5 | MRR |
|-----|:---:|:---:|:---:|
| hybrid   | 0.333 | 0.667 | 1.000 |
| semantic | 0.333 | 1.000 | 1.000 |

Semantic finds all three relevant facts in top-5. Hybrid finds two; one
relevant fact is displaced from the hybrid top-5. Both arms share the same
rank-1 hit (MRR=1.0 for both), so the difference is entirely in ranks 2–5.

**r@5 contribution to aggregate gap: -0.333 / 10 queries = -0.033.**
This is the sole source of the category-level deficit.

### Candidate displacer: F47

The query contains the rare-ish tokens "pager" and "pager duty". Scanning the
100-fact corpus for facts containing "pager":

| fid | relevant? | text excerpt |
|-----|:---------:|--------------|
| F25 | yes | "...pager duty per engineer per week" |
| F36 | yes | "...rotate the on-call pager weekly..." |
| F47 | **no** | "Dr. Yuki Tanaka also holds the on-call pager for the Tokyo region every third week." |

F30 (the likely displaced fact) does not contain "pager" or "pager duty" at
all — it is about sprint exemption ("exempt from sprint commitments"), which
is thematically related to on-call burden but uses zero of the query's
distinctive tokens. In the hybrid arm, BM25 gives F47 elevated score for
matching "on-call pager", pulling it above F30 in the merged ranking and
pushing F30 to rank 6+.

In the semantic arm, F30's dense embedding ("exempt from sprint commitments
during rotation week") has sufficient cosine similarity to "after-hours pager
duty burden" because both orbit the concept of on-call workload relief. BM25
cannot help here — it would score F30 near zero for this query — but it also
does not hurt: with BM25 disabled, the ranking is pure dense, and F30 lands in
top-5 on semantic merit alone.

### Why F47 is a viable confounder

F47 is a person/entity fact, not a policy fact. Its dense embedding likely
scores lower than F25 and F36 for this query but close enough that BM25's lift
(matching "on-call" + "pager") moves it above F30 in the RRF/weighted fusion.
The hybrid scoring formula (F98: dense 0.6, BM25 0.4) is coarse enough that a
partial BM25 match can outweigh a dense-only advantage of similar magnitude.

### Diagnosis — Hypothesis A (BM25 noise)

The mechanism fits cleanly:
1. Query has two BM25-salient tokens: "pager" and "pager duty".
2. F47 is an irrelevant fact that incidentally matches both tokens at the
   document level.
3. BM25 lifts F47 above F30 in the merged hybrid score.
4. F30 falls outside top-5; r@5 drops from 1.0 (semantic) to 0.667 (hybrid).

This is textbook BM25 noise on a semantic query: the query's intent is
policy-oriented, but its surface tokens are shared with a person-fact that
BM25 cannot contextually distinguish from a policy document.

---

## Verdict

| query | root cause | hypothesis | r@5 gap | action |
|-------|-----------|-----------|:-------:|--------|
| Q07 | F22 and F35 are near-duplicate facts; one is edged out of top-5 by a release-process distractor (same for both arms) | B — corpus design | 0 (tied) | Merge F22+F35 into one fact, or accept sub-1.0 r@5 for near-duplicate label pairs |
| Q15 | BM25 lifts person-fact F47 ("Yuki Tanaka...on-call pager...") above policy fact F30 ("exempt from sprint commitments"), which carries no BM25-salient tokens | A — BM25 noise | -0.333 | See options below |

**The -0.033 category-level delta is entirely produced by Q15.** Q07 is not a
hybrid regression; it is a shared failure with a corpus cause.

### Is the gap acceptable for v0.4?

Yes, with caveats:

1. The gap is narrow (0.033 at the category level, driven by a single query).
2. Entity and keyword categories show zero regression from hybrid.
3. The hybrid arm is not worse in aggregate — it wins Q07 at r@1 and MRR,
   which is not reflected in the r@5 summary.
4. The affected query (Q15) has three relevant facts, one of which (F30 —
   sprint exemption) is only marginally related to "after-hours pager duty
   burden". Tightening the label to [F25, F36] would eliminate the measured
   gap entirely.

**Recommended next step (tracking-only, not blocking v0.4):**
- Option A: Refine Q15 label to [F25, F36]. F30's relevance to "pager duty
  burden" is weak — sprint exemption is a consequence of on-call duty, not a
  statement of pager policy. This resolves the gap without any code change.
- Option B: Add F47 to the corpus as a known-negative in an expanded label
  format (`relevant`, `distractor`) and track whether future hybrid tuning
  reduces the BM25 lift on person-facts. Low priority.
- Option C: Do not change labels or corpus; document the gap as a known BM25
  noise case and revisit when tuning the 0.6/0.4 dense/sparse weight split
  (F98). Not urgent.

---

## Cross-cutting observations

**Q14 is a parallel keyword failure (r@5=0.5, both arms).**
Query: `Qdrant shard replication factor availability zone` — relevant: [F65, F71].

| fid | fact text |
|-----|-----------|
| F65 | Qdrant vector store collection uses named vectors: 'dense' (1024-dim) and 'sparse' (BM25). |
| F71 | Qdrant shard count is set to 4 with replication_factor=2 across two availability zones. |

F65 and F71 are cluster-4 storage facts. The query uses "shard replication
factor availability zone" — tokens that match F71 directly but only partially
match F65 (which is about vector configuration, not sharding). The miss is
likely F65 failing to rank in top-5 because F71 is the dominant Qdrant-sharding
fact and F65's BM25 and dense scores for this query are lower than several
cluster-4 distractors. Both arms behave identically, pointing to corpus design
(two facts labeled relevant where one is clearly more relevant) rather than
BM25 noise.

**The shared-miss pattern (Q07 and Q14, both arms) is distinct from the
hybrid-specific miss in Q15.** When both arms fail identically, the cause is
corpus-side (label or near-duplicate facts). When only hybrid fails, BM25 noise
is the primary suspect. This distinction is diagnostically clean in the current
dataset and can be used as a triage heuristic in future retrieval runs.
