# PHASE 2 — RETRIEVAL V2

## Objective: Maximize Candidate Recall Without Destroying Downstream Precision

We have already established and frozen the current baseline. Do NOT redo the baseline.

### Current Baseline

Validation:

* Dense MiniLM K=50:

  * Pair recall: 78.22%
  * S1/query coverage: 90.47%
* Exact:

  * Pair recall: 28.19%
  * S1/query coverage: 60.91%
* Dense + Exact hybrid:

  * Pair recall: 80.52%
  * S1/query coverage: 91.17%
  * True pairs: 7,761,612
  * Retrieved true pairs: 6,249,439
  * Missed true pairs: 1,512,173
* LightGBM:

  * Validation AUC: 0.9979
  * Best pairwise macro F0.5: 0.8042
  * Threshold: 0.95

The current system is an acceptable baseline.

The goal of this phase is to improve the retrieval/candidate-generation layer first.

---

# 1. Core objective

Current candidate recall is only 80.52%.

Approximately 19.48% of true relationships are currently invisible to the matcher because they never enter the candidate set.

Therefore:

> The primary objective of Phase 2 is to maximize incremental candidate recall.

Do NOT optimize only for raw candidate recall.

For every retriever, measure:

1. Pair recall
2. S1/query coverage
3. Number of candidates
4. Candidate reduction ratio
5. Incremental true pairs recovered over the existing Dense + Exact baseline
6. Overlap with existing retrievers
7. Candidate volume added per 1% incremental recall

We want complementary retrieval mechanisms, not several retrievers making the same mistakes.

---

# 2. Do NOT modify the existing baseline

Preserve all existing retrieval code and outputs.

Do not overwrite:

* existing Dense K=50 results
* existing Exact results
* existing validation metrics

Create a new Retrieval V2 implementation and new output directory.

Everything must remain reproducible.

---

# 3. Experiment A — Dense K sweep

Evaluate the existing dense retrieval model at:

* K=50
* K=100
* K=200
* K=500

Do this on validation first.

For each K report:

```text
K
candidate_count
pair_recall
S1_coverage
reduction_ratio
incremental_true_pairs_vs_K50
```

Determine where additional K stops being worthwhile.

Do not assume K=500 is better merely because recall increases.

---

# 4. Experiment B — Character TF-IDF retrieval

Build sparse character n-gram retrieval.

Test separately:

### Name

* char 3–5 grams
* char 3–6 grams

### Address

* char 3–5 grams
* char 3–6 grams

### Combined

* normalized name + normalized address

Use sparse matrices / efficient ANN-compatible sparse retrieval.

Do NOT materialize a massive dense similarity matrix.

Retrieve top-K candidates per query.

Initially test:

* K=20
* K=50
* K=100

Measure the same metrics as above.

Most importantly calculate:

```text
new_true_pairs =
true pairs found by Char-TFIDF
BUT NOT found by existing Dense + Exact
```

This incremental recall is the primary metric.

---

# 5. Experiment C — Word / BM25 retrieval

Implement efficient lexical retrieval for:

* business_name
* business_address
* business_name + business_address

Test top-K retrieval.

Again calculate:

```text
standalone recall
incremental recall over Dense + Exact
candidate count
overlap
```

Do not keep a retriever merely because its standalone recall is high.

Keep it if it recovers useful true pairs missed by the current hybrid.

---

# 6. Experiment D — Structured blocking

Create additional candidate generators using only fields actually present in the challenge:

* business_name
* business_address
* country

Do NOT invent unavailable fields.

Useful blocks to investigate:

### Name blocks

* normalized first token
* normalized name prefix
* rare name token
* token signature

### Address blocks

* numeric-token signature
* postal/PIN-like token when present
* house/building number
* rare address token
* locality/city-like token

### Combined blocks

* country + rare name token
* country + numeric address signature
* country + rare address token

These are candidate generators, NOT hard filters.

A true pair should be allowed to enter through any one retrieval mechanism.

---

# 7. Experiment E — Missed true-pair analysis

This is mandatory.

For every validation true pair missed by the current Dense + Exact baseline, determine whether it is recoverable by:

* Dense K100
* Dense K200
* Dense K500
* Char TF-IDF
* BM25
* structured blocking

Produce a coverage matrix:

```text
                         Dense  Exact  Char  BM25  Block
True pair A                1      0     0     1      0
True pair B                0      0     1     0      0
True pair C                0      0     0     0      1
...
```

Then summarize the failure population.

Example:

```text
Currently missed true pairs: 1,512,173

Recovered by Dense K100:       X
Recovered by Char TF-IDF:      X
Recovered by BM25:             X
Recovered by blocking:         X
Recovered only by new method:  X
Still missed:                  X
```

This analysis determines which retrieval mechanisms are actually valuable.

---

# 8. Build Hybrid Retrieval V2

After evaluating individual retrievers, create:

```text
Hybrid V2 =
    Dense
  ∪ Exact
  ∪ best Char-TFIDF
  ∪ best BM25
  ∪ best Structured Blocks
```

CRITICAL:

Deduplicate by:

```text
(query_id, candidate_id)
```

before writing the final candidate set.

A pair found by multiple retrievers must exist only once.

Retain metadata describing how it was retrieved:

```text
found_by_dense
found_by_exact
found_by_char
found_by_bm25
found_by_block
dense_rank
char_rank
bm25_rank
```

This metadata will later become LightGBM features.

---

# 9. Optimize for both recall and candidate efficiency

Do not simply maximize candidate count.

Produce a Pareto-style comparison:

```text
Retriever configuration
Candidate count
Pair recall
S1 coverage
Reduction ratio
Incremental recall
Candidates per 1% recall
```

The target is:

> substantially improve on 80.52% recall while keeping candidate volume computationally manageable.

Do not target an arbitrary recall number without measuring the cost.

---

# 10. Validation requirements

Every experiment must use exactly the same validation S1 split used by the current baseline.

Never change the split between experiments.

Use the existing ground truth.

Report:

### Retrieval metrics

* total true pairs
* retrieved true pairs
* missed true pairs
* pair recall
* S1 coverage
* candidate count
* reduction ratio

### Incremental metrics

* true pairs newly recovered over Dense + Exact
* percentage of current misses recovered
* overlap with existing retrieval mechanisms

---

# 11. Prepare for precision optimization

Phase 2 is primarily retrieval, but the output must preserve information needed to improve precision later.

For every final Hybrid V2 candidate pair retain:

```text
query_id
candidate_id
candidate_source
dense_rank
dense_score
retrieval_source(s)
found_by_dense
found_by_exact
found_by_char
found_by_bm25
found_by_block
```

This allows Phase 3/next stage to create retrieval-agreement features.

A pair found independently by multiple retrieval systems should be distinguishable from a pair found by only one system.

---

# 12. Memory / performance requirements

The dataset is millions of entities.

Do NOT implement anything that requires:

```text
N_query × N_candidate
```

dense similarity matrices.

Use:

* sparse matrices
* inverted indexes
* ANN
* chunked retrieval
* memory-mapped structures
* streaming/chunked processing

All retrieval must be resumable.

Use manifests/checkpoints where practical.

Never load hundreds of millions of candidate rows into a single Python object unnecessarily.

---

# 13. Do NOT implement these yet

Do NOT spend time on:

* XGBoost
* CatBoost
* model averaging
* Transformer reranking
* fine-tuning a cross-encoder
* replacing LightGBM

Those are downstream optimizations.

Our current bottleneck is candidate recall.

First improve the candidate set.

---

# 14. Success criteria

Phase 2 is successful only if we can demonstrate a meaningful improvement over:

```text
Dense + Exact K50
Pair recall = 80.52%
S1 coverage = 91.17%
```

The final report must clearly identify:

1. Best individual retriever
2. Best incremental retriever
3. Best Hybrid V2 configuration
4. New pair recall
5. New S1 coverage
6. Candidate count
7. Number of currently missed true pairs recovered
8. Number still missed
9. Computational cost
10. Recommended configuration for downstream LightGBM

Do not claim improvement merely because a new retriever produces more candidates.

The important result is:

> How many previously invisible true relationships did we recover, at what candidate-generation cost?

---

# Final expected architecture after Phase 2

```text
                         ┌── Dense K-best
                         ├── Exact
                         ├── Char TF-IDF
S1 ──────────────────────┼── BM25
                         └── Structured blocking
                                  │
                                  ▼
                            UNION + DEDUP
                                  │
                                  ▼
                         Hybrid V2 candidates
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
              retrieval metadata          pair features
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                              LightGBM
                                  │
                                  ▼
                       zero / one / many
                                  │
                                  ▼
                           final matching
```
