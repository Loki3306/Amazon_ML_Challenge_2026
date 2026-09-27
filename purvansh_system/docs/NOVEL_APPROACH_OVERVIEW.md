# Novel Approach: Winning Strategy for ML Challenge 2026
## Business Entity Resolution at Scale — Master Plan

---

## Executive Summary

This document lays out a **novel, competition-winning architecture** for the Amazon ML Challenge 2026 Business Entity Resolution task. The approach combines:

1. **Multi-Resolution Blocking** — 4-pass candidate generation with country-hard-partitioning, token inverted indexes, character n-gram TF-IDF sparse retrieval, phonetic encoding, and address-first fallback.
2. **Multi-View Feature Engineering** — 50+ redundant pairwise features across character, token, phonetic, structural, and optional embedding granularities.
3. **3-Stage Cascade Classifier** — Rule-based high-confidence filtering → LightGBM probability scoring → Consistency refinement with cardinality constraints.
4. **Self-Training Domain Adaptation for France** — Pseudo-label bootstrapping to transfer learned matching patterns from US/India to the zero-shot French domain.
5. **Graph-Based Transitive Consistency** — Post-processing with a bipartite matching graph to enforce global coherence and resolve conflicting assignments.

The entire pipeline is designed to run end-to-end within **a single 9-hour Kaggle session** (30GB RAM, T4 GPU).

---

## Why This Approach Wins

| Design Decision | Competitive Advantage |
| :--- | :--- |
| Multi-resolution blocking (4 passes) | Recall ceiling >98% vs. typical single-pass ~85% |
| 50+ multi-view features | Captures noise at every granularity; robust to any single failure mode |
| 3-stage cascade | Stage A resolves ~35% of pairs instantly; Stage B handles ambiguity; Stage C enforces consistency |
| Self-training for France | Most teams will have degraded France performance; this closes the gap |
| Graph-based post-processing | Resolves conflicting S2/S3 assignments that pairwise classifiers miss |
| Precision-optimized thresholding | F_0.5 rewards precision 2x over recall; most teams under-optimize this |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAW DATASETS                                 │
│   S1 (1.7M reference)    S2 (4.9M candidates)   S3 (5.1M candidates)│
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                     ┌─────────▼──────────┐
                     │  COUNTRY PARTITION  │
                     │  US | India | France│
                     └─────────┬──────────┘
                               │
              ┌────────────────▼────────────────┐
              │   STAGE 1: MULTI-RESOLUTION     │
              │   BLOCKING (4 Passes)           │
              │                                 │
              │  Pass 1: Token Inverted Index   │
              │  Pass 2: N-Gram TF-IDF Top-K    │
              │  Pass 3: Phonetic + Postal Code │
              │  Pass 4: Address-First Fallback │
              │                                 │
              │  Output: candidate_pairs.tsv    │
              │  (~15-30 candidates per S1)     │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │   FEATURE ENGINEERING           │
              │   50+ Pairwise Similarity       │
              │   Features (Name, Address,      │
              │   Cross-Field, Structural)      │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │   STAGE 2: CASCADE CLASSIFIER   │
              │                                 │
              │  A. Rule-Based Fast Filter      │
              │     (~35% resolved instantly)   │
              │                                 │
              │  B. LightGBM P(match) Scoring   │
              │     (1500 rounds, max_depth=7)  │
              │                                 │
              │  C. Threshold @ t* for F_0.5    │
              └────────────────┬────────────────┘
              ┌────────────────▼────────────────┐
              │   STAGE 3: POST-PROCESSING      │
              │                                 │
              │  Cardinality Enforcement         │
              │  Graph Consistency Check         │
              │  Singleton Verification          │
              │  Self-Training for France        │
              └────────────────┬────────────────┘
              ┌────────────────▼────────────────┐
              │   OUTPUT: matching_results.tsv   │
              │   (1,732,544 rows)              │
              └─────────────────────────────────┘
```

---

## Documentation Suite

This master plan is supported by the following detailed implementation guides:

| # | Document | Purpose |
| :---: | :--- | :--- |
| 01 | 01_DATASET_AND_EDA.md | Dataset statistics, noise patterns, and EDA findings |
| 02 | 02_PIPELINE_ARCHITECTURE_AND_BLOCKING.md | High-level pipeline architecture |
| 03 | 03_FEATURE_ENGINEERING_AND_MODELING.md | Feature taxonomy overview |
| 04 | 04_VALIDATION_METRICS_AND_THRESHOLDING.md | Macro F_0.5 metric and local validation |
| 05 | 05_SUBMISSION_AND_COMPLIANCE_GUIDE.md | Output formats, validation, and fair-play rules |
| 06 | 06_NOVEL_BLOCKING_STRATEGY.md | 4-pass multi-resolution candidate generation |
| 07 | 07_ADVANCED_FEATURE_ENGINEERING.md | 50+ multi-view pairwise features |
| 08 | 08_MODEL_TRAINING_AND_CASCADE.md | 3-stage cascade, LightGBM, threshold optimization |
| 09 | 09_FRANCE_DOMAIN_ADAPTATION.md | Self-training pseudo-label bootstrapping for France |
| 10 | 10_GRAPH_POST_PROCESSING.md | Bipartite consistency, cardinality, singleton verification |
| 11 | 11_KAGGLE_IMPLEMENTATION_GUIDE.md | Notebook structure, memory budget, execution timeline |

---

## Key Innovation Summary

### Innovation 1: Self-Training for France (Zero-Shot Domain Adaptation)
Most competitors will train on US + India and apply the same model naively to France, losing significant performance. Our approach:
1. Train LightGBM on US + India ground truth.
2. Run inference on France candidate pairs.
3. Select high-confidence predictions (P > 0.90 for matches, P < 0.10 for non-matches) as **pseudo-labels**.
4. Retrain LightGBM including French pseudo-labeled pairs.
5. Re-infer on France with the adapted model.
This iterative self-training bootstraps domain knowledge without external data.

### Innovation 2: Graph-Based Transitive Consistency
Pairwise classifiers treat each (S1, S2/S3) pair independently, which can produce inconsistencies:
- S2-A matched to both S1-X and S1-Y (cardinality violation if S2 entities are unique).
- S3-B matched to S1-X, and S3-C also matched to S1-X, but S3-B and S3-C are very different.

We build a bipartite assignment graph and use **greedy maximum weighted matching** to resolve conflicts, preferring the highest-confidence edge when assignments clash.

### Innovation 3: Adaptive Per-Country Thresholds
US, India, and France have different noise distributions. The optimal F_0.5 threshold differs by country. We optimize tau_US, tau_India, and tau_France independently on validation data.

### Innovation 4: Multi-Resolution Blocking Union
Instead of relying on a single blocking method, we use 4 complementary passes at different resolutions (token, character n-gram, phonetic, address). Their union achieves >98% recall while any single pass achieves only 70-85%.

### Innovation 5: Precision-First Cascade
Stage A of the cascade resolves ~35% of pairs using simple rules with near-perfect precision. This reduces the load on the LightGBM model and prevents it from making errors on easy cases.

---

## Expected Performance Targets

| Metric | Target | Rationale |
| :--- | :---: | :--- |
| Blocking Recall | >= 98.0% | 4-pass union with high-recall individual passes |
| Blocking Reduction Ratio | >= 99.999% | From 17T pairs to ~25M candidate pairs |
| LightGBM AUC-ROC | >= 0.985 | 50+ features on well-separated positives/negatives |
| Validation Macro F_0.5 (US+India) | >= 0.88 | Cascade + threshold optimization |
| Test Macro F_0.5 (US+India+France) | >= 0.85 | Self-training closes France gap |

---

## Risk Mitigation

| Risk | Mitigation |
| :--- | :--- |
| Blocking recall too low for France | Pass 2 (TF-IDF n-gram) is language-agnostic; augment with Pass 4 (address-first) |
| Self-training introduces noise | Conservative thresholds (0.92/0.08), sample weighting, monotonic improvement check |
| Kaggle memory overflow | Country partitioning, batch processing, float32/float16, sequential pass execution |
| Kaggle 9-hour time limit | Cascade Stage A resolves 35% instantly; profile and optimize bottlenecks |
| Model overfits to US/India patterns | Feature engineering is purely string-similarity-based (language agnostic) |
