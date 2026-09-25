# Amazon ML Challenge 2026: The Master Roadmap

This document outlines the strategic pivot from simple candidate generation to a rigorous, component-driven Retrieval Laboratory and Entity-Level Decision engine.

## The Core Philosophy
We must stop treating "Retrieval", "Features", and "Modeling" as overlapping problems. They solve fundamentally different issues.
*   **Retrieval/Blocking:** Solves the *Recall Ceiling*. If a true match is not in this set, no downstream model can ever predict it. 
*   **Matching/Features:** Solves *Pairwise Precision*. Given two items, how likely are they to be the same entity?
*   **Entity-Level Decision:** Solves *Global Consistency & Macro F0.5*. Translates raw pairwise probabilities into a final valid cluster (0, 1, or many matches) and handles singletons safely.

---

## Phase 7.1 — BASELINE AUDIT
*Our first task is to establish the true baseline on our Validation set using the output of the currently running pipeline.*
- [ ] Complete entity-level validation
- [ ] Measure Macro F0.5
- [ ] Measure Precision / Recall
- [ ] Measure Singleton False-Positive Rate (FPR)
- [ ] Breakdown of accuracy by 0-match, 1-match, and many-match queries
- [ ] **Candidate-Constrained Oracle F0.5:** Assume an Oracle perfectly selects true matches from our candidate sets. What is the mathematical maximum F0.5 score we can achieve? This tells us exactly whether we have a retrieval problem, a decision problem, or both.

## Phase 7.2 — RETRIEVAL LAB
*Do not immediately build one giant new pipeline. Build independent retrievers and measure their incremental value.*
- [ ] Dense K=100 / 200 / 500 sweep
- [ ] Char TF-IDF (name)
- [ ] Char TF-IDF (address)
- [ ] Word TF-IDF
- [ ] BM25 (name)
- [ ] BM25 (address)
- [ ] Structured address blocking (e.g., country + postal code, country + first name token)
- [ ] **Crucial Metric:** Measure *incremental recall* over the existing baseline for every retriever. We want different errors, not just more of the same.

## Phase 7.3 — MISSED-PAIR ANALYSIS
*Before throwing more embeddings at the wall, diagnose exactly why we miss the remaining 19.5% of true pairs.*
- [ ] Inspect all currently missed true pairs.
- [ ] Classify failure modes (Typo, Abbreviation, Transliteration, Address, DBA).
- [ ] Prioritize building the next retriever based on these actual, observed misses.

## Phase 7.4 — HYBRID RETRIEVAL V2
*Combine the winning components from the Retrieval Lab.*
- [ ] Union all complementary retrievers (C = union(C1, C2, C3...))
- [ ] Deduplicate (S1, Candidate) pairs.
- [ ] Generate Retrieval-Agreement metadata (`found_by_dense`, `found_by_bm25`, `dense_rank`).
- [ ] Target substantially > 80.5% recall.

## Phase 7.5 — FEATURE V2
*Make the existing LightGBM matcher smarter, particularly regarding addresses (our biggest opportunity).*
- [ ] Address components (Extract numeric tokens, postal codes, house numbers).
- [ ] Postal/house-number exact match flags.
- [ ] Char TF-IDF cosine similarity.
- [ ] Richer name normalization (initialism similarity, legal-suffix stripped).
- [ ] Token-order independent features (token sort JW, token jaccard).
- [ ] Transliteration / script features.
- [ ] Retrieval agreement features (Pass the metadata from 7.4 to LightGBM).

## Phase 7.6 — HARD NEGATIVE MATCHER
*Attack the region that matters at `threshold ≈ 0.95`. Sampled negatives are too easy.*
- [ ] Mine Dense hard negatives.
- [ ] Mine Lexical hard negatives.
- [ ] Mine Same-address but different-entity negatives.
- [ ] Mine Same-name but different-entity negatives.
- [ ] Retrain LightGBM strictly on these high-similarity false pairs.

## Phase 7.7 — DECISION V2
*Moving from a simple global `score >= 0.95` threshold to intelligent per-S1 logic.*
- [ ] Calibrate LightGBM probabilities.
- [ ] Per-S1 score margins (top score vs. second score ratio).
- [ ] Zero / One / Many assignment logic.
- [ ] Strict singleton rejection.
- [ ] Explicitly optimize for Macro F0.5.

## Phase 7.8 — EMBEDDING V2
*Only after the retrieval laboratory is built should we test stronger embeddings.*
- [ ] Test stronger embeddings (e.g., multilingual, semantic).
- [ ] Test name-only vs. address-only vs. concatenated embeddings.
- [ ] Compare complementary recall against our V1 embedding.

## Phase 7.9 — GLOBAL CONSISTENCY
*Enforce structural dataset constraints.*
- [ ] S2/S3 assignment constraints (satellites generally link to at most one S1).
- [ ] Cross-source consistency checking.

## Phase 7.10 & 7.11 — OPTIONAL UPGRADES
*Only to be explored if validation proves they are complementary.*
- [ ] LightGBM Top-N Neural Reranker (Only on difficult top candidates).
- [ ] XGBoost / CatBoost Ensemble (Average probabilities).
