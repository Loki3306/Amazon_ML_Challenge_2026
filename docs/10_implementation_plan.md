# Implementation and Experimentation Plan

This document outlines the strict execution phases for implementing the Hybrid Retrieval + LightGBM architecture. 

**Core Rule**: We must not implement the full system (especially LightGBM) until the candidate generation layer (hybrid retrieval) has been empirically validated to achieve sufficient candidate recall on the actual Kaggle dataset.

## Phase 1: Retrieval Audit & Candidate Generation Baseline (CURRENT GOAL)
Before modeling, we must understand the upper-bound of our recall and the size of the search space.

1. **Dataset Profiling & Schema Verification**
   - Load S1, S2, and S3.
   - Inspect row counts, analyze missingness, and determine true match cardinality from `train_ground_truth.tsv`.
2. **Text Normalization**
   - Implement basic deterministic normalization (lowercase, punctuation stripping, whitespace normalization).
3. **Lexical Retrieval (BM25)**
   - Build a BM25 index over the S2+S3 corpus.
4. **Dense Retrieval (Baseline ANN)**
   - Extract a basic baseline dense embedding (e.g., using a small sentence-transformer model that complies with <8B parameter and open-source constraints).
   - Build a FAISS index (Exact or HNSW) for the S2+S3 corpus.
5. **Retrieval Grid Search (The Mandatory Experiment)**
   We will compute the following matrix to determine if hybrid retrieval is actually justified:
   
   | Retriever | K | Candidate Recall | Avg Cand/S1 | P95 Cand/S1 | Runtime | Memory |
   | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
   | Exact | - | | | | | |
   | BM25 | 20 | | | | | |
   | BM25 | 50 | | | | | |
   | BM25 | 100 | | | | | |
   | Dense | 20 | | | | | |
   | Dense | 50 | | | | | |
   | Dense | 100 | | | | | |
   | Hybrid (BM25+Dense) | 20+20 | | | | | |
   | Hybrid (BM25+Dense) | 50+50 | | | | | |
   | Hybrid (BM25+Dense) | 100+100 | | | | | |

6. **Output**: Establish the optimal $K_{bm25}$ and $K_{dense}$ configuration that maximizes Candidate Recall without causing candidate explosion.

## Phase 2: Feature Engineering
Once a solid `candidate_pairs.tsv` is generated using the chosen $K$ parameters:
1. Construct the pairwise feature vectors.
2. Calculate lexical overlaps (Jaro-Winkler, Levenshtein, Jaccard).
3. Inject retrieval metadata (BM25 Rank, Dense Cosine, RRF).
4. Inject structural features (length ratios, numeric overlap).

## Phase 3: Matcher Ablation & LightGBM Training
Using the *fixed* candidate set from Phase 1, we will run controlled ablations to isolate LightGBM's value:
- **M0**: Simple weighted score baseline
- **M1**: Logistic Regression
- **M2**: LightGBM
- **M3**: XGBoost

We will also conduct **Feature Ablations**:
- **F0**: Name/address fuzzy only
- **F1**: + BM25 features
- **F2**: + Dense similarity features
- **F3**: + Retrieval rank features
- **F4**: + Retrieval agreement
- **F5**: + Structural features

*Crucial Training Detail*: We must use the retrieval-generated false positives as hard negatives for training the LightGBM classifier.

## Phase 4: Decision Policy Optimization
1. Analyze the raw LightGBM probability outputs.
2. Implement calibration logic to map scores to zero, one, or multiple matches (singleton handling).
3. Validate against the F0.5 Macro-Average metric.

## Phase 5: Advanced (Optional) Additions
Only if required to squeeze extra F0.5 performance (and only after Phase 1-4 are fully functioning):
- Cross-encoder reranking
- Country-aware index partitioning (if country proves 100% reliable)
- Entity-resolution specific fine-tuning of the embedding model via contrastive learning.
