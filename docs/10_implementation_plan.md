# Comprehensive Implementation Plan & Checklist

The project is broken down into 11 strictly gated phases. We will not proceed to a subsequent phase until the current phase's "Pass/Fail Gate" is met.

## [ ] Phase 1: Dataset Understanding
*Objective: Understand the empirical structure of S1, S2, S3 and the actual difficulty of matching them without assumptions.*
- [ ] 1.1 Load metadata via PyArrow/Polars (row count, dtype, null %, unique %, etc.)
- [ ] 1.2 Ground-truth cardinality analysis (singleton rate, one-match, multi-match)
- [ ] 1.3 Noise analysis (empirical examples of abbreviations, typos, missing data)
- [ ] 1.4 Collision analysis (most common normalized names/addresses)
- [ ] 1.5 Generate `dataset_profile.json` and `dataset_profile.html`
**Gate A (Dataset)**: Do we empirically understand the entity volume, singleton rate, and skew?

## [ ] Phase 2: Data Foundation & Normalization (Phase 2 & 3)
*Objective: Establish a deterministic data pipeline from raw TSVs to canonical Parquet files.*
- [ ] Create `src/preprocessing/` (`normalize_name.py`, `normalize_address.py`)
- [ ] Implement deterministic normalization (Unicode, lowercase, punctuation, whitespace, suffixes)
- [ ] Write unit tests for all normalization functions
- [ ] Convert normalized outputs to `processed/train_s1.parquet`, etc.
**Gate**: Are all TSVs transformed into clean, queryable Parquet files with both raw and normalized fields?

## [ ] Phase 3: Exact / Cheap Retrieval (Phase 4)
*Objective: Build dictionary/index lookups for computationally free candidate generation.*
- [ ] Exact normalized name lookup
- [ ] Exact normalized address lookup
- [ ] Exact rare token / numeric token lookup
**Gate**: Can we add high-confidence candidates for a fraction of the queries instantly?

## [ ] Phase 4: Lexical Retrieval (BM25 / n-gram) (Phase 5)
*Objective: High-recall string and token-level retrieval.*
- [ ] Build offline BM25 index over `S2 $\cup$ S3` corpus
- [ ] Evaluate separate indexes (name only, name+address, name+address+country)
- [ ] Test character n-gram indexes (3-gram, 4-gram)
- [ ] Produce `lexical_candidates.parquet`
**Gate**: Measure `Recall@K` - is the lexical recall sufficient?

## [ ] Phase 5: Dense ANN Retrieval (Phase 6)
*Objective: Semantic/noisy recall using embeddings.*
- [ ] Encode S2/S3 corpus using a frozen, <8B parameter pretrained encoder
- [ ] Build FAISS index (start with Exact/HNSW)
- [ ] Query index with S1 embeddings
**Gate**: Measure Dense `Recall@K`. Does it find true matches that BM25 missed?

## [ ] Phase 6: Hybrid Candidate Generation (Phase 7 & 8)
*Objective: Merge retrieval paths into a high-recall, bounded-volume candidate set.*
- [ ] Union Exact, Lexical, and Dense candidates (deduplicated)
- [ ] Retain retrieval metadata (`bm25_rank`, `dense_score`, etc.)
- [ ] Intersect with ground truth to calculate absolute Blocking Recall
- [ ] Export `artifacts/blocking_failures.parquet` for analysis
- [ ] Finalize `candidate_pairs.tsv`
**Gate B & C (Retrieval & Efficiency)**: Have we maximized candidate recall while keeping total candidate volume computationally manageable?

## [ ] Phase 7: Pair Feature Engine (Phase 9)
*Objective: Compute granular features for every candidate pair.*
- [ ] Generate Name features (Jaro-Winkler, Jaccard, overlaps)
- [ ] Generate Address features (numeric overlap, postal-code match)
- [ ] Incorporate Retrieval features (BM25/Dense ranks and scores)
- [ ] Vectorize operations (avoid Python `for` loops over millions of pairs)
**Gate**: Are feature matrices correctly computed without out-of-memory errors?

## [ ] Phase 8: LightGBM Matcher (Phase 10)
*Objective: Train a pairwise classifier on the candidate set.*
- [ ] Implement strict S1-entity-based training/validation splits (no leakage)
- [ ] Sample hard negatives (retrieved but incorrect candidates)
- [ ] Train LightGBM classifier to output `P(match)`
**Gate D (Matcher)**: Can the model cleanly separate true candidates from retrieval-generated false positives?

## [ ] Phase 9: Calibration & Zero/One/Many Decision (Phase 11 & 12)
*Objective: Convert raw pair probabilities into final entity-set decisions.*
- [ ] Calibrate LightGBM probabilities (Platt scaling / Isotonic regression)
- [ ] Implement dynamic thresholding policy
- [ ] Correctly handle empty match sets (Singletons)
**Gate E (Decision)**: Does the system achieve high macro $F_{0.5}$ without falsely merging singletons?

## [ ] Phase 10: Scale & Optimize (Phase 14-20)
*Objective: Ensure the pipeline scales to millions of test rows.*
- [ ] Implement batching for S1 queries (retrieval + feature + inference)
- [ ] Checkpoint system state (save indexes, models, and intermediate predictions)
- [ ] Add observability logging (rows/sec, RAM peak, retrieval failure rate)
**Gate**: Does the pipeline run end-to-end within memory and time constraints on Kaggle?

## [ ] Phase 11: Final Validation & Submission (Phase 21)
*Objective: Execute on the unseen Test data and package.*
- [ ] Freeze all preprocessing, indexes, models, and thresholds
- [ ] Run blind on `test_source1.tsv` mapping to `test_source2.tsv`/`test_source3.tsv`
- [ ] Validate outputs against competition rules (no S1 duplicates, IDs exist, etc.)
- [ ] Zip `matching_results.tsv`, `candidate_pairs.tsv`, code, and documentation
**Final Gate**: Successful run of `validate_submission.py`.
