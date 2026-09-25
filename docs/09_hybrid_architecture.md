# Hybrid Lexical + Dense ANN + LightGBM Architecture

The core philosophy of this architecture is to treat the entity resolution problem as a **three-stage system**: 
`high-recall multi-retrieval → evidence-rich pair scoring → zero/one/many decision`

This separation correctly addresses the fact that **candidate recall acts as the ceiling on final recall**. Retrieval failure and matching failure are fundamentally different problems.

## 1. Core Mental Model: S1 as a Query
Instead of computing an $O(N \times M)$ Cartesian product, we construct a retrieval corpus from **Source 2 $\cup$ Source 3**.
Every **Source 1** record acts as an independent query into this corpus.
```text
CORPUS (S2 + S3)
       ▲
       │ retrieval
       │
  S1_i query
```

## 2. Three-Stage Pipeline

### Stage A: Normalization
Before any indexing or matching occurs, deterministic normalization is applied to both S1 and S2/S3.
- Both the **raw** and **normalized** versions are retained (as the original spelling can contain unique matching evidence).
- Normalizations include: Unicode, lowercase, whitespace, punctuation, ampersand, legal suffixes, and abbreviations.

### Stage B: High-Recall Multi-Retrieval (Candidate Generation)
Because lexical similarity and dense (semantic) similarity fail in complementary ways, we use a hybrid retrieval approach. The goal here is **identity-sensitive recall + semantic/noisy recall**.

1. **Lexical Retrieval (BM25)**:
   - Indexes: Normalized text, Character n-grams.
   - Purpose: Extremely strong at matching exact/rare business terms and rare combinations. 
   - Operation: Query BM25 index with S1_i $\rightarrow$ Top K candidates.
2. **Dense Retrieval (FAISS/HNSW)**:
   - Indexes: Dense embeddings representing the *full entity profile* (e.g. `name: {name} address: {address} country: {country}`).
   - Purpose: Recovers entities suffering from vocabulary mismatch, aggressive typos, or semantic variations.
   - Operation: Query FAISS index with S1_i embedding $\rightarrow$ Top K candidates.
3. **Candidate Union**:
   - The results of Exact Matching, BM25, and Dense ANN are **unioned** and **deduplicated**.
   - These candidates are serialized to `candidate_pairs.tsv`.
   - *Note: Simple union guarantees maximum recall. Reciprocal Rank Fusion (RRF) can be computed as a feature for the next stage rather than used as a strict filter here.*

### Stage C: Evidence-Rich Pair Scoring (LightGBM)
For every candidate retrieved in Stage B, a pairwise feature vector is generated. LightGBM is chosen because business identity depends on complex interactions (e.g., high name similarity means something different if address similarity is extremely low vs. extremely high).

**Feature Families**:
- **Name Features**: Exact match, Jaro-Winkler, Levenshtein, token Jaccard, char n-gram cosine.
- **Address Features**: Token Jaccard, char similarity, numeric overlap, postal-code agreement.
- **Country**: Country equality (open-set).
- **Retrieval Features**: BM25 score/rank, Dense cosine/rank, RRF score, binary flags (`retrieved_by_bm25`, `retrieved_by_dense`).
- **Structural Features**: Name/address length ratios, shared digit/token counts.

### Stage D: Decision Policy
Because a Source 1 entity can map to zero, one, or multiple entities in S2/S3, the raw match score from LightGBM must be calibrated.
- A threshold or margin policy is applied to the LightGBM probability distribution.
- If no candidates exceed the threshold, it correctly outputs an empty list (Singleton).
- If multiple candidates exceed the threshold, it outputs all of them (Multi-match).

## 3. Data Flow Diagram
```text
┌─────────────────┐
│ S2 + S3 corpus  │
└────────┬────────┘
         │
 ┌───────┼───────┐
 │       │       │
 ▼       ▼       ▼
normalized  dense   metadata
  text    embeddings
 │       │
 ▼       ▼
BM25 index FAISS/HNSW
 │       │
 └───┬───┘
     │ ◄───────── S1 Query
     ▼
 BM25 top-K & ANN top-K
     │
     ▼
   UNION & Deduplicate
     │
     ▼
candidate_pairs.tsv
     │
     ▼
Pair Feature Engine
     │
     ▼
  LightGBM
     │
     ▼
calibrated score
     │
     ▼
 decision policy
     │
     ▼
   {Matches}
```

## 4. Risks & Mitigations
1. **Dense embeddings may introduce semantic false positives (e.g. ABC Pharma vs ABC Medical).** 
   *Mitigation*: Dense retrieval is only used for candidate generation. LightGBM will filter these using lexical features.
2. **Candidate Explosion (too many pairs for LightGBM).** 
   *Mitigation*: $K$ will be rigorously tuned based on marginal recall gains.
3. **Training Leakage.** 
   *Mitigation*: Validation splits will strictly separate S1 entities. We will *not* randomly split generated pairs.
4. **Lack of Hard Negatives.** 
   *Mitigation*: We will use retrieval-generated false positives (BM25 hard negatives, dense hard negatives) as training examples for LightGBM.
