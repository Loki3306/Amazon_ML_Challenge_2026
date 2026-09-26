# Context Handoff: Amazon ML Challenge 2026

## 1. Problem Statement
We are solving a massive-scale **Entity Resolution (Record Linkage)** problem. We are given 3 tables: `S1` (Queries), `S2` (Corpus), and `S3` (Corpus). 
The goal is to match every entity in `S1` to all its identical counterparts in `S2` and `S3`.
The dataset is highly noisy (typos, abbreviations, missing fields, shuffled words).
The evaluation metric is **Macro F0.5 Score**, which places a heavy penalty on false positives. Precision is absolutely critical.

## 2. Scale & The "Anti-Kaggle" Structure
**The Scale:**
- Train Queries (S1): 2.2 Million
- Test Queries (S1): ~130,000
- Total Corpus (S2 + S3): 10.3 Million
- A naive cross-join ($N \times M$) is over 10 Trillion comparisons, which is mathematically impossible.

**The Environment:**
We are running strictly on Kaggle Notebooks (30GB RAM, 20GB Disk, 2x T4 GPUs, 12-hour limit).
Because of the massive scale, standard Pandas/Scikit-learn workflows OOM instantly. 

**Our "Anti-Kaggle" Architecture constraints:**
- Everything must be chunked or streamed. There are no full-table merges.
- We exclusively use **Polars** (with `LazyFrame` scanning) instead of Pandas.
- Massive arrays (Dense Embeddings) are backed by `numpy.memmap` (FP16) on disk.
- Our FAISS index strictly uses `IndexIVFScalarQuantizer (SQ8)` because raw FP32 indices take 16GB RAM and crash Kaggle. SQ8 compresses it to 4GB.

## 3. Current Architecture (Candidate Generation + Re-Ranking)
Since we cannot do $O(N^2)$, we use a two-stage industrial search pipeline:

### Stage 1: Candidate Generation (High Recall)
1.  **Exact Blocking (`03_exact_blocking.py`)**: Hash-joins on normalized names and addresses. Extremely fast ($O(N)$), yields ~77% recall out of the box.
2.  **Dense Semantic Retrieval (`05c` & `05d`)**: 
    - We encode the corpus text (`name | address | country`) using a Bi-Encoder (`all-MiniLM-L6-v2`).
    - We store the 10.3M vectors as an FP16 memmap on disk.
    - We build an SQ8 compressed FAISS index on CPU and query the Top-50 nearest neighbors for every S1 query.
    - This rescues the remaining fuzzy/semantic matches that exact blocking misses.

### Stage 2: Feature Engineering & Re-Ranking (High Precision)
1.  **Feature Generation (`06a_features_lightgbm.py`)**: 
    - Takes the combined (Query, Candidate) pairs.
    - Computes string distances on the fly: Jaro-Winkler, Levenshtein, Token Jaccard.
2.  **LightGBM Classifier**: 
    - Trained on the generated features using `train_ground_truth.tsv`.
    - Employs Hard Negative Mining (using retriever mistakes as label=0).
3.  **Inference (`07_inference.py`)**: 
    - Streams test candidates in massive chunks of 2,000,000 to prevent RAM OOMs.
    - Computes features, predicts with LightGBM, filters scores above threshold, and streams output directly to `submission.tsv`.

## 4. Current Baseline Stats
- **Exact Match Recall:** 77.15%
- **Dense Retrieval (Top-50) Recall:** ~93%+ Expected.
- **LightGBM F-Score:** 
  - We recently trained a model achieving **0.8040 Macro F0.5**.
  - **Validation AUC:** 0.9988
  - **Best Threshold:** 0.80 - 0.95 (Classes are highly imbalanced, precision requires high thresholds).
- **Top Features (Gain):** `addr_token_jaccard`, `addr_jaro_winkler`, `name_jw_x_addr_jw`, `addr_levenshtein`.

## 5. The Goal (Recall & Precision Progression)
We need to push the F0.5 score from 0.80 to 0.85+. We have a solid engineering foundation, so the focus is entirely on ML progression.

**Known Gaps / Next Steps:**
1.  **Sparse Lexical Retrieval (BM25)**: Dense retrieval misses exact serial numbers or rare acronyms. We need a high-recall lexical booster, but standard BM25 OOMs or takes hours. We need a GPU-accelerated (CuPy) sparse index or a fast FlashText/TF-IDF chunked implementation.
2.  **Cross-Encoder Re-Ranking**: `all-MiniLM-L6-v2` compresses everything into a single vector. A Cross-Encoder (`ms-marco`) would massively boost precision, but we can only afford to run it as a LightGBM feature on the Top 3-5 candidates per query due to execution speed constraints.
3.  **Advanced Features**: Injecting Token Sort Ratio (RapidFuzz) and exact TF-IDF Cosine Similarity directly into LightGBM as numerical features.
