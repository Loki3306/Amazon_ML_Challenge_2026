# Phase 4: Lexical Blocking (Kaggle Execution)

## Purpose
Phase 3 captured the "easy" exact matches. Phase 4 captures fuzzy lexical matches (e.g. typos, slightly different wordings) by generating TF-IDF vectors for all business names and retrieving the Top-K most similar S2/S3 candidates for every S1 query.

To ensure this runs cleanly on Kaggle CPU for 10.3M corpus records, it uses a high-performance `scikit-learn` memory-efficient `HashingVectorizer` combined with batch-processed SciPy sparse matrix dot products.

## Kaggle Environment
- **Environment**: Kaggle Notebook (Python)
- **Compute**: CPU
- **RAM**: Standard 30GB is sufficient because the script leverages sparse matrices and garbage-collects raw strings.
- **Required Packages**:
  ```bash
  pip install polars pyarrow scikit-learn scipy numpy
  ```
- **Required Data**: The output of Phase 2 (`train_source1.parquet`, etc.).

## Execution Command
Assuming Phase 2 data is mounted at `/kaggle/input/phase-2-canonical-data/`:

```bash
# Generate train lexical candidates (Top 5 matches per query)
!python scripts/04_lexical_blocking.py \
    --data-dir /kaggle/input/phase-2-canonical-data/train \
    --split train \
    --output-dir /kaggle/working/data/candidates \
    --artifacts-dir /kaggle/working/artifacts \
    --top-k 5

# Generate test lexical candidates
!python scripts/04_lexical_blocking.py \
    --data-dir /kaggle/input/phase-2-canonical-data/test \
    --split test \
    --output-dir /kaggle/working/data/candidates \
    --artifacts-dir /kaggle/working/artifacts \
    --top-k 5
```

## Output
1. `data/candidates/{split}_lexical_candidates.parquet`
   - Schema: `query_id`, `candidate_id`, `candidate_source`, `match_lexical` (boolean = True).
2. `artifacts/lexical_blocking_report_{split}.json`
   - Contains execution metrics.

## Notes
- Phase 3 generated ~13 candidates per query instantly.
- Phase 4 generates exactly K candidates per query.
- In Phase 7 (Hybrid Candidate Generation), we will simply UNION the exact candidate pool and the lexical candidate pool.
