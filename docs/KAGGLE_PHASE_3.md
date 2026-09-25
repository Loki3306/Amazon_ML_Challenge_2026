# Phase 3: Exact / Cheap Blocking (Kaggle Execution)

## Purpose
This script generates highly precise candidate pairs by performing exact-match joins on normalized names and normalized addresses between S1 (queries) and S2+S3 (corpus).

Because exact hashing is extremely fast and Polars is highly optimized for large joins, we don't need to manually build indexes (like dictionaries) to disk—Polars handles the hash join in memory and streams the pairs to Parquet. 

## Kaggle Environment
- **Environment**: Kaggle Notebook (Python)
- **Compute**: CPU
- **RAM**: Standard 30GB is more than enough.
- **Required Data**: The output of Phase 2 (`train_source1.parquet`, etc.) must be available. 
  - *Best Practice:* Add your Phase 2 Kaggle output as an attached Dataset to your Phase 3 notebook.

## Execution Command
Assuming your Phase 2 output is mounted at `/kaggle/input/phase-2-canonical-data/`:

```bash
# Generate train candidates
!python scripts/03_exact_blocking.py \
    --data-dir /kaggle/input/phase-2-canonical-data/train \
    --split train \
    --output-dir /kaggle/working/data/candidates \
    --artifacts-dir /kaggle/working/artifacts

# Generate test candidates (if needed now)
!python scripts/03_exact_blocking.py \
    --data-dir /kaggle/input/phase-2-canonical-data/test \
    --split test \
    --output-dir /kaggle/working/data/candidates \
    --artifacts-dir /kaggle/working/artifacts
```

*(Note: Adjust the `--data-dir` argument to point to the directory containing the `.parquet` files for that split.)*

## Output
The script outputs:
1. `data/candidates/{split}_exact_candidates.parquet`
   - Schema: `query_id`, `candidate_id`, `candidate_source`, `match_name` (boolean), `match_address` (boolean).
2. `artifacts/exact_blocking_report_{split}.json`
   - Contains candidate reduction metrics (e.g., how many S1 queries got at least one exact match, average candidates per query).

## Handling Empty Strings
The script actively prevents Cartesian explosions by filtering out `""` (empty strings) before performing joins. Matching on empty strings would incorrectly link millions of entities that are simply missing addresses.
