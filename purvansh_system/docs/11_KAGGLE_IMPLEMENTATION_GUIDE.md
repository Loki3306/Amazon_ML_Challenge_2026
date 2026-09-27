# Kaggle Implementation Guide: End-to-End Execution Plan

---

## 1. Kaggle Environment Specifications

| Resource | Limit |
| :--- | :--- |
| RAM | 30 GB (CPU) / 13 GB (GPU) |
| GPU | 2x NVIDIA T4 (16 GB VRAM each) |
| Session Duration | 9 hours max |
| Disk | 73 GB (persistent via datasets) |
| Internet | Disabled during submission runs |
| Python | 3.10+ |

---

## 2. Recommended Notebook Structure

Organize your solution as **3 separate Kaggle notebooks** (chained via output datasets) to avoid the 9-hour limit:

### Notebook 1: Blocking & Candidate Generation (~2.5 hours)
```
Notebook 1: blocking_pipeline
├── Cell 1: Import libraries, load datasets
├── Cell 2: Preprocessing (normalize names, strip legal suffixes, extract postal codes)
├── Cell 3: Country partitioning
├── Cell 4: Pass 1 — Token inverted index blocking
├── Cell 5: Pass 2 — Character n-gram TF-IDF sparse retrieval
├── Cell 6: Pass 3 — Phonetic + postal code blocking
├── Cell 7: Pass 4 — Address-first fallback
├── Cell 8: Union & deduplication
├── Cell 9: Save candidate_pairs.tsv as output dataset
└── Cell 10: Blocking quality metrics (on training data)
```

### Notebook 2: Feature Engineering & Model Training (~3.5 hours)
```
Notebook 2: feature_engineering_and_training
├── Cell 1: Load candidate pairs from Notebook 1 output
├── Cell 2: Load source datasets
├── Cell 3: Compute name similarity features (rapidfuzz batch)
├── Cell 4: Compute address similarity features
├── Cell 5: Compute cross-field & structural features
├── Cell 6: (Optional) Compute embedding features with all-MiniLM-L6-v2 on GPU
├── Cell 7: Construct training labels from ground truth
├── Cell 8: Train/validation split (entity-level, stratified by country)
├── Cell 9: LightGBM training with early stopping
├── Cell 10: Threshold optimization for F_0.5
├── Cell 11: Save trained model, features, and threshold as output dataset
└── Cell 12: Validation metrics report
```

### Notebook 3: Inference & Submission (~2 hours)
```
Notebook 3: inference_and_submission
├── Cell 1: Load model, features pipeline, and threshold from Notebook 2
├── Cell 2: Load test candidate pairs from Notebook 1
├── Cell 3: Compute features for test candidate pairs
├── Cell 4: LightGBM inference (predict probabilities)
├── Cell 5: Self-training for France (2-3 iterations)
├── Cell 6: Post-processing (cardinality, singletons, match cap)
├── Cell 7: Generate matching_results.tsv
├── Cell 8: Run validation script
├── Cell 9: Save final submission files
└── Cell 10: Summary statistics
```

---

## 3. Library Requirements

```python
# Core
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

# String Matching (CRITICAL: rapidfuzz is 10-100x faster than fuzzywuzzy)
# pip install rapidfuzz
from rapidfuzz import fuzz, distance
from rapidfuzz.process import cdist

# Phonetic Encoding
# pip install metaphone
from metaphone import doublemetaphone

# ML Model
import lightgbm as lgb

# Optional: Sentence Embeddings
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

# Hyperparameter Tuning
import optuna
```

### Kaggle requirements.txt
```
rapidfuzz>=3.0
metaphone>=0.6
lightgbm>=4.0
sentence-transformers>=2.2  # optional
optuna>=3.0  # optional
```

---

## 4. Memory Management Strategy

### 4.1 Country-Sequential Processing
Process one country at a time and free memory between countries:
```python
import gc

for country in ['France', 'US', 'India']:  # France first (smallest)
    s1_c = s1_df[s1_df['country'] == country].copy()
    s2_c = s2_df[s2_df['country'] == country].copy()
    s3_c = s3_df[s3_df['country'] == country].copy()
    
    candidates = run_blocking(s1_c, s2_c, s3_c)
    save_candidates(candidates, country)
    
    del s1_c, s2_c, s3_c, candidates
    gc.collect()
```

### 4.2 Data Type Optimization
```python
# Use category dtype for country column
df['country'] = df['country'].astype('category')

# Use string[pyarrow] for text columns (50% memory savings)
df['business_name'] = df['business_name'].astype('string[pyarrow]')
df['business_address'] = df['business_address'].astype('string[pyarrow]')

# Use float32 for features instead of float64
features = features.astype(np.float32)
```

### 4.3 Sparse Matrix Management
```python
# For TF-IDF blocking, use float32 sparse matrices
vectorizer = TfidfVectorizer(
    analyzer='char_wb', ngram_range=(3, 4),
    max_features=500_000, dtype=np.float32,
    sublinear_tf=True
)

# Process in batches to limit memory
BATCH = 50_000
for i in range(0, n_s1, BATCH):
    batch_scores = s1_tfidf[i:i+BATCH] @ s2s3_tfidf.T
    # Extract top-K candidates from batch_scores
    del batch_scores
    gc.collect()
```

### 4.4 Expected Memory Profile
| Stage | Peak RAM | Duration | Notes |
| :--- | :--- | :--- | :--- |
| Data loading (all sources) | ~8 GB | 3 min | pyarrow strings |
| Blocking Pass 2 (TF-IDF India) | ~14 GB | 15 min | Largest country |
| Feature computation (batch) | ~12 GB | 45 min | Process 100K pairs at a time |
| LightGBM training | ~10 GB | 15 min | ~17M training rows |
| LightGBM inference | ~8 GB | 10 min | ~25M test pairs |
| Self-training (France) | ~6 GB | 20 min | France only (~4M pairs) |

---

## 5. Execution Timeline (Single 9-Hour Session)

If running everything in a single notebook instead of 3:

| Step | Start | End | Duration | Cumulative |
| :--- | :--- | :--- | :--- | :--- |
| Load & preprocess data | 0:00 | 0:15 | 15 min | 0:15 |
| Blocking: France | 0:15 | 0:35 | 20 min | 0:35 |
| Blocking: US | 0:35 | 1:05 | 30 min | 1:05 |
| Blocking: India | 1:05 | 1:40 | 35 min | 1:40 |
| Save candidate_pairs.tsv | 1:40 | 1:45 | 5 min | 1:45 |
| Feature eng (train pairs) | 1:45 | 2:45 | 60 min | 2:45 |
| LightGBM training + tuning | 2:45 | 4:15 | 90 min | 4:15 |
| Threshold optimization | 4:15 | 4:30 | 15 min | 4:30 |
| Feature eng (test pairs) | 4:30 | 5:30 | 60 min | 5:30 |
| LightGBM inference | 5:30 | 5:45 | 15 min | 5:45 |
| Self-training France (3 iter) | 5:45 | 6:30 | 45 min | 6:30 |
| Post-processing | 6:30 | 6:45 | 15 min | 6:45 |
| Validation & save output | 6:45 | 7:00 | 15 min | 7:00 |
| **Total** | | | | **7:00** |
| **Buffer remaining** | | | | **2:00** |

---

## 6. Debugging & Monitoring Checklist

### Blocking Phase
- [ ] Blocking recall on training data >= 98%
- [ ] Average candidates per S1 entity: 15-30
- [ ] candidate_pairs.tsv has exactly 1,732,544 rows (one per test S1)
- [ ] All S1 entity IDs from test_source1.tsv are present

### Feature Engineering Phase
- [ ] No NaN values in feature matrix (fill missing address features with 0)
- [ ] Feature distributions look reasonable (plot histograms for top 5 features)
- [ ] rapidfuzz producing sane similarity scores (spot-check 10 pairs manually)

### Training Phase
- [ ] Validation AUC-ROC >= 0.98
- [ ] Validation Macro F_0.5 >= 0.85
- [ ] No significant drop when evaluating cross-country (US-train/India-val)
- [ ] Learning curve shows no overfitting (val metric stable after 1000 rounds)

### Inference Phase
- [ ] matching_results.tsv has exactly 1,732,544 rows
- [ ] No S1- IDs in matched_entity_ids column
- [ ] No duplicate IDs within any row
- [ ] validate_submission.py prints PASS

### France Specific
- [ ] Self-training increases French pseudo-labels across iterations
- [ ] Macro F_0.5 on US+India validation does NOT decrease after self-training
- [ ] French predictions have reasonable match distribution (not all singletons, not all 10+ matches)

---

## 7. Speed Optimization Tips

1. **Use `rapidfuzz.process.cdist`** for batch string similarity computation instead of Python loops.
2. **Precompute normalized names/addresses** once and reuse across all passes and features.
3. **Use `scipy.sparse` consistently** — never convert sparse matrices to dense.
4. **LightGBM `predict` with `num_threads=-1`** for parallel inference.
5. **Cache TF-IDF vectorizers** — fit once on S2+S3, transform S1 and test S1 separately.
6. **Use `polars` instead of `pandas`** for data loading and joins (2-5x faster on large TSVs).

---

## 8. Iterative Improvement Workflow

Follow this loop to systematically improve your score:

```
Iteration 1: Baseline
  → Single-pass TF-IDF blocking + 10 features + LightGBM
  → Target: F_0.5 ~ 0.70

Iteration 2: Multi-pass Blocking
  → Add passes 1, 3, 4 to blocking
  → Target: F_0.5 ~ 0.78 (higher recall ceiling)

Iteration 3: Full Feature Engineering
  → Expand to 50+ features
  → Target: F_0.5 ~ 0.83

Iteration 4: Threshold + Post-Processing
  → Optimize threshold, add cardinality enforcement, singleton protection
  → Target: F_0.5 ~ 0.86

Iteration 5: Self-Training for France
  → Bootstrap pseudo-labels for France
  → Target: F_0.5 ~ 0.87

Iteration 6: Ensemble + Final Tuning
  → Blend LightGBM + CatBoost, per-country thresholds
  → Target: F_0.5 ~ 0.88+
```

---

## 9. Data Upload to Kaggle

The dataset is ~2GB total (unzipped). Upload as a Kaggle Dataset:

1. Go to kaggle.com/datasets → New Dataset
2. Upload the following files:
   - `dataset/train/train_source1.tsv`
   - `dataset/train/train_source2.tsv`
   - `dataset/train/train_source3.tsv`
   - `dataset/train/train_ground_truth.tsv`
   - `dataset/test/test_source1.tsv`
   - `dataset/test/test_source2.tsv`
   - `dataset/test/test_source3.tsv`
   - `utils/validate_submission.py`
3. Name it `ml-challenge-2026-entity-resolution`
4. In notebooks, access via `/kaggle/input/ml-challenge-2026-entity-resolution/`
