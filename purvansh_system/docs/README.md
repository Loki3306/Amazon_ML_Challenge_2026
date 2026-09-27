# ML Challenge 2026: Business Entity Resolution
## Complete Documentation Suite & Roadmap

---

## 🏗️ Novel Approach Overview

> [See Full Strategy Document →](file:///D:/ML_Challenge/docs/NOVEL_APPROACH_OVERVIEW.md) *(if created)*

Our winning strategy combines **5 key innovations**:
1. **Multi-Resolution Blocking** — 4-pass union candidate generation achieving >98% recall.
2. **50+ Multi-View Features** — Character, token, phonetic, structural, and embedding features.
3. **3-Stage Cascade Classifier** — Rules → LightGBM → Post-processing.
4. **Self-Training for France** — Pseudo-label bootstrapping for the zero-shot country.
5. **Graph-Based Consistency** — Cardinality enforcement and singleton protection.

---

## 📚 Documentation Modules

### Foundation Documents (Problem & Data Understanding)

| # | Document | What It Covers |
| :---: | :--- | :--- |
| — | [PROBLEM_STATEMENT.md](file:///D:/ML_Challenge/docs/PROBLEM_STATEMENT.md) | Official rules, file formats, evaluation metric, submission requirements |
| 01 | [Dataset & EDA](file:///D:/ML_Challenge/docs/01_DATASET_AND_EDA.md) | 24.2M records breakdown, country distributions, France shift, singleton stats, noise examples |
| 02 | [Pipeline Architecture](file:///D:/ML_Challenge/docs/02_PIPELINE_ARCHITECTURE_AND_BLOCKING.md) | High-level 2-stage ER system design, reduction ratio targets |
| 03 | [Feature Engineering Overview](file:///D:/ML_Challenge/docs/03_FEATURE_ENGINEERING_AND_MODELING.md) | Feature taxonomy, GBDT vs. Transformer comparison |
| 04 | [Validation & Metrics](file:///D:/ML_Challenge/docs/04_VALIDATION_METRICS_AND_THRESHOLDING.md) | Macro F_0.5 metric, Python implementation, threshold optimization |
| 05 | [Submission & Compliance](file:///D:/ML_Challenge/docs/05_SUBMISSION_AND_COMPLIANCE_GUIDE.md) | Output formats, validation script, zip packaging, fair-play rules |

### Novel Approach — Deep Technical Guides

| # | Document | What It Covers |
| :---: | :--- | :--- |
| 06 | [**Novel Blocking Strategy**](file:///D:/ML_Challenge/docs/06_NOVEL_BLOCKING_STRATEGY.md) | 4-pass multi-resolution blocking: token inverted index, TF-IDF sparse retrieval, phonetic+postal, address fallback |
| 07 | [**Advanced Feature Engineering**](file:///D:/ML_Challenge/docs/07_ADVANCED_FEATURE_ENGINEERING.md) | 50+ pairwise features: string distances, token metrics, n-grams, phonetic, structural, embeddings |
| 08 | [**Model Training & Cascade**](file:///D:/ML_Challenge/docs/08_MODEL_TRAINING_AND_CASCADE.md) | 3-stage cascade, LightGBM training, hard negatives, Optuna tuning, ensemble strategy |
| 09 | [**France Domain Adaptation**](file:///D:/ML_Challenge/docs/09_FRANCE_DOMAIN_ADAPTATION.md) | Self-training pseudo-labeling, diacritics handling, cross-country validation, safeguards |
| 10 | [**Graph Post-Processing**](file:///D:/ML_Challenge/docs/10_GRAPH_POST_PROCESSING.md) | Cardinality enforcement, singleton protection, transitivity checks, match count cap |
| 11 | [**Kaggle Implementation Guide**](file:///D:/ML_Challenge/docs/11_KAGGLE_IMPLEMENTATION_GUIDE.md) | 3-notebook structure, memory management, 9-hour timeline, debugging checklist |
| 12 | [**Kaggle Dataset Creation Guide**](file:///D:/ML_Challenge/docs/12_KAGGLE_DATASET_CREATION_GUIDE.md) | Step-by-step dataset packaging, uploading, and chained notebooks setup |

---

## 💻 Ready-to-Run Jupyter Notebooks

Located in [`notebooks/`](file:///D:/ML_Challenge/notebooks/):
- 🚀 [**`kaggle_runner.ipynb`**](file:///D:/ML_Challenge/notebooks/kaggle_runner.ipynb) — **Production Runner Notebook** (Clones GitHub repo or runs the modular `.py` pipeline with live progress and automatic validation).
- 🌟 [**`master_entity_resolution_pipeline.ipynb`**](file:///D:/ML_Challenge/notebooks/master_entity_resolution_pipeline.ipynb) — **All-in-One Standalone Master Notebook** (Vectorized zero-iterrows lookups, CSR pointer slicing, 7 decisive features, LightGBM, and bipartite graph post-processing).

---

## 📦 Production Code Package (Submission-Ready)

Located in [`code/business_entity_resolution/`](file:///D:/ML_Challenge/code/business_entity_resolution/):
- `run_pipeline.py`: Production CLI runner with checkpointing and auto-validation.
- `src/preprocessing.py`: C-speed vectorization, diacritics stripping, postal extraction.
- `src/blocking.py`: TF-IDF sparse matrix cosine retrieval with direct CSR pointer indexing.
- `src/features.py`: SIMD-accelerated 7-feature extractor.
- `src/model.py`: LightGBM binary classifier with GroupKFold cross-validation and Macro $F_{0.5}$ threshold tuner.
- `src/postprocessing.py`: Greedy Maximum Weighted Bipartite Matching (1-to-1 cardinality).
- `requirements.txt`: Pinned dependencies.
- `README.md`: Official reproduction guide.



---

## 🗺️ Recommended Reading Order

```
START HERE
    │
    ▼
┌─────────────────────────────────┐
│  PROBLEM_STATEMENT.md           │  Understand the rules
│  01_DATASET_AND_EDA.md          │  Understand the data
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  06_NOVEL_BLOCKING_STRATEGY.md  │  Build the candidate set
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  07_ADVANCED_FEATURE_ENGINEERING│  Compute pairwise features
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  08_MODEL_TRAINING_AND_CASCADE  │  Train the classifier
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  09_FRANCE_DOMAIN_ADAPTATION    │  Handle zero-shot France
│  10_GRAPH_POST_PROCESSING       │  Enforce consistency
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  11_KAGGLE_IMPLEMENTATION_GUIDE │  Execute on Kaggle
│  05_SUBMISSION_AND_COMPLIANCE   │  Package & submit
└─────────────────────────────────┘
```

---

## ⏱️ Implementation Timeline on Kaggle

| Phase | Notebooks | Est. Time | Deliverable |
| :--- | :--- | :--- | :--- |
| **Phase 1:** Blocking | Notebook 1 | ~2.5 hrs | `candidate_pairs.tsv` |
| **Phase 2:** Features & Training | Notebook 2 | ~3.5 hrs | Trained LightGBM model |
| **Phase 3:** Inference & Submission | Notebook 3 | ~2.0 hrs | `matching_results.tsv` |
| **Total** | 3 notebooks | **~8.0 hrs** | Complete submission |

---

## 🎯 Target Performance

| Metric | Target |
| :--- | :---: |
| Blocking Recall | ≥ 98.0% |
| LightGBM AUC-ROC | ≥ 0.985 |
| Validation Macro F_0.5 (US+India) | ≥ 0.88 |
| Test Macro F_0.5 (with France) | ≥ 0.85 |
