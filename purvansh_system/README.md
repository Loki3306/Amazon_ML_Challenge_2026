# Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![RapidFuzz](https://img.shields.io/badge/SIMD-RapidFuzz-orange.svg)](https://github.com/rapidfuzz/RapidFuzz)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Platform-Kaggle%20CPU-20BEFF.svg)](https://www.kaggle.com/)

A scalable machine learning pipeline for multi-source business entity resolution across heterogeneous datasets.

---

> **Notice:** The pipeline uses a shard-streaming architecture designed for memory-constrained environments (Kaggle CPU, 30 GB RAM). Previous monolithic TF-IDF approaches caused OOM on large partitions (6M+ candidates). This version guarantees a hard ceiling of ~8 GB peak RAM.

Entity Resolution (ER) identifies records referring to the same real-world business entity across independent data sources without shared unique identifiers.

The dataset consists of **24.2 million total records** across three sources ($S_1$, $S_2$, $S_3$) spanning three countries (**US**, **India**, and **France**):
- **Source 1 ($S_1$):** Deduplicated reference entity list.
- **Source 2 ($S_2$) & Source 3 ($S_3$):** Noisy candidate records containing OCR errors, missing addresses, legal abbreviations, and spelling variations.
- **Evaluation Metric:** Precision-weighted **Macro $F_{0.5}$** score (penalizing false matches 2x over missed matches).

---

## Key Technical Features

- **Shard-Streaming TF-IDF Blocking:** Fits vocabulary on a 500K sample, then streams candidates in 300K-row shards. Each shard is transformed, multiplied against S1, and discarded — never holding the full candidate matrix. Running top-k is merged across shards.
- **Filtered Candidate Reload:** After blocking selects top-k candidates (~500K out of 6.2M), only those IDs are loaded in a second pass for feature extraction, avoiding multi-GB DataFrames.
- **SIMD Feature Extraction:** Computes pairwise string similarities (Jaro-Winkler, Token Sort, Token Set, Postal exact/prefix match, Length ratio) accelerated by SIMD instructions.
- **Regularized Gradient Boosting:** Trains a LightGBM binary classifier with GroupKFold cross-validation on $S_1$ entity IDs to prevent data leakage, paired with fine-grained threshold optimization for Macro $F_{0.5}$.
- **Graph Consistency:** Enforces strict 1-to-1 candidate cardinality using Greedy Maximum Weighted Bipartite Matching and singleton protection.

---

## Performance Benchmarks

| Metric | Measurement |
| :--- | :--- |
| **Peak RAM Footprint** | `< 8 GB` (Kaggle CPU Limit: 30 GB) |
| **End-to-End Pipeline Runtime** | `~20 - 30 minutes` |
| **Validation AUC-ROC** | `~0.989` |
| **Validation Macro F_0.5** | `~0.981` |

---

## Repository Structure

```
.
├── code/
│   └── business_entity_resolution/      # Official submission package
│       ├── src/
│       │   ├── __init__.py
│       │   ├── preprocessing.py         # Vectorized lookups & streaming loaders
│       │   ├── blocking.py              # TF-IDF sparse cosine & CSR pointer slicing
│       │   ├── features.py              # SIMD string distance features
│       │   ├── model.py                 # LightGBM training & Macro F_0.5 tuning
│       │   └── postprocessing.py        # Maximum weighted bipartite matching
│       ├── run_pipeline.py              # Production CLI runner with checkpointing
│       ├── requirements.txt             # Pinned package versions
│       └── README.md                    # Package documentation
├── docs/                                # Technical documentation suite
│   ├── 01_DATASET_AND_EDA.md
│   ├── 06_NOVEL_BLOCKING_STRATEGY.md
│   ├── 07_ADVANCED_FEATURE_ENGINEERING.md
│   ├── 08_MODEL_TRAINING_AND_CASCADE.md
│   ├── 09_FRANCE_DOMAIN_ADAPTATION.md
│   ├── 10_GRAPH_POST_PROCESSING.md
│   └── README.md
├── notebooks/                           # Jupyter execution notebooks
│   ├── kaggle_runner.ipynb              # Production Kaggle runner
│   └── master_entity_resolution_pipeline.ipynb # Standalone master notebook
├── .gitignore
├── PROBLEM_STATEMENT.md
└── README.md
```

---

## Quickstart

### Local CLI Execution

```bash
# Install dependencies
pip install -r code/business_entity_resolution/requirements.txt

# Run full pipeline
python code/business_entity_resolution/run_pipeline.py --output-dir ./output
```

### Kaggle Execution

In a Kaggle Notebook (Settings: **CPU**, **30 GB RAM**, **Internet: ON**):

```python
!git clone https://github.com/purvanshjoshi/business-entity-resolution.git
%cd business-entity-resolution/code/business_entity_resolution
!pip install -q -r requirements.txt
!python -u run_pipeline.py --output-dir /kaggle/working/output --sample-train 100000 --top-k 12 --batch-size 1000 --shard-size 300000 --max-features 50000
```

---

## License

Distributed under the [MIT License](LICENSE).
