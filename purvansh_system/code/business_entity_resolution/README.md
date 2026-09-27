# Business Entity Resolution Package

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![RapidFuzz](https://img.shields.io/badge/SIMD-RapidFuzz-orange.svg)](https://github.com/rapidfuzz/RapidFuzz)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Standalone production package for large-scale multi-source business entity resolution.

> **Notice:** This package uses a shard-streaming architecture that guarantees a hard ceiling of ~8 GB peak RAM, regardless of candidate pool size.

---

## Package Architecture

- **`src/preprocessing.py`**: Stream-loading TSV readers (`load_country_slice`, `load_filtered_candidates`), diacritics stripping via Unicode `NFKD`, legal suffix removal, and $O(1)$ vectorized lookup builders.
- **`src/blocking.py`**: Shard-streaming TF-IDF blocking — fits vocabulary on a 500K sample, then streams candidates in 300K-row shards. Never holds the full candidate matrix in memory. Running top-k merged across shards.
- **`src/features.py`**: SIMD-accelerated pairwise feature extractor (Jaro-Winkler, Token Sort, Token Set, Postal exact/prefix match, Length ratio, TF-IDF cosine score).
- **`src/model.py`**: LightGBM binary classifier with GroupKFold cross-validation on $S_1$ entity IDs and fine-grained Macro $F_{0.5}$ decision threshold grid search.
- **`src/postprocessing.py`**: Greedy Maximum Weighted Bipartite Matching for 1-to-1 candidate cardinality resolution, singleton protection, and TSV submission generation.
- **`run_pipeline.py`**: Command-line orchestrator with shard-streaming blocking, filtered candidate reload, checkpointing, and automated submission validation.

---

## Installation & Dependencies

```bash
pip install -r requirements.txt
```

---

## CLI Options

```bash
python run_pipeline.py [OPTIONS]
```

| Option | Default | Description |
| :--- | :--- | :--- |
| `--data-dir` | Auto-detect | Root path containing `train/`, `test/`, and `utils/` |
| `--output-dir` | `./output` | Path to save `matching_results.tsv` and `candidate_pairs.tsv` |
| `--sample-train` | `100000` | Number of $S_1$ reference entities to sample for training |
| `--top-k` | `12` | Maximum candidates to retrieve per reference entity |
| `--batch-size` | `1000` | $S_1$ sub-batch size for sparse matrix multiplication |
| `--shard-size` | `300000` | Candidate shard size for streaming TF-IDF (rows per shard) |
| `--max-features` | `50000` | TF-IDF vocabulary size cap |
| `--max-neg` | `4` | Max hard negatives per $S_1$ entity during training |
| `--checkpoint-dir` | `./checkpoints` | Path to save/load intermediate execution checkpoints |

---

## License

Distributed under the [MIT License](../../LICENSE).
