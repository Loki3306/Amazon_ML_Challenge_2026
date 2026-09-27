# V2 Advanced Fine-Tuning & Dual Stacking Ensemble Package

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production fine-tuning and dual-model stacking ensemble package for large-scale multi-source business entity resolution.

---

## Technical Features

- **Pre-Trained State & Artifact Injection:** Loads 4 pre-trained XGBoost fold models (`model_0..3.json`) and 832 mined domain equivalences from `state.pkl`.
- **Warm-Start Booster Expansion:** Performs XGBoost `xgb.train(..., xgb_model=existing_booster)` warm-start fine-tuning (`+150` rounds, `lr=0.01`).
- **Dual-Model Stacking Ensemble:** Trains a LightGBM classifier on the feature matrix and performs probability blending (`alpha_xgb=0.60`).
- **Shard-Streaming Blocking:** Guaranteed memory safety `< 8 GB` RAM on Kaggle CPU.

---

## Package Structure

- **`src/artifact_loader.py`**: Discovers and loads `state.pkl` and `model_0..3.json` files.
- **`src/finetuner.py`**: Warm-start XGBoost booster expansion, LightGBM model training, ensemble blending, and Macro $F_{0.5}$ decision threshold grid optimization.
- **`src/postprocessing.py`**: Greedy Maximum Weighted Bipartite Matching for 1-to-1 candidate cardinality resolution.
- **`run_finetune_pipeline.py`**: Production CLI runner for fine-tuning, inference, and submission export.

---

## CLI Usage

```bash
python run_finetune_pipeline.py \
    --artifact-dir "../../Trained model with output/97" \
    --output-dir "./output_v2" \
    --fine-tune-rounds 150 \
    --alpha-xgb 0.60
```
