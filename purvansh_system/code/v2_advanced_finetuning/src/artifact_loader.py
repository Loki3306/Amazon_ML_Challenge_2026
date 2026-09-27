"""
Artifact & Pre-Trained Model Loader
Loads pre-trained XGBoost fold models (model_0..3.json) and state pickle (state.pkl).
"""

import os
import pickle
import xgboost as xgb


def find_artifact_files(artifact_dir: str) -> tuple:
    """
    Scan artifact directory and locate state.pkl and model_0..3.json files.
    Handles filename variations like 'state (1).pkl' or 'model_0 (1).json'.
    """
    assert os.path.exists(artifact_dir), f"Artifact directory not found: {artifact_dir}"

    state_file = None
    model_files = []

    for root, _, files in os.walk(artifact_dir):
        for f in files:
            full_path = os.path.join(root, f)
            if f.endswith(".pkl") and ("state" in f or "artifacts" in f):
                state_file = full_path
            elif f.endswith(".json") and "model_" in f:
                model_files.append(full_path)

    model_files.sort()
    return state_file, model_files


def load_pretrained_artifacts(artifact_dir: str) -> tuple:
    """
    Load pre-trained state.pkl and XGBoost booster fold models.
    
    Returns:
        tuple: (state_dict, list_of_xgboost_boosters)
    """
    state_file, model_files = find_artifact_files(artifact_dir)

    print("=" * 60, flush=True)
    print("  LOADING PRE-TRAINED ARTIFACTS (98.36% Baseline)", flush=True)
    print("=" * 60, flush=True)
    
    # 1. Load State Pickle
    state_dict = {}
    if state_file and os.path.exists(state_file):
        print(f"  [Artifacts] Loading state pickle: {os.path.basename(state_file)}...", flush=True)
        with open(state_file, "rb") as f:
            state_dict = pickle.load(f)
        print(f"  [Artifacts] Successfully loaded state object.", flush=True)
    else:
        print("  [Warning] State pickle not found in artifact directory.", flush=True)

    # 2. Load XGBoost Boosters
    boosters = []
    if model_files:
        print(f"  [Artifacts] Found {len(model_files)} pre-trained XGBoost model JSON files.", flush=True)
        for mf in model_files:
            booster = xgb.Booster()
            booster.load_model(mf)
            boosters.append(booster)
            print(f"    Loaded XGBoost booster: {os.path.basename(mf)}", flush=True)
    else:
        print("  [Warning] No model_*.json files found in artifact directory.", flush=True)

    print("=" * 60, flush=True)
    return state_dict, boosters
