#!/usr/bin/env python3
"""
V2 Advanced Fine-Tuning & Ensemble Pipeline Orchestrator
Amazon ML Challenge 2026: Business Entity Resolution

Integrates pre-trained artifacts (state.pkl + model_0..3.json), performs XGBoost booster
warm-start expansion, LightGBM model stacking, and dual-model ensemble probability blending.
Guarantees a hard memory ceiling of ~8 GB peak RAM.

Usage:
    python run_finetune_pipeline.py --artifact-dir "/path/to/artifacts" --output-dir "/path/to/output"
"""

import os
import gc
import sys
import time

# Guarantee unbuffered streaming output in Kaggle subprocesses
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import argparse
import pickle
import psutil
import numpy as np
import pandas as pd
from collections import Counter
import xgboost as xgb

# Ensure current script directory is in sys.path so 'src' is found regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import core modules
from src.preprocessing import (
    clean_text, build_entity_lookup, load_country_slice,
    load_filtered_candidates, TSV_DTYPES
)
from src.blocking import shard_streaming_tfidf_blocking
from src.features import extract_pair_features
from src.artifact_loader import load_pretrained_artifacts
from src.finetuner import train_ensemble_and_tune_threshold
from src.postprocessing import (
    apply_graph_postprocessing, write_submission_tsv, write_candidates_tsv
)


def get_ram_usage() -> str:
    """Return current process RAM usage in GB."""
    return f"{psutil.Process().memory_info().rss / 1e9:.2f} GB"


def auto_detect_paths():
    """Locate dataset and artifact directories across Kaggle and local environments."""
    search_roots = [
        "/kaggle/input",
        ".",
        "..",
        "D:/ML_Challenge",
        "D:/ML_Challenge/Trained model with output/97"
    ]
    train_dir, test_dir, utils_dir, artifact_dir = None, None, None, None
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dp, dn, fn in os.walk(root):
            if "train_source1.tsv" in fn and not train_dir:
                train_dir = dp
            if "test_source1.tsv" in fn and not test_dir:
                test_dir = dp
            if "validate_submission.py" in fn and not utils_dir:
                utils_dir = dp
            if ("model_0 (1).json" in fn or "model_0.json" in fn or "state (1).pkl" in fn) and not artifact_dir:
                artifact_dir = dp

    if not artifact_dir and os.path.exists("D:/ML_Challenge/Trained model with output/97"):
        artifact_dir = "D:/ML_Challenge/Trained model with output/97"

    return train_dir, test_dir, utils_dir, artifact_dir


def main():
    parser = argparse.ArgumentParser(description="V2 Advanced Fine-Tuning & Ensemble Pipeline")
    parser.add_argument("--data-dir", type=str, default=None, help="Root dataset directory")
    parser.add_argument("--artifact-dir", type=str, default=None, help="Path to pre-trained model_*.json and state.pkl")
    parser.add_argument("--output-dir", type=str, default="./output_v2", help="Output directory for submissions")
    parser.add_argument("--sample-train", type=int, default=100000, help="S1 sample size for fine-tuning training")
    parser.add_argument("--top-k", type=int, default=12, help="Top candidates per S1 entity")
    parser.add_argument("--batch-size", type=int, default=1000, help="S1 sub-batch size for sparse multiply")
    parser.add_argument("--shard-size", type=int, default=300000, help="Candidate shard size for streaming TF-IDF")
    parser.add_argument("--fine-tune-rounds", type=int, default=150, help="Number of warm-start booster expansion rounds")
    parser.add_argument("--alpha-xgb", type=float, default=0.60, help="Ensemble weight for XGBoost (vs LightGBM)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints_v2", help="Directory for caching checkpoints")
    args = parser.parse_args()

    t_start = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 75, flush=True)
    print("  AMAZON ML CHALLENGE 2026: V2 ADVANCED FINE-TUNING & ENSEMBLE PIPELINE", flush=True)
    print("  Architecture: Pre-Trained State Injection + Dual Model Stacking Ensemble", flush=True)
    print("=" * 75, flush=True)
    print(f"Initial RAM: {get_ram_usage()}", flush=True)

    # 1. Resolve Directories
    if args.data_dir:
        train_dir = os.path.join(args.data_dir, "train") if os.path.exists(os.path.join(args.data_dir, "train")) else args.data_dir
        test_dir = os.path.join(args.data_dir, "test") if os.path.exists(os.path.join(args.data_dir, "test")) else args.data_dir
        utils_dir = os.path.join(args.data_dir, "utils") if os.path.exists(os.path.join(args.data_dir, "utils")) else None
        artifact_dir = args.artifact_dir
    else:
        train_dir, test_dir, utils_dir, artifact_dir = auto_detect_paths()
        if args.artifact_dir:
            artifact_dir = args.artifact_dir

    assert train_dir and os.path.exists(train_dir), f"Could not find train directory: {train_dir}"
    assert test_dir and os.path.exists(test_dir), f"Could not find test directory: {test_dir}"
    print(f"Train Directory:    {train_dir}", flush=True)
    print(f"Test Directory:     {test_dir}", flush=True)
    print(f"Artifact Directory: {artifact_dir}", flush=True)
    print(f"Output Directory:   {args.output_dir}", flush=True)

    # Load Pre-Trained Artifacts
    state_dict, pretrained_boosters = {}, []
    if artifact_dir and os.path.exists(artifact_dir):
        state_dict, pretrained_boosters = load_pretrained_artifacts(artifact_dir)

    train_cand_paths = [os.path.join(train_dir, "train_source2.tsv"), os.path.join(train_dir, "train_source3.tsv")]
    test_cand_paths = [os.path.join(test_dir, "test_source2.tsv"), os.path.join(test_dir, "test_source3.tsv")]

    # =========================================================================
    # STAGE 1: FINE-TUNING TRAINING & DUAL-MODEL STACKING
    # =========================================================================
    model_checkpoint = os.path.join(args.checkpoint_dir, "v2_ensemble_model.pkl")

    if os.path.exists(model_checkpoint):
        print(f"\n[Stage 1] Loading cached fine-tuned ensemble from {model_checkpoint}...", flush=True)
        with open(model_checkpoint, "rb") as f:
            ckpt = pickle.load(f)
            models_dict = ckpt["models_dict"]
            best_tau    = ckpt["best_tau"]
            best_f05    = ckpt["best_f05"]
            val_auc     = ckpt["val_auc"]
        print(f"  Loaded ensemble: optimal tau* = {best_tau:.4f}, Val F_0.5 = {best_f05:.5f}", flush=True)
    else:
        print(f"\n[Stage 1] Building Training Pairs & Fine-Tuning Dual Ensemble...", flush=True)
        t_s1 = time.time()

        # Load Ground Truth
        gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
        gt_df = pd.read_csv(gt_path, sep="\t")
        gt_map = {}
        for sid, mids in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"].fillna("")):
            gt_map[sid] = set(m.strip() for m in mids.split(",") if m.strip())
        del gt_df
        gc.collect()

        # Load & Sample S1 Train
        s1_train = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", dtype=TSV_DTYPES)
        if len(s1_train) > args.sample_train:
            s1_train = s1_train.sample(n=args.sample_train, random_state=42).reset_index(drop=True)
        s1_train["entity_id"] = s1_train["entity_id"].fillna("").astype(str)
        s1_train["business_name"] = s1_train["business_name"].fillna("").astype(str)
        s1_train["clean_name"] = [clean_text(n, eid) for n, eid in zip(s1_train["business_name"], s1_train["entity_id"])]

        X_list, y_list, group_list, pair_meta = [], [], [], []

        for ctry in ["US", "India"]:
            print(f"\n  --- Training Slice: {ctry} ---", flush=True)
            s1_c = s1_train[s1_train["country"] == ctry].copy()
            if s1_c.empty:
                continue

            train_cands = shard_streaming_tfidf_blocking(
                s1_c, cand_paths=train_cand_paths, country=ctry, top_k=10,
                s1_batch_size=args.batch_size, shard_chunksize=args.shard_size
            )

            needed_ids = set()
            for sid_key, ctuples in train_cands.items():
                for cid, _ in ctuples:
                    needed_ids.add(cid)
            for sid_key in s1_c["entity_id"].values:
                for mid in gt_map.get(sid_key, set()):
                    needed_ids.add(mid)

            s2s3_filt = load_filtered_candidates(train_cand_paths, ctry, needed_ids)
            del needed_ids
            gc.collect()

            s1_lookup = build_entity_lookup(s1_c, country=ctry)
            cand_lookup = build_entity_lookup(s2s3_filt, country=ctry)
            del s1_c, s2s3_filt
            gc.collect()

            for sid_key, cand_tuples in train_cands.items():
                if sid_key not in s1_lookup:
                    continue
                sn, sa, sp = s1_lookup[sid_key]
                true_set = gt_map.get(sid_key, set())
                cand_score_map = dict(cand_tuples)

                # Positives
                for mid in true_set:
                    if mid in cand_lookup:
                        cn, ca, cp = cand_lookup[mid]
                        score = cand_score_map.get(mid, 0.5)
                        X_list.append(extract_pair_features(sn, sa, sp, cn, ca, cp, score))
                        y_list.append(1)
                        group_list.append(sid_key)
                        pair_meta.append((sid_key, mid))

                # Hard Negatives
                negs = [c for c in cand_tuples if c[0] not in true_set and c[0] in cand_lookup]
                if len(negs) > 4:
                    negs = negs[:4]
                for nid, score in negs:
                    cn, ca, cp = cand_lookup[nid]
                    X_list.append(extract_pair_features(sn, sa, sp, cn, ca, cp, score))
                    y_list.append(0)
                    group_list.append(sid_key)
                    pair_meta.append((sid_key, nid))

            del s1_lookup, cand_lookup, train_cands
            gc.collect()

        del s1_train
        gc.collect()

        X_arr = np.array(X_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)
        group_arr = np.array(group_list)
        del X_list, y_list, group_list
        gc.collect()

        # Perform Fine-Tuning & Dual Stacking Training
        models_dict, best_tau, best_f05, val_auc = train_ensemble_and_tune_threshold(
            X_arr, y_arr, group_arr, pair_meta, gt_map,
            pretrained_boosters=pretrained_boosters,
            alpha_xgb=args.alpha_xgb
        )

        with open(model_checkpoint, "wb") as f:
            pickle.dump({
                "models_dict": models_dict,
                "best_tau": best_tau,
                "best_f05": best_f05,
                "val_auc": val_auc
            }, f)

        del X_arr, y_arr, group_arr, pair_meta, gt_map
        gc.collect()
        print(f"  Stage 1 complete in {(time.time() - t_s1)/60:.1f} min | RAM: {get_ram_usage()}", flush=True)

    # =========================================================================
    # STAGE 2: TEST INFERENCE & DUAL ENSEMBLE SCORING
    # =========================================================================
    print(f"\n[Stage 2] Running Test Inference & Ensemble Scoring (France -> US -> India)...", flush=True)
    t_s2 = time.time()

    s1_ids_df = pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t", usecols=["entity_id"], dtype={"entity_id": "str"})
    all_test_s1_ids = s1_ids_df["entity_id"].tolist()
    del s1_ids_df
    gc.collect()

    all_candidates_dict = {}
    all_test_pairs = []
    all_test_probs = []

    lgb_model = models_dict["lgb_model"]
    xgb_models = models_dict.get("xgb_models", [])
    alpha_xgb  = models_dict.get("alpha_xgb", 0.60)

    for ctry in ["France", "US", "India"]:
        print(f"\n  --- Processing {ctry} ---", flush=True)
        t_ctry = time.time()

        s1_c = load_country_slice(os.path.join(test_dir, "test_source1.tsv"), ctry)
        if s1_c.empty:
            continue

        ctry_ckpt = os.path.join(args.checkpoint_dir, f"v2_blocking_{ctry}.pkl")
        cand_results = shard_streaming_tfidf_blocking(
            s1_c, cand_paths=test_cand_paths, country=ctry, top_k=args.top_k,
            s1_batch_size=args.batch_size, shard_chunksize=args.shard_size,
            checkpoint_file=ctry_ckpt
        )

        for sid_key, ctuples in cand_results.items():
            all_candidates_dict[sid_key] = [ct[0] for ct in ctuples]

        needed_ids = set()
        for sid_key, ctuples in cand_results.items():
            for cid, _ in ctuples:
                needed_ids.add(cid)

        s2s3_filt = load_filtered_candidates(test_cand_paths, ctry, needed_ids)
        del needed_ids
        gc.collect()

        s1_lookup = build_entity_lookup(s1_c, country=ctry)
        cand_lookup = build_entity_lookup(s2s3_filt, country=ctry)
        del s1_c, s2s3_filt
        gc.collect()

        ctry_X = []
        ctry_pairs = []
        for sid_key, ctuples in cand_results.items():
            if sid_key not in s1_lookup:
                continue
            sn, sa, sp = s1_lookup[sid_key]
            for cid, score in ctuples:
                if cid in cand_lookup:
                    cn, ca, cp = cand_lookup[cid]
                    ctry_X.append(extract_pair_features(sn, sa, sp, cn, ca, cp, score))
                    ctry_pairs.append((sid_key, cid))

        del s1_lookup, cand_lookup, cand_results
        gc.collect()

        if ctry_X:
            ctry_X_arr = np.array(ctry_X, dtype=np.float32)
            
            # Predict LightGBM
            lgb_p = lgb_model.predict(ctry_X_arr, num_iteration=lgb_model.best_iteration)
            
            # Predict XGBoost Boosters
            if xgb_models:
                dtest_xgb = xgb.DMatrix(ctry_X_arr)
                xgb_p_list = [bm.predict(dtest_xgb) for bm in xgb_models]
                xgb_p = np.mean(xgb_p_list, axis=0)
                ctry_probs = alpha_xgb * xgb_p + (1.0 - alpha_xgb) * lgb_p
            else:
                ctry_probs = lgb_p

            all_test_pairs.extend(ctry_pairs)
            all_test_probs.extend(ctry_probs.tolist())
            del ctry_X_arr, ctry_probs

        del ctry_X, ctry_pairs
        gc.collect()
        print(f"  Finished {ctry} in {(time.time() - t_ctry)/60:.1f} min | Scored: {len(all_test_pairs):,} total", flush=True)

    # =========================================================================
    # STAGE 3: POST-PROCESSING & SUBMISSION EXPORTS
    # =========================================================================
    print(f"\n[Stage 3] Writing Submission TSVs & Enforcing Bipartite Graph Consistency...", flush=True)

    cand_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    print(f"  Writing {cand_path} ({len(all_test_s1_ids):,} rows)...", flush=True)
    write_candidates_tsv(cand_path, all_test_s1_ids, all_candidates_dict)

    pair_records = sorted(zip(all_test_pairs, all_test_probs), key=lambda x: x[1], reverse=True)
    del all_test_pairs, all_test_probs, all_candidates_dict
    gc.collect()

    final_matches = apply_graph_postprocessing(
        pair_records,
        tau_match=best_tau,
        tau_singleton=0.30,
        max_matches=11
    )

    match_path = os.path.join(args.output_dir, "matching_results.tsv")
    print(f"  Writing {match_path} ({len(all_test_s1_ids):,} rows)...", flush=True)
    write_submission_tsv(match_path, all_test_s1_ids, final_matches)

    # Run Official Validator if Present
    if utils_dir:
        val_script = os.path.join(utils_dir, "validate_submission.py")
        if os.path.exists(val_script):
            print("\n" + "=" * 55, flush=True)
            print("RUNNING OFFICIAL SUBMISSION VALIDATOR:", flush=True)
            print("=" * 55, flush=True)
            os.system(f'python "{val_script}" --matching "{match_path}" --candidate "{cand_path}" --test-dir "{test_dir}"')
            print("=" * 55, flush=True)

    total_time = (time.time() - t_start) / 60.0
    print("\n" + "=" * 75, flush=True)
    print("  V2 PIPELINE EXECUTION COMPLETE", flush=True)
    print("=" * 75, flush=True)
    print(f"  Total Runtime:              {total_time:.1f} minutes", flush=True)
    print(f"  Validation AUC-ROC:          {val_auc:.5f}", flush=True)
    print(f"  Validation Macro F_0.5:      {best_f05:.5f}", flush=True)
    print(f"  Optimal Decision Threshold:  {best_tau:.4f}", flush=True)
    print(f"  Submissions Ready:", flush=True)
    print(f"    -> {match_path}", flush=True)
    print(f"    -> {cand_path}", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
