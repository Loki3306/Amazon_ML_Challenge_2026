#!/usr/bin/env python3
"""
Production Pipeline Runner
Amazon ML Challenge 2026: Business Entity Resolution

Shard-streaming architecture — guaranteed hard ceiling of ~8 GB peak RAM.
Never loads full S2/S3 candidate matrices into memory.

Usage:
    python run_pipeline.py --output-dir /path/to/output
"""

import os
import gc
import sys
import time

# Guarantee unbuffered line-by-line streaming output in Kaggle subprocesses
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import argparse
import pickle
import psutil
import numpy as np
import pandas as pd
from collections import Counter

# Package imports
from src.preprocessing import (
    clean_text, build_entity_lookup, load_country_slice,
    load_filtered_candidates, TSV_DTYPES,
)
from src.blocking import shard_streaming_tfidf_blocking
from src.features import extract_pair_features, FEATURE_NAMES
from src.model import train_lightgbm_model
from src.postprocessing import (
    apply_graph_postprocessing, write_submission_tsv, write_candidates_tsv,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_ram_usage() -> str:
    """Return current process RAM usage in GB."""
    return f"{psutil.Process().memory_info().rss / 1e9:.2f} GB"


def ram_gb() -> float:
    """Return current process RAM in GB as a float."""
    return psutil.Process().memory_info().rss / 1e9


def auto_detect_paths():
    """Locate dataset directories across Kaggle and local environments."""
    search_roots = [
        "/kaggle/input",
        ".",
        "..",
        "D:/ML_Challenge",
        "D:/ML_Challenge/DATA/UNZIPPED",
    ]
    train_dir, test_dir, utils_dir = None, None, None
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
    return train_dir, test_dir, utils_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026: Business Entity Resolution Pipeline"
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Root dataset directory")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Output directory for TSV submissions")
    parser.add_argument("--sample-train", type=int, default=100000,
                        help="S1 sample size for training")
    parser.add_argument("--top-k", type=int, default=12,
                        help="Top candidates per S1 entity (test)")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="S1 sub-batch size for sparse multiply")
    parser.add_argument("--shard-size", type=int, default=300000,
                        help="Candidate shard size for streaming TF-IDF")
    parser.add_argument("--max-neg", type=int, default=4,
                        help="Max hard negatives per S1 entity")
    parser.add_argument("--max-features", type=int, default=50000,
                        help="TF-IDF vocabulary size cap")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory for caching checkpoints")
    args = parser.parse_args()

    t_start = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 70, flush=True)
    print("  AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION PIPELINE", flush=True)
    print("  Architecture: Shard-Streaming TF-IDF (hard RAM ceiling ~8 GB)", flush=True)
    print("=" * 70, flush=True)
    print(f"Initial RAM: {get_ram_usage()}", flush=True)

    # ------------------------------------------------------------------
    # 1. Resolve directories
    # ------------------------------------------------------------------
    if args.data_dir:
        train_dir = (os.path.join(args.data_dir, "train")
                     if os.path.exists(os.path.join(args.data_dir, "train"))
                     else args.data_dir)
        test_dir = (os.path.join(args.data_dir, "test")
                    if os.path.exists(os.path.join(args.data_dir, "test"))
                    else args.data_dir)
        utils_dir = (os.path.join(args.data_dir, "utils")
                     if os.path.exists(os.path.join(args.data_dir, "utils"))
                     else None)
    else:
        train_dir, test_dir, utils_dir = auto_detect_paths()

    assert train_dir and os.path.exists(train_dir), \
        f"Could not find train directory: {train_dir}"
    assert test_dir and os.path.exists(test_dir), \
        f"Could not find test directory: {test_dir}"
    print(f"Train Directory:  {train_dir}", flush=True)
    print(f"Test Directory:   {test_dir}", flush=True)
    print(f"Output Directory: {args.output_dir}", flush=True)

    # Build S2/S3 file paths
    s2_train = os.path.join(train_dir, "train_source2.tsv")
    s3_train = os.path.join(train_dir, "train_source3.tsv")
    train_cand_paths = [s2_train, s3_train]

    s2_test = os.path.join(test_dir, "test_source2.tsv")
    s3_test = os.path.join(test_dir, "test_source3.tsv")
    test_cand_paths = [s2_test, s3_test]

    # ==================================================================
    # STAGE 1: TRAINING — SHARD-STREAMING BLOCKING + FILTERED FEATURE EXTRACTION
    # ==================================================================
    model_checkpoint = os.path.join(args.checkpoint_dir, "lgbm_model.pkl")

    if os.path.exists(model_checkpoint):
        print(f"\n[Stage 1] Loading cached model from {model_checkpoint}...",
              flush=True)
        with open(model_checkpoint, "rb") as f:
            ckpt = pickle.load(f)
            model     = ckpt["model"]
            best_tau  = ckpt["best_tau"]
            best_f05  = ckpt["best_f05"]
            val_auc   = ckpt["val_auc"]
        print(f"  Loaded: tau*={best_tau:.4f}, Val F_0.5={best_f05:.5f}",
              flush=True)
    else:
        print(f"\n[Stage 1] Building Training Pairs & Training LightGBM...",
              flush=True)
        t_s1 = time.time()

        # ---- Load ground truth ----
        gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
        gt_df = pd.read_csv(gt_path, sep="\t")
        gt_map = {}
        for sid, mids in zip(
            gt_df["source1_entity_id"],
            gt_df["matched_entity_ids"].fillna("")
        ):
            gt_map[sid] = set(m.strip() for m in mids.split(",") if m.strip())
        del gt_df
        gc.collect()

        # ---- Load & sample S1 train ----
        s1_train = pd.read_csv(
            os.path.join(train_dir, "train_source1.tsv"),
            sep="\t", dtype=TSV_DTYPES,
        )
        if len(s1_train) > args.sample_train:
            s1_train = s1_train.sample(
                n=args.sample_train, random_state=42
            ).reset_index(drop=True)
        s1_train["entity_id"] = s1_train["entity_id"].fillna("").astype(str)
        s1_train["business_name"] = s1_train["business_name"].fillna("").astype(str)
        s1_train["clean_name"] = [
            clean_text(n, eid)
            for n, eid in zip(s1_train["business_name"], s1_train["entity_id"])
        ]
        print(f"  Sampled S1 reference: {len(s1_train):,} | RAM: {get_ram_usage()}",
              flush=True)

        X_list, y_list, group_list, pair_meta = [], [], [], []

        # ---- Process each country independently ----
        for ctry in ["US", "India"]:
            print(f"\n  --- Training: {ctry} ---", flush=True)
            s1_c = s1_train[s1_train["country"] == ctry].copy()
            if s1_c.empty:
                print(f"    No S1 entities for {ctry}, skipping.", flush=True)
                continue

            # 1) Shard-streaming blocking (never loads full candidate matrix)
            print(f"    S1 entities: {len(s1_c):,} | RAM: {get_ram_usage()}",
                  flush=True)
            train_cands = shard_streaming_tfidf_blocking(
                s1_c,
                cand_paths=train_cand_paths,
                country=ctry,
                top_k=10,
                s1_batch_size=args.batch_size,
                shard_chunksize=args.shard_size,
                max_features=args.max_features,
            )

            # 2) Collect IDs needed for feature extraction
            #    Include both blocking results AND true-match IDs
            needed_ids = set()
            for sid_key, ctuples in train_cands.items():
                for cid, _ in ctuples:
                    needed_ids.add(cid)
            for sid_key in s1_c["entity_id"].values:
                for mid in gt_map.get(sid_key, set()):
                    needed_ids.add(mid)

            print(f"    Unique candidate IDs needed: {len(needed_ids):,} | "
                  f"RAM: {get_ram_usage()}", flush=True)

            # 3) Load ONLY the needed candidates (second pass, filtered)
            s2s3_filt = load_filtered_candidates(
                train_cand_paths, ctry, needed_ids
            )
            del needed_ids
            gc.collect()
            print(f"    Loaded {len(s2s3_filt):,} filtered candidates | "
                  f"RAM: {get_ram_usage()}", flush=True)

            # 4) Build lookups
            s1_lookup = build_entity_lookup(s1_c, country=ctry)
            cand_lookup = build_entity_lookup(s2s3_filt, country=ctry)
            del s1_c, s2s3_filt
            gc.collect()

            # 5) Extract features: positives + hard negatives
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
                        X_list.append(extract_pair_features(
                            sn, sa, sp, cn, ca, cp, score
                        ))
                        y_list.append(1)
                        group_list.append(sid_key)
                        pair_meta.append((sid_key, mid))

                # Hard negatives
                negs = [
                    c for c in cand_tuples
                    if c[0] not in true_set and c[0] in cand_lookup
                ]
                if len(negs) > args.max_neg:
                    negs = negs[:args.max_neg]
                for nid, score in negs:
                    cn, ca, cp = cand_lookup[nid]
                    X_list.append(extract_pair_features(
                        sn, sa, sp, cn, ca, cp, score
                    ))
                    y_list.append(0)
                    group_list.append(sid_key)
                    pair_meta.append((sid_key, nid))

            del s1_lookup, cand_lookup, train_cands
            gc.collect()
            print(f"    {ctry} done | Pairs: {len(X_list):,} | "
                  f"RAM: {get_ram_usage()}", flush=True)

        del s1_train
        gc.collect()

        # ---- Train LightGBM ----
        X_arr = np.array(X_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)
        group_arr = np.array(group_list)
        del X_list, y_list, group_list
        gc.collect()

        print(f"\n  Total training pairs: {len(X_arr):,} "
              f"(Pos={int(y_arr.sum()):,}, Neg={int((1-y_arr).sum()):,}) | "
              f"RAM: {get_ram_usage()}", flush=True)

        model, best_tau, best_f05, val_auc = train_lightgbm_model(
            X_arr, y_arr, group_arr, pair_meta, gt_map, random_state=42
        )

        # Save model checkpoint
        with open(model_checkpoint, "wb") as f:
            pickle.dump({
                "model": model,
                "best_tau": best_tau,
                "best_f05": best_f05,
                "val_auc": val_auc,
            }, f)

        del X_arr, y_arr, group_arr, pair_meta, gt_map
        gc.collect()
        print(f"  Stage 1 complete in {(time.time() - t_s1)/60:.1f} min | "
              f"RAM after purge: {get_ram_usage()}", flush=True)

    # ==================================================================
    # STAGE 2: TEST INFERENCE — SHARD-STREAMING PER COUNTRY
    # ==================================================================
    print(f"\n[Stage 2] Streaming Test Inference (France -> US -> India)...",
          flush=True)
    t_s2 = time.time()

    # Read all test S1 entity IDs (tiny, < 15 MB)
    s1_ids_df = pd.read_csv(
        os.path.join(test_dir, "test_source1.tsv"),
        sep="\t", usecols=["entity_id"], dtype={"entity_id": "str"},
    )
    all_test_s1_ids = s1_ids_df["entity_id"].tolist()
    del s1_ids_df
    gc.collect()
    print(f"  Total S1 test entities: {len(all_test_s1_ids):,} | "
          f"RAM: {get_ram_usage()}", flush=True)

    all_candidates_dict = {}
    all_test_pairs = []
    all_test_probs = []

    for ctry in ["France", "US", "India"]:
        print(f"\n  --- Test: {ctry} ---", flush=True)
        t_ctry = time.time()

        # 1) Load S1 test for this country (small: uses load_country_slice)
        s1_c = load_country_slice(
            os.path.join(test_dir, "test_source1.tsv"), ctry
        )
        if s1_c.empty:
            print(f"    No S1 entities for {ctry}, skipping.", flush=True)
            continue
        print(f"    S1: {len(s1_c):,} entities | RAM: {get_ram_usage()}",
              flush=True)

        # 2) Shard-streaming blocking
        ctry_ckpt = os.path.join(args.checkpoint_dir, f"blocking_{ctry}.pkl")
        cand_results = shard_streaming_tfidf_blocking(
            s1_c,
            cand_paths=test_cand_paths,
            country=ctry,
            top_k=args.top_k,
            s1_batch_size=args.batch_size,
            shard_chunksize=args.shard_size,
            max_features=args.max_features,
            checkpoint_file=ctry_ckpt,
        )

        # Save candidate IDs for candidate_pairs.tsv
        for sid_key, ctuples in cand_results.items():
            all_candidates_dict[sid_key] = [ct[0] for ct in ctuples]

        # 3) Collect needed candidate IDs
        needed_ids = set()
        for sid_key, ctuples in cand_results.items():
            for cid, _ in ctuples:
                needed_ids.add(cid)
        print(f"    Unique candidate IDs: {len(needed_ids):,} | "
              f"RAM: {get_ram_usage()}", flush=True)

        # 4) Load ONLY needed candidates
        s2s3_filt = load_filtered_candidates(
            test_cand_paths, ctry, needed_ids
        )
        del needed_ids
        gc.collect()
        print(f"    Loaded {len(s2s3_filt):,} filtered candidates | "
              f"RAM: {get_ram_usage()}", flush=True)

        # 5) Build lookups
        s1_lookup = build_entity_lookup(s1_c, country=ctry)
        cand_lookup = build_entity_lookup(s2s3_filt, country=ctry)
        del s1_c, s2s3_filt
        gc.collect()

        # 6) Extract features & score
        ctry_X = []
        ctry_pairs = []
        for sid_key, ctuples in cand_results.items():
            if sid_key not in s1_lookup:
                continue
            sn, sa, sp = s1_lookup[sid_key]
            for cid, score in ctuples:
                if cid in cand_lookup:
                    cn, ca, cp = cand_lookup[cid]
                    ctry_X.append(extract_pair_features(
                        sn, sa, sp, cn, ca, cp, score
                    ))
                    ctry_pairs.append((sid_key, cid))

        del s1_lookup, cand_lookup, cand_results
        gc.collect()

        # 7) Model scoring
        if ctry_X:
            ctry_X_arr = np.array(ctry_X, dtype=np.float32)
            ctry_probs = model.predict(
                ctry_X_arr, num_iteration=model.best_iteration
            )
            all_test_pairs.extend(ctry_pairs)
            all_test_probs.extend(ctry_probs.tolist())
            del ctry_X_arr, ctry_probs

        del ctry_X, ctry_pairs
        gc.collect()

        elapsed_ctry = (time.time() - t_ctry) / 60
        print(f"    {ctry} done in {elapsed_ctry:.1f} min | "
              f"Scored: {len(all_test_pairs):,} total | "
              f"RAM: {get_ram_usage()}", flush=True)

    print(f"  Stage 2 complete in {(time.time() - t_s2)/60:.1f} min!",
          flush=True)

    # ==================================================================
    # STAGE 3: POST-PROCESSING & TSV SUBMISSIONS
    # ==================================================================
    print(f"\n[Stage 3] Writing TSV Submissions & Enforcing Graph Consistency...",
          flush=True)

    # 1. Write candidate_pairs.tsv
    cand_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    print(f"  Writing {cand_path} ({len(all_test_s1_ids):,} rows)...",
          flush=True)
    write_candidates_tsv(cand_path, all_test_s1_ids, all_candidates_dict)

    # 2. Maximum Weighted Bipartite Matching
    print("  Running greedy maximum weighted bipartite matching...", flush=True)
    pair_records = sorted(
        zip(all_test_pairs, all_test_probs),
        key=lambda x: x[1], reverse=True,
    )
    del all_test_pairs, all_test_probs, all_candidates_dict
    gc.collect()

    final_matches = apply_graph_postprocessing(
        pair_records,
        tau_match=best_tau,
        tau_singleton=0.30,
        max_matches=11,
    )

    # 3. Write matching_results.tsv
    match_path = os.path.join(args.output_dir, "matching_results.tsv")
    print(f"  Writing {match_path} ({len(all_test_s1_ids):,} rows)...",
          flush=True)
    write_submission_tsv(match_path, all_test_s1_ids, final_matches)

    # 4. Optional validation check
    if utils_dir:
        val_script = os.path.join(utils_dir, "validate_submission.py")
        if os.path.exists(val_script):
            print("\n" + "=" * 55, flush=True)
            print("RUNNING OFFICIAL SUBMISSION VALIDATOR:", flush=True)
            print("=" * 55, flush=True)
            os.system(
                f'python "{val_script}" '
                f'--matching "{match_path}" '
                f'--candidate "{cand_path}" '
                f'--test-dir "{test_dir}"'
            )
            print("=" * 55, flush=True)

    # ==================================================================
    # SUMMARY DASHBOARD
    # ==================================================================
    match_counts = [len(final_matches.get(sid, set())) for sid in all_test_s1_ids]
    singletons = sum(1 for c in match_counts if c == 0)
    total_time = (time.time() - t_start) / 60.0

    print("\n" + "=" * 70, flush=True)
    print("  PIPELINE EXECUTION SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"  Total Pipeline Runtime:      {total_time:.1f} minutes", flush=True)
    print(f"  Validation AUC-ROC:          {val_auc:.5f}", flush=True)
    print(f"  Validation Macro F_0.5:      {best_f05:.5f}", flush=True)
    print(f"  Optimal Decision Threshold:  {best_tau:.4f}", flush=True)
    print(f"  Total Test Entities (S1):    {len(all_test_s1_ids):,}", flush=True)
    print(f"  Predicted Singletons:        {singletons:,} "
          f"({singletons/len(all_test_s1_ids)*100:.2f}%)", flush=True)
    print(f"  Average Matches per Entity:  {np.mean(match_counts):.2f}", flush=True)
    print(f"  Maximum Matches per Entity:  {max(match_counts)}", flush=True)
    print(f"  Peak RAM Usage:              {get_ram_usage()}", flush=True)
    print(f"\n  Match Distribution:", flush=True)
    dist = Counter(match_counts)
    for k in range(min(12, max(dist.keys()) + 1)):
        count = dist.get(k, 0)
        bar = "#" * min(40, int(count / max(dist.values()) * 40))
        print(f"    {k:2d} matches: {count:>8,} "
              f"({count/len(all_test_s1_ids)*100:5.2f}%) {bar}", flush=True)
    print("=" * 70, flush=True)
    print(f"Submissions ready:", flush=True)
    print(f"  -> {match_path}", flush=True)
    print(f"  -> {cand_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
