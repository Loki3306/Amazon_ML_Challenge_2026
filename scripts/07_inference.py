import os
import argparse
import time
import gc
import numpy as np
import polars as pl
import lightgbm as lgb
from multiprocessing import Pool
import sys

import importlib.util

# Dynamically import 06a_features_lightgbm since its name starts with a number
script_dir = os.path.dirname(os.path.abspath(__file__))
feat_script_path = os.path.join(script_dir, "06a_features_lightgbm.py")
spec = importlib.util.spec_from_file_location("feat_gen", feat_script_path)
feat_gen = importlib.util.module_from_spec(spec)
sys.modules["feat_gen"] = feat_gen
spec.loader.exec_module(feat_gen)

load_entity_tables = feat_gen.load_entity_tables
generate_features = feat_gen.generate_features

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 7: Inference & Submission")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--candidates-dir", type=str, default="data/candidates")
    parser.add_argument("--model-path", type=str, default="models/lightgbm/model_6a.txt")
    parser.add_argument("--output-csv", type=str, default="submission.csv")
    parser.add_argument("--threshold", type=float, default=0.95, help="F0.5 optimized threshold")
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()

def process_chunk(args):
    """Wrapper for feature generation to run in pool."""
    chunk, s1_df, cand_df = args
    return generate_features(chunk, s1_df, cand_df)

def main():
    args = parse_args()
    
    print("==================================================")
    print(" PHASE 7: INFERENCE & SUBMISSION")
    print(f" Threshold: {args.threshold}")
    print("==================================================")
    
    # 1. Load Model
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at {args.model_path}")
        return
    print("Loading LightGBM model...")
    model = lgb.Booster(model_file=args.model_path)
    # Get feature names from model
    feature_names = model.feature_name()
    
    # 2. Load Entity Tables
    print("Loading Test Entity tables...")
    s1_df, cand_df = load_entity_tables(args.data_dir, "test")
    
    # 3. Process Sources
    sources = [
        ("dense", os.path.join(args.candidates_dir, f"test_dense_candidates_K{args.top_k}.parquet")),
        ("exact", os.path.join(args.candidates_dir, "test_exact_candidates.parquet"))
    ]
    
    predicted_matches = []
    total_candidates_processed = 0
    t0 = time.time()
    
    for source_name, source_path in sources:
        if not os.path.exists(source_path):
            print(f"Warning: Candidate file {source_path} not found. Skipping.")
            continue
            
        print(f"\nProcessing {source_name} candidates from {source_path}")
        # Lazy load candidates
        lazy_cands = pl.scan_parquet(source_path)
        total_rows = lazy_cands.select(pl.len()).collect().item()
        
        num_chunks = (total_rows + args.chunk_size - 1) // args.chunk_size
        
        for i in range(num_chunks):
            start_idx = i * args.chunk_size
            chunk = lazy_cands.slice(start_idx, args.chunk_size).collect()
            
            chunk_size = chunk.height
            total_candidates_processed += chunk_size
            
            t_chunk = time.perf_counter()
            
            # Compute features
            if args.workers == 1:
                feat_df = generate_features(chunk, s1_df, cand_df)
            else:
                raise ValueError("Inference script currently only supports workers=1 for memory safety.")
                
            # Keep track of IDs
            query_ids = feat_df["query_id"].to_list()
            cand_ids = feat_df["candidate_id"].to_list()
            
            # Extract features for LightGBM
            X = feat_df.select(feature_names).to_numpy()
            
            # Free feat_df
            del feat_df
            del chunk
            gc.collect()
            
            # Predict
            scores = model.predict(X)
            
            # Free X
            del X
            gc.collect()
            
            # Filter matches by threshold
            match_mask = scores > args.threshold
            match_indices = np.where(match_mask)[0]
            
            for idx in match_indices:
                predicted_matches.append((query_ids[idx], cand_ids[idx]))
                
            speed = chunk_size / (time.perf_counter() - t_chunk)
            print(f"  chunk_{i:05d}_{source_name} [{start_idx}+{chunk_size}] "
                  f"found {len(match_indices)} matches | {speed:.0f} rows/s")
                  
    print(f"\nFeature generation & prediction complete in {time.time()-t0:.1f}s")
    
    # 4. Format Submission
    print(f"\nFormatting submission for {len(predicted_matches)} matched pairs...")
    
    # Group matches by query_id
    from collections import defaultdict
    matches_by_query = defaultdict(list)
    
    # Filter duplicates (in case exact and dense returned the same pair)
    unique_matches = list(set(predicted_matches))
    print(f"Unique matched pairs: {len(unique_matches)}")
    
    for q_id, c_id in unique_matches:
        matches_by_query[q_id].append(c_id)
        
    # Get ALL query IDs from S1 so we output a prediction for every S1 entity
    # (even if the prediction is empty)
    all_query_ids = s1_df["entity_id"].to_list()
    
    submission_rows = []
    for q_id in all_query_ids:
        cands = matches_by_query.get(q_id, [])
        # Space separated list of candidates
        cands_str = " ".join(sorted(cands))
        submission_rows.append({"source1_id": q_id, "predicted_match_ids": cands_str})
        
    sub_df = pl.DataFrame(submission_rows)
    sub_df.write_csv(args.output_csv)
    print(f"Saved submission to {args.output_csv} with {sub_df.height} rows.")
    print("Done!")

if __name__ == "__main__":
    main()
