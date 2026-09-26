import argparse
import os
import gc
import json
import cudf
import cupy as cp
import lightgbm as lgb
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-dir", type=str, default="data/candidates")
    parser.add_argument("--data-dir", type=str, default="data/processed/train")
    parser.add_argument("--ground-truth", type=str, required=True)
    parser.add_argument("--models-dir", type=str, default="data/models")
    return parser.parse_args()

def load_gt(path):
    import polars as pl
    print(f"Loading ground truth from {path}")
    df = pl.read_csv(path, separator="\t")
    
    rows = []
    for row in df.iter_rows(named=True):
        s1_id = str(row["source1_entity_id"])
        raw = row["matched_entity_ids"]
        if raw:
            for c in str(raw).split(","):
                rows.append({"query_id": s1_id, "candidate_id": c})
    
    if not rows:
        return pl.DataFrame({"query_id": [], "candidate_id": [], "label": []})
        
    gt_df = pl.DataFrame(rows).with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    return gt_df

def compute_gpu_features(chunk: cudf.DataFrame) -> cudf.DataFrame:
    # Handle nulls
    chunk['name_s1'] = chunk['name_s1'].fillna("")
    chunk['name_cand'] = chunk['name_cand'].fillna("")
    chunk['addr_s1'] = chunk['addr_s1'].fillna("")
    chunk['addr_cand'] = chunk['addr_cand'].fillna("")
    
    # Exact matches
    chunk['name_exact'] = (chunk['name_s1'] == chunk['name_cand']).astype(cp.float32)
    chunk['addr_exact'] = (chunk['addr_s1'] == chunk['addr_cand']).astype(cp.float32)
    
    # Edit distance (Levenshtein)
    name_ed = chunk['name_s1'].str.edit_distance(chunk['name_cand'])
    name_max_len = cp.maximum(chunk['name_s1'].str.len(), chunk['name_cand'].str.len()).astype(cp.float32)
    name_max_len = cp.maximum(name_max_len, 1.0)
    chunk['name_lev_sim'] = (1.0 - (name_ed.astype(cp.float32) / name_max_len)).astype(cp.float32)
    
    addr_ed = chunk['addr_s1'].str.edit_distance(chunk['addr_cand'])
    addr_max_len = cp.maximum(chunk['addr_s1'].str.len(), chunk['addr_cand'].str.len()).astype(cp.float32)
    addr_max_len = cp.maximum(addr_max_len, 1.0)
    chunk['addr_lev_sim'] = (1.0 - (addr_ed.astype(cp.float32) / addr_max_len)).astype(cp.float32)
    
    # Length diff
    chunk['name_len_diff'] = cp.abs(chunk['name_s1'].str.len() - chunk['name_cand'].str.len()).astype(cp.float32)
    
    features = ['name_exact', 'addr_exact', 'name_lev_sim', 'addr_lev_sim', 'name_len_diff']
    return chunk[['query_id', 'candidate_id', 'label'] + features]

def main():
    args = parse_args()
    gt = load_gt(args.ground_truth)
    
    print("Loading entity tables to GPU...")
    s1_df = cudf.read_parquet(os.path.join(args.data_dir, "train_source1.parquet"))
    s2 = cudf.read_parquet(os.path.join(args.data_dir, "train_source2.parquet"))
    s3 = cudf.read_parquet(os.path.join(args.data_dir, "train_source3.parquet"))
    cand_df = cudf.concat([s2, s3])
    del s2, s3
    
    s1_df = s1_df[['entity_id', 'name_norm', 'address_norm']].rename(
        columns={'entity_id': 'query_id', 'name_norm': 'name_s1', 'address_norm': 'addr_s1'})
    cand_df = cand_df[['entity_id', 'name_norm', 'address_norm']].rename(
        columns={'entity_id': 'candidate_id', 'name_norm': 'name_cand', 'address_norm': 'addr_cand'})
    
    train_shards = [os.path.join(args.candidates_dir, f) for f in os.listdir(args.candidates_dir) 
                   if f.startswith("train_") and f.endswith(".parquet")]
    
    print(f"Found {len(train_shards)} candidate files.")
    
    features = ['name_exact', 'addr_exact', 'name_lev_sim', 'addr_lev_sim', 'name_len_diff']
    all_X = []
    all_y = []
    
    for p in train_shards:
        print(f"Processing {p} on CPU first to prevent GPU OOM...")
        import polars as pl
        import pandas as pd
        
        # Read candidate pairs on CPU
        pairs = pl.read_parquet(p)
        
        # Label on CPU instantly via join
        print("  Labeling on CPU via Polars join...")
        pairs = pairs.join(gt, on=['query_id', 'candidate_id'], how='left').with_columns(
            pl.col("label").fill_null(0)
        )
        
        # Downsample negatives on CPU BEFORE merging strings!
        print("  Downsampling on CPU...")
        pos = pairs.filter(pl.col("label") == 1)
        neg = pairs.filter(pl.col("label") == 0)
        
        n_keep = max(pos.height, 1) * 10
        if neg.height > n_keep:
            # Polars random sample
            neg = neg.sample(n=n_keep, seed=42)
            
        sampled = pl.concat([pos, neg])
        sampled_pdf = sampled.to_pandas()
        del pos, neg, pairs, sampled; gc.collect()
        
        print("  Transferring to GPU in chunks to preserve VRAM...")
        chunk_size = 500_000
        for i in range(0, len(sampled_pdf), chunk_size):
            print(f"    GPU Chunk {i//chunk_size + 1}/{len(sampled_pdf)//chunk_size + 1}...")
            sub_pdf = sampled_pdf.iloc[i:i+chunk_size]
            df = cudf.DataFrame(sub_pdf)
            df = df.merge(s1_df, on='query_id', how='inner')
            df = df.merge(cand_df, on='candidate_id', how='inner')
            
            feat_df = compute_gpu_features(df)
            
            # Immediately extract to numpy arrays on CPU to free the huge strings from GPU memory
            X_chunk = feat_df[features].to_pandas().values
            y_chunk = feat_df['label'].to_pandas().values
            all_X.append(X_chunk)
            all_y.append(y_chunk)
            
            del df, feat_df, sub_pdf
            gc.collect()
            
        del sampled_pdf
        gc.collect()
        
    print("Concatenating all chunks...")
    import numpy as np
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    del all_X, all_y; gc.collect()
    
    print("Training LightGBM on GPU...")
    params = {
        "objective": "binary", "metric": "auc",
        "boosting_type": "gbdt", "learning_rate": 0.05,
        "n_estimators": 100, "device": "gpu"
    }
    
    lgb_tr = lgb.Dataset(X, label=y, feature_name=features)
    model = lgb.train(params, lgb_tr)
    
    os.makedirs(args.models_dir, exist_ok=True)
    model.save_model(os.path.join(args.models_dir, "model_gpu.txt"))
    print("Done! Saved model_gpu.txt")

if __name__ == "__main__":
    main()
