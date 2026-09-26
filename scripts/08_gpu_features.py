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
    parser.add_argument("--data-dir", type=str, default="data/canonical")
    parser.add_argument("--ground-truth", type=str, required=True)
    parser.add_argument("--models-dir", type=str, default="data/models")
    return parser.parse_args()

def load_gt(path):
    import polars as pl
    print(f"Loading ground truth from {path}")
    df = pl.read_csv(path, separator="\t")
    gt = {}
    for row in df.iter_rows(named=True):
        s1_id = str(row["source1_entity_id"])
        raw = row["matched_entity_ids"]
        if raw:
            gt[s1_id] = set(str(x) for x in raw.split(","))
        else:
            gt[s1_id] = set()
    return gt

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
    s1_df = cudf.read_parquet(os.path.join(args.data_dir, "train_s1.parquet"))
    cand_df = cudf.read_parquet(os.path.join(args.data_dir, "train_candidates.parquet"))
    
    s1_df = s1_df[['entity_id', 'name_norm', 'address_norm']].rename(
        columns={'entity_id': 'query_id', 'name_norm': 'name_s1', 'address_norm': 'addr_s1'})
    cand_df = cand_df[['entity_id', 'name_norm', 'address_norm']].rename(
        columns={'entity_id': 'candidate_id', 'name_norm': 'name_cand', 'address_norm': 'addr_cand'})
    
    train_shards = [os.path.join(args.candidates_dir, f) for f in os.listdir(args.candidates_dir) 
                   if f.startswith("train_") and f.endswith(".parquet")]
    
    print(f"Found {len(train_shards)} candidate files.")
    
    all_features = []
    for p in train_shards:
        print(f"Processing {p} on GPU...")
        df = cudf.read_parquet(p)
        df = df.merge(s1_df, on='query_id', how='inner')
        df = df.merge(cand_df, on='candidate_id', how='inner')
        
        # Labeling (must do on CPU for dict lookup, then move to GPU)
        pdf = df[['query_id', 'candidate_id']].to_pandas()
        labels = []
        for _, row in pdf.iterrows():
            q, c = str(row['query_id']), str(row['candidate_id'])
            labels.append(1 if q in gt and c in gt[q] else 0)
        df['label'] = cudf.Series(labels, dtype=cp.int8)
        
        # Downsample negatives to save memory
        pos = df[df['label'] == 1]
        neg = df[df['label'] == 0]
        n_keep = max(len(pos), 1) * 10
        if len(neg) > n_keep:
            neg = neg.sample(n=n_keep, random_state=42)
        df = cudf.concat([pos, neg])
        
        feat_df = compute_gpu_features(df)
        all_features.append(feat_df)
        
        del df, pos, neg, pdf
        gc.collect()
        
    print("Concatenating all features...")
    final_df = cudf.concat(all_features)
    del all_features; gc.collect()
    
    features = ['name_exact', 'addr_exact', 'name_lev_sim', 'addr_lev_sim', 'name_len_diff']
    X = final_df[features].to_pandas().values
    y = final_df['label'].to_pandas().values
    
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
