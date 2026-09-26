import os
import gc
import cudf
import pandas as pd
import lightgbm as lgb
import argparse

import sys
import importlib.util
script_dir = os.path.dirname(os.path.abspath(__file__))
feat_script_path = os.path.join(script_dir, "08_gpu_features.py")
spec = importlib.util.spec_from_file_location("gpu_feat", feat_script_path)
gpu_feat = importlib.util.module_from_spec(spec)
sys.modules["gpu_feat"] = gpu_feat
spec.loader.exec_module(gpu_feat)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()

    print(f"Loading Model for split: {args.split}...")
    model = lgb.Booster(model_file="data/models/model_gpu.txt")
    features = ['name_exact', 'addr_exact', 'name_lev_sim', 'addr_lev_sim', 'name_len_diff']

    data_dir = f"data/processed/{args.split}"
    s1_df = cudf.read_parquet(os.path.join(data_dir, f"{args.split}_source1.parquet"))
    s2 = cudf.read_parquet(os.path.join(data_dir, f"{args.split}_source2.parquet"))
    s3 = cudf.read_parquet(os.path.join(data_dir, f"{args.split}_source3.parquet"))
    cand_df = cudf.concat([s2, s3])
    del s2, s3

    s1_df = s1_df[['entity_id', 'name_norm', 'address_norm']].rename(
        columns={'entity_id': 'query_id', 'name_norm': 'name_s1', 'address_norm': 'addr_s1'})
    cand_df = cand_df[['entity_id', 'name_norm', 'address_norm']].rename(
        columns={'entity_id': 'candidate_id', 'name_norm': 'name_cand', 'address_norm': 'addr_cand'})

    shards = [os.path.join("data/candidates", f) for f in os.listdir("data/candidates") 
              if f.startswith(f"{args.split}_") and f.endswith(".parquet")]

    all_preds = []
    for p in shards:
        print(f"Scoring {p}...")
        pairs = pd.read_parquet(p, columns=['query_id', 'candidate_id'])
        chunk_size = 500_000
        for i in range(0, len(pairs), chunk_size):
            print(f"  Chunk {i//chunk_size + 1}/{len(pairs)//chunk_size + 1}")
            sub_pdf = pairs.iloc[i:i+chunk_size].copy()
            df = cudf.DataFrame(sub_pdf)
            df = df.merge(s1_df, on='query_id', how='inner')
            df = df.merge(cand_df, on='candidate_id', how='inner')
            
            feat_df = gpu_feat.compute_gpu_features(df)
            X_chunk = feat_df[features].to_pandas().values
            preds = model.predict(X_chunk)
            
            res_pdf = feat_df[['query_id', 'candidate_id']].to_pandas()
            res_pdf['score'] = preds
            all_preds.append(res_pdf[res_pdf['score'] > args.threshold])
            
            del df, feat_df, res_pdf
            gc.collect()

    final_preds = pd.concat(all_preds)
    # Deduplicate in case a candidate was found by both Exact and BM25
    final_preds = final_preds.sort_values('score', ascending=False).drop_duplicates(['query_id', 'candidate_id'])
    
    submission = final_preds.groupby('query_id')['candidate_id'].apply(lambda x: ','.join(x.astype(str))).reset_index()
    submission.columns = ['source1_entity_id', 'matched_entity_ids']
    
    # Add queries that had no matches
    all_s1 = s1_df['query_id'].to_pandas().to_frame(name='source1_entity_id')
    submission = all_s1.merge(submission, on='source1_entity_id', how='left').fillna("")
    
    out_name = f"submission_{args.split}.tsv"
    submission.to_csv(out_name, sep='\t', index=False)
    print(f"Done! Saved {out_name}")

if __name__ == "__main__":
    main()
