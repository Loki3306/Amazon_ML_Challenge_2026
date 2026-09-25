import os
import time
import argparse
import polars as pl
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from datetime import datetime
import json
import gc
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src")))

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4: Lexical Blocking (TF-IDF Top-K)")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Canonical Parquet directory")
    parser.add_argument("--split", type=str, default="train", help="Which split to run")
    parser.add_argument("--output-dir", type=str, default="data/candidates", help="Output directory")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Artifacts directory")
    parser.add_argument("--top-k", type=int, default=5, help="Number of lexical candidates to retrieve per query")
    parser.add_argument("--batch-size", type=int, default=50000, help="Query batch size for sparse dot product")
    return parser.parse_args()

def get_top_k_sparse(sparse_matrix, k):
    """
    Given a sparse matrix of scores (queries x corpus),
    returns the top K indices and scores for each query.
    Optimized for CSR matrices.
    """
    # Convert to CSR if not already, for row slicing
    sparse_matrix = sparse_matrix.tocsr()
    
    n_queries = sparse_matrix.shape[0]
    top_k_indices = np.zeros((n_queries, k), dtype=np.int32)
    
    for i in range(n_queries):
        row = sparse_matrix[i]
        if row.nnz == 0:
            top_k_indices[i] = -1  # No candidates
            continue
            
        # Get data and column indices
        data = row.data
        indices = row.indices
        
        # If fewer non-zeros than k, pad with -1
        if len(data) <= k:
            sorted_idx = np.argsort(-data)
            top_k_indices[i, :len(data)] = indices[sorted_idx]
            if len(data) < k:
                top_k_indices[i, len(data):] = -1
        else:
            # argpartition to get top K efficiently, then sort just those K
            part_idx = np.argpartition(-data, k - 1)[:k]
            # sort the top k
            sorted_part = part_idx[np.argsort(-data[part_idx])]
            top_k_indices[i] = indices[sorted_part]
            
    return top_k_indices

from joblib import Parallel, delayed

def process_batch(start_idx, end_idx, batch_names, batch_ids, vectorizer, tfidf, corpus_tfidf_T, top_k):
    # Transform queries
    batch_hash = vectorizer.transform(batch_names)
    batch_tfidf = tfidf.transform(batch_hash)
    
    # Dot product: (B, V) dot (V, C) -> (B, C)
    scores = batch_tfidf.dot(corpus_tfidf_T)
    
    # Get Top K indices
    top_k_idx = get_top_k_sparse(scores, top_k)
    
    out_q = []
    out_c_idx = []
    
    for i in range(len(batch_ids)):
        q_id = batch_ids[i]
        for rank in range(top_k):
            c_idx = top_k_idx[i, rank]
            if c_idx != -1:
                out_q.append(q_id)
                out_c_idx.append(c_idx)
                
    return out_q, out_c_idx

def run_lexical_blocking(data_dir, split, output_dir, artifacts_dir, top_k, batch_size):
    start_time = time.time()
    
    s1_path = os.path.join(data_dir, split, f"{split}_source1.parquet")
    s2_path = os.path.join(data_dir, split, f"{split}_source2.parquet")
    s3_path = os.path.join(data_dir, split, f"{split}_source3.parquet")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    print(f"[{split}] Loading Corpus (S2 + S3)...")
    select_cols = ["entity_id", "source", "name_norm"]
    
    s2_df = pl.read_parquet(s2_path, columns=select_cols)
    s3_df = pl.read_parquet(s3_path, columns=select_cols)
    corpus_df = pl.concat([s2_df, s3_df])
    
    corpus_ids = corpus_df["entity_id"].to_numpy()
    corpus_sources = corpus_df["source"].to_numpy()
    corpus_names = corpus_df["name_norm"].fill_null("").to_list()
    
    del s2_df, s3_df, corpus_df
    gc.collect()
    
    print(f"[{split}] Building Lexical Vectorizer (Hashing TF-IDF)...")
    vectorizer = HashingVectorizer(n_features=2**21, analyzer="word", ngram_range=(1, 2), lowercase=False)
    tfidf = TfidfTransformer()
    
    corpus_hash = vectorizer.transform(corpus_names)
    corpus_tfidf = tfidf.fit_transform(corpus_hash)
    
    del corpus_names, corpus_hash
    gc.collect()
    
    corpus_tfidf_T = corpus_tfidf.T.tocsr()
    
    print(f"[{split}] Loading Queries (S1)...")
    s1_df = pl.read_parquet(s1_path, columns=select_cols)
    s1_ids = s1_df["entity_id"].to_numpy()
    s1_names = s1_df["name_norm"].fill_null("").to_list()
    
    del s1_df
    gc.collect()
    
    n_queries = len(s1_names)
    
    print(f"[{split}] Processing {n_queries} queries in parallel across all CPU cores...")
    
    # Create batch arguments
    batches = []
    for start_idx in range(0, n_queries, batch_size):
        end_idx = min(start_idx + batch_size, n_queries)
        batches.append((
            start_idx, end_idx, 
            s1_names[start_idx:end_idx], 
            s1_ids[start_idx:end_idx]
        ))
        
    # Run in parallel to maximize CPU cores
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_batch)(
            b[0], b[1], b[2], b[3], 
            vectorizer, tfidf, corpus_tfidf_T, top_k
        ) for b in batches
    )
    
    print(f"[{split}] Aggregating results...")
    out_query_ids = []
    out_candidate_ids = []
    out_candidate_sources = []
    
    for res_q, res_c_idx in results:
        out_query_ids.extend(res_q)
        for idx in res_c_idx:
            out_candidate_ids.append(corpus_ids[idx])
            out_candidate_sources.append(corpus_sources[idx])
            
    print(f"[{split}] Saving candidates to Parquet...")
    candidates_df = pl.DataFrame({
        "query_id": out_query_ids,
        "candidate_id": out_candidate_ids,
        "candidate_source": out_candidate_sources,
        "match_lexical": True
    })
    
    output_path = os.path.join(output_dir, f"{split}_lexical_candidates.parquet")
    candidates_df.write_parquet(output_path, compression="snappy")
    
    processing_time = time.time() - start_time
    total_pairs = len(out_query_ids)
    
    stats = {
        "split": split,
        "total_candidate_pairs": total_pairs,
        "total_s1_queries": n_queries,
        "avg_candidates_per_query": round(total_pairs / n_queries, 2) if n_queries > 0 else 0,
        "processing_time_sec": round(processing_time, 2),
        "output_size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 2)
    }
    
    print(f"[{split}] DONE.")
    print(json.dumps(stats, indent=2))
    return stats

def main():
    args = parse_args()
    print("==================================================")
    print("PHASE 4: LEXICAL BLOCKING (TF-IDF TOP-K)")
    print("==================================================")
    
    reports = []
    
    for split in [args.split] if args.split != "both" else ["train", "test"]:
        stats = run_lexical_blocking(args.data_dir, split, args.output_dir, args.artifacts_dir, args.top_k, args.batch_size)
        if stats:
            reports.append(stats)
            
    if reports:
        report_path = os.path.join(args.artifacts_dir, f"lexical_blocking_report_{args.split}.json")
        with open(report_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "reports": reports
            }, f, indent=2)
        print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    main()
