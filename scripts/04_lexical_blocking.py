import os
import time
import argparse
import polars as pl
from datetime import datetime
import json
import gc
import sys
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src")))

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4: Lexical Blocking (sparse_dot_topn)")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Canonical Parquet directory")
    parser.add_argument("--split", type=str, default="train", help="Which split to run")
    parser.add_argument("--output-dir", type=str, default="data/candidates", help="Output directory")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Artifacts directory")
    parser.add_argument("--top-k", type=int, default=5, help="Number of lexical candidates to retrieve")
    return parser.parse_args()

def get_fast_dot_function():
    try:
        from sparse_dot_topn import sp_matmul_topn
        def fast_dot(A, B, ntop):
            return sp_matmul_topn(A, B, top_n=ntop, n_threads=-1)
        return fast_dot
    except ImportError:
        try:
            from sparse_dot_topn import awesome_cossim_topn
            def fast_dot(A, B, ntop):
                return awesome_cossim_topn(A, B, ntop=ntop, lower_bound=0.0, use_threads=True, n_jobs=-1)
            return fast_dot
        except ImportError:
            raise ImportError("Please install sparse_dot_topn: pip install sparse_dot_topn")

def run_lexical_blocking(data_dir, split, output_dir, artifacts_dir, top_k):
    start_time = time.time()
    
    fast_dot = get_fast_dot_function()
    
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
    
    print(f"[{split}] Fitting TfidfVectorizer (max_df=0.01) to auto-drop zipfian stopwords...")
    # max_df=0.01 absolutely DESTROYS the zipfian bottleneck by mathematically removing 
    # any token appearing in >1% of documents (inc, llc, etc).
    # min_df=2 drops unique misspellings to save massive amounts of RAM.
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_df=0.01, min_df=2, dtype=np.float32)
    
    corpus_tfidf = vectorizer.fit_transform(corpus_names)
    del corpus_names
    gc.collect()
    
    corpus_tfidf_T = corpus_tfidf.T.tocsr()
    del corpus_tfidf
    gc.collect()
    
    print(f"[{split}] Loading Queries (S1)...")
    s1_df = pl.read_parquet(s1_path, columns=select_cols)
    s1_ids = s1_df["entity_id"].to_numpy()
    s1_names = s1_df["name_norm"].fill_null("").to_list()
    del s1_df
    gc.collect()
    
    n_queries = len(s1_names)
    print(f"[{split}] Transform {n_queries} queries...")
    query_tfidf = vectorizer.transform(s1_names).tocsr()
    del s1_names
    gc.collect()
    
    print(f"[{split}] Computing C++ Multi-Threaded Sparse Dot Top-K...")
    # This C++ function computes cosine similarity and keeps ONLY top-K per row instantly in C++ memory
    matches = fast_dot(query_tfidf, corpus_tfidf_T, top_k)
    
    del query_tfidf, corpus_tfidf_T
    gc.collect()
    
    print(f"[{split}] Mapping results back to entity IDs...")
    # matches is a CSR matrix of shape (n_queries, n_corpus)
    out_query_ids = []
    out_candidate_ids = []
    out_candidate_sources = []
    
    # Fast extraction from CSR
    indptr = matches.indptr
    indices = matches.indices
    
    for q_idx in range(n_queries):
        start = indptr[q_idx]
        end = indptr[q_idx+1]
        
        q_id = s1_ids[q_idx]
        
        # Get top-k indices for this query
        for idx in range(start, end):
            c_idx = indices[idx]
            out_query_ids.append(q_id)
            out_candidate_ids.append(corpus_ids[c_idx])
            out_candidate_sources.append(corpus_sources[c_idx])
            
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
    print("PHASE 4: LEXICAL BLOCKING (sparse_dot_topn)")
    print("==================================================")
    
    reports = []
    
    for split in [args.split] if args.split != "both" else ["train", "test"]:
        stats = run_lexical_blocking(args.data_dir, split, args.output_dir, args.artifacts_dir, args.top_k)
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
