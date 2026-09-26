"""
Phase 4B: BM25 / Word-Level TF-IDF Retrieval (Experiment C)
=============================================================
Implements efficient lexical retrieval for name, address, and combined fields.
Uses word-level TF-IDF with chunked sparse matrix multiplication.
Never materialises N×M dense matrix.

Outputs: {split}_bm25_candidates_{config}.parquet
Metrics: standalone recall, incremental recall over Dense+Exact, overlap analysis.
"""
import os
import time
import argparse
import gc
import json
from datetime import datetime

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp


def get_topk_fn(top_k: int):
    """Returns the fastest available top-K sparse matmul function."""
    try:
        from sparse_dot_topn import sp_matmul_topn
        print(f"  [sparse_dot_topn] Using C++ extension for Top-{top_k} retrieval (fastest path)")
        def fn(A, B_T):
            return sp_matmul_topn(A, B_T, top_n=top_k, n_threads=-1, threshold=0.0)
        return fn, True
    except ImportError:
        try:
            from sparse_dot_topn import awesome_cossim_topn
            print(f"  [sparse_dot_topn legacy] Using C++ extension")
            def fn(A, B_T):
                return awesome_cossim_topn(A, B_T, ntop=top_k, lower_bound=0.0, use_threads=True, n_jobs=4)
            return fn, True
        except ImportError:
            print(f"  [scipy fallback] sparse_dot_topn not found, using chunked scipy (slower)")
            return None, False


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4B: BM25/Word TF-IDF Retrieval")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=str, default="data/candidates_v2")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--ground-truth", type=str, default="")
    parser.add_argument("--baseline-dir", type=str, default="data/candidates")
    parser.add_argument("--fields", type=str, default="name", choices=["name", "address", "combined"])
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=50000)
    # BM25 approximation: sublinear_tf + max_df is a strong BM25 approximation
    parser.add_argument("--max-df", type=float, default=0.001, help="Drop terms in >X% of docs (removes stopwords)")
    parser.add_argument("--min-df", type=int, default=2)
    return parser.parse_args()


def load_ground_truth(gt_path: str) -> pl.DataFrame:
    gt_df = pl.read_csv(gt_path, separator="\t").rename({
        "source1_entity_id": "query_id",
        "matched_entity_ids": "candidate_id"
    })
    gt_df = gt_df.with_columns(pl.col("candidate_id").str.split(",")).explode("candidate_id")
    return gt_df.with_columns([
        pl.col("query_id").cast(pl.Utf8),
        pl.col("candidate_id").cast(pl.Utf8).str.strip_chars()
    ]).select(["query_id", "candidate_id"]).unique()


def evaluate_recall(gt_df: pl.DataFrame, cand_df: pl.DataFrame, name: str, baseline_pairs=None):
    truth_pairs = gt_df.select(["query_id", "candidate_id"]).unique()
    total_true = truth_pairs.height
    total_queries = truth_pairs["query_id"].n_unique()

    cand_pairs = cand_df.select(["query_id", "candidate_id"]).with_columns([
        pl.col("query_id").cast(pl.Utf8),
        pl.col("candidate_id").cast(pl.Utf8)
    ]).unique()

    matches = truth_pairs.join(cand_pairs, on=["query_id", "candidate_id"], how="inner")
    matched = matches.height
    covered_queries = matches["query_id"].n_unique()

    result = {
        "retriever": name,
        "total_true_pairs": total_true,
        "retrieved_true_pairs": matched,
        "missed_true_pairs": total_true - matched,
        "pair_recall_pct": round(matched / total_true * 100, 4) if total_true > 0 else 0,
        "s1_coverage_pct": round(covered_queries / total_queries * 100, 4) if total_queries > 0 else 0,
        "total_candidates": cand_pairs.height,
        "candidates_per_query": round(cand_pairs.height / total_queries, 2) if total_queries > 0 else 0,
    }

    if baseline_pairs is not None:
        baseline_matched = truth_pairs.join(
            baseline_pairs.select(["query_id","candidate_id"]).with_columns([
                pl.col("query_id").cast(pl.Utf8), pl.col("candidate_id").cast(pl.Utf8)
            ]).unique(), on=["query_id", "candidate_id"], how="inner"
        ).height
        new_pairs = cand_pairs.join(
            baseline_pairs.select(["query_id","candidate_id"]).with_columns([
                pl.col("query_id").cast(pl.Utf8), pl.col("candidate_id").cast(pl.Utf8)
            ]),
            on=["query_id", "candidate_id"], how="anti"
        )
        new_true = truth_pairs.join(new_pairs, on=["query_id", "candidate_id"], how="inner").height
        misses = total_true - baseline_matched
        result["new_true_pairs_vs_baseline"] = new_true
        result["pct_of_misses_recovered"] = round(new_true / max(1, misses) * 100, 2)

    print(f"\n--- {name} ---")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


def build_text_series(df: pl.DataFrame, fields: str) -> list[str]:
    if fields == "name":
        return df["name_norm"].fill_null("").to_list()
    elif fields == "address":
        return df["address_norm"].fill_null("").to_list()
    else:
        return df.select(
            pl.concat_str([
                pl.col("name_norm").fill_null(""),
                pl.col("address_norm").fill_null("")
            ], separator=" ")
        ).to_series().to_list()


def chunked_topk_sparse(query_matrix: sp.csr_matrix, corpus_matrix_T: sp.csr_matrix,
                          top_k: int, chunk_size: int):
    topk_fn, use_fast = get_topk_fn(top_k)
    n_queries = query_matrix.shape[0]
    all_rows, all_cols, all_scores = [], [], []

    if use_fast:
        import time
        t_start = time.time()
        for start in range(0, n_queries, chunk_size):
            t_chunk = time.time()
            end = min(start + chunk_size, n_queries)
            q_chunk = query_matrix[start:end]
            result = topk_fn(q_chunk, corpus_matrix_T)
            cx = result.tocoo()
            
            all_rows.extend((cx.row + start).tolist())
            all_cols.extend(cx.col.tolist())
            all_scores.extend(cx.data.tolist())
            
            elapsed = time.time() - t_start
            throughput = end / elapsed if elapsed > 0 else 0
            eta_s = (n_queries - end) / throughput if throughput > 0 else 0
            
            from datetime import timedelta
            eta_str = str(timedelta(seconds=int(eta_s)))
            print(f"  [C++ Fast Path] {end}/{n_queries} queries | {throughput:.1f} q/s | ETA: {eta_str}")
        return np.array(all_rows, dtype=np.int32), np.array(all_cols, dtype=np.int32), np.array(all_scores, dtype=np.float32)

    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)
        q_chunk = query_matrix[start:end]
        sim = q_chunk.dot(corpus_matrix_T)

        if sp.issparse(sim):
            sim_dense = sim.toarray()
        else:
            sim_dense = np.asarray(sim)

        n_corpus = sim_dense.shape[1]
        k = min(top_k, n_corpus)
        top_indices = np.argpartition(sim_dense, -k, axis=1)[:, -k:]
        top_scores = np.take_along_axis(sim_dense, top_indices, axis=1)

        for local_row in range(end - start):
            global_row = start + local_row
            valid_mask = top_scores[local_row] > 0
            cols = top_indices[local_row][valid_mask]
            scrs = top_scores[local_row][valid_mask]
            if len(cols) > 0:
                all_rows.extend([global_row] * len(cols))
                all_cols.extend(cols.tolist())
                all_scores.extend(scrs.tolist())

        if (start // chunk_size) % 10 == 0:
            print(f"  Chunk {start}/{n_queries} done")

        del sim, sim_dense
        gc.collect()

    return (np.array(all_rows, dtype=np.int32),
            np.array(all_cols, dtype=np.int32),
            np.array(all_scores, dtype=np.float32))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    config_name = f"{args.fields}_word_K{args.top_k}"
    print("=" * 60)
    print(f" PHASE 4B: BM25/WORD TF-IDF — {config_name.upper()}")
    print("=" * 60)

    s1_path = os.path.join(args.data_dir, args.split, f"{args.split}_source1.parquet")
    s2_path = os.path.join(args.data_dir, args.split, f"{args.split}_source2.parquet")
    s3_path = os.path.join(args.data_dir, args.split, f"{args.split}_source3.parquet")

    select_cols = ["entity_id", "source", "name_norm", "address_norm"]
    print("Loading corpus (S2+S3)...")
    s2_df = pl.read_parquet(s2_path, columns=select_cols)
    s3_df = pl.read_parquet(s3_path, columns=select_cols)
    corpus_df = pl.concat([s2_df, s3_df])
    del s2_df, s3_df
    gc.collect()

    corpus_ids = corpus_df["entity_id"].to_list()
    corpus_sources = corpus_df["source"].to_list()
    corpus_texts = build_text_series(corpus_df, args.fields)
    del corpus_df
    gc.collect()

    print(f"Building word TF-IDF (max_df={args.max_df}, min_df={args.min_df}) on {len(corpus_texts)} docs...")
    t0 = time.time()
    # sublinear_tf=True is the key BM25 approximation: replaces tf with 1+log(tf)
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_df=args.max_df,
        min_df=args.min_df,
        dtype=np.float32,
        sublinear_tf=True
    )
    corpus_matrix = vectorizer.fit_transform(corpus_texts)
    print(f"  Corpus matrix: {corpus_matrix.shape}, nnz={corpus_matrix.nnz}, time={time.time()-t0:.1f}s")
    del corpus_texts
    gc.collect()

    corpus_matrix_T = corpus_matrix.T.tocsr()
    del corpus_matrix
    gc.collect()

    print("Loading queries (S1)...")
    s1_df = pl.read_parquet(s1_path, columns=select_cols)
    query_ids = s1_df["entity_id"].to_list()
    query_texts = build_text_series(s1_df, args.fields)
    del s1_df
    gc.collect()

    print(f"Transforming {len(query_texts)} queries...")
    t0 = time.time()
    query_matrix = vectorizer.transform(query_texts).tocsr()
    print(f"  Query matrix: {query_matrix.shape}, time={time.time()-t0:.1f}s")
    del query_texts
    gc.collect()

    print(f"Computing Top-{args.top_k} in chunks of {args.chunk_size}...")
    t0 = time.time()
    row_idxs, col_idxs, scores = chunked_topk_sparse(query_matrix, corpus_matrix_T, args.top_k, args.chunk_size)
    print(f"  Retrieval done in {time.time()-t0:.1f}s, {len(row_idxs)} raw pairs")
    del query_matrix, corpus_matrix_T
    gc.collect()

    print("Building raw results dataframe...")
    out_df = pl.DataFrame({
        "q_idx": row_idxs,
        "c_idx": col_idxs,
        "bm25_score": scores,
        "found_by_bm25": True
    })
    del row_idxs, col_idxs, scores
    gc.collect()
    
    print("Joining query IDs...")
    q_df = pl.DataFrame({
        "q_idx": np.arange(len(query_ids), dtype=np.int32),
        "query_id": query_ids
    })
    out_df = out_df.join(q_df, on="q_idx", how="left").drop("q_idx")
    del q_df, query_ids
    gc.collect()

    print("Joining corpus IDs...")
    c_df = pl.DataFrame({
        "c_idx": np.arange(len(corpus_ids), dtype=np.int32),
        "candidate_id": corpus_ids,
        "candidate_source": corpus_sources
    })
    out_df = out_df.join(c_df, on="c_idx", how="left").drop("c_idx")
    del c_df, corpus_ids, corpus_sources
    gc.collect()

    out_df = out_df.filter(pl.col("query_id") != pl.col("candidate_id"))

    output_path = os.path.join(args.output_dir, f"{args.split}_bm25_candidates_{config_name}.parquet")
    out_df.write_parquet(output_path, compression="snappy")
    print(f"Saved {out_df.height} candidates to {output_path}")

    if args.ground_truth and os.path.exists(args.ground_truth) and args.split == "train":
        print("\nEvaluating recall...")
        gt_df = load_ground_truth(args.ground_truth)

        baseline_pairs = None
        dense_path = os.path.join(args.baseline_dir, "train_dense_candidates_K50.parquet")
        exact_path = os.path.join(args.baseline_dir, "train_exact_candidates.parquet")
        if os.path.exists(dense_path) and os.path.exists(exact_path):
            baseline_pairs = pl.concat([
                pl.scan_parquet(dense_path).select(["query_id", "candidate_id"]),
                pl.scan_parquet(exact_path).select(["query_id", "candidate_id"])
            ]).unique(subset=["query_id", "candidate_id"]).collect().with_columns([
                pl.col("query_id").cast(pl.Utf8),
                pl.col("candidate_id").cast(pl.Utf8)
            ])

        metrics = evaluate_recall(gt_df, out_df, config_name, baseline_pairs)
        report_path = os.path.join(args.artifacts_dir, f"bm25_recall_{config_name}.json")
        with open(report_path, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "config": config_name, "metrics": metrics}, f, indent=2)
        print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
