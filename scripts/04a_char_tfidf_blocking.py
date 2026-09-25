"""
Phase 4A: Character TF-IDF Retrieval (Experiment B)
=====================================================
Builds char n-gram TF-IDF sparse retrievers over name and address fields.
Memory-safe: uses chunked sparse matrix multiplication, never materialises
the full NxM dense similarity matrix.

Outputs: {split}_char_candidates_{config}.parquet
Metrics: pair_recall, s1_coverage, candidate_count, incremental_recall over Dense+Exact baseline.
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


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4A: Character TF-IDF Retrieval")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=str, default="data/candidates_v2")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--ground-truth", type=str, default="", help="TSV path for recall evaluation (train only)")
    parser.add_argument("--baseline-dir", type=str, default="data/candidates", help="Dir with existing Dense+Exact candidates for incremental recall")
    parser.add_argument("--fields", type=str, default="name", choices=["name", "address", "combined"], help="Which field(s) to build char TF-IDF on")
    parser.add_argument("--ngram-min", type=int, default=3)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=50000, help="Query chunk size for matrix multiply")
    return parser.parse_args()


def load_ground_truth(gt_path: str) -> pl.DataFrame:
    gt_df = pl.read_csv(gt_path, separator="\t").rename({
        "source1_entity_id": "query_id",
        "matched_entity_ids": "candidate_id"
    })
    gt_df = gt_df.with_columns(pl.col("candidate_id").str.split(",")).explode("candidate_id")
    gt_df = gt_df.with_columns([
        pl.col("query_id").cast(pl.Utf8),
        pl.col("candidate_id").cast(pl.Utf8).str.strip_chars()
    ])
    return gt_df.select(["query_id", "candidate_id"]).unique()


def evaluate_recall(gt_df: pl.DataFrame, cand_df: pl.DataFrame, name: str, baseline_pairs: pl.DataFrame | None = None):
    truth_pairs = gt_df.select(["query_id", "candidate_id"]).unique()
    total_true = truth_pairs.height
    total_queries = truth_pairs["query_id"].n_unique()

    cand_pairs = cand_df.select(["query_id", "candidate_id"]).with_columns([
        pl.col("query_id").cast(pl.Utf8),
        pl.col("candidate_id").cast(pl.Utf8)
    ]).unique()
    total_cands = cand_pairs.height

    matches = truth_pairs.join(cand_pairs, on=["query_id", "candidate_id"], how="inner")
    matched = matches.height
    covered_queries = matches["query_id"].n_unique()

    pair_recall = matched / total_true if total_true > 0 else 0
    s1_coverage = covered_queries / total_queries if total_queries > 0 else 0

    incremental = None
    if baseline_pairs is not None:
        # True pairs found by THIS retriever but NOT in existing baseline
        new_pairs = cand_pairs.join(baseline_pairs.select(["query_id", "candidate_id"]), on=["query_id", "candidate_id"], how="anti")
        new_true = truth_pairs.join(new_pairs, on=["query_id", "candidate_id"], how="inner").height
        incremental = {
            "new_true_pairs_vs_baseline": new_true,
            "pct_of_misses_recovered": round(new_true / max(1, total_true - baseline_pairs.join(truth_pairs, on=["query_id","candidate_id"], how="inner").height) * 100, 2)
        }

    result = {
        "retriever": name,
        "total_true_pairs": total_true,
        "retrieved_true_pairs": matched,
        "missed_true_pairs": total_true - matched,
        "pair_recall_pct": round(pair_recall * 100, 4),
        "s1_coverage_pct": round(s1_coverage * 100, 4),
        "total_candidates": total_cands,
        "candidates_per_query": round(total_cands / total_queries, 2) if total_queries > 0 else 0,
    }
    if incremental:
        result.update(incremental)

    print(f"\n--- {name} ---")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


def build_text_series(df: pl.DataFrame, fields: str) -> list[str]:
    if fields == "name":
        return df["name_norm"].fill_null("").to_list()
    elif fields == "address":
        return df["address_norm"].fill_null("").to_list()
    else:  # combined
        return df.select(
            pl.concat_str([
                pl.col("name_norm").fill_null(""),
                pl.col("address_norm").fill_null("")
            ], separator=" ")
        ).to_series().to_list()


def chunked_topk_sparse(query_matrix: sp.csr_matrix, corpus_matrix_T: sp.csr_matrix,
                         top_k: int, chunk_size: int):
    """
    Memory-safe Top-K retrieval via chunked sparse dot product.
    Never materialises the full NxM matrix.
    Returns: (row_indices, col_indices, scores)
    """
    n_queries = query_matrix.shape[0]
    all_rows, all_cols, all_scores = [], [], []

    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)
        q_chunk = query_matrix[start:end]  # (chunk_size, vocab)

        # Sparse dot: (chunk_size, vocab) x (vocab, n_corpus) -> (chunk_size, n_corpus)
        # But we must avoid full dense materialisation.
        # Use scipy.sparse matmul and immediately extract top-K per row.
        sim = q_chunk.dot(corpus_matrix_T)  # sparse result

        # Convert to dense only for top-K selection (chunk_size x n_corpus is manageable per chunk)
        if sp.issparse(sim):
            sim_dense = sim.toarray()
        else:
            sim_dense = np.asarray(sim)

        # argpartition is O(n) per row, much faster than full sort
        n_corpus = sim_dense.shape[1]
        k = min(top_k, n_corpus)
        top_indices = np.argpartition(sim_dense, -k, axis=1)[:, -k:]
        top_scores = np.take_along_axis(sim_dense, top_indices, axis=1)

        # Filter out zero-score matches
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

    return np.array(all_rows, dtype=np.int32), np.array(all_cols, dtype=np.int32), np.array(all_scores, dtype=np.float32)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    config_name = f"{args.fields}_char{args.ngram_min}{args.ngram_max}_K{args.top_k}"
    print("=" * 60)
    print(f" PHASE 4A: CHAR TF-IDF — {config_name.upper()}")
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

    print(f"Building char TF-IDF ({args.ngram_min},{args.ngram_max})-gram vectorizer on {len(corpus_texts)} corpus docs...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(args.ngram_min, args.ngram_max),
        max_df=0.95,
        min_df=2,
        dtype=np.float32,
        sublinear_tf=True
    )
    corpus_matrix = vectorizer.fit_transform(corpus_texts)
    print(f"  Corpus matrix: {corpus_matrix.shape}, nnz={corpus_matrix.nnz}, time={time.time()-t0:.1f}s")
    del corpus_texts
    gc.collect()

    # Transpose corpus matrix for efficient query dot product
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

    # Map indices to entity IDs
    corpus_ids_np = np.array(corpus_ids)
    corpus_sources_np = np.array(corpus_sources)
    query_ids_np = np.array(query_ids)

    out_df = pl.DataFrame({
        "query_id": query_ids_np[row_idxs],
        "candidate_id": corpus_ids_np[col_idxs],
        "candidate_source": corpus_sources_np[col_idxs],
        "char_score": scores,
        "found_by_char": True
    })

    # Remove self-matches
    out_df = out_df.filter(pl.col("query_id") != pl.col("candidate_id"))

    output_path = os.path.join(args.output_dir, f"{args.split}_char_candidates_{config_name}.parquet")
    out_df.write_parquet(output_path, compression="snappy")
    print(f"Saved {out_df.height} candidates to {output_path}")

    # Evaluate if ground truth provided
    if args.ground_truth and os.path.exists(args.ground_truth) and args.split == "train":
        print("\nEvaluating recall...")
        gt_df = load_ground_truth(args.ground_truth)

        # Load baseline for incremental metric
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

        report_path = os.path.join(args.artifacts_dir, f"char_recall_{config_name}.json")
        with open(report_path, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "config": config_name, "metrics": metrics}, f, indent=2)
        print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
