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
import sys
from datetime import datetime

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src")))
from business_entity_resolution.regeneration import (  # noqa: E402
    enforce_memory_budget,
    sparse_memory_projection,
    write_sparse_candidates_from_texts,
)


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
    parser.add_argument("--max-memory-fraction", type=float, default=0.75,
                        help="Refuse retrieval when projected working set exceeds this fraction of available RAM")
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
        max_df=0.01,
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

    print("Loading queries (S1)...")
    s1_df = pl.read_parquet(s1_path, columns=select_cols)
    query_ids = s1_df["entity_id"].to_list()
    query_texts = build_text_series(s1_df, args.fields)
    del s1_df
    gc.collect()

    sample_size = min(len(query_texts), args.chunk_size)
    query_sample = vectorizer.transform(query_texts[:sample_size]).tocsr()
    projection = sparse_memory_projection(
        corpus_matrix, corpus_matrix_T, query_sample, len(query_ids), args.top_k, args.chunk_size
    )
    enforce_memory_budget(projection, args.max_memory_fraction)
    del corpus_matrix, query_sample
    gc.collect()

    print(f"Computing Top-{args.top_k} in bounded chunks of {args.chunk_size}...")
    t0 = time.time()
    output_path = os.path.join(args.output_dir, f"{args.split}_char_candidates_{config_name}.parquet")
    rows_written = write_sparse_candidates_from_texts(
        output_path=output_path,
        vectorizer=vectorizer,
        query_texts=query_texts,
        corpus_matrix_t=corpus_matrix_T,
        query_ids=np.asarray(query_ids),
        corpus_ids=np.asarray(corpus_ids),
        corpus_sources=np.asarray(corpus_sources),
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        score_column="char_score",
        flag_column="found_by_char",
    )
    print(f"  Retrieval and streaming write done in {time.time()-t0:.1f}s")
    print(f"Saved {rows_written} candidates to {output_path}")
    del query_texts, corpus_matrix_T
    gc.collect()

    # Evaluate if ground truth provided
    if args.ground_truth and os.path.exists(args.ground_truth) and args.split == "train":
        print("\nEvaluating recall...")
        gt_df = load_ground_truth(args.ground_truth)
        out_df = pl.read_parquet(output_path)

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
