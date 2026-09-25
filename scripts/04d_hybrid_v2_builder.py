"""
Phase 4D: Missed Pair Analysis (Experiment E) + Hybrid V2 Builder
==================================================================
Two tasks in one script:

TASK 1 — Missed-Pair Analysis:
  For every true validation pair missed by Dense+Exact baseline, determine
  which new retrievers (Char TF-IDF, BM25, Structured) can recover it.
  Outputs a coverage summary and Pareto-style comparison table.

TASK 2 — Hybrid V2 Builder:
  Merges all retrievers into one deduplicated candidate set with full
  retrieval metadata columns for downstream LightGBM features.

  Output schema:
    query_id, candidate_id, candidate_source,
    found_by_dense, found_by_exact,
    found_by_char, found_by_bm25, found_by_block,
    dense_rank, dense_score,
    char_score, bm25_score,
    num_retrievers
"""
import os
import time
import argparse
import gc
import json
from datetime import datetime

import polars as pl
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4D: Missed Pair Analysis + Hybrid V2 Builder")
    parser.add_argument("--candidates-dir", type=str, default="data/candidates", help="Baseline Dense+Exact directory")
    parser.add_argument("--candidates-v2-dir", type=str, default="data/candidates_v2", help="New retriever outputs directory")
    parser.add_argument("--output-dir", type=str, default="data/candidates_v2", help="Where to write Hybrid V2")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--ground-truth", type=str, default="", help="TSV path for validation evaluation")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dense-k", type=int, default=50, help="K of the dense candidate file to use as baseline")
    # Config names that match what 04a/04b generated
    parser.add_argument("--char-config", type=str, default="name_char35_K50", help="Char TF-IDF config suffix")
    parser.add_argument("--bm25-config", type=str, default="name_word_K50", help="BM25 config suffix")
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


def evaluate_recall_df(gt_pairs: pl.DataFrame, cand_pairs: pl.DataFrame, name: str) -> dict:
    total_true = gt_pairs.height
    total_queries = gt_pairs["query_id"].n_unique()

    cand_u = cand_pairs.select(["query_id", "candidate_id"]).with_columns([
        pl.col("query_id").cast(pl.Utf8),
        pl.col("candidate_id").cast(pl.Utf8)
    ]).unique()

    matched = gt_pairs.join(cand_u, on=["query_id", "candidate_id"], how="inner")
    n_matched = matched.height
    n_covered = matched["query_id"].n_unique()

    result = {
        "retriever": name,
        "pair_recall_pct": round(n_matched / total_true * 100, 4) if total_true > 0 else 0,
        "s1_coverage_pct": round(n_covered / total_queries * 100, 4) if total_queries > 0 else 0,
        "retrieved_true_pairs": n_matched,
        "missed_true_pairs": total_true - n_matched,
        "total_candidates": cand_u.height,
        "candidates_per_query": round(cand_u.height / total_queries, 2) if total_queries > 0 else 0,
    }
    return result


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    sp = args.split
    print("=" * 60)
    print(f" PHASE 4D: MISSED PAIR ANALYSIS + HYBRID V2 ({sp.upper()})")
    print("=" * 60)

    # ── File paths ─────────────────────────────────────────────────────────
    dense_path   = os.path.join(args.candidates_dir, f"{sp}_dense_candidates_K{args.dense_k}.parquet")
    exact_path   = os.path.join(args.candidates_dir, f"{sp}_exact_candidates.parquet")
    char_path    = os.path.join(args.candidates_v2_dir, f"{sp}_char_candidates_{args.char_config}.parquet")
    bm25_path    = os.path.join(args.candidates_v2_dir, f"{sp}_bm25_candidates_{args.bm25_config}.parquet")
    struct_path  = os.path.join(args.candidates_v2_dir, f"{sp}_structured_candidates.parquet")

    # ── Load all retrievers lazily ─────────────────────────────────────────
    def safe_load(path, cols=None):
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping.")
            return None
        lf = pl.scan_parquet(path)
        if cols:
            lf = lf.select([c for c in cols if c in pl.scan_parquet(path).columns])
        return lf

    dense_lf   = safe_load(dense_path)
    exact_lf   = safe_load(exact_path)
    char_lf    = safe_load(char_path)
    bm25_lf    = safe_load(bm25_path)
    struct_lf  = safe_load(struct_path)

    # ── TASK 1: Missed Pair Analysis ────────────────────────────────────────
    if args.ground_truth and os.path.exists(args.ground_truth):
        print("\n[TASK 1] Missed Pair Analysis")
        gt_df = load_ground_truth(args.ground_truth)
        total_true = gt_df.height

        # Baseline pairs (Dense + Exact)
        baseline_parts = []
        if dense_lf is not None:
            baseline_parts.append(dense_lf.select(["query_id","candidate_id"]))
        if exact_lf is not None:
            baseline_parts.append(exact_lf.select(["query_id","candidate_id"]))

        if baseline_parts:
            baseline_pairs = (pl.concat(baseline_parts)
                .unique(subset=["query_id","candidate_id"])
                .collect()
                .with_columns([pl.col("query_id").cast(pl.Utf8), pl.col("candidate_id").cast(pl.Utf8)])
            )
        else:
            print("No baseline candidates found.")
            return

        # Missed pairs = true pairs NOT in baseline
        missed_pairs = gt_df.join(
            baseline_pairs.select(["query_id","candidate_id"]),
            on=["query_id","candidate_id"], how="anti"
        )
        n_missed = missed_pairs.height
        print(f"\n  Total true pairs:          {total_true}")
        print(f"  Retrieved by baseline:     {total_true - n_missed}")
        print(f"  Missed by baseline:        {n_missed} ({n_missed/total_true*100:.2f}%)")

        coverage = {"total_missed": n_missed}
        retriever_dfs = {"char": char_lf, "bm25": bm25_lf, "block": struct_lf}

        for ret_name, ret_lf in retriever_dfs.items():
            if ret_lf is None:
                coverage[ret_name] = "NOT RUN"
                continue
            ret_df = ret_lf.select(["query_id","candidate_id"]).collect().with_columns([
                pl.col("query_id").cast(pl.Utf8),
                pl.col("candidate_id").cast(pl.Utf8)
            ])
            # How many of the missed pairs does this retriever recover?
            recovered = missed_pairs.join(ret_df, on=["query_id","candidate_id"], how="inner").height
            coverage[f"{ret_name}_recovers_of_missed"] = recovered
            coverage[f"{ret_name}_pct_of_missed"] = round(recovered / max(1, n_missed) * 100, 2)
            del ret_df
            gc.collect()

        print("\n  Coverage of Missed Pairs:")
        for k, v in coverage.items():
            print(f"    {k}: {v}")

        coverage["timestamp"] = datetime.now().isoformat()
        with open(os.path.join(args.artifacts_dir, "missed_pair_analysis.json"), "w") as f:
            json.dump(coverage, f, indent=2)

    # ── TASK 2: Build Hybrid V2 ─────────────────────────────────────────────
    print("\n[TASK 2] Building Hybrid V2 candidate set")

    # Build each retriever's contribution with metadata flags
    frames = []

    if dense_lf is not None:
        print("  Loading Dense...")
        dense_cols = pl.scan_parquet(dense_path).columns
        select_dense = ["query_id", "candidate_id", "candidate_source"]
        if "dense_rank" in dense_cols: select_dense.append("dense_rank")
        if "dense_score" in dense_cols: select_dense.append("dense_score")
        dense_df = (dense_lf.select(select_dense).collect()
            .with_columns([
                pl.col("query_id").cast(pl.Utf8),
                pl.col("candidate_id").cast(pl.Utf8),
                pl.lit(True).alias("found_by_dense"),
                pl.lit(False).alias("found_by_exact"),
                pl.lit(False).alias("found_by_char"),
                pl.lit(False).alias("found_by_bm25"),
                pl.lit(False).alias("found_by_block"),
            ])
        )
        if "dense_rank" not in dense_df.columns:
            dense_df = dense_df.with_columns(pl.lit(None).cast(pl.Int32).alias("dense_rank"))
        if "dense_score" not in dense_df.columns:
            dense_df = dense_df.with_columns(pl.lit(None).cast(pl.Float32).alias("dense_score"))
        dense_df = dense_df.with_columns([
            pl.lit(None).cast(pl.Float32).alias("char_score"),
            pl.lit(None).cast(pl.Float32).alias("bm25_score"),
        ])
        frames.append(dense_df)
        del dense_df
        gc.collect()

    if exact_lf is not None:
        print("  Loading Exact...")
        exact_df = (exact_lf.select(["query_id", "candidate_id", "candidate_source"]).collect()
            .with_columns([
                pl.col("query_id").cast(pl.Utf8),
                pl.col("candidate_id").cast(pl.Utf8),
                pl.lit(False).alias("found_by_dense"),
                pl.lit(True).alias("found_by_exact"),
                pl.lit(False).alias("found_by_char"),
                pl.lit(False).alias("found_by_bm25"),
                pl.lit(False).alias("found_by_block"),
                pl.lit(None).cast(pl.Int32).alias("dense_rank"),
                pl.lit(None).cast(pl.Float32).alias("dense_score"),
                pl.lit(None).cast(pl.Float32).alias("char_score"),
                pl.lit(None).cast(pl.Float32).alias("bm25_score"),
            ])
        )
        frames.append(exact_df)
        del exact_df
        gc.collect()

    if char_lf is not None:
        print("  Loading Char TF-IDF...")
        char_df = (char_lf.select(["query_id", "candidate_id", "candidate_source", "char_score"]).collect()
            .with_columns([
                pl.col("query_id").cast(pl.Utf8),
                pl.col("candidate_id").cast(pl.Utf8),
                pl.lit(False).alias("found_by_dense"),
                pl.lit(False).alias("found_by_exact"),
                pl.lit(True).alias("found_by_char"),
                pl.lit(False).alias("found_by_bm25"),
                pl.lit(False).alias("found_by_block"),
                pl.lit(None).cast(pl.Int32).alias("dense_rank"),
                pl.lit(None).cast(pl.Float32).alias("dense_score"),
                pl.lit(None).cast(pl.Float32).alias("bm25_score"),
            ])
        )
        frames.append(char_df)
        del char_df
        gc.collect()

    if bm25_lf is not None:
        print("  Loading BM25...")
        bm25_df = (bm25_lf.select(["query_id", "candidate_id", "candidate_source", "bm25_score"]).collect()
            .with_columns([
                pl.col("query_id").cast(pl.Utf8),
                pl.col("candidate_id").cast(pl.Utf8),
                pl.lit(False).alias("found_by_dense"),
                pl.lit(False).alias("found_by_exact"),
                pl.lit(False).alias("found_by_char"),
                pl.lit(True).alias("found_by_bm25"),
                pl.lit(False).alias("found_by_block"),
                pl.lit(None).cast(pl.Int32).alias("dense_rank"),
                pl.lit(None).cast(pl.Float32).alias("dense_score"),
                pl.lit(None).cast(pl.Float32).alias("char_score"),
            ])
        )
        frames.append(bm25_df)
        del bm25_df
        gc.collect()

    if struct_lf is not None:
        print("  Loading Structured...")
        struct_df = (struct_lf.select(["query_id", "candidate_id", "candidate_source"]).collect()
            .with_columns([
                pl.col("query_id").cast(pl.Utf8),
                pl.col("candidate_id").cast(pl.Utf8),
                pl.lit(False).alias("found_by_dense"),
                pl.lit(False).alias("found_by_exact"),
                pl.lit(False).alias("found_by_char"),
                pl.lit(False).alias("found_by_bm25"),
                pl.lit(True).alias("found_by_block"),
                pl.lit(None).cast(pl.Int32).alias("dense_rank"),
                pl.lit(None).cast(pl.Float32).alias("dense_score"),
                pl.lit(None).cast(pl.Float32).alias("char_score"),
                pl.lit(None).cast(pl.Float32).alias("bm25_score"),
            ])
        )
        frames.append(struct_df)
        del struct_df
        gc.collect()

    if not frames:
        print("ERROR: No candidate files found.")
        return

    print("\n  Concatenating all retrievers...")
    t0 = time.time()
    combined = pl.concat(frames, how="diagonal_relaxed")
    del frames
    gc.collect()
    print(f"  Combined: {combined.height} rows ({time.time()-t0:.1f}s)")

    print("  Merging metadata per (query_id, candidate_id)...")
    t0 = time.time()
    # Aggregate flags and scores per pair using max (True dominates False, max score wins)
    hybrid_v2 = combined.group_by(["query_id", "candidate_id", "candidate_source"]).agg([
        pl.col("found_by_dense").any(),
        pl.col("found_by_exact").any(),
        pl.col("found_by_char").any(),
        pl.col("found_by_bm25").any(),
        pl.col("found_by_block").any(),
        pl.col("dense_rank").min().alias("dense_rank"),       # best rank
        pl.col("dense_score").max().alias("dense_score"),
        pl.col("char_score").max().alias("char_score"),
        pl.col("bm25_score").max().alias("bm25_score"),
    ])

    # Add num_retrievers column
    hybrid_v2 = hybrid_v2.with_columns(
        (pl.col("found_by_dense").cast(pl.Int8) +
         pl.col("found_by_exact").cast(pl.Int8) +
         pl.col("found_by_char").cast(pl.Int8) +
         pl.col("found_by_bm25").cast(pl.Int8) +
         pl.col("found_by_block").cast(pl.Int8)).alias("num_retrievers")
    )

    # Remove self-matches
    hybrid_v2 = hybrid_v2.filter(pl.col("query_id") != pl.col("candidate_id"))

    print(f"  Hybrid V2: {hybrid_v2.height} unique pairs ({time.time()-t0:.1f}s)")

    output_path = os.path.join(args.output_dir, f"{sp}_hybrid_v2_candidates.parquet")
    hybrid_v2.write_parquet(output_path, compression="snappy")
    print(f"\nSaved Hybrid V2 to: {output_path}")

    # ── Final evaluation ────────────────────────────────────────────────────
    if args.ground_truth and os.path.exists(args.ground_truth):
        print("\n  Evaluating Hybrid V2 recall...")
        gt_df = load_ground_truth(args.ground_truth)
        result = evaluate_recall_df(gt_df, hybrid_v2, "Hybrid_V2")

        print("\n  === PARETO COMPARISON ===")
        print(f"  {'Retriever':<25} {'Recall':>8} {'Coverage':>10} {'Candidates':>12} {'C/Query':>8}")
        print(f"  {'Baseline (Dense+Exact)':<25} {'80.52%':>8} {'91.17%':>10} {'~128M':>12} {'~58':>8}")
        print(f"  {'Hybrid V2':<25} {result['pair_recall_pct']:>7}% {result['s1_coverage_pct']:>9}% {result['total_candidates']:>12} {result['candidates_per_query']:>8}")

        result["timestamp"] = datetime.now().isoformat()
        with open(os.path.join(args.artifacts_dir, "hybrid_v2_recall.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Recall report saved.")


if __name__ == "__main__":
    main()
