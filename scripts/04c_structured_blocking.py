"""
Phase 4C: Structured Blocking (Experiment D)
=============================================
Creates candidate generators using entity field structure:
  - Name token blocks (first token, prefix, token signature, rare token)
  - Address blocks (numeric signature, postal token, house number, rare address token)
  - Combined blocks (country + rare name token, country + numeric address signature)

These are candidate GENERATORS, not hard filters.
Any pair retrieved by any block becomes a candidate.

Outputs: {split}_structured_candidates.parquet
Metrics: incremental recall over Dense+Exact baseline.
"""
import os
import time
import argparse
import gc
import json
import re
from datetime import datetime
from collections import Counter

import polars as pl
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4C: Structured Blocking")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=str, default="data/candidates_v2")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--ground-truth", type=str, default="")
    parser.add_argument("--baseline-dir", type=str, default="data/candidates")
    # Rarity threshold: tokens appearing in fewer than X entities are "rare" (discriminative)
    parser.add_argument("--rare-token-max-freq", type=int, default=500,
                        help="Max corpus frequency for a token to be considered 'rare'")
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
        "candidates_per_query": round(cand_pairs.height / total_queries, 2),
    }

    if baseline_pairs is not None:
        baseline_matched = truth_pairs.join(
            baseline_pairs.select(["query_id", "candidate_id"]).with_columns([
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


def extract_numeric_signature(text: str) -> str:
    """Extract numeric tokens from address (house numbers, postal codes)."""
    if not text:
        return ""
    nums = re.findall(r'\d+', text)
    return "_".join(sorted(nums)) if nums else ""


def extract_postal_token(text: str) -> str:
    """Heuristically extract postal/PIN-like token (4-8 digit sequence)."""
    if not text:
        return ""
    match = re.search(r'\b\d{4,8}\b', text)
    return match.group(0) if match else ""


def get_first_token(text: str) -> str:
    if not text:
        return ""
    tokens = text.split()
    return tokens[0] if tokens else ""


def get_name_prefix(text: str, n: int = 4) -> str:
    """First n characters of normalized name."""
    return text[:n].strip() if text else ""


def get_token_signature(text: str) -> str:
    """Sorted, deduplicated token set — catches word reorderings."""
    if not text:
        return ""
    tokens = sorted(set(text.split()))
    return "|".join(tokens[:6])  # cap at 6 to avoid huge keys


def compute_token_frequencies(texts: list[str]) -> Counter:
    freq = Counter()
    for t in texts:
        for tok in t.split():
            freq[tok] += 1
    return freq


def get_rare_tokens(text: str, freq: Counter, max_freq: int) -> list[str]:
    """Return tokens appearing <= max_freq times in the corpus."""
    return [t for t in text.split() if 0 < freq[t] <= max_freq]


def join_on_key(s1_df: pl.DataFrame, corpus_df: pl.DataFrame,
                s1_key_col: str, corpus_key_col: str,
                block_name: str) -> pl.DataFrame:
    """Performs an inner join on a blocking key, returns (query_id, candidate_id, candidate_source, block_name)."""
    # Filter out empty keys
    s1_filt = s1_df.filter(pl.col(s1_key_col) != "").filter(pl.col(s1_key_col).is_not_null())
    corpus_filt = corpus_df.filter(pl.col(corpus_key_col) != "").filter(pl.col(corpus_key_col).is_not_null())

    if s1_filt.is_empty() or corpus_filt.is_empty():
        return pl.DataFrame({"query_id": [], "candidate_id": [], "candidate_source": [], "block_name": []})

    joined = s1_filt.join(
        corpus_filt,
        left_on=s1_key_col,
        right_on=corpus_key_col,
        how="inner",
        suffix="_cand"
    ).select([
        pl.col("entity_id").alias("query_id"),
        pl.col("entity_id_cand").alias("candidate_id"),
        pl.col("source_cand").alias("candidate_source"),
        pl.lit(block_name).alias("block_name")
    ])
    return joined


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    print("=" * 60)
    print(f" PHASE 4C: STRUCTURED BLOCKING ({args.split.upper()})")
    print("=" * 60)

    s1_path = os.path.join(args.data_dir, args.split, f"{args.split}_source1.parquet")
    s2_path = os.path.join(args.data_dir, args.split, f"{args.split}_source2.parquet")
    s3_path = os.path.join(args.data_dir, args.split, f"{args.split}_source3.parquet")

    select_cols = ["entity_id", "source", "name_norm", "address_norm", "country_norm"]
    print("Loading S1 and corpus...")
    s1_df = pl.read_parquet(s1_path, columns=select_cols)
    s2_df = pl.read_parquet(s2_path, columns=select_cols)
    s3_df = pl.read_parquet(s3_path, columns=select_cols)
    corpus_df = pl.concat([s2_df, s3_df])
    del s2_df, s3_df
    gc.collect()

    # Fill nulls
    for col in ["name_norm", "address_norm", "country_norm"]:
        s1_df = s1_df.with_columns(pl.col(col).fill_null(""))
        corpus_df = corpus_df.with_columns(pl.col(col).fill_null(""))

    # Compute token frequencies on the FULL corpus (S1 + S2 + S3) for rare-token detection
    all_names = s1_df["name_norm"].to_list() + corpus_df["name_norm"].to_list()
    all_addrs = s1_df["address_norm"].to_list() + corpus_df["address_norm"].to_list()
    print(f"Computing token frequencies on {len(all_names)} name docs...")
    name_freq = compute_token_frequencies(all_names)
    addr_freq = compute_token_frequencies(all_addrs)
    del all_names, all_addrs
    gc.collect()

    max_freq = args.rare_token_max_freq

    # ── Build blocking keys for S1 ──────────────────────────────────────────
    print("Building S1 blocking keys...")
    s1_names = s1_df["name_norm"].to_list()
    s1_addrs = s1_df["address_norm"].to_list()
    s1_countries = s1_df["country_norm"].to_list()
    s1_ids = s1_df["entity_id"].to_list()

    s1_first_name_token = [get_first_token(n) for n in s1_names]
    s1_name_prefix4 = [get_name_prefix(n, 4) for n in s1_names]
    s1_token_sig = [get_token_signature(n) for n in s1_names]
    s1_rare_name = ["|".join(get_rare_tokens(n, name_freq, max_freq)[:3]) for n in s1_names]
    s1_numeric_sig = [extract_numeric_signature(a) for a in s1_addrs]
    s1_postal = [extract_postal_token(a) for a in s1_addrs]
    s1_rare_addr = ["|".join(get_rare_tokens(a, addr_freq, max_freq)[:3]) for a in s1_addrs]
    s1_country_rare_name = [f"{c}||{r}" for c, r in zip(s1_countries, s1_rare_name)]
    s1_country_numeric = [f"{c}||{n}" for c, n in zip(s1_countries, s1_numeric_sig)]

    s1_keyed = pl.DataFrame({
        "entity_id": s1_ids,
        "first_name_token": s1_first_name_token,
        "name_prefix4": s1_name_prefix4,
        "token_sig": s1_token_sig,
        "rare_name": s1_rare_name,
        "numeric_sig": s1_numeric_sig,
        "postal": s1_postal,
        "rare_addr": s1_rare_addr,
        "country_rare_name": s1_country_rare_name,
        "country_numeric": s1_country_numeric,
    })

    # ── Build blocking keys for corpus ──────────────────────────────────────
    print("Building corpus blocking keys...")
    c_names = corpus_df["name_norm"].to_list()
    c_addrs = corpus_df["address_norm"].to_list()
    c_countries = corpus_df["country_norm"].to_list()
    c_ids = corpus_df["entity_id"].to_list()
    c_sources = corpus_df["source"].to_list()

    c_first_name_token = [get_first_token(n) for n in c_names]
    c_name_prefix4 = [get_name_prefix(n, 4) for n in c_names]
    c_token_sig = [get_token_signature(n) for n in c_names]
    c_rare_name = ["|".join(get_rare_tokens(n, name_freq, max_freq)[:3]) for n in c_names]
    c_numeric_sig = [extract_numeric_signature(a) for a in c_addrs]
    c_postal = [extract_postal_token(a) for a in c_addrs]
    c_rare_addr = ["|".join(get_rare_tokens(a, addr_freq, max_freq)[:3]) for a in c_addrs]
    c_country_rare_name = [f"{c}||{r}" for c, r in zip(c_countries, c_rare_name)]
    c_country_numeric = [f"{c}||{n}" for c, n in zip(c_countries, c_numeric_sig)]

    corpus_keyed = pl.DataFrame({
        "entity_id": c_ids,
        "source": c_sources,
        "first_name_token": c_first_name_token,
        "name_prefix4": c_name_prefix4,
        "token_sig": c_token_sig,
        "rare_name": c_rare_name,
        "numeric_sig": c_numeric_sig,
        "postal": c_postal,
        "rare_addr": c_rare_addr,
        "country_rare_name": c_country_rare_name,
        "country_numeric": c_country_numeric,
    })

    del c_names, c_addrs, c_countries, c_ids, c_sources
    del s1_names, s1_addrs, s1_countries, s1_ids
    gc.collect()

    # ── Run all blocks ──────────────────────────────────────────────────────
    blocks = [
        ("first_name_token", "first_name_token", "name_first_token"),
        ("name_prefix4",     "name_prefix4",     "name_prefix4"),
        ("token_sig",        "token_sig",         "name_token_sig"),
        ("rare_name",        "rare_name",         "rare_name_token"),
        ("numeric_sig",      "numeric_sig",        "addr_numeric_sig"),
        ("postal",           "postal",             "addr_postal"),
        ("rare_addr",        "rare_addr",          "rare_addr_token"),
        ("country_rare_name","country_rare_name",  "country_rare_name"),
        ("country_numeric",  "country_numeric",    "country_numeric"),
    ]

    all_frames = []
    for s1_key, corpus_key, block_name in blocks:
        t0 = time.time()
        block_df = join_on_key(s1_keyed, corpus_keyed, s1_key, corpus_key, block_name)
        n_pairs = block_df.height
        print(f"  {block_name}: {n_pairs} pairs ({time.time()-t0:.1f}s)")
        if n_pairs > 0:
            all_frames.append(block_df)
        gc.collect()

    if not all_frames:
        print("No structured candidates found.")
        return

    print("\nConsolidating and deduplicating blocks...")
    t0 = time.time()
    combined = pl.concat(all_frames)
    del all_frames
    gc.collect()

    # Pivot: keep all block_names for each (query_id, candidate_id) pair
    # We add a `found_by_block=True` flag and a `block_names` column listing which blocks fired
    combined_dedup = (combined
        .group_by(["query_id", "candidate_id", "candidate_source"])
        .agg(pl.col("block_name").str.concat("|").alias("block_names"))
        .with_columns(pl.lit(True).alias("found_by_block"))
    )
    # Remove self-matches
    combined_dedup = combined_dedup.filter(pl.col("query_id") != pl.col("candidate_id"))
    print(f"  {combined.height} raw → {combined_dedup.height} unique pairs ({time.time()-t0:.1f}s)")

    output_path = os.path.join(args.output_dir, f"{args.split}_structured_candidates.parquet")
    combined_dedup.write_parquet(output_path, compression="snappy")
    print(f"Saved to {output_path}")

    if args.ground_truth and os.path.exists(args.ground_truth) and args.split == "train":
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

        metrics = evaluate_recall(gt_df, combined_dedup, "structured_blocking", baseline_pairs)
        report_path = os.path.join(args.artifacts_dir, "structured_blocking_recall.json")
        with open(report_path, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "metrics": metrics}, f, indent=2)
        print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
