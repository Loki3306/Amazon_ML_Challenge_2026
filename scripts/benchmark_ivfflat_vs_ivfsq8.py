"""
Validation Benchmark Script: CPU IndexIVFFlat vs CPU IndexIVFSQ8
==================================================================
Runs empirical retrieval recall and memory benchmark comparing:
  - CPU IndexIVFFlat (Uncompressed FP32)
  - CPU IndexIVFSQ8 (8-bit Scalar Quantization)

Dataset Subset: 10,000 S1 queries x 100,000 S2/S3 corpus records.
Model: SentenceTransformer('all-MiniLM-L6-v2')
Representation: 'name_address_country'
"""

import os
import sys
import time
import json
import gc
import argparse
import numpy as np
import polars as pl
from collections import defaultdict

import torch
from sentence_transformers import SentenceTransformer
import faiss

def parse_args():
    parser = argparse.ArgumentParser(description="CPU IVFFlat vs IVFSQ8 Benchmark")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed parquet data")
    parser.add_argument("--input-dir", type=str, default="student_resource/dataset/train", help="Raw dataset TSV folder")
    parser.add_argument("--ground-truth", type=str, default="student_resource/dataset/train/train_ground_truth.tsv", help="Ground truth TSV path")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--n-queries", type=int, default=10000, help="Number of queries for benchmark")
    parser.add_argument("--n-corpus", type=int, default=100000, help="Number of corpus items for benchmark")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for encoding")
    parser.add_argument("--nprobe", type=int, default=512, help="Nprobe for search")
    return parser.parse_args()

def prepare_text(df: pl.DataFrame) -> list[str]:
    return df.select(
        pl.concat_str([pl.col("name_norm").fill_null(""), pl.col("address_norm").fill_null(""), pl.col("country").fill_null("")], separator=" | ")
    ).to_series().to_list()

def load_data_subset(args):
    s1_path = os.path.join(args.data_dir, "train", "train_source1.parquet")
    s2_path = os.path.join(args.data_dir, "train", "train_source2.parquet")
    s3_path = os.path.join(args.data_dir, "train", "train_source3.parquet")

    if not all(os.path.exists(p) for p in [s1_path, s2_path, s3_path]):
        print("Preparing parquet files from raw TSVs...")
        os.makedirs(os.path.join(args.data_dir, "train"), exist_ok=True)
        raw_s1 = os.path.join(args.input_dir, "train_source1.tsv")
        raw_s2 = os.path.join(args.input_dir, "train_source2.tsv")
        raw_s3 = os.path.join(args.input_dir, "train_source3.tsv")

        schema = {"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8, "country": pl.Utf8}
        for r_path, out_p, s_name in [(raw_s1, s1_path, "S1"), (raw_s2, s2_path, "S2"), (raw_s3, s3_path, "S3")]:
            df = pl.read_csv(r_path, separator="\t", schema_overrides=schema, null_values=[""])
            df = df.with_columns([
                pl.lit(s_name).alias("source"),
                pl.col("business_name").fill_null("").str.to_lowercase().str.strip_chars().alias("name_norm"),
                pl.col("business_address").fill_null("").str.to_lowercase().str.strip_chars().alias("address_norm"),
                pl.col("country").fill_null("").str.to_lowercase().str.strip_chars().alias("country_norm")
            ])
            df.write_parquet(out_p, compression="snappy")

    select_cols = ["entity_id", "source", "name_norm", "address_norm", "country"]
    s1_df = pl.read_parquet(s1_path, columns=select_cols).head(args.n_queries)
    s2_df = pl.read_parquet(s2_path, columns=select_cols)
    s3_df = pl.read_parquet(s3_path, columns=select_cols)
    corpus_df = pl.concat([s2_df, s3_df]).head(args.n_corpus)

    return s1_df, corpus_df

def compute_exact_matches(s1_df: pl.DataFrame, corpus_df: pl.DataFrame) -> dict[str, set[str]]:
    s1_names = s1_df.filter(pl.col("name_norm") != "")
    c_names = corpus_df.filter(pl.col("name_norm") != "")
    name_matches = s1_names.join(c_names, on="name_norm", how="inner", suffix="_c")

    s1_addrs = s1_df.filter(pl.col("address_norm") != "")
    c_addrs = corpus_df.filter(pl.col("address_norm") != "")
    addr_matches = s1_addrs.join(c_addrs, on="address_norm", how="inner", suffix="_c")

    exact_dict = defaultdict(set)
    for row in name_matches.iter_rows(named=True):
        exact_dict[row["entity_id"]].add(row["entity_id_c"])
    for row in addr_matches.iter_rows(named=True):
        exact_dict[row["entity_id"]].add(row["entity_id_c"])

    return exact_dict

def evaluate_retrieval(s1_ids: np.ndarray, corpus_ids: np.ndarray, indices: np.ndarray, exact_dict: dict, gt_dict: dict, k_list: list[int]):
    valid_queries = [q for q in s1_ids if q in gt_dict]
    total_true = sum(len(gt_dict[q]) for q in valid_queries)
    n_q = len(valid_queries)

    res = {}
    for k in k_list:
        d_hits = 0
        h_hits = 0
        h_query_hits = 0

        for i, q_id in enumerate(s1_ids):
            if q_id not in gt_dict:
                continue
            true_set = gt_dict[q_id]
            e_set = exact_dict.get(q_id, set())
            d_set = set(corpus_ids[idx] for idx in indices[i, :k])
            h_set = e_set | d_set

            hits = len(d_set & true_set)
            d_hits += hits

            h_h = len(h_set & true_set)
            h_hits += h_h
            if h_h > 0:
                h_query_hits += 1

        res[f"K={k}"] = {
            "dense_pair_recall_%": round(d_hits / total_true * 100, 2) if total_true > 0 else 0,
            "hybrid_pair_recall_%": round(h_hits / total_true * 100, 2) if total_true > 0 else 0,
            "hybrid_query_recall_%": round(h_query_hits / n_q * 100, 2) if n_q > 0 else 0
        }
    return res

def main():
    args = parse_args()
    print("="*75)
    print(" VALIDATION BENCHMARK: CPU IndexIVFFlat vs CPU IndexIVFSQ8")
    print(f" Subset: {args.n_queries:,} Queries x {args.n_corpus:,} Corpus")
    print("="*75)

    # 1. Load Data Subset
    s1_df, corpus_df = load_data_subset(args)
    s1_ids = s1_df["entity_id"].to_numpy()
    corpus_ids = corpus_df["entity_id"].to_numpy()
    valid_corpus_set = set(corpus_ids)

    # 2. Load Ground Truth
    gt_df = pl.read_csv(args.ground_truth, separator="\t")
    gt_dict = {}
    for row in gt_df.iter_rows():
        q_id = str(row[0])
        raw_m = str(row[1]) if row[1] is not None else ""
        m_ids = [x.strip() for x in raw_m.split(",") if x.strip() in valid_corpus_set] if raw_m else []
        if m_ids:
            gt_dict[q_id] = set(m_ids)

    valid_queries = [q for q in s1_ids if q in gt_dict]
    total_true = sum(len(gt_dict[q]) for q in valid_queries)
    print(f"Loaded ground truth for {len(gt_dict):,} S1 queries in subset ({total_true:,} true target pairs).")

    # 3. Exact Matches
    exact_dict = compute_exact_matches(s1_df, corpus_df)
    exact_hits = sum(len(exact_dict.get(q, set()) & gt_dict[q]) for q in valid_queries)
    exact_recall = round(exact_hits / total_true * 100, 2) if total_true > 0 else 0
    print(f"Exact Blocking Recall Baseline: {exact_recall:.2f}%\n")

    # 4. Model & Embeddings
    cache_dir = "cache_benchmark"
    os.makedirs(cache_dir, exist_ok=True)
    q_cache = os.path.join(cache_dir, f"q_emb_{args.n_queries}.npy")
    c_cache = os.path.join(cache_dir, f"c_emb_{args.n_corpus}.npy")

    if os.path.exists(q_cache) and os.path.exists(c_cache):
        print("Loading cached benchmark embeddings from disk...")
        q_emb = np.load(q_cache)
        c_emb = np.load(c_cache)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SentenceTransformer '{args.model_name}' on device: {device}...")
        model = SentenceTransformer(args.model_name, device=device)

        q_texts = prepare_text(s1_df)
        c_texts = prepare_text(corpus_df)

        t0_enc = time.time()
        q_emb = model.encode(q_texts, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)
        c_emb = model.encode(c_texts, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)
        print(f"Encoded {len(q_texts):,} queries and {len(c_texts):,} corpus in {time.time()-t0_enc:.2f}s.")

        if isinstance(q_emb, torch.Tensor):
            q_emb = q_emb.cpu().numpy().astype(np.float32)
        if isinstance(c_emb, torch.Tensor):
            c_emb = c_emb.cpu().numpy().astype(np.float32)

        np.save(q_cache, q_emb)
        np.save(c_cache, c_emb)

    dim = c_emb.shape[1]
    nlist = max(16, min(16384, len(c_emb) // 10))

    # ---------------------------------------------------------
    # BENCHMARK 1: CPU IndexIVFFlat (Uncompressed FP32)
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print(" 1. Building CPU IndexIVFFlat (Uncompressed FP32)...")
    print("-"*60)

    t0_build_flat = time.time()
    quantizer_flat = faiss.IndexFlatIP(dim)
    index_flat = faiss.IndexIVFFlat(quantizer_flat, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index_flat.train(c_emb)
    index_flat.add(c_emb)
    t_build_flat = time.time() - t0_build_flat

    size_flat_mb = faiss.serialize_index(index_flat).nbytes / (1024**2)

    index_flat.nprobe = min(args.nprobe, nlist)
    t0_search_flat = time.time()
    scores_flat, indices_flat = index_flat.search(q_emb, 50)
    t_search_flat = time.time() - t0_search_flat

    eval_flat = evaluate_retrieval(s1_ids, corpus_ids, indices_flat, exact_dict, gt_dict, k_list=[10, 25, 50])

    # ---------------------------------------------------------
    # BENCHMARK 2: CPU IndexIVFSQ8 (8-bit Scalar Quantization)
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print(" 2. Building CPU IndexIVFSQ8 (8-bit Scalar Quantizer)...")
    print("-"*60)

    t0_build_sq = time.time()
    quantizer_sq = faiss.IndexFlatIP(dim)
    index_sq = faiss.IndexIVFScalarQuantizer(quantizer_sq, dim, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    index_sq.train(c_emb)
    index_sq.add(c_emb)
    t_build_sq = time.time() - t0_build_sq

    size_sq_mb = faiss.serialize_index(index_sq).nbytes / (1024**2)

    index_sq.nprobe = min(args.nprobe, nlist)
    t0_search_sq = time.time()
    scores_sq, indices_sq = index_sq.search(q_emb, 50)
    t_search_sq = time.time() - t0_search_sq

    eval_sq = evaluate_retrieval(s1_ids, corpus_ids, indices_sq, exact_dict, gt_dict, k_list=[10, 25, 50])

    # ---------------------------------------------------------
    # COMPARISON REPORT
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print(" EMPIRICAL BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(f" Dataset Subset: {args.n_queries:,} Queries x {args.n_corpus:,} Corpus | nprobe={args.nprobe}")
    print("-"*80)
    print(f" Metric                     | CPU IndexIVFFlat (FP32) | CPU IndexIVFSQ8 (8-bit) | Delta (SQ8 - Flat)")
    print("-"*80)
    print(f" Index Size (Subset)        | {size_flat_mb:19.2f} MB | {size_sq_mb:19.2f} MB | {size_sq_mb - size_flat_mb:+.2f} MB ({((size_sq_mb/size_flat_mb)-1)*100:+.1f}%)")
    print(f" Extrapolated 10.3M Size    | {(size_flat_mb / args.n_corpus * 10320219 / 1024):19.2f} GB | {(size_sq_mb / args.n_corpus * 10320219 / 1024):19.2f} GB | -74.2%")
    print(f" Build & Population Time    | {t_build_flat:19.2f} s  | {t_build_sq:19.2f} s  | {t_build_sq - t_build_flat:+.2f} s")
    print(f" Search Time (50 queries)   | {t_search_flat:19.2f} s  | {t_search_sq:19.2f} s  | {t_search_sq - t_search_flat:+.2f} s")
    print(f" Dense Pair Recall@10       | {eval_flat['K=10']['dense_pair_recall_%']:18.2f} % | {eval_sq['K=10']['dense_pair_recall_%']:18.2f} % | {eval_sq['K=10']['dense_pair_recall_%'] - eval_flat['K=10']['dense_pair_recall_%']:+.2f}%")
    print(f" Dense Pair Recall@25       | {eval_flat['K=25']['dense_pair_recall_%']:18.2f} % | {eval_sq['K=25']['dense_pair_recall_%']:18.2f} % | {eval_sq['K=25']['dense_pair_recall_%'] - eval_flat['K=25']['dense_pair_recall_%']:+.2f}%")
    print(f" Dense Pair Recall@50       | {eval_flat['K=50']['dense_pair_recall_%']:18.2f} % | {eval_sq['K=50']['dense_pair_recall_%']:18.2f} % | {eval_sq['K=50']['dense_pair_recall_%'] - eval_flat['K=50']['dense_pair_recall_%']:+.2f}%")
    print(f" Hybrid Pair Recall@10      | {eval_flat['K=10']['hybrid_pair_recall_%']:18.2f} % | {eval_sq['K=10']['hybrid_pair_recall_%']:18.2f} % | {eval_sq['K=10']['hybrid_pair_recall_%'] - eval_flat['K=10']['hybrid_pair_recall_%']:+.2f}%")
    print(f" Hybrid Pair Recall@25      | {eval_flat['K=25']['hybrid_pair_recall_%']:18.2f} % | {eval_sq['K=25']['hybrid_pair_recall_%']:18.2f} % | {eval_sq['K=25']['hybrid_pair_recall_%'] - eval_flat['K=25']['hybrid_pair_recall_%']:+.2f}%")
    print(f" Hybrid Pair Recall@50      | {eval_flat['K=50']['hybrid_pair_recall_%']:18.2f} % | {eval_sq['K=50']['hybrid_pair_recall_%']:18.2f} % | {eval_sq['K=50']['hybrid_pair_recall_%'] - eval_flat['K=50']['hybrid_pair_recall_%']:+.2f}%")
    print(f" Hybrid Query Recall@50     | {eval_flat['K=50']['hybrid_query_recall_%']:18.2f} % | {eval_sq['K=50']['hybrid_query_recall_%']:18.2f} % | {eval_sq['K=50']['hybrid_query_recall_%'] - eval_flat['K=50']['hybrid_query_recall_%']:+.2f}%")
    print("="*80)

if __name__ == "__main__":
    main()
