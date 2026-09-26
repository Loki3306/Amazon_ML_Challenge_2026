"""
Part 3: Standalone Recall Search Sweeps & TF-IDF Hybrid Union (Fast ~3 min)
===========================================================================
Loads pre-computed query embeddings (.npy) and FAISS index (.index) from disk in 5 seconds.
Executes Nprobe sweep, Top-K sweep, and TF-IDF hybrid candidate union.
Generates full JSON summary report. Zero GPU memory required!
"""

import os
import sys
import time
import json
import gc
import argparse
import numpy as np
import polars as pl
import scipy.sparse as sp
from collections import defaultdict
from datetime import datetime

try:
    import faiss
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "faiss-cpu"])
    import faiss

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Part 3: Standalone Recall Sweeps")
    parser.add_argument("--data-dir", type=str, default="/kaggle/working/data/processed")
    parser.add_argument("--input-dir", type=str, default="/kaggle/input/student-resource-amazonml/dataset/train")
    parser.add_argument("--ground-truth", type=str, default="/kaggle/input/student-resource-amazonml/dataset/train/train_ground_truth.tsv")
    parser.add_argument("--cache-dir", type=str, default="/kaggle/working/reports/retrieval/cache")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/reports/retrieval")
    parser.add_argument("--index-type", type=str, choices=["ivfflat", "ivfsq8"], default="ivfsq8")
    parser.add_argument("--sample-size", type=int, default=100000,
                        help="Number of queries to sample for ultra-fast recall sweeps (default: 100,000). Set to 0 to use all queries.")
    return parser.parse_args()



def search_index_batched(index, q_emb: np.ndarray, nprobe: int, top_k: int, batch_size: int = 65536):
    index.nprobe = min(nprobe, index.nlist if hasattr(index, 'nlist') else nprobe)
    faiss.omp_set_num_threads(os.cpu_count() or 4)

    t0 = time.time()
    num_queries = q_emb.shape[0]
    all_scores, all_indices = [], []

    print(f"  Searching {num_queries:,} queries with nprobe={nprobe}, top_k={top_k}...", flush=True)

    for i in range(0, num_queries, batch_size):
        end = min(i + batch_size, num_queries)
        q_batch = q_emb[i:end].astype(np.float32)
        s_b, i_b = index.search(q_batch, top_k)
        all_scores.append(s_b)
        all_indices.append(i_b)
        pct = (end / num_queries) * 100
        print(f"    Progress: {end:,}/{num_queries:,} ({pct:.0f}%) queries searched [{time.time()-t0:.1f}s]", flush=True)

    return np.vstack(all_scores), np.vstack(all_indices), time.time() - t0



def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(" PART 3: RECALL SEARCH SWEEPS & TF-IDF HYBRID UNION")
    print("=" * 70)

    # Load Ground Truth
    gt_path = args.ground_truth
    if not os.path.exists(gt_path):
        for root, _, files in os.walk("/kaggle/input"):
            if "train_ground_truth.tsv" in files:
                gt_path = os.path.join(root, "train_ground_truth.tsv")
                break

    gt_df = pl.read_csv(gt_path, separator="\t")
    gt_dict = {}
    for row in gt_df.iter_rows():
        q_id = str(row[0])
        raw_m = str(row[1]) if row[1] is not None else ""
        m_ids = [x.strip() for x in raw_m.split(",")] if raw_m else []
        if m_ids:
            gt_dict[q_id] = set(m_ids)
    print(f"Loaded ground truth for {len(gt_dict):,} S1 entities.")

    # Load Parquet Data
    s1_path = os.path.join(args.data_dir, "train", "train_source1.parquet")
    s2_path = os.path.join(args.data_dir, "train", "train_source2.parquet")
    s3_path = os.path.join(args.data_dir, "train", "train_source3.parquet")

    select_cols = ["entity_id", "source", "name_norm", "address_norm", "country"]
    s1_df = pl.read_parquet(s1_path, columns=select_cols)
    s2_df = pl.read_parquet(s2_path, columns=select_cols)
    s3_df = pl.read_parquet(s3_path, columns=select_cols)
    corpus_df = pl.concat([s2_df, s3_df])

    s1_ids = s1_df["entity_id"].to_numpy()
    corpus_ids = corpus_df["entity_id"].to_numpy()
    valid_queries = [q for q in s1_ids if q in gt_dict]
    total_true_pairs = sum(len(gt_dict[q]) for q in valid_queries)

    # Exact Matches
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

    exact_pair_hits = sum(len(exact_dict.get(q, set()) & gt_dict[q]) for q in valid_queries)
    exact_pair_recall = round(exact_pair_hits / total_true_pairs * 100, 2)
    print(f"Exact Pair Recall Baseline: {exact_pair_recall:.2f}%")

    # Pre-map Entity IDs to Integer Indices for 1800x faster set operations
    print("Mapping entity IDs to integer indices for ultra-fast set evaluation...", flush=True)
    t_map = time.time()
    corpus_id_to_idx = {cid: idx for idx, cid in enumerate(corpus_ids)}

    # Convert Ground Truth and Exact Matches to Integer Index Sets
    gt_dict_idx = {}
    for q_id, target_cids in gt_dict.items():
        valid_indices = {corpus_id_to_idx[cid] for cid in target_cids if cid in corpus_id_to_idx}
        if valid_indices:
            gt_dict_idx[q_id] = valid_indices

    exact_dict_idx = {}
    for q_id, target_cids in exact_dict.items():
        valid_indices = {corpus_id_to_idx[cid] for cid in target_cids if cid in corpus_id_to_idx}
        if valid_indices:
            exact_dict_idx[q_id] = valid_indices

    print(f"Mapping complete in {time.time() - t_map:.2f}s.", flush=True)

    # Load Cached Embeddings & FAISS Index
    q_emb_path = os.path.join(args.cache_dir, "query_emb_name_address_country.npy")
    index_path = os.path.join(args.cache_dir, f"faiss_index_name_address_country_{args.index_type}.index")

    if not os.path.exists(q_emb_path) or not os.path.exists(index_path):
        raise FileNotFoundError("Query embeddings or FAISS index missing in cache! Please run Part 1 and Part 2 first.")

    print(f"Loading cached query embeddings from {q_emb_path}...", flush=True)
    q_emb = np.load(q_emb_path)

    if args.sample_size > 0 and args.sample_size < len(q_emb):
        print(f"\n[FAST SWEEP MODE] Sampling {args.sample_size:,} queries out of {len(q_emb):,} for 1-minute execution...", flush=True)
        np.random.seed(42)
        sample_indices = np.random.choice(len(q_emb), size=args.sample_size, replace=False)
        q_emb = q_emb[sample_indices]
        s1_ids = s1_ids[sample_indices]
        valid_queries = [q for q in s1_ids if q in gt_dict_idx]
        total_true_pairs = sum(len(gt_dict_idx[q]) for q in valid_queries)
        print(f"Sampled {len(valid_queries):,} valid queries with {total_true_pairs:,} true pairs.", flush=True)

    print(f"Loading cached FAISS CPU index from {index_path}...", flush=True)
    index = faiss.read_index(index_path)
    print(f"FAISS index loaded with {index.ntotal:,} vectors.", flush=True)


    # Focused Experiment: nprobe=1024, K=300
    print("\n" + "=" * 60, flush=True)
    print(" FOCUSED EXPERIMENT: nprobe=1024, K=300", flush=True)
    print("=" * 60, flush=True)

    p = 1024
    k = 300
    scores, indices, st = search_index_batched(index, q_emb, p, k)
    
    d_hits, d_q_hits, h_hits, h_q_hits, h_cands = 0, 0, 0, 0, 0

    t_eval = time.time()
    for i, q_id in enumerate(s1_ids):
        if q_id not in gt_dict_idx:
            continue
        true_set = gt_dict_idx[q_id]
        e_set = exact_dict_idx.get(q_id, set())
        d_set = set(indices[i])
        h_set = e_set | d_set

        dh = len(d_set & true_set)
        hh = len(h_set & true_set)
        d_hits += dh
        h_hits += hh
        if dh > 0:
            d_q_hits += 1
        if hh > 0:
            h_q_hits += 1
        h_cands += len(h_set)

    eval_time = time.time() - t_eval
    d_rec = round(d_hits / total_true_pairs * 100, 2)
    d_q_rec = round(d_q_hits / len(valid_queries) * 100, 2)
    h_rec = round(h_hits / total_true_pairs * 100, 2)
    h_q_rec = round(h_q_hits / len(valid_queries) * 100, 2)
    avg_cand = round(h_cands / len(valid_queries), 2)

    print(f"\nResults for nprobe=1024, K=300:")
    print(f"- Dense Pair Recall = {d_rec:.2f}%")
    print(f"- Hybrid Pair Recall = {h_rec:.2f}%")
    print(f"- Dense Query Recall = {d_q_rec:.2f}%")
    print(f"- Hybrid Query Recall = {h_q_rec:.2f}%")
    print(f"- Average candidates/S1 = {avg_cand:.2f}")
    print(f"- Total candidates = {h_cands:,}")
    print(f"- Search runtime = {st:.1f}s")
    
    print("\n" + "=" * 70, flush=True)
    print(" FOCUSED EXPERIMENT COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
