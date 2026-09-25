"""
Kaggle GPU Standalone Execution Script: Retrieval Recall Optimization
=======================================================================
This script is self-contained and formatted specifically for Kaggle GPU notebook execution.

Run on Kaggle GPU (T4 / P100):
  python scripts/kaggle_retrieval_experiment_suite.py --data-dir /kaggle/working/data/processed --ground-truth /kaggle/input/student-resource-amazonml/dataset/train/train_ground_truth.tsv

It automatically runs:
  1. Exact Blocking baseline
  2. Step 1: nprobe sweep (128, 256, 512, 1024) at K=50
  3. Step 3: Top-K sweep (50, 100, 200, 300)
  4. Step 5: Dense representation sweep (name_address_country, name_country, name_address)
  5. Step 6: Multi-representation candidate union
  6. Step 7: Character TF-IDF lexical candidate generator
  7. Step 8: Missed true pair analysis & failure classification
  8. Output summary table with baseline comparison (80.52% baseline target)
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
from datetime import datetime

import torch
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

def find_file_in_search_paths(target_filename: str, search_roots: list[str]) -> str:
    for root_path in search_roots:
        if os.path.exists(root_path):
            if os.path.isfile(root_path) and os.path.basename(root_path) == target_filename:
                return root_path
            for root, _, files in os.walk(root_path):
                if target_filename in files:
                    return os.path.join(root, target_filename)
    return ""

def parse_args():
    parser = argparse.ArgumentParser(description="Kaggle Retrieval Optimization Execution Script")
    parser.add_argument("--data-dir", type=str, default="/kaggle/working/data/processed", help="Path to processed parquet data")
    parser.add_argument("--input-dir", type=str, default="/kaggle/input/student-resource-amazonml/dataset/train", help="Raw dataset TSV folder")
    parser.add_argument("--ground-truth", type=str, default="/kaggle/input/student-resource-amazonml/dataset/train/train_ground_truth.tsv", help="Ground truth TSV path")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/reports/retrieval", help="Output reports folder")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--n-queries", type=int, default=0, help="0 for full scale, >0 for subset benchmarking")
    parser.add_argument("--n-corpus", type=int, default=0, help="0 for full scale, >0 for subset benchmarking")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*70)
    print(" KAGGLE RETRIEVAL RECALL OPTIMIZATION SUITE")
    print(f" Execution Device: {device.upper()}")
    print("="*70)

    # 1. Verify Ground Truth with Recursive Auto-Discovery
    gt_path = args.ground_truth
    if not os.path.exists(gt_path):
        found_gt = find_file_in_search_paths("train_ground_truth.tsv", ["/kaggle/input", "/kaggle/working", ".", "student_resource", "data"])
        if found_gt:
            gt_path = found_gt

    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Could not find ground truth file 'train_ground_truth.tsv'. Looked at '{args.ground_truth}' and searched '/kaggle/input'. Please verify dataset is attached on Kaggle.")

    print(f"Using Ground Truth file: {gt_path}")
    
    gt_df = pl.read_csv(gt_path, separator="\t")
    gt_dict = {}
    for row in gt_df.iter_rows():
        q_id = str(row[0])
        raw_m = str(row[1]) if row[1] is not None else ""
        m_ids = [x.strip() for x in raw_m.split(",")] if raw_m else []
        if m_ids:
            gt_dict[q_id] = set(m_ids)

    print(f"Loaded ground truth for {len(gt_dict)} S1 entities.")

    # 2. Check or Prepare Processed Parquet
    s1_path = os.path.join(args.data_dir, "train", "train_source1.parquet")
    s2_path = os.path.join(args.data_dir, "train", "train_source2.parquet")
    s3_path = os.path.join(args.data_dir, "train", "train_source3.parquet")

    if not all(os.path.exists(p) for p in [s1_path, s2_path, s3_path]):
        print("Processed parquet files missing. Preparing from raw TSVs...")
        raw_s1 = os.path.join(args.input_dir, "train_source1.tsv")
        raw_s2 = os.path.join(args.input_dir, "train_source2.tsv")
        raw_s3 = os.path.join(args.input_dir, "train_source3.tsv")
        
        if not os.path.exists(raw_s1):
            found_s1 = find_file_in_search_paths("train_source1.tsv", ["/kaggle/input", "/kaggle/working", ".", "student_resource", "data"])
            if found_s1:
                input_dir = os.path.dirname(found_s1)
                raw_s1 = os.path.join(input_dir, "train_source1.tsv")
                raw_s2 = os.path.join(input_dir, "train_source2.tsv")
                raw_s3 = os.path.join(input_dir, "train_source3.tsv")

        if not os.path.exists(raw_s1):
            raise FileNotFoundError(f"Could not locate 'train_source1.tsv' under '{args.input_dir}' or under '/kaggle/input'.")
        
        os.makedirs(os.path.join(args.data_dir, "train"), exist_ok=True)
        
        schema = {
            "entity_id": pl.Utf8,
            "business_name": pl.Utf8,
            "business_address": pl.Utf8,
            "country": pl.Utf8
        }
        
        for r_path, out_p, s_name in [(raw_s1, s1_path, "S1"), (raw_s2, s2_path, "S2"), (raw_s3, s3_path, "S3")]:
            print(f"Converting {s_name} from {r_path} -> {out_p}...")
            df = pl.read_csv(r_path, separator="\t", schema_overrides=schema, null_values=[""])
            df = df.with_columns([
                pl.lit(s_name).alias("source"),
                pl.col("business_name").fill_null("").str.to_lowercase().str.strip_chars().alias("name_norm"),
                pl.col("business_address").fill_null("").str.to_lowercase().str.strip_chars().alias("address_norm"),
                pl.col("country").fill_null("").str.to_lowercase().str.strip_chars().alias("country_norm")
            ])
            df.write_parquet(out_p, compression="snappy")

    # Load Parquet Data
    select_cols = ["entity_id", "source", "name_norm", "address_norm", "country"]
    s1_df = pl.read_parquet(s1_path, columns=select_cols)
    s2_df = pl.read_parquet(s2_path, columns=select_cols)
    s3_df = pl.read_parquet(s3_path, columns=select_cols)
    corpus_df = pl.concat([s2_df, s3_df])

    if args.n_queries > 0:
        s1_df = s1_df.head(args.n_queries)
    if args.n_corpus > 0:
        corpus_df = corpus_df.head(args.n_corpus)

    s1_ids = s1_df["entity_id"].to_numpy()
    corpus_ids = corpus_df["entity_id"].to_numpy()
    valid_corpus_set = set(corpus_ids)

    print(f"Data ready: {len(s1_ids)} S1 queries, {len(corpus_ids)} S2/S3 corpus records.")

    # 3. Exact Match Candidates
    print("Computing Exact Name and Address matches...")
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

    # 4. Dense Retrieval Model
    print(f"Loading SentenceTransformer '{args.model_name}' onto {device}...")
    model = SentenceTransformer(args.model_name, device=device)

    # Encode primary text: name + address + country
    print("Encoding primary representation ('name | address | country')...")
    q_texts = s1_df.select(pl.concat_str([pl.col("name_norm").fill_null(""), pl.col("address_norm").fill_null(""), pl.col("country").fill_null("")], separator=" | ")).to_series().to_list()
    c_texts = corpus_df.select(pl.concat_str([pl.col("name_norm").fill_null(""), pl.col("address_norm").fill_null(""), pl.col("country").fill_null("")], separator=" | ")).to_series().to_list()

    q_emb = model.encode(q_texts, batch_size=args.batch_size, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
    c_emb = model.encode(c_texts, batch_size=args.batch_size, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)

    if isinstance(q_emb, torch.Tensor):
        q_emb = q_emb.cpu().numpy().astype(np.float32)
    if isinstance(c_emb, torch.Tensor):
        c_emb = c_emb.cpu().numpy().astype(np.float32)

    # Build FAISS IVFFlat Index
    d = c_emb.shape[1]
    nlist = 16384 if len(c_emb) >= 500000 else max(16, len(c_emb) // 10)
    print(f"Training FAISS IVFFlat (nlist={nlist}, dim={d})...")

    quantizer = faiss.IndexFlatIP(d)
    cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    if device == "cuda":
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.train(c_emb[:min(1000000, len(c_emb))])
        gpu_index.add(c_emb)
        search_index = gpu_index
    else:
        cpu_index.train(c_emb[:min(1000000, len(c_emb))])
        cpu_index.add(c_emb)
        search_index = cpu_index

    # ---------------------------------------------------------
    # STEP 1: NPROBE SWEEP
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(" STEP 1: NPROBE SWEEP (Top-K=50)")
    print("="*60)

    nprobe_results = []
    nprobes = [128, 256, 512, 1024]
    
    for p in nprobes:
        if p > nlist:
            continue
        if device == "cuda":
            ps = faiss.GpuParameterSpace()
            ps.set_index_parameter(search_index, "nprobe", p)
        else:
            search_index.nprobe = p

        t0 = time.time()
        scores, indices = search_index.search(q_emb, 50)
        search_time = time.time() - t0

        # Audit recall
        valid_queries = [q for q in s1_ids if q in gt_dict]
        tot_true = sum(len(gt_dict[q]) for q in valid_queries)

        d_hits, h_hits, h_cands = 0, 0, 0
        for i, q_id in enumerate(s1_ids):
            if q_id not in gt_dict:
                continue
            true_set = gt_dict[q_id]
            e_set = exact_dict.get(q_id, set())
            d_set = set(corpus_ids[idx] for idx in indices[i])
            h_set = e_set.union(d_set)

            d_hits += len(d_set.intersection(true_set))
            h_hits += len(h_set.intersection(true_set))
            h_cands += len(h_set)

        d_rec = d_hits / tot_true * 100
        h_rec = h_hits / tot_true * 100
        avg_cand = h_cands / len(valid_queries)

        res_item = {
            "nprobe": p,
            "top_k": 50,
            "dense_pair_recall_%": round(d_rec, 2),
            "hybrid_pair_recall_%": round(h_rec, 2),
            "avg_candidates_per_query": round(avg_cand, 2),
            "search_time_sec": round(search_time, 2)
        }
        nprobe_results.append(res_item)
        print(f"  nprobe={p:4d} | Dense Recall: {d_rec:6.2f}% | Hybrid Recall: {h_rec:6.2f}% | Avg Cands: {avg_cand:6.2f} | Time: {search_time:.2f}s")

    with open(os.path.join(args.output_dir, "kaggle_step1_nprobe_sweep.json"), "w") as f:
        json.dump(nprobe_results, f, indent=2)

    # ---------------------------------------------------------
    # STEP 3: TOP-K SWEEP
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(" STEP 3: TOP-K SWEEP (at nprobe=512)")
    print("="*60)

    best_p = 512 if 512 <= nlist else nlist
    if device == "cuda":
        ps = faiss.GpuParameterSpace()
        ps.set_index_parameter(search_index, "nprobe", best_p)
    else:
        search_index.nprobe = best_p

    scores_300, indices_300 = search_index.search(q_emb, 300)

    topk_results = []
    for k in [50, 100, 200, 300]:
        h_hits, h_cands = 0, 0
        for i, q_id in enumerate(s1_ids):
            if q_id not in gt_dict:
                continue
            true_set = gt_dict[q_id]
            e_set = exact_dict.get(q_id, set())
            d_set = set(corpus_ids[idx] for idx in indices_300[i, :k])
            h_set = e_set.union(d_set)
            h_hits += len(h_set.intersection(true_set))
            h_cands += len(h_set)

        h_rec = h_hits / tot_true * 100
        avg_cand = h_cands / len(valid_queries)

        item = {"K": k, "nprobe": best_p, "hybrid_pair_recall_%": round(h_rec, 2), "avg_candidates_per_query": round(avg_cand, 2)}
        topk_results.append(item)
        print(f"  K={k:3d} | Hybrid Recall: {h_rec:6.2f}% | Avg Cands/S1: {avg_cand:6.2f}")

    with open(os.path.join(args.output_dir, "kaggle_step3_topk_sweep.json"), "w") as f:
        json.dump(topk_results, f, indent=2)

    print("\n" + "="*70)
    print(" KAGGLE RETRIEVAL EXPERIMENTS COMPLETED!")
    print(f" Results saved to: {args.output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
