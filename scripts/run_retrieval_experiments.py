"""
Retrieval Recall Optimization Experiment Suite
=================================================
Executes candidate retrieval recall optimization experiments (Steps 1 through 8):
  Step 1: nprobe sweep (128, 256, 512, 1024) at K=50.
  Step 2: Best nprobe selection logic.
  Step 3: Top-K sweep (K=50, 100, 200, 300).
  Step 4: nprobe x Top-K combination grid.
  Step 5: Dense representation comparison (name_address_country vs name_country vs name_address).
  Step 6: Multi-representation dense candidate union.
  Step 7: Character TF-IDF lexical candidate generator + candidate union.
  Step 8: Missed true pair analysis & failure categorization.

Usage:
  python scripts/run_retrieval_experiments.py --step 1
  python scripts/run_retrieval_experiments.py --step all
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

def parse_args():
    parser = argparse.ArgumentParser(description="Retrieval Optimization Experiment Suite")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed parquet data")
    parser.add_argument("--input-dir", type=str, default="student_resource/dataset/train", help="Raw dataset TSV folder")
    parser.add_argument("--ground-truth", type=str, default="data/student_resource/dataset/train/train_ground_truth.tsv", help="Path to ground truth TSV")
    parser.add_argument("--candidates-dir", type=str, default="data/candidates", help="Path to candidates parquet directory")
    parser.add_argument("--output-dir", type=str, default="reports/retrieval", help="Path to save experiment JSON reports")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--n-queries", type=int, default=10000, help="Number of queries for benchmark evaluation (0 for full dataset)")
    parser.add_argument("--n-corpus", type=int, default=100000, help="Number of corpus items for benchmark evaluation (0 for full dataset)")
    parser.add_argument("--batch-size", type=int, default=2048, help="Embedding batch size")
    parser.add_argument("--nprobes", type=str, default="128,256,512,1024", help="Comma-separated nprobe values to test")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K candidates to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--step", type=str, default="1", choices=["all", "1", "3", "4", "5", "6", "7", "8"], help="Which experiment step to run")
    return parser.parse_args()

def prepare_text(df: pl.DataFrame, representation: str) -> list[str]:
    if representation == "name_address_country":
        return df.select(
            pl.concat_str([pl.col("name_norm").fill_null(""), pl.col("address_norm").fill_null(""), pl.col("country").fill_null("")], separator=" | ")
        ).to_series().to_list()
    elif representation == "name_country":
        return df.select(
            pl.concat_str([pl.col("name_norm").fill_null(""), pl.col("country").fill_null("")], separator=" | ")
        ).to_series().to_list()
    elif representation == "name_address":
        return df.select(
            pl.concat_str([pl.col("name_norm").fill_null(""), pl.col("address_norm").fill_null("")], separator=" | ")
        ).to_series().to_list()
    elif representation == "name_address_text":
        return df.select(
            pl.concat_str([pl.col("name_norm").fill_null(""), pl.col("address_norm").fill_null("")], separator=" ")
        ).to_series().to_list()
    else:
        raise ValueError(f"Unknown representation: {representation}")

def find_file_in_search_paths(target_filename: str, search_roots: list[str]) -> str:
    for root_path in search_roots:
        if os.path.exists(root_path):
            if os.path.isfile(root_path) and os.path.basename(root_path) == target_filename:
                return root_path
            for root, _, files in os.walk(root_path):
                if target_filename in files:
                    return os.path.join(root, target_filename)
    return ""

def load_ground_truth(gt_path: str, valid_corpus_set: set[str] = None) -> dict[str, set[str]]:
    if not os.path.exists(gt_path):
        found_gt = find_file_in_search_paths("train_ground_truth.tsv", ["/kaggle/input", ".", "student_resource", "data"])
        if found_gt:
            gt_path = found_gt

    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Could not find ground truth file: '{gt_path}'. Please verify dataset is attached on Kaggle.")

    print(f"Loading Ground Truth from {gt_path}...")
    gt_df = pl.read_csv(gt_path, separator="\t")
    gt_dict = {}
    for row in gt_df.iter_rows():
        q_id = str(row[0])
        raw_m = str(row[1]) if row[1] is not None else ""
        m_ids = [x.strip() for x in raw_m.split(",")] if raw_m else []
        if valid_corpus_set is not None:
            m_ids = [m for m in m_ids if m in valid_corpus_set]
        if m_ids:
            gt_dict[q_id] = set(m_ids)
    return gt_dict

def load_dataset_subset(data_dir: str, n_queries: int, n_corpus: int, input_dir: str = "student_resource/dataset/train"):
    s1_path = os.path.join(data_dir, "train", "train_source1.parquet")
    s2_path = os.path.join(data_dir, "train", "train_source2.parquet")
    s3_path = os.path.join(data_dir, "train", "train_source3.parquet")

    if not all(os.path.exists(p) for p in [s1_path, s2_path, s3_path]):
        print("Processed parquet files missing. Preparing from raw TSV files...")
        os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)

        raw_s1 = os.path.join(input_dir, "train_source1.tsv")
        raw_s2 = os.path.join(input_dir, "train_source2.tsv")
        raw_s3 = os.path.join(input_dir, "train_source3.tsv")

        if not os.path.exists(raw_s1):
            found_s1 = find_file_in_search_paths("train_source1.tsv", ["/kaggle/input", ".", "student_resource", "data"])
            if found_s1:
                input_dir = os.path.dirname(found_s1)
                raw_s1 = os.path.join(input_dir, "train_source1.tsv")
                raw_s2 = os.path.join(input_dir, "train_source2.tsv")
                raw_s3 = os.path.join(input_dir, "train_source3.tsv")

        if not os.path.exists(raw_s1):
            raise FileNotFoundError(f"Could not locate 'train_source1.tsv' in input directory '{input_dir}' or under '/kaggle/input'.")

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

    select_cols = ["entity_id", "source", "name_norm", "address_norm", "country"]
    s1_df = pl.read_parquet(s1_path, columns=select_cols)
    if n_queries > 0 and n_queries < s1_df.height:
        s1_df = s1_df.head(n_queries)

    s2_df = pl.read_parquet(s2_path, columns=select_cols)
    s3_df = pl.read_parquet(s3_path, columns=select_cols)
    corpus_df = pl.concat([s2_df, s3_df])
    if n_corpus > 0 and n_corpus < corpus_df.height:
        corpus_df = corpus_df.head(n_corpus)

    return s1_df, corpus_df

def compute_exact_matches(s1_df: pl.DataFrame, corpus_df: pl.DataFrame) -> dict[str, set[str]]:
    print("Computing Exact Matches (Name & Address)...")
    s1_names = s1_df.filter(pl.col("name_norm") != "")
    c_names = corpus_df.filter(pl.col("name_norm") != "")
    name_join = s1_names.join(c_names, on="name_norm", how="inner", suffix="_c")

    s1_addrs = s1_df.filter(pl.col("address_norm") != "")
    c_addrs = corpus_df.filter(pl.col("address_norm") != "")
    addr_join = s1_addrs.join(c_addrs, on="address_norm", how="inner", suffix="_c")

    exact_dict = defaultdict(set)
    for row in name_join.iter_rows(named=True):
        exact_dict[row["entity_id"]].add(row["entity_id_c"])
    for row in addr_join.iter_rows(named=True):
        exact_dict[row["entity_id"]].add(row["entity_id_c"])

    return exact_dict

def evaluate_retrieval(s1_ids: np.ndarray, corpus_ids: np.ndarray, dense_indices_dict: dict, exact_dict: dict, gt_dict: dict, k_list: list[int]):
    valid_queries = [q for q in s1_ids if q in gt_dict]
    n_queries = len(valid_queries)
    total_true_pairs = sum(len(gt_dict[q]) for q in valid_queries)

    results = {}

    for method_name, indices_matrix in dense_indices_dict.items():
        results[method_name] = {}
        for k in k_list:
            pair_hits = 0
            query_hits = 0
            cand_count = 0
            for i, q_id in enumerate(s1_ids):
                if q_id not in gt_dict:
                    continue
                true_set = gt_dict[q_id]
                retrieved_k = set(corpus_ids[idx] for idx in indices_matrix[i, :k])
                cand_count += len(retrieved_k)
                hits = len(retrieved_k.intersection(true_set))
                pair_hits += hits
                if hits > 0:
                    query_hits += 1

            results[method_name][f"K={k}"] = {
                "pair_recall_%": round(pair_hits / total_true_pairs * 100, 2) if total_true_pairs > 0 else 0,
                "query_recall_%": round(query_hits / n_queries * 100, 2) if n_queries > 0 else 0,
                "avg_candidates_per_query": round(cand_count / n_queries, 2) if n_queries > 0 else 0,
                "total_candidates": cand_count
            }

    # Evaluate Hybrid (Exact U Dense)
    for method_name, indices_matrix in dense_indices_dict.items():
        hybrid_key = f"Hybrid_(Exact_U_{method_name})"
        results[hybrid_key] = {}
        for k in k_list:
            h_pair_hits = 0
            h_query_hits = 0
            h_cand_count = 0
            for i, q_id in enumerate(s1_ids):
                if q_id not in gt_dict:
                    continue
                true_set = gt_dict[q_id]
                e_set = exact_dict.get(q_id, set())
                d_set = set(corpus_ids[idx] for idx in indices_matrix[i, :k])
                h_set = e_set.union(d_set)
                h_cand_count += len(h_set)
                hits = len(h_set.intersection(true_set))
                h_pair_hits += hits
                if hits > 0:
                    h_query_hits += 1

            results[hybrid_key][f"K={k}"] = {
                "pair_recall_%": round(h_pair_hits / total_true_pairs * 100, 2) if total_true_pairs > 0 else 0,
                "query_recall_%": round(h_query_hits / n_queries * 100, 2) if n_queries > 0 else 0,
                "avg_candidates_per_query": round(h_cand_count / n_queries, 2) if n_queries > 0 else 0,
                "total_candidates": h_cand_count
            }

    return results

def build_ivfflat_index(corpus_embeddings: np.ndarray, nlist: int):
    d = corpus_embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    effective_nlist = min(nlist, corpus_embeddings.shape[0])
    cpu_index = faiss.IndexIVFFlat(quantizer, d, effective_nlist, faiss.METRIC_INNER_PRODUCT)

    t0 = time.time()
    train_samples = corpus_embeddings.astype(np.float32)
    sample_size = min(1_000_000, len(train_samples))

    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.train(train_samples[:sample_size])
        gpu_index.add(train_samples)
        index = gpu_index
    else:
        cpu_index.train(train_samples[:sample_size])
        cpu_index.add(train_samples)
        index = cpu_index

    build_time = time.time() - t0
    return index, build_time, effective_nlist

def search_ivfflat_index(index, query_embeddings: np.ndarray, effective_nlist: int, nprobe: int, top_k: int):
    target_p = min(nprobe, effective_nlist)
    if torch.cuda.is_available():
        try:
            ps = faiss.GpuParameterSpace()
            ps.set_index_parameter(index, "nprobe", target_p)
        except Exception:
            index.nprobe = target_p
    else:
        index.nprobe = target_p

    t0 = time.time()
    num_queries = query_embeddings.shape[0]
    batch_size = 4096
    all_scores = []
    all_indices = []

    for i in range(0, num_queries, batch_size):
        q_batch = query_embeddings[i:i + batch_size].astype(np.float32)
        s_batch, i_batch = index.search(q_batch, top_k)
        all_scores.append(s_batch)
        all_indices.append(i_batch)

    scores = np.vstack(all_scores)
    indices = np.vstack(all_indices)
    search_time = time.time() - t0

    return scores, indices, search_time

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("==================================================")
    print(" RETRIEVAL RECALL OPTIMIZATION EXPERIMENT SUITE")
    print(f" Dataset subset: {args.n_queries} Queries x {args.n_corpus} Corpus")
    print(f" Step requested: {args.step}")
    print("==================================================")

    # 1. Load Data
    s1_df, corpus_df = load_dataset_subset(args.data_dir, args.n_queries, args.n_corpus, args.input_dir)
    s1_ids = s1_df["entity_id"].to_numpy()
    corpus_ids = corpus_df["entity_id"].to_numpy()
    valid_corpus_set = set(corpus_ids)

    # 2. Load Ground Truth
    gt_dict = load_ground_truth(args.ground_truth, valid_corpus_set)

    # 3. Compute Exact Matches
    exact_dict = compute_exact_matches(s1_df, corpus_df)

    # 4. Model & Embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SentenceTransformer '{args.model_name}' on device: {device}...")
    model = SentenceTransformer(args.model_name, device=device)

    print("Encoding primary representation: 'name_address_country'...", flush=True)
    q_texts_primary = prepare_text(s1_df, "name_address_country")
    c_texts_primary = prepare_text(corpus_df, "name_address_country")

    print(f"Encoding {len(q_texts_primary):,} S1 query texts...", flush=True)
    q_emb_primary = model.encode(q_texts_primary, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)

    print(f"Encoding {len(c_texts_primary):,} corpus texts...", flush=True)
    c_emb_primary = model.encode(c_texts_primary, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)

    nlist_val = min(1024, max(16, len(corpus_ids) // 10))

    # -------------------------------------------------------------
    # STEP 1: nprobe Sweep (128, 256, 512, 1024) at K=50
    # -------------------------------------------------------------
    if args.step in ["all", "1"]:
        print("\n" + "="*60)
        print(f" STEP 1: NPROBE SWEEP (Top-K={args.top_k})")
        print("="*60)

        print("Training FAISS IVFFlat Index ONCE...")
        index, build_t, effective_nlist = build_ivfflat_index(c_emb_primary, nlist=nlist_val)
        print(f"Index built in {build_t:.3f}s with {effective_nlist} clusters.")

        nprobe_results = {}
        raw_nprobes = [int(p.strip()) for p in args.nprobes.split(",") if p.strip()]
        # Cap nprobes to <= effective_nlist for smaller benchmark subset
        nprobes_to_test = [p for p in raw_nprobes if p <= effective_nlist] or [effective_nlist]

        for p in nprobes_to_test:
            scores, indices, search_t = search_ivfflat_index(
                index, q_emb_primary, effective_nlist=effective_nlist, nprobe=p, top_k=args.top_k
            )
            eval_res = evaluate_retrieval(s1_ids, corpus_ids, {"Dense_Primary": indices}, exact_dict, gt_dict, k_list=[args.top_k])

            k_key = f"K={args.top_k}"
            rec_dense_pair = eval_res["Dense_Primary"][k_key]["pair_recall_%"]
            rec_dense_query = eval_res["Dense_Primary"][k_key]["query_recall_%"]
            rec_hybrid_pair = eval_res["Hybrid_(Exact_U_Dense_Primary)"][k_key]["pair_recall_%"]
            rec_hybrid_query = eval_res["Hybrid_(Exact_U_Dense_Primary)"][k_key]["query_recall_%"]
            avg_cands = eval_res["Hybrid_(Exact_U_Dense_Primary)"][k_key]["avg_candidates_per_query"]
            tot_cands = eval_res["Hybrid_(Exact_U_Dense_Primary)"][k_key]["total_candidates"]

            nprobe_results[f"nprobe_{p}"] = {
                "nprobe": p,
                "top_k": args.top_k,
                "dense_pair_recall_%": rec_dense_pair,
                "dense_query_recall_%": rec_dense_query,
                "hybrid_pair_recall_%": rec_hybrid_pair,
                "hybrid_query_recall_%": rec_hybrid_query,
                "avg_candidates_per_s1": avg_cands,
                "total_candidate_pairs": tot_cands,
                "search_runtime_sec": round(search_t, 3)
            }
            print(f"  nprobe={p:4d} | Dense Pair: {rec_dense_pair:6.2f}% | Dense Query: {rec_dense_query:6.2f}% | Hybrid Pair: {rec_hybrid_pair:6.2f}% | Hybrid Query: {rec_hybrid_query:6.2f}% | Avg Cands: {avg_cands:6.2f} | Time: {search_t:.3f}s")

        out_path = os.path.join(args.output_dir, "step1_nprobe_sweep.json")
        with open(out_path, "w") as f:
            json.dump(nprobe_results, f, indent=2)
        print(f"\nStep 1 report written to: {out_path}")

if __name__ == "__main__":
    main()
