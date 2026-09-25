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
    parser.add_argument("--nprobe-fixed", type=int, default=512, help="Fixed nprobe value for Step 3 Top-K sweep")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K candidates to evaluate")
    parser.add_argument("--top-ks", type=str, default="50,100,200,300", help="Comma-separated Top-K values for Step 3 sweep")
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
    # Smaller search batch size (512) prevents FAISS GPU TemporaryMemoryOverflow when nprobe and top_k are large
    batch_size = 512
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

    # Free PyTorch model to reclaim GPU VRAM for FAISS search
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

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

    # -------------------------------------------------------------
    # STEP 3: Top-K Sweep (K=50, 100, 200, 300) at fixed nprobe (512)
    # -------------------------------------------------------------
    if args.step in ["all", "3"]:
        print("\n" + "="*60)
        print(f" STEP 3: TOP-K SWEEP (at nprobe={args.nprobe_fixed})")
        print("="*60)

        if 'index' not in locals():
            print("Training FAISS IVFFlat Index...")
            index, build_t, effective_nlist = build_ivfflat_index(c_emb_primary, nlist=nlist_val)
        else:
            effective_nlist = min(nlist_val, c_emb_primary.shape[0])

        target_nprobe = min(args.nprobe_fixed, effective_nlist)
        top_k_values = [int(k.strip()) for k in args.top_ks.split(",") if k.strip()]
        max_k = max(top_k_values)

        print(f"Searching Top-{max_k} candidates at nprobe={target_nprobe}...")
        scores, indices, search_t = search_ivfflat_index(
            index, q_emb_primary, effective_nlist=effective_nlist, nprobe=target_nprobe, top_k=max_k
        )

        eval_res = evaluate_retrieval(s1_ids, corpus_ids, {"Dense_Primary": indices}, exact_dict, gt_dict, k_list=top_k_values)

        topk_results = {}
        for k in top_k_values:
            k_key = f"K={k}"
            rec_dense_pair = eval_res["Dense_Primary"][k_key]["pair_recall_%"]
            rec_dense_query = eval_res["Dense_Primary"][k_key]["query_recall_%"]
            rec_hybrid_pair = eval_res["Hybrid_(Exact_U_Dense_Primary)"][k_key]["pair_recall_%"]
            rec_hybrid_query = eval_res["Hybrid_(Exact_U_Dense_Primary)"][k_key]["query_recall_%"]
            avg_cands = eval_res["Hybrid_(Exact_U_Dense_Primary)"][k_key]["avg_candidates_per_query"]
            tot_cands = eval_res["Hybrid_(Exact_U_Dense_Primary)"][k_key]["total_candidates"]

            topk_results[f"K_{k}"] = {
                "top_k": k,
                "nprobe": target_nprobe,
                "dense_pair_recall_%": rec_dense_pair,
                "dense_query_recall_%": rec_dense_query,
                "hybrid_pair_recall_%": rec_hybrid_pair,
                "hybrid_query_recall_%": rec_hybrid_query,
                "avg_candidates_per_s1": avg_cands,
                "total_candidate_pairs": tot_cands,
                "search_runtime_sec": round(search_t, 3)
            }
            print(f"  K={k:3d} | Dense Pair: {rec_dense_pair:6.2f}% | Dense Query: {rec_dense_query:6.2f}% | Hybrid Pair: {rec_hybrid_pair:6.2f}% | Hybrid Query: {rec_hybrid_query:6.2f}% | Avg Cands: {avg_cands:6.2f} | Time: {search_t:.3f}s")

        out_path = os.path.join(args.output_dir, "step3_topk_sweep.json")
        with open(out_path, "w") as f:
            json.dump(topk_results, f, indent=2)
        print(f"\nStep 3 report written to: {out_path}")

    # -------------------------------------------------------------
    # STEP 5: Dense Representation Comparison at nprobe=512, K=50
    # Compares: name_address_country vs name_country vs name_address
    # -------------------------------------------------------------
    if args.step in ["all", "5"]:
        print("\n" + "="*60)
        print(f" STEP 5: DENSE REPRESENTATION SWEEP (nprobe={args.nprobe_fixed}, K={args.top_k})")
        print("="*60)

        # Re-load model if it was freed
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SentenceTransformer '{args.model_name}' on device: {device}...")
        model_step5 = SentenceTransformer(args.model_name, device=device)

        representations = ["name_address_country", "name_country", "name_address"]
        rep_results = {}

        effective_nlist_5 = min(nlist_val, len(corpus_ids))
        target_nprobe_5 = min(args.nprobe_fixed, effective_nlist_5)

        for rep in representations:
            print(f"\n  Encoding representation: '{rep}'...")
            q_texts_r = prepare_text(s1_df, rep)
            c_texts_r = prepare_text(corpus_df, rep)

            q_emb_r = model_step5.encode(q_texts_r, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)
            c_emb_r = model_step5.encode(c_texts_r, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)

            idx_r, bt_r, eff_nlist_r = build_ivfflat_index(c_emb_r, nlist=nlist_val)
            sc_r, ind_r, st_r = search_ivfflat_index(idx_r, q_emb_r, effective_nlist=eff_nlist_r, nprobe=target_nprobe_5, top_k=args.top_k)

            k_key = f"K={args.top_k}"
            k_list_for_rep = [1, 5, 10, 25, args.top_k]
            # Clamp k_list values to top_k
            k_list_for_rep = sorted(set(min(k, args.top_k) for k in k_list_for_rep))
            eval_r = evaluate_retrieval(s1_ids, corpus_ids, {rep: ind_r}, exact_dict, gt_dict, k_list=k_list_for_rep)

            recall_at_k = {}
            for k_val in k_list_for_rep:
                kk = f"K={k_val}"
                recall_at_k[f"recall_at_{k_val}"] = eval_r[rep][kk]["pair_recall_%"]

            pair_recall_50 = eval_r[rep][k_key]["pair_recall_%"] if k_key in eval_r[rep] else None
            query_recall_50 = eval_r[rep][k_key]["query_recall_%"] if k_key in eval_r[rep] else None
            hybrid_key = f"Hybrid_(Exact_U_{rep})"
            h_pair_recall_50 = eval_r[hybrid_key][k_key]["pair_recall_%"] if hybrid_key in eval_r and k_key in eval_r[hybrid_key] else None
            h_query_recall_50 = eval_r[hybrid_key][k_key]["query_recall_%"] if hybrid_key in eval_r and k_key in eval_r[hybrid_key] else None
            avg_cands = eval_r[hybrid_key][k_key]["avg_candidates_per_query"] if hybrid_key in eval_r and k_key in eval_r[hybrid_key] else None

            rep_results[rep] = {
                "representation": rep,
                "nprobe": target_nprobe_5,
                "top_k": args.top_k,
                **recall_at_k,
                "dense_pair_recall_%": pair_recall_50,
                "dense_query_recall_%": query_recall_50,
                "hybrid_pair_recall_%": h_pair_recall_50,
                "hybrid_query_recall_%": h_query_recall_50,
                "avg_candidates_per_s1": avg_cands,
                "search_runtime_sec": round(st_r, 3)
            }
            print(f"  [{rep:30s}] Dense Pair: {pair_recall_50:6.2f}% | Hybrid Pair: {h_pair_recall_50:6.2f}% | Query: {query_recall_50:6.2f}% | Time: {st_r:.3f}s")
            print(f"    Recall@1={recall_at_k['recall_at_1']:.2f}% Recall@5={recall_at_k['recall_at_5']:.2f}% Recall@10={recall_at_k['recall_at_10']:.2f}% Recall@25={recall_at_k['recall_at_25']:.2f}% Recall@50={recall_at_k.get('recall_at_50', pair_recall_50):.2f}%")

            del q_emb_r, c_emb_r, idx_r, sc_r, ind_r
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        del model_step5
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        out_path = os.path.join(args.output_dir, "step5_representation_sweep.json")
        with open(out_path, "w") as f:
            json.dump(rep_results, f, indent=2)
        print(f"\nStep 5 report written to: {out_path}")

    # -------------------------------------------------------------
    # STEP 6: Multi-Representation Dense Union + Exact
    # Combines Dense(name_address_country) + Dense(name_country) + Exact
    # Deduplicates strictly by (query_id, candidate_id)
    # -------------------------------------------------------------
    if args.step in ["all", "6"]:
        print("\n" + "="*60)
        print(f" STEP 6: MULTI-REPRESENTATION UNION (nprobe={args.nprobe_fixed}, K={args.top_k})")
        print("="*60)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_step6 = SentenceTransformer(args.model_name, device=device)

        representations_6 = ["name_address_country", "name_country"]
        all_rep_indices = {}
        effective_nlist_6 = min(nlist_val, len(corpus_ids))
        target_nprobe_6 = min(args.nprobe_fixed, effective_nlist_6)

        for rep in representations_6:
            print(f"\n  Encoding '{rep}'...")
            q_texts_r = prepare_text(s1_df, rep)
            c_texts_r = prepare_text(corpus_df, rep)
            q_emb_r = model_step6.encode(q_texts_r, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)
            c_emb_r = model_step6.encode(c_texts_r, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)

            idx_r, _, eff_r = build_ivfflat_index(c_emb_r, nlist=nlist_val)
            _, ind_r, _ = search_ivfflat_index(idx_r, q_emb_r, effective_nlist=eff_r, nprobe=target_nprobe_6, top_k=args.top_k)
            all_rep_indices[rep] = ind_r

            del q_emb_r, c_emb_r, idx_r
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        del model_step6
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # Build union candidate sets per query (deduplicated by candidate_id)
        valid_queries = [q for q in s1_ids if q in gt_dict]
        n_q = len(valid_queries)
        total_true = sum(len(gt_dict[q]) for q in valid_queries)

        union_pair_hits = 0
        union_query_hits = 0
        union_cand_count = 0

        for i, q_id in enumerate(s1_ids):
            if q_id not in gt_dict:
                continue
            true_set = gt_dict[q_id]
            e_set = exact_dict.get(q_id, set())

            # Dense union across all representations
            d_union = set()
            for rep, ind_matrix in all_rep_indices.items():
                d_union |= set(corpus_ids[idx] for idx in ind_matrix[i])

            full_union = e_set | d_union
            union_cand_count += len(full_union)
            hits = len(full_union & true_set)
            union_pair_hits += hits
            if hits > 0:
                union_query_hits += 1

        union_pair_recall = round(union_pair_hits / total_true * 100, 2) if total_true > 0 else 0
        union_query_recall = round(union_query_hits / n_q * 100, 2) if n_q > 0 else 0
        avg_cands_union = round(union_cand_count / n_q, 2) if n_q > 0 else 0

        # Baseline single-rep (name_address_country only) for incremental gain comparison
        base_indices = all_rep_indices.get("name_address_country", None)
        base_pair_hits = 0
        base_cand_count = 0
        if base_indices is not None:
            for i, q_id in enumerate(s1_ids):
                if q_id not in gt_dict:
                    continue
                true_set = gt_dict[q_id]
                e_set = exact_dict.get(q_id, set())
                d_set = set(corpus_ids[idx] for idx in base_indices[i])
                h_set = e_set | d_set
                base_cand_count += len(h_set)
                base_pair_hits += len(h_set & true_set)
        base_pair_recall = round(base_pair_hits / total_true * 100, 2) if total_true > 0 else 0

        incremental_gain = round(union_pair_recall - base_pair_recall, 2)
        print(f"\n  Baseline Hybrid (name_address_country + Exact): {base_pair_recall:.2f}%")
        print(f"  Multi-Rep Union (name_address_country + name_country + Exact): {union_pair_recall:.2f}%")
        print(f"  Incremental Gain: +{incremental_gain:.2f} percentage points")
        print(f"  Avg Candidates/S1: {avg_cands_union:.2f}  |  Query Recall: {union_query_recall:.2f}%")

        step6_results = {
            "nprobe": target_nprobe_6,
            "top_k": args.top_k,
            "representations_used": representations_6,
            "baseline_hybrid_pair_recall_%": base_pair_recall,
            "multi_rep_union_pair_recall_%": union_pair_recall,
            "multi_rep_union_query_recall_%": union_query_recall,
            "incremental_gain_%": incremental_gain,
            "avg_candidates_per_s1": avg_cands_union,
            "total_candidate_pairs": union_cand_count
        }

        out_path = os.path.join(args.output_dir, "step6_multirep_union.json")
        with open(out_path, "w") as f:
            json.dump(step6_results, f, indent=2)
        print(f"\nStep 6 report written to: {out_path}")

    # -------------------------------------------------------------
    # STEP 7: Character TF-IDF Lexical Candidate Generator
    # Uses ngram_range=(3,5) on business_name + business_address
    # Creates: Exact + Dense + TF-IDF candidate union
    # TF-IDF is ONLY used as a candidate generator, NOT as a classifier
    # -------------------------------------------------------------
    if args.step in ["all", "7"]:
        print("\n" + "="*60)
        print(f" STEP 7: CHARACTER TF-IDF CANDIDATE GENERATOR (K={args.top_k})")
        print("="*60)

        # Build character TF-IDF representations
        print("Building character TF-IDF (ngram_range=(3,5)) on name + address...")
        q_tfidf_texts = s1_df.select(
            pl.concat_str([pl.col("name_norm").fill_null(""), pl.col("address_norm").fill_null("")], separator=" ")
        ).to_series().to_list()
        c_tfidf_texts = corpus_df.select(
            pl.concat_str([pl.col("name_norm").fill_null(""), pl.col("address_norm").fill_null("")], separator=" ")
        ).to_series().to_list()

        t0_tfidf = time.time()
        tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=200000,
            sublinear_tf=True
        )
        # Fit on corpus + queries
        all_tfidf_texts = c_tfidf_texts + q_tfidf_texts
        tfidf.fit(all_tfidf_texts)

        c_tfidf_mat = tfidf.transform(c_tfidf_texts)
        q_tfidf_mat = tfidf.transform(q_tfidf_texts)
        print(f"TF-IDF matrix built: corpus={c_tfidf_mat.shape}, queries={q_tfidf_mat.shape} in {time.time()-t0_tfidf:.2f}s", flush=True)

        # Retrieve Top-K candidates via TF-IDF cosine similarity (batched to avoid memory issues)
        from sklearn.metrics.pairwise import cosine_similarity
        tfidf_top_k = args.top_k
        t0_tfidf_search = time.time()

        tfidf_indices_list = []
        query_batch_size = 500
        n_queries_tfidf = q_tfidf_mat.shape[0]

        print(f"Searching TF-IDF Top-{tfidf_top_k} for {n_queries_tfidf} queries...", flush=True)
        for i in range(0, n_queries_tfidf, query_batch_size):
            q_batch = q_tfidf_mat[i:i + query_batch_size]
            sims = cosine_similarity(q_batch, c_tfidf_mat)
            top_k_batch = np.argpartition(sims, -tfidf_top_k, axis=1)[:, -tfidf_top_k:]
            # Sort by similarity score within each row
            for row_idx in range(top_k_batch.shape[0]):
                row_sorted = top_k_batch[row_idx][np.argsort(sims[row_idx, top_k_batch[row_idx]])[::-1]]
                tfidf_indices_list.append(row_sorted)
            if (i // query_batch_size) % 5 == 0:
                print(f"  TF-IDF searched {min(i + query_batch_size, n_queries_tfidf)}/{n_queries_tfidf} queries...", flush=True)

        tfidf_indices = np.vstack(tfidf_indices_list)
        tfidf_search_time = time.time() - t0_tfidf_search
        print(f"TF-IDF search completed in {tfidf_search_time:.2f}s")

        # Evaluate individual and union recall contributions
        valid_queries = [q for q in s1_ids if q in gt_dict]
        n_q = len(valid_queries)
        total_true = sum(len(gt_dict[q]) for q in valid_queries)

        tfidf_pair_hits = 0
        tfidf_query_hits = 0

        exact_tfidf_pair_hits = 0
        dense_tfidf_pair_hits = 0
        dense_tfidf_query_hits = 0

        full_union_pair_hits = 0
        full_union_query_hits = 0
        full_union_cand_count = 0

        # Use primary dense (name_address_country) for dense component in union
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_step7 = SentenceTransformer(args.model_name, device=device)
        q_texts_primary_7 = prepare_text(s1_df, "name_address_country")
        c_texts_primary_7 = prepare_text(corpus_df, "name_address_country")
        q_emb_7 = model_step7.encode(q_texts_primary_7, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)
        c_emb_7 = model_step7.encode(c_texts_primary_7, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)
        del model_step7
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        effective_nlist_7 = min(nlist_val, len(corpus_ids))
        target_nprobe_7 = min(args.nprobe_fixed, effective_nlist_7)
        idx_7, _, eff_7 = build_ivfflat_index(c_emb_7, nlist=nlist_val)
        _, dense_indices_7, _ = search_ivfflat_index(idx_7, q_emb_7, effective_nlist=eff_7, nprobe=target_nprobe_7, top_k=args.top_k)
        del c_emb_7, q_emb_7, idx_7
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        for i, q_id in enumerate(s1_ids):
            if q_id not in gt_dict:
                continue
            true_set = gt_dict[q_id]
            e_set = exact_dict.get(q_id, set())
            d_set = set(corpus_ids[idx] for idx in dense_indices_7[i])
            t_set = set(corpus_ids[idx] for idx in tfidf_indices[i])

            # TF-IDF only
            t_hits = len(t_set & true_set)
            tfidf_pair_hits += t_hits
            if t_hits > 0:
                tfidf_query_hits += 1

            # Exact + TF-IDF
            et_set = e_set | t_set
            exact_tfidf_pair_hits += len(et_set & true_set)

            # Dense + TF-IDF
            dt_set = d_set | t_set
            dt_hits = len(dt_set & true_set)
            dense_tfidf_pair_hits += dt_hits
            if dt_hits > 0:
                dense_tfidf_query_hits += 1

            # Exact + Dense + TF-IDF union
            full_union = e_set | d_set | t_set
            full_union_cand_count += len(full_union)
            fu_hits = len(full_union & true_set)
            full_union_pair_hits += fu_hits
            if fu_hits > 0:
                full_union_query_hits += 1

        tfidf_pair_recall = round(tfidf_pair_hits / total_true * 100, 2) if total_true > 0 else 0
        tfidf_query_recall = round(tfidf_query_hits / n_q * 100, 2) if n_q > 0 else 0
        exact_tfidf_recall = round(exact_tfidf_pair_hits / total_true * 100, 2) if total_true > 0 else 0
        dense_tfidf_recall = round(dense_tfidf_pair_hits / total_true * 100, 2) if total_true > 0 else 0
        full_union_recall = round(full_union_pair_hits / total_true * 100, 2) if total_true > 0 else 0
        avg_cands_full = round(full_union_cand_count / n_q, 2) if n_q > 0 else 0

        print(f"\n  TF-IDF Pair Recall@{tfidf_top_k}:              {tfidf_pair_recall:.2f}%")
        print(f"  Exact + TF-IDF Pair Recall:              {exact_tfidf_recall:.2f}%")
        print(f"  Dense + TF-IDF Pair Recall:              {dense_tfidf_recall:.2f}%")
        print(f"  Exact + Dense + TF-IDF Pair Recall:      {full_union_recall:.2f}%")
        print(f"  Avg Candidates/S1 (Full Union):           {avg_cands_full:.2f}")

        step7_results = {
            "nprobe": target_nprobe_7,
            "top_k": args.top_k,
            "tfidf_ngram_range": [3, 5],
            "tfidf_max_features": 200000,
            "tfidf_pair_recall_%": tfidf_pair_recall,
            "tfidf_query_recall_%": tfidf_query_recall,
            "exact_tfidf_pair_recall_%": exact_tfidf_recall,
            "dense_tfidf_pair_recall_%": dense_tfidf_recall,
            "exact_dense_tfidf_pair_recall_%": full_union_recall,
            "avg_candidates_per_s1_full_union": avg_cands_full,
            "total_candidates_full_union": full_union_cand_count,
            "tfidf_search_time_sec": round(tfidf_search_time, 3)
        }

        out_path = os.path.join(args.output_dir, "step7_tfidf_union.json")
        with open(out_path, "w") as f:
            json.dump(step7_results, f, indent=2)
        print(f"\nStep 7 report written to: {out_path}")

if __name__ == "__main__":
    main()
