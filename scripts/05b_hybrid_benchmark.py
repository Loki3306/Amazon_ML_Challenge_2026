import os
import time
import argparse
import polars as pl
import numpy as np
import json
import gc
from datetime import datetime
from collections import defaultdict

import torch
from sentence_transformers import SentenceTransformer
import faiss
import faiss.contrib.torch_utils

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5B: Hybrid (Exact U Dense) GPU Benchmark")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Canonical Parquet directory")
    parser.add_argument("--ground-truth", type=str, default="data/student_resource/dataset/train/train_ground_truth.tsv", help="Path to ground truth labels")
    parser.add_argument("--output-dir", type=str, default="artifacts/benchmark", help="Output directory")
    parser.add_argument("--n-queries", type=int, default=10000, help="Number of queries to subset")
    parser.add_argument("--n-corpus", type=int, default=100000, help="Number of corpus documents to subset")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--representation", type=str, default="name_address_country", choices=["name", "name_country", "name_address_country"])
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for embedding")
    return parser.parse_args()

def prepare_text(df: pl.DataFrame, rep: str) -> list:
    if rep == "name":
        return df["name_norm"].fill_null("").to_list()
    elif rep == "name_country":
        return df.select(pl.concat_str([pl.col("name_norm").fill_null(""), pl.col("country").fill_null("")], separator=" | ")).to_series().to_list()
    elif rep == "name_address_country":
        return df.select(pl.concat_str([pl.col("name_norm").fill_null(""), pl.col("address_norm").fill_null(""), pl.col("country").fill_null("")], separator=" | ")).to_series().to_list()

def run_hybrid_benchmark():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"==================================================")
    print(f" PHASE 5B: HYBRID (EXACT ∪ DENSE) BENCHMARK")
    print(f" Model: {args.model_name} | Rep: {args.representation}")
    print(f"==================================================")
    
    # 1. Load Data
    s1_path = os.path.join(args.data_dir, "train", "train_source1.parquet")
    s2_path = os.path.join(args.data_dir, "train", "train_source2.parquet")
    s3_path = os.path.join(args.data_dir, "train", "train_source3.parquet")
    
    select_cols = ["entity_id", "source", "name_norm", "address_norm", "country"]
    
    print("Loading datasets...")
    s1_df = pl.read_parquet(s1_path, columns=select_cols).head(args.n_queries)
    s2_df = pl.read_parquet(s2_path, columns=select_cols)
    s3_df = pl.read_parquet(s3_path, columns=select_cols)
    corpus_df = pl.concat([s2_df, s3_df]).head(args.n_corpus)
    
    s1_ids = s1_df["entity_id"].to_numpy()
    corpus_ids = corpus_df["entity_id"].to_numpy()
    valid_corpus_set = set(corpus_ids)
    
    # 2. Parse Ground Truth
    print("Loading Ground Truth...")
    gt_dict = {}
    if args.ground_truth.endswith(".tsv"):
        gt_df = pl.read_csv(args.ground_truth, separator="\t")
    else:
        gt_df = pl.read_csv(args.ground_truth)
        
    for row in gt_df.iter_rows():
        q_id = str(row[0])
        m_ids_raw = str(row[1])
        for sep in ["|", ",", " "]:
            if sep in m_ids_raw:
                m_ids = [x.strip() for x in m_ids_raw.split(sep)]
                break
        else:
            m_ids = [m_ids_raw.strip()]
        
        valid_targets = [m for m in m_ids if m in valid_corpus_set]
        if valid_targets:
            gt_dict[q_id] = set(valid_targets)
            
    valid_queries = len([q for q in s1_ids if q in gt_dict])
    total_valid_pairs = sum(len(gt_dict[q]) for q in s1_ids if q in gt_dict)
    print(f"Evaluating on {valid_queries} queries with {total_valid_pairs} total true pairs in subset.")

    # 3. Exact Blocking (In-Memory on Subset)
    print("Running Exact Blocking (Name & Address)...")
    s1_valid_names = s1_df.filter(pl.col("name_norm") != "")
    corpus_valid_names = corpus_df.filter(pl.col("name_norm") != "")
    name_matches = s1_valid_names.join(corpus_valid_names, on="name_norm", how="inner", suffix="_c")
    
    s1_valid_addrs = s1_df.filter(pl.col("address_norm") != "")
    corpus_valid_addrs = corpus_df.filter(pl.col("address_norm") != "")
    addr_matches = s1_valid_addrs.join(corpus_valid_addrs, on="address_norm", how="inner", suffix="_c")
    
    exact_match_dict = defaultdict(set)
    for row in name_matches.iter_rows(named=True):
        exact_match_dict[row["entity_id"]].add(row["entity_id_c"])
    for row in addr_matches.iter_rows(named=True):
        exact_match_dict[row["entity_id"]].add(row["entity_id_c"])
        
    # 4. Dense Retrieval
    print("Encoding texts for Dense Retrieval...")
    queries_text = prepare_text(s1_df, args.representation)
    corpus_text = prepare_text(corpus_df, args.representation)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model_name, device=device)
    dim = model.get_sentence_embedding_dimension()
    
    corpus_emb = model.encode(corpus_text, batch_size=args.batch_size, convert_to_tensor=True, normalize_embeddings=True)
    query_emb = model.encode(queries_text, batch_size=args.batch_size, convert_to_tensor=True, normalize_embeddings=True)
    
    print("Building FAISS and Searching Top-50...")
    if device == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, dim)
        index.add(corpus_emb.cpu().numpy().astype(np.float32))
        scores, dense_indices = index.search(query_emb.cpu().numpy().astype(np.float32), 50)
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(corpus_emb)
        scores, dense_indices = index.search(query_emb, 50)
        
    # 5. Hybrid Evaluation
    print("Evaluating Hybrid Recall...")
    k_vals = [1, 5, 10, 25, 50]
    
    metrics = {
        "query_recall": {"Exact": 0, "Dense": {k: 0 for k in k_vals}, "Hybrid": {k: 0 for k in k_vals}},
        "pair_recall": {"Exact": 0, "Dense": {k: 0 for k in k_vals}, "Hybrid": {k: 0 for k in k_vals}},
        "candidates_generated": {"Exact": 0, "Dense": {k: 0 for k in k_vals}, "Hybrid": {k: 0 for k in k_vals}}
    }
    
    for i, q_id in enumerate(s1_ids):
        if q_id not in gt_dict:
            continue
            
        true_set = gt_dict[q_id]
        n_true = len(true_set)
        
        # Exact sets
        exact_set = exact_match_dict.get(q_id, set())
        
        # Dense sets
        dense_list = [corpus_ids[idx] for idx in dense_indices[i]]
        
        # Exact metrics
        metrics["candidates_generated"]["Exact"] += len(exact_set)
        exact_hits = len(exact_set.intersection(true_set))
        metrics["pair_recall"]["Exact"] += exact_hits
        if exact_hits > 0:
            metrics["query_recall"]["Exact"] += 1
            
        # K-level metrics
        for k in k_vals:
            dense_set_k = set(dense_list[:k])
            hybrid_set_k = exact_set.union(dense_set_k)
            
            # Dense metrics
            metrics["candidates_generated"]["Dense"][k] += len(dense_set_k)
            dense_hits = len(dense_set_k.intersection(true_set))
            metrics["pair_recall"]["Dense"][k] += dense_hits
            if dense_hits > 0:
                metrics["query_recall"]["Dense"][k] += 1
                
            # Hybrid metrics
            metrics["candidates_generated"]["Hybrid"][k] += len(hybrid_set_k)
            hybrid_hits = len(hybrid_set_k.intersection(true_set))
            metrics["pair_recall"]["Hybrid"][k] += hybrid_hits
            if hybrid_hits > 0:
                metrics["query_recall"]["Hybrid"][k] += 1

    # Format Output
    report = {
        "subset": f"{args.n_queries}Q x {args.n_corpus}C",
        "valid_queries": valid_queries,
        "total_true_pairs": total_valid_pairs,
        "results": {
            "query_recall_%": {
                "Exact": round(metrics["query_recall"]["Exact"] / valid_queries * 100, 2),
                "Dense": {str(k): round(metrics["query_recall"]["Dense"][k] / valid_queries * 100, 2) for k in k_vals},
                "Hybrid": {str(k): round(metrics["query_recall"]["Hybrid"][k] / valid_queries * 100, 2) for k in k_vals}
            },
            "pair_recall_%": {
                "Exact": round(metrics["pair_recall"]["Exact"] / total_valid_pairs * 100, 2),
                "Dense": {str(k): round(metrics["pair_recall"]["Dense"][k] / total_valid_pairs * 100, 2) for k in k_vals},
                "Hybrid": {str(k): round(metrics["pair_recall"]["Hybrid"][k] / total_valid_pairs * 100, 2) for k in k_vals}
            },
            "avg_candidates_per_query": {
                "Exact": round(metrics["candidates_generated"]["Exact"] / valid_queries, 2),
                "Dense": {str(k): round(metrics["candidates_generated"]["Dense"][k] / valid_queries, 2) for k in k_vals},
                "Hybrid": {str(k): round(metrics["candidates_generated"]["Hybrid"][k] / valid_queries, 2) for k in k_vals}
            }
        }
    }
    
    print("\nHYBRID EVALUATION RESULTS:")
    print(json.dumps(report, indent=2))
    
    report_path = os.path.join(args.output_dir, "hybrid_benchmark.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    run_hybrid_benchmark()
