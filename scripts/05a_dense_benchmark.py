import os
import time
import argparse
import polars as pl
import numpy as np
import json
import gc
from datetime import datetime

import torch
from sentence_transformers import SentenceTransformer
import faiss
import faiss.contrib.torch_utils  # Allows feeding torch tensors directly to faiss GPU

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5A: Dense Retrieval GPU Benchmark")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Canonical Parquet directory")
    parser.add_argument("--ground-truth", type=str, default="data/student_resource/dataset/train/train.csv", help="Path to ground truth labels")
    parser.add_argument("--output-dir", type=str, default="artifacts/benchmark", help="Output directory")
    parser.add_argument("--n-queries", type=int, default=10000, help="Number of queries to subset")
    parser.add_argument("--n-corpus", type=int, default=100000, help="Number of corpus documents to subset")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 for Faiss Index to save GPU memory")
    parser.add_argument("--representation", type=str, default="name_country", choices=["name", "name_country", "name_address_country"], help="Text representation")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for embedding")
    return parser.parse_args()

def prepare_text(df: pl.DataFrame, rep: str) -> list:
    if rep == "name":
        texts = df["name_norm"].fill_null("").to_list()
    elif rep == "name_country":
        texts = df.select(
            pl.concat_str(
                [pl.col("name_norm").fill_null(""), pl.col("country").fill_null("")], separator=" | "
            )
        ).to_series().to_list()
    elif rep == "name_address_country":
        texts = df.select(
            pl.concat_str(
                [pl.col("name_norm").fill_null(""), pl.col("address_norm").fill_null(""), pl.col("country").fill_null("")], separator=" | "
            )
        ).to_series().to_list()
    return texts

def run_benchmark():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"==================================================")
    print(f" PHASE 5A: DENSE GPU BENCHMARK")
    print(f" Model: {args.model_name} | Rep: {args.representation} | FP16: {args.fp16}")
    print(f" Subset: {args.n_queries} Queries x {args.n_corpus} Corpus")
    print(f"==================================================")
    
    # 1. Load Data Subset
    s1_path = os.path.join(args.data_dir, "train", "train_source1.parquet")
    s2_path = os.path.join(args.data_dir, "train", "train_source2.parquet")
    s3_path = os.path.join(args.data_dir, "train", "train_source3.parquet")
    
    # We only need enough cols for representation + id
    select_cols = ["entity_id", "source", "name_norm", "address_norm", "country"]
    
    print("Loading queries subset...")
    s1_df = pl.read_parquet(s1_path, columns=select_cols).head(args.n_queries)
    s1_ids = s1_df["entity_id"].to_numpy()
    
    print("Loading corpus subset...")
    s2_df = pl.read_parquet(s2_path, columns=select_cols)
    s3_df = pl.read_parquet(s3_path, columns=select_cols)
    corpus_df = pl.concat([s2_df, s3_df]).head(args.n_corpus)
    corpus_ids = corpus_df["entity_id"].to_numpy()
    
    del s2_df, s3_df
    
    # Parse ground truth to find matches that ACTUALLY EXIST in our corpus subset
    valid_corpus_set = set(corpus_ids)
    
    print("Loading Ground Truth...")
    try:
        if args.ground_truth.endswith(".tsv"):
            gt_df = pl.read_csv(args.ground_truth, separator="\t")
        else:
            gt_df = pl.read_csv(args.ground_truth)
            
        col_s1 = gt_df.columns[0]
        col_s2 = gt_df.columns[1]
        
        gt_dict = {}
        for row in gt_df.iter_rows():
            q_id = str(row[0])
            m_ids_raw = str(row[1])
            # Handle possible separators
            for sep in ["|", ",", " "]:
                if sep in m_ids_raw:
                    m_ids = [x.strip() for x in m_ids_raw.split(sep)]
                    break
            else:
                m_ids = [m_ids_raw.strip()]
            
            # Only keep targets that are in our corpus subset
            valid_targets = [m for m in m_ids if m in valid_corpus_set]
            if valid_targets:
                gt_dict[q_id] = set(valid_targets)
    except Exception as e:
        print(f"Warning: Could not parse ground truth properly: {e}")
        gt_dict = {}

    
    # 2. Text Representation
    print("Preparing text representations...")
    queries_text = prepare_text(s1_df, args.representation)
    corpus_text = prepare_text(corpus_df, args.representation)
    del s1_df, corpus_df
    gc.collect()
    
    # 3. Model Loading
    print("Loading SentenceTransformer onto GPU...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not detected! Benchmark will run on CPU.")
    
    model = SentenceTransformer(args.model_name, device=device)
    dim = model.get_sentence_embedding_dimension()
    
    # 4. Generate Embeddings
    print("Encoding Corpus...")
    t0 = time.time()
    corpus_embeddings = model.encode(corpus_text, batch_size=args.batch_size, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
    corpus_time = time.time() - t0
    
    print("Encoding Queries...")
    t0 = time.time()
    query_embeddings = model.encode(queries_text, batch_size=args.batch_size, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
    query_time = time.time() - t0
    
    del queries_text, corpus_text
    
    # 5. Build FAISS Index
    print("Building FAISS Index...")
    t0 = time.time()
    
    # CPU fallback or GPU index
    if device == "cuda":
        res = faiss.StandardGpuResources()
        if args.fp16:
            # useGpu=True, useFloat16=True
            config = faiss.GpuIndexFlatConfig()
            config.useFloat16 = True
            config.device = 0
            index = faiss.GpuIndexFlatIP(res, dim, config)
        else:
            index = faiss.GpuIndexFlatIP(res, dim)
    else:
        index = faiss.IndexFlatIP(dim)
    
    # Move to numpy if needed, or use faiss torch bindings
    if isinstance(corpus_embeddings, torch.Tensor) and device == "cuda":
        if args.fp16:
            corpus_np = corpus_embeddings.cpu().numpy().astype(np.float16)
            index.add(corpus_np)
        else:
            index.add(corpus_embeddings.cpu().numpy().astype(np.float32))
    else:
        index.add(corpus_embeddings)
        
    index_time = time.time() - t0
    
    # 6. Search
    print("Searching FAISS Index...")
    t0 = time.time()
    
    k_max = 50
    if isinstance(query_embeddings, torch.Tensor):
        if args.fp16:
            query_np = query_embeddings.cpu().numpy().astype(np.float16)
            scores, indices = index.search(query_np, k_max)
        else:
            scores, indices = index.search(query_embeddings.cpu().numpy().astype(np.float32), k_max)
    else:
        scores, indices = index.search(query_embeddings, k_max)
        
    search_time = time.time() - t0
    
    # 7. Evaluate Recall
    print("Evaluating Recall...")
    recall_k = [1, 5, 10, 25, 50]
    hits = {k: 0 for k in recall_k}
    valid_queries = 0
    
    for i, q_id in enumerate(s1_ids):
        if q_id not in gt_dict:
            continue # Skip queries where true match is not in our corpus subset
            
        valid_queries += 1
        true_targets = gt_dict[q_id]
        
        retrieved_ids = [corpus_ids[idx] for idx in indices[i]]
        
        for k in recall_k:
            retrieved_k = set(retrieved_ids[:k])
            # If ANY true target is in the top-K, it's a hit for candidate generation
            if true_targets.intersection(retrieved_k):
                hits[k] += 1
                
    results = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "timing_sec": {
            "corpus_encode": round(corpus_time, 2),
            "query_encode": round(query_time, 2),
            "faiss_build": round(index_time, 2),
            "faiss_search": round(search_time, 2),
        },
        "throughput": {
            "corpus_encode_per_sec": round(args.n_corpus / corpus_time, 2),
            "query_search_per_sec": round(args.n_queries / search_time, 2),
        },
        "evaluation": {
            "valid_queries_in_subset": valid_queries,
            "recall_metrics": {f"Recall@{k}": round(hits[k] / valid_queries, 4) if valid_queries > 0 else 0 for k in recall_k}
        }
    }
    
    print("\nBENCHMARK RESULTS:")
    print(json.dumps(results, indent=2))
    
    report_path = os.path.join(args.output_dir, f"dense_bench_{args.representation}_{'fp16' if args.fp16 else 'fp32'}.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSaved benchmark report to {report_path}")
    print("Phase 5A Complete.")

if __name__ == "__main__":
    run_benchmark()
