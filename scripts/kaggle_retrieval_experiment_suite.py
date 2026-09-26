"""
Kaggle GPU Standalone Execution Script: Retrieval Recall Optimization Suite (OOM-Safe)
===================================================================================
This script is self-contained and memory-bounded for Kaggle GPU notebook execution (14-16GB VRAM & 30GB RAM).

Key Memory Management & OOM Prevention Features:
  1. Chunked Corpus Encoding: Encodes 10.3M corpus in 200,000 text chunks on CPU numpy.
  2. Immediate Memory Reclamation: GPU VRAM peak during encoding is <300 MB.
  3. FAISS IVFFlat Index: Trained ONCE on 1M sample embeddings.
  4. Streamed Index Population: Chunks are added directly to FAISS index.
  5. Index Reuse: Reused across nprobe (128, 256, 512, 1024) and Top-K (50, 100, 200, 300) sweeps.
  6. Query Search Batching: Search executed in 4096-query batches to avoid FAISS GPU TemporaryMemoryOverflow.
  7. SentenceTransformer Unloading: Model freed from VRAM before search and TF-IDF steps.
  8. Memory-Bounded TF-IDF: TF-IDF similarity computed in 50-query batches (~2.0 GB CPU RAM per batch).

Run on Kaggle GPU (T4 / P100):
  python scripts/kaggle_retrieval_experiment_suite.py \
    --data-dir /kaggle/working/data/processed \
    --ground-truth /kaggle/input/student-resource-amazonml/dataset/train/train_ground_truth.tsv \
    --input-dir /kaggle/input/student-resource-amazonml/dataset/train \
    --output-dir /kaggle/working/reports/retrieval
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
from sklearn.metrics.pairwise import cosine_similarity


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
    parser = argparse.ArgumentParser(description="Kaggle Retrieval Optimization Execution Suite")
    parser.add_argument("--data-dir", type=str, default="/kaggle/working/data/processed", help="Path to processed parquet data")
    parser.add_argument("--input-dir", type=str, default="/kaggle/input/student-resource-amazonml/dataset/train", help="Raw dataset TSV folder")
    parser.add_argument("--ground-truth", type=str, default="/kaggle/input/student-resource-amazonml/dataset/train/train_ground_truth.tsv", help="Ground truth TSV path")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/reports/retrieval", help="Output reports folder")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--n-queries", type=int, default=0, help="0 for full scale, >0 for subset benchmarking")
    parser.add_argument("--n-corpus", type=int, default=0, help="0 for full scale, >0 for subset benchmarking")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for model encoding")
    parser.add_argument("--chunk-size", type=int, default=200000, help="Chunk size for streaming encoding")
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
    else:
        raise ValueError(f"Unknown representation: {representation}")


def encode_texts_chunked(model, texts: list[str], batch_size: int = 2048, chunk_size: int = 200000) -> np.ndarray:
    """Encodes texts in chunks to prevent PyTorch CUDA OOM. Memory stays bounded to 1 batch."""
    n = len(texts)
    dim = model.get_sentence_embedding_dimension()
    embeddings = np.empty((n, dim), dtype=np.float32)

    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunk_texts = texts[i:end]
        chunk_emb = model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embeddings[i:end] = chunk_emb.astype(np.float32)
        del chunk_emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return embeddings


def build_and_populate_ivfflat_index(model, c_texts: list[str], nlist_target: int = 16384, chunk_size: int = 200000, batch_size: int = 2048):
    """Trains FAISS IVFFlat ONCE on a sample and populates it incrementally in chunks."""
    total_corpus = len(c_texts)
    dim = model.get_sentence_embedding_dimension()
    effective_nlist = min(nlist_target, max(16, total_corpus // 10))

    print(f"  Sample-encoding {min(1000000, total_corpus):,} corpus texts for FAISS training...")
    train_sample_texts = c_texts[:min(1000000, total_corpus)]
    train_sample = encode_texts_chunked(model, train_sample_texts, batch_size=batch_size, chunk_size=chunk_size)

    print(f"  Training FAISS IVFFlat Index (nlist={effective_nlist}, dim={dim})...")
    quantizer = faiss.IndexFlatIP(dim)
    cpu_index = faiss.IndexIVFFlat(quantizer, dim, effective_nlist, faiss.METRIC_INNER_PRODUCT)

    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        gpu_train_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_train_index.train(train_sample)
        cpu_index = faiss.index_gpu_to_cpu(gpu_train_index)
        del gpu_train_index, res
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        cpu_index.train(train_sample)

    del train_sample
    gc.collect()

    print(f"  Populating FAISS index with {total_corpus:,} vectors in streaming {chunk_size:,} chunks...")
    if torch.cuda.is_available():
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        try:
            res = faiss.StandardGpuResources()
            search_index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
            print("  FAISS index transferred to GPU (FP16 mode).")
        except Exception as e:
            print(f"  GPU Index transfer warning ({e}). Using CPU Index...")
            search_index = cpu_index
    else:
        search_index = cpu_index

    t0 = time.time()
    for i in range(0, total_corpus, chunk_size):
        end = min(i + chunk_size, total_corpus)
        chunk_texts = c_texts[i:end]
        chunk_emb = model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        search_index.add(chunk_emb)
        del chunk_emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print(f"  FAISS index populated in {time.time()-t0:.2f}s.")
    return search_index, effective_nlist


def search_ivfflat_index_batched(index, q_emb: np.ndarray, effective_nlist: int, nprobe: int, top_k: int, batch_size: int = 4096):
    """Executes FAISS search in mini-batches to prevent FAISS GPU TemporaryMemoryOverflow."""
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
    num_queries = q_emb.shape[0]
    all_scores = []
    all_indices = []

    for i in range(0, num_queries, batch_size):
        end = min(i + batch_size, num_queries)
        q_batch = q_emb[i:end].astype(np.float32)
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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*75)
    print(" KAGGLE GPU RETRIEVAL RECALL OPTIMIZATION EXPERIMENT SUITE (OOM-SAFE)")
    print(f" Execution Device: {device.upper()}")
    print("="*75)

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

    print(f"Data ready: {len(s1_ids):,} S1 queries, {len(corpus_ids):,} S2/S3 corpus records.")

    valid_queries = [q for q in s1_ids if q in gt_dict]
    total_true_pairs = sum(len(gt_dict[q]) for q in valid_queries)

    # 3. Exact Match Baseline Candidates
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

    exact_pair_hits = sum(len(exact_dict.get(q, set()) & gt_dict[q]) for q in valid_queries)
    exact_pair_recall = round(exact_pair_hits / total_true_pairs * 100, 2) if total_true_pairs > 0 else 0
    print(f"  Exact Pair Recall Baseline: {exact_pair_recall:.2f}%")

    # 4. Dense Retrieval Model Setup
    print(f"\nLoading SentenceTransformer '{args.model_name}' onto {device}...")
    model = SentenceTransformer(args.model_name, device=device)

    # Encode primary text: name + address + country in chunked mode
    print("Encoding primary representation ('name | address | country')...")
    q_texts_primary = prepare_text(s1_df, "name_address_country")
    c_texts_primary = prepare_text(corpus_df, "name_address_country")

    print(f"  Encoding {len(q_texts_primary):,} queries in chunked mode...")
    q_emb_primary = encode_texts_chunked(model, q_texts_primary, batch_size=args.batch_size, chunk_size=args.chunk_size)

    print(f"  Building & populating FAISS index for {len(c_texts_primary):,} corpus items...")
    nlist_val = 16384 if len(corpus_ids) >= 500000 else max(16, len(corpus_ids) // 10)
    index_primary, effective_nlist = build_and_populate_ivfflat_index(
        model, c_texts_primary, nlist_target=nlist_val, chunk_size=args.chunk_size, batch_size=args.batch_size
    )

    # ---------------------------------------------------------
    # STEP 1: NPROBE SWEEP (Top-K=50)
    # ---------------------------------------------------------
    print("\n" + "="*65)
    print(" STEP 1: NPROBE SWEEP (128, 256, 512, 1024 at K=50)")
    print("="*65)

    nprobe_results = []
    nprobes = [128, 256, 512, 1024]
    
    for p in nprobes:
        if p > effective_nlist:
            continue
        scores, indices, search_time = search_ivfflat_index_batched(index_primary, q_emb_primary, effective_nlist, p, 50)

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

        d_rec = round(d_hits / total_true_pairs * 100, 2)
        h_rec = round(h_hits / total_true_pairs * 100, 2)
        avg_cand = round(h_cands / len(valid_queries), 2)

        res_item = {
            "nprobe": p,
            "top_k": 50,
            "dense_pair_recall_%": d_rec,
            "hybrid_pair_recall_%": h_rec,
            "avg_candidates_per_query": avg_cand,
            "search_time_sec": round(search_time, 2)
        }
        nprobe_results.append(res_item)
        print(f"  nprobe={p:4d} | Dense Recall: {d_rec:6.2f}% | Hybrid Recall: {h_rec:6.2f}% | Avg Cands: {avg_cand:6.2f} | Time: {search_time:.2f}s")

    with open(os.path.join(args.output_dir, "kaggle_step1_nprobe_sweep.json"), "w") as f:
        json.dump(nprobe_results, f, indent=2)

    # ---------------------------------------------------------
    # STEP 3: TOP-K SWEEP (at nprobe=512)
    # ---------------------------------------------------------
    print("\n" + "="*65)
    print(" STEP 3: TOP-K SWEEP (50, 100, 200, 300 at nprobe=512)")
    print("="*65)

    best_p = min(512, effective_nlist)
    scores_300, indices_300, search_t_300 = search_ivfflat_index_batched(index_primary, q_emb_primary, effective_nlist, best_p, 300)

    topk_results = []
    for k in [50, 100, 200, 300]:
        h_hits, h_cands = 0, 0
        d_hits = 0
        for i, q_id in enumerate(s1_ids):
            if q_id not in gt_dict:
                continue
            true_set = gt_dict[q_id]
            e_set = exact_dict.get(q_id, set())
            d_set = set(corpus_ids[idx] for idx in indices_300[i, :k])
            h_set = e_set.union(d_set)
            d_hits += len(d_set.intersection(true_set))
            h_hits += len(h_set.intersection(true_set))
            h_cands += len(h_set)

        d_rec = round(d_hits / total_true_pairs * 100, 2)
        h_rec = round(h_hits / total_true_pairs * 100, 2)
        avg_cand = round(h_cands / len(valid_queries), 2)

        item = {"K": k, "nprobe": best_p, "dense_pair_recall_%": d_rec, "hybrid_pair_recall_%": h_rec, "avg_candidates_per_query": avg_cand}
        topk_results.append(item)
        print(f"  K={k:3d} | Dense Recall: {d_rec:6.2f}% | Hybrid Recall: {h_rec:6.2f}% | Avg Cands/S1: {avg_cand:6.2f}")

    with open(os.path.join(args.output_dir, "kaggle_step3_topk_sweep.json"), "w") as f:
        json.dump(topk_results, f, indent=2)

    # Free memory
    del index_primary, q_emb_primary
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---------------------------------------------------------
    # STEP 5: DENSE REPRESENTATION COMPARISON (nprobe=512, K=50)
    # Compares: name_address_country vs name_country vs name_address
    # ---------------------------------------------------------
    print("\n" + "="*65)
    print(" STEP 5: DENSE REPRESENTATION SWEEP (at nprobe=512, K=50)")
    print("="*65)

    representations = ["name_address_country", "name_country", "name_address"]
    rep_results = {}
    rep_indices_dict = {}

    for rep in representations:
        print(f"\n  Encoding representation: '{rep}'...")
        q_texts_r = prepare_text(s1_df, rep)
        c_texts_r = prepare_text(corpus_df, rep)

        q_emb_r = encode_texts_chunked(model, q_texts_r, batch_size=args.batch_size, chunk_size=args.chunk_size)
        idx_r, eff_r = build_and_populate_ivfflat_index(model, c_texts_r, nlist_target=nlist_val, chunk_size=args.chunk_size, batch_size=args.batch_size)

        target_p = min(512, eff_r)
        sc_r, ind_r, st_r = search_ivfflat_index_batched(idx_r, q_emb_r, eff_r, target_p, 50)
        rep_indices_dict[rep] = ind_r

        d_hits, h_hits, h_cands = 0, 0, 0
        for i, q_id in enumerate(s1_ids):
            if q_id not in gt_dict:
                continue
            true_set = gt_dict[q_id]
            e_set = exact_dict.get(q_id, set())
            d_set = set(corpus_ids[idx] for idx in ind_r[i])
            h_set = e_set.union(d_set)
            d_hits += len(d_set.intersection(true_set))
            h_hits += len(h_set.intersection(true_set))
            h_cands += len(h_set)

        d_rec = round(d_hits / total_true_pairs * 100, 2)
        h_rec = round(h_hits / total_true_pairs * 100, 2)
        avg_cand = round(h_cands / len(valid_queries), 2)

        rep_results[rep] = {
            "representation": rep,
            "nprobe": target_p,
            "top_k": 50,
            "dense_pair_recall_%": d_rec,
            "hybrid_pair_recall_%": h_rec,
            "avg_candidates_per_query": avg_cand,
            "search_time_sec": round(st_r, 2)
        }
        print(f"  [{rep:25s}] Dense Recall: {d_rec:6.2f}% | Hybrid Recall: {h_rec:6.2f}% | Avg Cands: {avg_cand:6.2f} | Time: {st_r:.2f}s")

        del idx_r, q_emb_r
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    with open(os.path.join(args.output_dir, "kaggle_step5_representation_sweep.json"), "w") as f:
        json.dump(rep_results, f, indent=2)

    # Free model completely before TF-IDF step
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---------------------------------------------------------
    # STEP 6: MULTI-REPRESENTATION DENSE UNION
    # Combines Dense(name_address_country) + Dense(name_country) + Exact
    # ---------------------------------------------------------
    print("\n" + "="*65)
    print(" STEP 6: MULTI-REPRESENTATION DENSE UNION (K=50)")
    print("="*65)

    base_hybrid_recall = rep_results["name_address_country"]["hybrid_pair_recall_%"]

    union_pair_hits = 0
    union_cands = 0
    for i, q_id in enumerate(s1_ids):
        if q_id not in gt_dict:
            continue
        true_set = gt_dict[q_id]
        e_set = exact_dict.get(q_id, set())

        d_union = set()
        for rep in ["name_address_country", "name_country"]:
            if rep in rep_indices_dict:
                d_union |= set(corpus_ids[idx] for idx in rep_indices_dict[rep][i])

        full_union = e_set | d_union
        union_cands += len(full_union)
        union_pair_hits += len(full_union & true_set)

    union_pair_recall = round(union_pair_hits / total_true_pairs * 100, 2)
    avg_cands_union = round(union_cands / len(valid_queries), 2)
    incremental_gain = round(union_pair_recall - base_hybrid_recall, 2)

    print(f"  Baseline Hybrid (name_address_country + Exact): {base_hybrid_recall:.2f}%")
    print(f"  Multi-Rep Union (name_address_country + name_country + Exact): {union_pair_recall:.2f}%")
    print(f"  Incremental Gain: +{incremental_gain:.2f} percentage points")
    print(f"  Avg Candidates/S1: {avg_cands_union:.2f}")

    step6_results = {
        "nprobe": 512,
        "top_k": 50,
        "representations_used": ["name_address_country", "name_country"],
        "baseline_hybrid_pair_recall_%": base_hybrid_recall,
        "multi_rep_union_pair_recall_%": union_pair_recall,
        "incremental_gain_%": incremental_gain,
        "avg_candidates_per_query": avg_cands_union
    }
    with open(os.path.join(args.output_dir, "kaggle_step6_multirep_union.json"), "w") as f:
        json.dump(step6_results, f, indent=2)

    # ---------------------------------------------------------
    # STEP 7: CHARACTER TF-IDF CANDIDATE GENERATOR & HYBRID RECALL EVALUATION
    # Uses ngram_range=(3,5) on business_name + business_address
    # Evaluates Exact + Dense + TF-IDF union
    # ---------------------------------------------------------
    print("\n" + "="*65)
    print(" STEP 7: CHARACTER TF-IDF CANDIDATE GENERATOR (K=50)")
    print("="*65)

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
        sublinear_tf=True,
        dtype=np.float32
    )
    all_tfidf_texts = c_tfidf_texts + q_tfidf_texts
    tfidf.fit(all_tfidf_texts)
    del all_tfidf_texts
    gc.collect()

    c_tfidf_mat = tfidf.transform(c_tfidf_texts)
    q_tfidf_mat = tfidf.transform(q_tfidf_texts)
    del c_tfidf_texts, q_tfidf_texts
    gc.collect()

    print(f"TF-IDF matrix built: corpus={c_tfidf_mat.shape}, queries={q_tfidf_mat.shape} in {time.time()-t0_tfidf:.2f}s", flush=True)

    tfidf_top_k = 50
    t0_tfidf_search = time.time()

    tfidf_indices_list = []
    # Mini-batch size of 50 queries keeps peak intermediate matrix at ~2.0 GB System RAM
    query_batch_size = 50
    n_queries_tfidf = q_tfidf_mat.shape[0]

    print(f"Searching TF-IDF Top-{tfidf_top_k} for {n_queries_tfidf} queries...", flush=True)
    for i in range(0, n_queries_tfidf, query_batch_size):
        q_batch = q_tfidf_mat[i:i + query_batch_size]
        sims = cosine_similarity(q_batch, c_tfidf_mat)
        top_k_batch = np.argpartition(sims, -tfidf_top_k, axis=1)[:, -tfidf_top_k:]
        for row_idx in range(top_k_batch.shape[0]):
            row_sorted = top_k_batch[row_idx][np.argsort(sims[row_idx, top_k_batch[row_idx]])[::-1]]
            tfidf_indices_list.append(row_sorted)
        del sims
        if (i // query_batch_size) % 100 == 0:
            print(f"  TF-IDF searched {min(i + query_batch_size, n_queries_tfidf)}/{n_queries_tfidf} queries...", flush=True)

    tfidf_indices = np.vstack(tfidf_indices_list)
    tfidf_search_time = time.time() - t0_tfidf_search
    print(f"TF-IDF search completed in {tfidf_search_time:.2f}s")

    tfidf_pair_hits = 0
    exact_tfidf_pair_hits = 0
    dense_tfidf_pair_hits = 0
    full_union_pair_hits = 0
    full_union_cand_count = 0

    primary_dense_indices = rep_indices_dict.get("name_address_country", None)

    for i, q_id in enumerate(s1_ids):
        if q_id not in gt_dict:
            continue
        true_set = gt_dict[q_id]
        e_set = exact_dict.get(q_id, set())
        d_set = set(corpus_ids[idx] for idx in primary_dense_indices[i]) if primary_dense_indices is not None else set()
        t_set = set(corpus_ids[idx] for idx in tfidf_indices[i])

        tfidf_pair_hits += len(t_set & true_set)
        exact_tfidf_pair_hits += len((e_set | t_set) & true_set)
        dense_tfidf_pair_hits += len((d_set | t_set) & true_set)

        full_union = e_set | d_set | t_set
        full_union_cand_count += len(full_union)
        full_union_pair_hits += len(full_union & true_set)

    tfidf_pair_recall = round(tfidf_pair_hits / total_true_pairs * 100, 2)
    exact_tfidf_recall = round(exact_tfidf_pair_hits / total_true_pairs * 100, 2)
    dense_tfidf_recall = round(dense_tfidf_pair_hits / total_true_pairs * 100, 2)
    full_union_recall = round(full_union_pair_hits / total_true_pairs * 100, 2)
    avg_cands_full = round(full_union_cand_count / len(valid_queries), 2)

    print(f"\n  TF-IDF Pair Recall@50:                     {tfidf_pair_recall:.2f}%")
    print(f"  Exact + TF-IDF Pair Recall:               {exact_tfidf_recall:.2f}%")
    print(f"  Dense + TF-IDF Pair Recall:               {dense_tfidf_recall:.2f}%")
    print(f"  Exact + Dense + TF-IDF Pair Recall:       {full_union_recall:.2f}%")
    print(f"  Avg Candidates/S1 (Full Union):            {avg_cands_full:.2f}")

    step7_results = {
        "nprobe": 512,
        "top_k": 50,
        "tfidf_ngram_range": [3, 5],
        "tfidf_max_features": 200000,
        "tfidf_pair_recall_%": tfidf_pair_recall,
        "exact_tfidf_pair_recall_%": exact_tfidf_recall,
        "dense_tfidf_pair_recall_%": dense_tfidf_recall,
        "exact_dense_tfidf_pair_recall_%": full_union_recall,
        "avg_candidates_per_query_full_union": avg_cands_full,
        "tfidf_search_time_sec": round(tfidf_search_time, 2)
    }
    with open(os.path.join(args.output_dir, "kaggle_step7_tfidf_union.json"), "w") as f:
        json.dump(step7_results, f, indent=2)

    # Master Summary Report
    master_summary = {
        "timestamp": datetime.now().isoformat(),
        "baseline_targets": {
            "exact_pair_recall_%": 28.19,
            "dense_top50_pair_recall_%": 78.22,
            "hybrid_pair_recall_%": 80.52
        },
        "kaggle_measured_results": {
            "exact_pair_recall_%": exact_pair_recall,
            "nprobe_sweep_k50": nprobe_results,
            "topk_sweep_nprobe512": topk_results,
            "representation_sweep": rep_results,
            "multirep_union": step6_results,
            "tfidf_hybrid_union": step7_results
        }
    }
    summary_path = os.path.join(args.output_dir, "retrieval_summary_report.json")
    with open(summary_path, "w") as f:
        json.dump(master_summary, f, indent=2)

    # Save copy to /kaggle/working/ if running under Kaggle
    if os.path.exists("/kaggle/working"):
        kw_summary = "/kaggle/working/retrieval_summary_report.json"
        with open(kw_summary, "w") as f:
            json.dump(master_summary, f, indent=2)

    print("\n" + "="*75)
    print(" KAGGLE RETRIEVAL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f" Saved full summary report to: {summary_path}")
    print("="*75)

if __name__ == "__main__":
    main()
