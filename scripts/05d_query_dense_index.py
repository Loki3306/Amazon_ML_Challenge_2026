import os
import time
import argparse
import polars as pl
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
import gc

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5D: Query Dense GPU Index")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Parquet directory")
    parser.add_argument("--split", type=str, default="train", help="Which split to process (train or test)")
    parser.add_argument("--index-dir", type=str, default="data/dense_index", help="Output directory of 5C")
    parser.add_argument("--output-dir", type=str, default="data/candidates", help="Where to save candidates")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--top-k", type=int, default=10, help="Number of candidates to retrieve")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for query embedding")
    parser.add_argument("--nprobe", type=int, default=32, help="Number of clusters to search (higher = slower but better recall)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure index dir is split-specific if it's the default
    if args.index_dir == "data/dense_index" and args.split == "test":
        args.index_dir = "data/dense_index_test"
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("==================================================")
    print(f" PHASE 5D: DENSE QUERY RETRIEVAL ({args.split.upper()})")
    print(f" Top-K: {args.top_k}")
    print("==================================================")
    
    emb_path = os.path.join(args.index_dir, "corpus_embeddings_fp16.npy")
    mapping_path = os.path.join(args.index_dir, "corpus_mapping.parquet")
    
    if not os.path.exists(emb_path) or not os.path.exists(mapping_path):
        print(f"Missing index files in {args.index_dir}. Run 05c_build_dense_index.py first.")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ---------------------------------------------------------
    # STEP 1: ENCODE QUERIES FIRST (To avoid memory fragmentation)
    # ---------------------------------------------------------
    print("Loading S1 Queries...")
    s1_path = os.path.join(args.data_dir, args.split, f"{args.split}_source1.parquet")
    select_cols = ["entity_id", "source", "name_norm", "address_norm", "country"]
    s1_df = pl.read_parquet(s1_path, columns=select_cols)
    
    print("Preparing query text...")
    queries_text = s1_df.select(
        pl.concat_str([pl.col("name_norm").fill_null(""), pl.col("address_norm").fill_null(""), pl.col("country").fill_null("")], separator=" | ")
    ).to_series().to_list()
    
    query_ids = s1_df["entity_id"].to_list()
    del s1_df
    
    query_emb_path = os.path.join(args.index_dir, "query_embeddings_fp32.npy")
    if os.path.exists(query_emb_path):
        print(f"Loading CACHED query embeddings from {query_emb_path} (Skipping 10-minute encoding!)")
        t0 = time.time()
        query_np = np.load(query_emb_path)
        print(f"Loaded in {time.time()-t0:.1f}s.")
    else:
        print("Encoding Queries...")
        t0 = time.time()
        model = SentenceTransformer(args.model_name, device=device)
        query_emb = model.encode(queries_text, batch_size=args.batch_size, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
        
        if isinstance(query_emb, torch.Tensor):
            query_np = query_emb.cpu().numpy().astype(np.float32)
        else:
            query_np = query_emb.astype(np.float32)
            
        print(f"Queries encoded in {time.time()-t0:.1f}s.")
        print(f"Saving query embeddings to cache: {query_emb_path}")
        np.save(query_emb_path, query_np)
        
        # Free the model and clear GPU cache completely!
        del model
        del query_emb
        if device == "cuda":
            torch.cuda.empty_cache()
            
    gc.collect()

    # ---------------------------------------------------------
    # STEP 2: TRAIN IVFFLAT INDEX
    # ---------------------------------------------------------
    print("Loading FAISS row mapping...")
    mapping_df = pl.read_parquet(mapping_path)
    total_corpus_rows = mapping_df.height
    
    print(f"Loading Memmap ({total_corpus_rows} rows) for IVFFlat Training...")
    memmap_array = np.memmap(emb_path, dtype='float16', mode='r', shape=(total_corpus_rows, 384))
    
    d = 384
    nlist = 16384
    quantizer = faiss.IndexFlatIP(d)
    cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    
    t0 = time.time()
    print(f"Training IVFFlat on 1,000,000 samples for {nlist} clusters...")
    train_sample = memmap_array[:1_000_000].astype(np.float32)
    
    if device == "cuda":
        # Train on a single GPU to be fast and safe
        res = faiss.StandardGpuResources()
        gpu_train_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_train_index.train(train_sample)
        # Pull trained index back to CPU
        cpu_index = faiss.index_gpu_to_cpu(gpu_train_index)
        del gpu_train_index
        del res
        torch.cuda.empty_cache()
    else:
        cpu_index.train(train_sample)
        
    del train_sample
    print(f"Training completed in {time.time()-t0:.1f}s.")
    
    # ---------------------------------------------------------
    # STEP 3: KEEP INDEX ON CPU (FAISS GPU OOMs on Kaggle T4)
    # ---------------------------------------------------------
    print("Keeping FAISS index on CPU to prevent CUDA OOM...")
    search_index = cpu_index

    # ---------------------------------------------------------
    # STEP 4: ADD VECTORS DIRECTLY TO GPU INDEX (Prevents CPU OOM)
    # ---------------------------------------------------------
    print("Populating Index with 10.3M vectors directly to GPUs...")
    t0 = time.time()
    chunk_size = 200_000
    for i in range(0, total_corpus_rows, chunk_size):
        end_idx = min(i + chunk_size, total_corpus_rows)
        fp32_chunk = memmap_array[i:end_idx].astype(np.float32)
        # Adds directly to VRAM in FP16, perfectly distributed
        search_index.add(fp32_chunk)
        print(f"  Added {end_idx}/{total_corpus_rows} vectors.")
        
    print(f"Index populated in {time.time()-t0:.1f}s.")
    
    # Free up Memmap RAM
    del memmap_array
    gc.collect()
        
    # Set nprobe
    ps = faiss.GpuParameterSpace()
    ps.set_index_parameter(search_index, "nprobe", args.nprobe)
    print(f"Set nprobe to {args.nprobe}.")
    
    print(f"Searching Top-{args.top_k} candidates across {total_corpus_rows} corpus...")
    t0 = time.time()
    
    # We must chunk the queries during search.
    # 50,000 queries caused a 2.45 GB TemporaryMemoryOverflow because FAISS computes distances to all 16k centroids for the batch.
    # We drop the batch size to 4096 to keep the intermediate matrix at ~260 MB, perfectly safe for VRAM.
    all_scores = []
    all_indices = []
    search_batch_size = 4096
    num_queries = query_np.shape[0]
    
    for i in range(0, num_queries, search_batch_size):
        end_idx = min(i + search_batch_size, num_queries)
        q_batch = query_np[i:end_idx]
        s_batch, i_batch = search_index.search(q_batch, args.top_k)
        all_scores.append(s_batch)
        all_indices.append(i_batch)
        
        # Only print every ~100k queries to avoid spamming the Kaggle logs
        if end_idx % (search_batch_size * 25) < search_batch_size or end_idx == num_queries:
            print(f"  Searched {end_idx}/{num_queries} queries.")
        
    scores = np.vstack(all_scores)
    indices = np.vstack(all_indices)
    
    print(f"Search completed in {time.time()-t0:.1f}s.")
    
    # ---------------------------------------------------------
    # STEP 5: FORMAT OUTPUT
    # ---------------------------------------------------------
    print("Formatting candidates...")
    
    mapping_entity_ids = mapping_df["entity_id"].to_numpy()
    mapping_sources = mapping_df["source"].to_numpy()
    
    num_queries = len(query_ids)
    flat_query_ids = np.repeat(query_ids, args.top_k)
    flat_ranks = np.tile(np.arange(1, args.top_k + 1), num_queries)
    
    flat_indices = indices.flatten()
    flat_scores = scores.flatten()
    
    flat_candidate_ids = mapping_entity_ids[flat_indices]
    flat_candidate_sources = mapping_sources[flat_indices]
    
    candidates_df = pl.DataFrame({
        "query_id": flat_query_ids,
        "candidate_id": flat_candidate_ids,
        "candidate_source": flat_candidate_sources,
        "dense_rank": flat_ranks,
        "dense_score": flat_scores,
        "retrieval_method": ["dense"] * len(flat_query_ids)
    })
    
    output_path = os.path.join(args.output_dir, f"{args.split}_dense_candidates_K{args.top_k}.parquet")
    candidates_df.write_parquet(output_path, compression="snappy")
    
    print(f"\nSaved {candidates_df.height} candidate pairs to {output_path}")
    print("Phase 5D Complete.")

if __name__ == "__main__":
    main()
