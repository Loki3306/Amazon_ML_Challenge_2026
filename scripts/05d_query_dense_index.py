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
    parser.add_argument("--index-dir", type=str, default="data/dense_index", help="Output directory of 5C")
    parser.add_argument("--output-dir", type=str, default="data/candidates", help="Where to save candidates")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--top-k", type=int, default=50, help="Number of candidates to retrieve")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for query embedding")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("==================================================")
    print(" PHASE 5D: DENSE QUERY RETRIEVAL")
    print(f" Top-K: {args.top_k}")
    print("==================================================")
    
    emb_path = os.path.join(args.index_dir, "corpus_embeddings_fp16.npy")
    mapping_path = os.path.join(args.index_dir, "corpus_mapping.parquet")
    
    if not os.path.exists(emb_path) or not os.path.exists(mapping_path):
        print(f"Missing index files in {args.index_dir}. Run 05c_build_dense_index.py first.")
        return
        
    # 1. Load Mapping
    print("Loading FAISS row mapping...")
    mapping_df = pl.read_parquet(mapping_path)
    total_corpus_rows = mapping_df.height
    
    # 2. Build FAISS Index (IVFFlat Approximate Nearest Neighbors)
    print(f"Loading Memmap ({total_corpus_rows} rows) for IVFFlat Training...")
    t0 = time.time()
    
    # Memory map the FP16 array
    memmap_array = np.memmap(emb_path, dtype='float16', mode='r', shape=(total_corpus_rows, 384))
    
    # We use IVFFlat to reduce 22 trillion calculations to a tiny fraction
    d = 384
    nlist = 65536  # Number of Voronoi cells (clusters)
    quantizer = faiss.IndexFlatIP(d)
    cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("Transferring untrained IVFFlat to GPUs (Sharded & FP16)...")
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(cpu_index, co=co)
    else:
        index = cpu_index
        print("WARNING: CUDA not detected, running on CPU.")

    print("Training IVFFlat on 2,000,000 samples...")
    # Train on first 2M rows (sufficient for 10M dataset)
    train_sample = memmap_array[:2_000_000].astype(np.float32)
    index.train(train_sample)
    del train_sample
    print(f"Training completed in {time.time()-t0:.1f}s.")
    
    print("Populating Index with 10.3M vectors...")
    t0 = time.time()
    chunk_size = 1_000_000
    for i in range(0, total_corpus_rows, chunk_size):
        end_idx = min(i + chunk_size, total_corpus_rows)
        fp32_chunk = memmap_array[i:end_idx].astype(np.float32)
        index.add(fp32_chunk)
        print(f"  Added {end_idx}/{total_corpus_rows} vectors.")
        
    print(f"Index populated in {time.time()-t0:.1f}s.")
    
    # Set nprobe (number of clusters to search). 64 out of 65536 = ~0.1% of the corpus searched per query
    # faiss.GpuIndexIVF has a setNumProbes method in Python (via SWIG) or we can set nprobe on CPU index and clone again, 
    # but the easiest way to set nprobe on a sharded GPU index is through GpuParameterSpace.
    ps = faiss.GpuParameterSpace()
    ps.set_index_parameter(index, "nprobe", 64)
    
    # Free up RAM (we don't need memmap anymore)
    del memmap_array
    gc.collect()
    
    # 3. Load Queries
    print("Loading S1 Queries...")
    s1_path = os.path.join(args.data_dir, "train", "train_source1.parquet")
    select_cols = ["entity_id", "source", "name_norm", "address_norm", "country"]
    s1_df = pl.read_parquet(s1_path, columns=select_cols)
    
    print("Preparing query text...")
    queries_text = s1_df.select(
        pl.concat_str([pl.col("name_norm").fill_null(""), pl.col("address_norm").fill_null(""), pl.col("country").fill_null("")], separator=" | ")
    ).to_series().to_list()
    
    query_ids = s1_df["entity_id"].to_list()
    del s1_df
    
    # 4. Encode Queries
    print("Encoding Queries...")
    t0 = time.time()
    model = SentenceTransformer(args.model_name, device=device)
    query_emb = model.encode(queries_text, batch_size=args.batch_size, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
    
    if isinstance(query_emb, torch.Tensor):
        query_np = query_emb.cpu().numpy().astype(np.float32)
    else:
        query_np = query_emb.astype(np.float32)
        
    print(f"Queries encoded in {time.time()-t0:.1f}s.")
    
    # 5. Search
    print(f"Searching Top-{args.top_k} candidates across {total_corpus_rows} corpus...")
    t0 = time.time()
    
    # FAISS search
    scores, indices = index.search(query_np, args.top_k)
    print(f"Search completed in {time.time()-t0:.1f}s.")
    
    # 6. Format Output
    print("Formatting candidates...")
    
    # mapping_df has faiss_row_id (index), entity_id, source
    # We convert mapping_df to fast numpy arrays for lookup
    mapping_entity_ids = mapping_df["entity_id"].to_numpy()
    mapping_sources = mapping_df["source"].to_numpy()
    
    # Flatten outputs
    num_queries = len(query_ids)
    
    # Repeat query_ids for K results
    flat_query_ids = np.repeat(query_ids, args.top_k)
    flat_ranks = np.tile(np.arange(1, args.top_k + 1), num_queries)
    
    flat_indices = indices.flatten()
    flat_scores = scores.flatten()
    
    # Lookup entity_id and source using the indices
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
    
    output_path = os.path.join(args.output_dir, f"train_dense_candidates_K{args.top_k}.parquet")
    candidates_df.write_parquet(output_path, compression="snappy")
    
    print(f"\nSaved {candidates_df.height} candidate pairs to {output_path}")
    print("Phase 5D Complete.")

if __name__ == "__main__":
    main()
