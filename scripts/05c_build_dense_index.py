import os
import time
import argparse
import json
import polars as pl
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5C: Build Dense GPU Index (Resumable Memmap)")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Parquet directory")
    parser.add_argument("--split", type=str, default="train", help="Which split to process (train or test)")
    parser.add_argument("--index-dir", type=str, default="data/dense_index", help="Output directory for index and mapping")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Rows per checkpointed chunk")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for embedding")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure index dir is split-specific if it's the default
    if args.index_dir == "data/dense_index" and args.split == "test":
        args.index_dir = "data/dense_index_test"
        
    os.makedirs(args.index_dir, exist_ok=True)
    
    print("==================================================")
    print(f" PHASE 5C: CORPUS INDEXING ({args.split.upper()})")
    print("==================================================")
    
    # 1. Discover Total Corpus Size
    s2_path = os.path.join(args.data_dir, args.split, f"{args.split}_source2.parquet")
    s3_path = os.path.join(args.data_dir, args.split, f"{args.split}_source3.parquet")
    
    print("Scanning corpus size...")
    # Lazy scan to get total rows
    s2_lazy = pl.scan_parquet(s2_path)
    s3_lazy = pl.scan_parquet(s3_path)
    corpus_lazy = pl.concat([s2_lazy, s3_lazy])
    
    total_rows = corpus_lazy.select(pl.len()).collect().item()
    print(f"Total Corpus Rows (S2 + S3): {total_rows}")
    
    # 2. Setup Memmap and Checkpointing
    emb_path = os.path.join(args.index_dir, "corpus_embeddings_fp16.npy")
    mapping_path = os.path.join(args.index_dir, "corpus_mapping.parquet")
    ckpt_path = os.path.join(args.index_dir, "checkpoint.json")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model_name, device=device)
    dim = model.get_sentence_embedding_dimension()
    
    # Load checkpoint if exists
    start_offset = 0
    mapping_dfs = []
    
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            ckpt = json.load(f)
            if ckpt.get("total_rows") == total_rows and ckpt.get("model") == args.model_name:
                start_offset = ckpt["processed_rows"]
                print(f"Resuming from row {start_offset} / {total_rows}...")
                
                # If we are resuming, we MUST already have the partial mapping files
                # (For simplicity in this script, we'll write partial mappings and concat at the end)
            else:
                print("Checkpoint mismatch (different model or dataset size). Starting fresh.")
                start_offset = 0
    
    # Pre-allocate memmap (only if starting fresh)
    if start_offset == 0:
        print(f"Pre-allocating {total_rows} x {dim} FP16 memmap on disk...")
        # FP16 = 2 bytes per float. 10.3M * 384 * 2 = ~7.9 GB
        # Using mode='w+' overwrites any existing file
        memmap_array = np.memmap(emb_path, dtype='float16', mode='w+', shape=(total_rows, dim))
    else:
        # Open existing for append
        memmap_array = np.memmap(emb_path, dtype='float16', mode='r+', shape=(total_rows, dim))
        
    if start_offset >= total_rows:
        print("Corpus already fully encoded! Exiting.")
        return
        
    # 3. Stream and Encode in Chunks
    # We use Polars iter_slices (requires loading into DataFrame, but we can do it via slice or scan offsets)
    # Actually, scan_parquet doesn't guarantee row order if we slice later.
    # To be deterministic, we read the entire dataframe but only select needed columns.
    print("Loading corpus columns for text representation...")
    # This takes a few GBs of RAM but perfectly safe on Kaggle (30GB available)
    select_cols = ["entity_id", "source", "name_norm", "address_norm", "country"]
    s2_df = pl.read_parquet(s2_path, columns=select_cols)
    s3_df = pl.read_parquet(s3_path, columns=select_cols)
    full_corpus = pl.concat([s2_df, s3_df])
    del s2_df, s3_df
    
    print("Generating text representations...")
    # Create the concatenated text column
    texts_series = full_corpus.select(
        pl.concat_str([pl.col("name_norm").fill_null(""), pl.col("address_norm").fill_null(""), pl.col("country").fill_null("")], separator=" | ")
    ).to_series()
    
    print("Starting chunked encoding...")
    
    for offset in range(start_offset, total_rows, args.chunk_size):
        end_offset = min(offset + args.chunk_size, total_rows)
        chunk_texts = texts_series[offset:end_offset].to_list()
        
        t0 = time.time()
        # Encode (normalize=True because we use Inner Product FAISS)
        embeddings = model.encode(chunk_texts, batch_size=args.batch_size, show_progress_bar=False, convert_to_tensor=True, normalize_embeddings=True)
        
        # Cast to FP16 and save to memmap
        emb_fp16 = embeddings.cpu().numpy().astype(np.float16)
        memmap_array[offset:end_offset] = emb_fp16
        
        # Flush to disk immediately
        memmap_array.flush()
        
        # Save mapping chunk
        chunk_mapping = full_corpus[offset:end_offset].select([
            pl.col("entity_id"),
            pl.col("source"),
            pl.Series("faiss_row_id", range(offset, end_offset), dtype=pl.UInt32)
        ])
        mapping_chunk_path = os.path.join(args.index_dir, f"mapping_{offset}.parquet")
        chunk_mapping.write_parquet(mapping_chunk_path)
        
        elapsed = time.time() - t0
        speed = len(chunk_texts) / elapsed
        print(f"Encoded {offset} -> {end_offset} | {speed:.1f} rows/sec | Checkpointed.")
        
        # Update checkpoint
        with open(ckpt_path, "w") as f:
            json.dump({
                "processed_rows": end_offset,
                "total_rows": total_rows,
                "model": args.model_name
            }, f)
            
    # 4. Finalize Mapping
    print("Consolidating mapping files...")
    all_mapping_files = [os.path.join(args.index_dir, f) for f in os.listdir(args.index_dir) if f.startswith("mapping_") and f.endswith(".parquet")]
    all_mapping_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    
    final_mapping_df = pl.concat([pl.read_parquet(f) for f in all_mapping_files])
    final_mapping_df.write_parquet(mapping_path)
    
    # Cleanup temp mapping files
    for f in all_mapping_files:
        os.remove(f)
        
    print(f"\nCorpus Indexing Complete!")
    print(f"Saved {total_rows} embeddings to: {emb_path} (FP16 memmap)")
    print(f"Saved FAISS row mapping to: {mapping_path}")

if __name__ == "__main__":
    main()
