"""
Part 1: Standalone Query Embedding Encoder (OOM-Safe & Accelerated)
===================================================================
Encodes 2.2M S1 query entities using SentenceTransformers with FP16 AMP autocast.
Saves query embeddings (.npy) to disk cache in ~15 minutes.
"""

import os
import sys
import time
import gc
import argparse
import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Part 1: Standalone Query Encoder")
    parser.add_argument("--data-dir", type=str, default="/kaggle/working/data/processed")
    parser.add_argument("--input-dir", type=str, default="/kaggle/input/student-resource-amazonml/dataset/train")
    parser.add_argument("--cache-dir", type=str, default="/kaggle/working/reports/retrieval/cache")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=500000)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    print("=" * 70)
    print(" PART 1: STANDALONE QUERY EMBEDDING ENCODER")
    print("=" * 70)

    # Load S1 data
    s1_path = os.path.join(args.data_dir, "train", "train_source1.parquet")
    if not os.path.exists(s1_path):
        raw_s1 = os.path.join(args.input_dir, "train_source1.tsv")
        if not os.path.exists(raw_s1):
            # search fallback
            for root, _, files in os.walk("/kaggle/input"):
                if "train_source1.tsv" in files:
                    raw_s1 = os.path.join(root, "train_source1.tsv")
                    break

        print(f"Converting raw S1 TSV from {raw_s1} -> {s1_path}...")
        df = pl.read_csv(raw_s1, separator="\t", null_values=[""])
        df = df.with_columns([
            pl.lit("S1").alias("source"),
            pl.col("business_name").fill_null("").str.to_lowercase().str.strip_chars().alias("name_norm"),
            pl.col("business_address").fill_null("").str.to_lowercase().str.strip_chars().alias("address_norm"),
            pl.col("country").fill_null("").str.to_lowercase().str.strip_chars().alias("country_norm")
        ])
        os.makedirs(os.path.dirname(s1_path), exist_ok=True)
        df.write_parquet(s1_path, compression="snappy")

    s1_df = pl.read_parquet(s1_path)
    print(f"Loaded {len(s1_df):,} S1 queries.")

    q_texts = s1_df.select(
        pl.concat_str([pl.col("name_norm").fill_null(""), pl.col("address_norm").fill_null(""), pl.col("country").fill_null("")], separator=" | ")
    ).to_series().to_list()

    cache_path = os.path.join(args.cache_dir, "query_emb_name_address_country.npy")
    if os.path.exists(cache_path):
        print(f"Query embeddings already cached at {cache_path}. Exiting.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading '{args.model_name}' on device: {device.upper()}...")
    model = SentenceTransformer(args.model_name, device=device)
    dim = model.get_sentence_embedding_dimension()

    n = len(q_texts)
    embeddings = np.empty((n, dim), dtype=np.float32)
    use_amp = torch.cuda.is_available()

    t0 = time.time()
    print(f"Encoding {n:,} queries with batch_size={args.batch_size} (AMP FP16={use_amp})...")
    for i in range(0, n, args.chunk_size):
        end = min(i + args.chunk_size, n)
        chunk_texts = q_texts[i:end]

        if use_amp:
            with torch.cuda.amp.autocast():
                chunk_emb = model.encode(
                    chunk_texts, batch_size=args.batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
                )
        else:
            chunk_emb = model.encode(
                chunk_texts, batch_size=args.batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
            )

        embeddings[i:end] = chunk_emb.astype(np.float32)
        del chunk_emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"  Encoded query chunk {end:,}/{n:,} ({(end/n)*100:.1f}%)...", flush=True)

    print(f"Finished query encoding in {time.time()-t0:.2f}s. Saving to cache: {cache_path}")
    np.save(cache_path, embeddings)
    print("[PART 1 COMPLETED SUCCESSFULLY]")


if __name__ == "__main__":
    main()
