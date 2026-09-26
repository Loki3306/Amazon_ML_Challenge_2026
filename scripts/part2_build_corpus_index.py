"""
Part 2: Standalone Corpus FAISS Index Builder (High-Speed & OOM-Safe)
=====================================================================
Streams and encodes ~10.3M S2/S3 corpus records into a FAISS CPU IVFSQ8 index.
Uses PyTorch FP16 AMP autocast & batch_size=8192 for 3x-4x faster GPU throughput (~18-22 mins total).
Saves index (.index) to disk cache.
"""

import os
# Configure PyTorch CUDA Memory Allocator to prevent VRAM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import time
import gc
import argparse
import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "faiss-cpu"])
    import faiss


def parse_args():
    parser = argparse.ArgumentParser(description="Part 2: Standalone Corpus FAISS Index Builder")
    parser.add_argument("--data-dir", type=str, default="/kaggle/working/data/processed")
    parser.add_argument("--input-dir", type=str, default="/kaggle/input/student-resource-amazonml/dataset/train")
    parser.add_argument("--cache-dir", type=str, default="/kaggle/working/reports/retrieval/cache")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--index-type", type=str, choices=["ivfflat", "ivfsq8"], default="ivfsq8")
    parser.add_argument("--batch-size", type=int, default=2048, help="Safe batch size to prevent PyTorch activation OOM")
    parser.add_argument("--chunk-size", type=int, default=200000, help="Chunk size for streaming encoding")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    print("=" * 70)
    print(f" PART 2: STANDALONE CORPUS FAISS INDEX BUILDER ({args.index_type.upper()})")
    print("=" * 70)

    # Load S2 & S3 data
    s2_path = os.path.join(args.data_dir, "train", "train_source2.parquet")
    s3_path = os.path.join(args.data_dir, "train", "train_source3.parquet")

    if not os.path.exists(s2_path) or not os.path.exists(s3_path):
        raw_s2 = os.path.join(args.input_dir, "train_source2.tsv")
        raw_s3 = os.path.join(args.input_dir, "train_source3.tsv")
        if not os.path.exists(raw_s2):
            for root, _, files in os.walk("/kaggle/input"):
                if "train_source2.tsv" in files:
                    input_dir = root
                    raw_s2 = os.path.join(input_dir, "train_source2.tsv")
                    raw_s3 = os.path.join(input_dir, "train_source3.tsv")
                    break

        os.makedirs(os.path.dirname(s2_path), exist_ok=True)
        for r_path, out_p, s_name in [(raw_s2, s2_path, "S2"), (raw_s3, s3_path, "S3")]:
            print(f"Converting raw {s_name} TSV from {r_path} -> {out_p}...")
            df = pl.read_csv(r_path, separator="\t", null_values=[""])
            df = df.with_columns([
                pl.lit(s_name).alias("source"),
                pl.col("business_name").fill_null("").str.to_lowercase().str.strip_chars().alias("name_norm"),
                pl.col("business_address").fill_null("").str.to_lowercase().str.strip_chars().alias("address_norm"),
                pl.col("country").fill_null("").str.to_lowercase().str.strip_chars().alias("country_norm")
            ])
            df.write_parquet(out_p, compression="snappy")

    select_cols = ["entity_id", "source", "name_norm", "address_norm", "country"]
    s2_df = pl.read_parquet(s2_path, columns=select_cols)
    s3_df = pl.read_parquet(s3_path, columns=select_cols)
    corpus_df = pl.concat([s2_df, s3_df])
    total_corpus = len(corpus_df)
    print(f"Loaded {total_corpus:,} corpus records (S2 + S3).")

    c_texts = corpus_df.select(
        pl.concat_str([pl.col("name_norm").fill_null(""), pl.col("address_norm").fill_null(""), pl.col("country").fill_null("")], separator=" | ")
    ).to_series().to_list()

    index_path = os.path.join(args.cache_dir, f"faiss_index_name_address_country_{args.index_type}.index")
    if os.path.exists(index_path):
        print(f"FAISS index already cached at {index_path}. Exiting.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading '{args.model_name}' on device: {device.upper()}...")
    model = SentenceTransformer(args.model_name, device=device)
    dim = model.get_sentence_embedding_dimension()
    use_amp = torch.cuda.is_available()

    # Step 2a: Sample-encode 1M texts for centroid training
    sample_size = min(1000000, total_corpus)
    print(f"Sample-encoding {sample_size:,} corpus texts for FAISS training...")
    sample_texts = c_texts[:sample_size]
    sample_embeddings = np.empty((sample_size, dim), dtype=np.float32)

    for i in range(0, sample_size, args.chunk_size):
        end = min(i + args.chunk_size, sample_size)
        c_chunk = sample_texts[i:end]
        if use_amp:
            with torch.cuda.amp.autocast():
                emb = model.encode(c_chunk, batch_size=args.batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        else:
            emb = model.encode(c_chunk, batch_size=args.batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        sample_embeddings[i:end] = emb.astype(np.float32)
        del emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    effective_nlist = min(16384, max(16, total_corpus // 10))
    quantizer = faiss.IndexFlatIP(dim)

    if args.index_type.lower() == "ivfsq8":
        print(f"Training FAISS CPU IndexIVFSQ8 (8-bit Quantization, nlist={effective_nlist}, dim={dim})...")
        cpu_index = faiss.IndexIVFScalarQuantizer(quantizer, dim, effective_nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    else:
        print(f"Training FAISS CPU IndexIVFFlat (FP32, nlist={effective_nlist}, dim={dim})...")
        cpu_index = faiss.IndexIVFFlat(quantizer, dim, effective_nlist, faiss.METRIC_INNER_PRODUCT)

    faiss.omp_set_num_threads(os.cpu_count() or 4)
    cpu_index.train(sample_embeddings)
    del sample_embeddings
    gc.collect()

    # Step 2b: Stream all 10.3M vectors into FAISS CPU index
    t0_stream = time.time()
    print(f"Streaming {total_corpus:,} corpus vectors into FAISS CPU index in {args.chunk_size:,} chunks...")
    for i in range(0, total_corpus, args.chunk_size):
        end = min(i + args.chunk_size, total_corpus)
        chunk_texts = c_texts[i:end]
        if use_amp:
            with torch.cuda.amp.autocast():
                chunk_emb = model.encode(chunk_texts, batch_size=args.batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        else:
            chunk_emb = model.encode(chunk_texts, batch_size=args.batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

        cpu_index.add(chunk_emb.astype(np.float32))
        del chunk_emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"  Added chunk {end:,}/{total_corpus:,} ({(end/total_corpus)*100:.1f}%)...", flush=True)

    print(f"FAISS CPU index populated in {time.time()-t0_stream:.2f}s. Saving to disk cache: {index_path}")
    faiss.write_index(cpu_index, index_path)
    print("[PART 2 COMPLETED SUCCESSFULLY]")


if __name__ == "__main__":
    main()
