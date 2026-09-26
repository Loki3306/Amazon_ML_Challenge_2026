"""
Part 2 (NEW): Corpus Encoder Only - Saves shards to disk, NO FAISS index building here.
========================================================================================
DESIGN:
  - Encodes 10.3M corpus records in 10 shards of ~1M texts each.
  - Each shard is saved to disk as a .npy float32 file immediately after encoding.
  - If interrupted, re-running skips already-saved shards (disk checkpoint).
  - Keeps GPU VRAM strictly bounded to 1 small batch at a time.
  - model.eval() + torch.no_grad() are always active.
  - Texts sorted by length within each shard to minimize padding waste.
  - Uses batch_size=512 to stay safely within T4 VRAM budget.

Peak VRAM: ~2.5 GB (model weights ~90MB + activations for 512 texts).
Shard size on disk: ~380 MB each × 10 shards = ~3.8 GB total shard storage.
"""

import os
# MUST be set before any torch import to fix PyTorch CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import gc
import time
import argparse
import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Part 2a: Encode corpus to disk shards")
    parser.add_argument("--data-dir", type=str, default="/kaggle/working/data/processed")
    parser.add_argument("--input-dir", type=str, default="/kaggle/input/student-resource-amazonml/dataset/train")
    parser.add_argument("--cache-dir", type=str, default="/kaggle/working/reports/retrieval/cache")
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="GPU batch size. 512 is safe for T4 (14.5GB). Reduce to 256 if OOM persists.")
    parser.add_argument("--n-shards", type=int, default=10,
                        help="Number of corpus shards to split into (default=10 → ~1M texts each)")
    return parser.parse_args()


def load_corpus(data_dir: str, input_dir: str) -> pl.DataFrame:
    s2_path = os.path.join(data_dir, "train", "train_source2.parquet")
    s3_path = os.path.join(data_dir, "train", "train_source3.parquet")

    if not os.path.exists(s2_path) or not os.path.exists(s3_path):
        # Auto-discover raw TSV files
        raw_s2 = os.path.join(input_dir, "train_source2.tsv")
        raw_s3 = os.path.join(input_dir, "train_source3.tsv")
        if not os.path.exists(raw_s2):
            for root, _, files in os.walk("/kaggle/input"):
                if "train_source2.tsv" in files:
                    raw_s2 = os.path.join(root, "train_source2.tsv")
                    raw_s3 = os.path.join(root, "train_source3.tsv")
                    break

        os.makedirs(os.path.dirname(s2_path), exist_ok=True)
        for r_path, out_p, s_name in [(raw_s2, s2_path, "S2"), (raw_s3, s3_path, "S3")]:
            print(f"  Converting raw {s_name} TSV -> parquet: {r_path}")
            df = pl.read_csv(r_path, separator="\t", null_values=[""])
            df = df.with_columns([
                pl.lit(s_name).alias("source"),
                pl.col("business_name").fill_null("").str.to_lowercase().str.strip_chars().alias("name_norm"),
                pl.col("business_address").fill_null("").str.to_lowercase().str.strip_chars().alias("address_norm"),
                pl.col("country").fill_null("").str.to_lowercase().str.strip_chars().alias("country_norm")
            ])
            df.write_parquet(out_p, compression="snappy")
            print(f"  Saved {len(df):,} records to {out_p}")

    select_cols = ["entity_id", "source", "name_norm", "address_norm", "country"]
    s2_df = pl.read_parquet(s2_path, columns=select_cols)
    s3_df = pl.read_parquet(s3_path, columns=select_cols)
    return pl.concat([s2_df, s3_df])


def encode_shard_to_disk(model, texts: list[str], shard_path: str, batch_size: int):
    """
    Encode a list of texts in small batches and save the resulting float32
    numpy array to disk. Sorts by text length to minimize padding waste.
    model.eval() + torch.no_grad() MUST be active before calling this.
    """
    n = len(texts)
    # Sort by length to reduce padding waste, keep track of original order
    order = sorted(range(n), key=lambda i: len(texts[i]))
    sorted_texts = [texts[i] for i in order]

    dim = model.get_embedding_dimension()
    embeddings_sorted = np.empty((n, dim), dtype=np.float32)

    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch = sorted_texts[i:end]
        emb = model.encode(
            batch,
            batch_size=len(batch),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embeddings_sorted[i:end] = emb.astype(np.float32)
        del emb
        torch.cuda.empty_cache()

    # Unsort back to original order
    embeddings = np.empty_like(embeddings_sorted)
    for new_pos, orig_pos in enumerate(order):
        embeddings[orig_pos] = embeddings_sorted[new_pos]
    del embeddings_sorted

    np.save(shard_path, embeddings)
    gc.collect()
    return len(embeddings)


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    print("=" * 70)
    print(" PART 2a: CORPUS ENCODER → DISK SHARDS")
    print(f"  batch_size={args.batch_size} | n_shards={args.n_shards}")
    print(f"  PYTORCH_CUDA_ALLOC_CONF = {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'not set')}")
    print("=" * 70)

    corpus_df = load_corpus(args.data_dir, args.input_dir)
    total = len(corpus_df)
    print(f"Loaded {total:,} corpus records (S2 + S3).")

    c_texts = corpus_df.select(
        pl.concat_str([
            pl.col("name_norm").fill_null(""),
            pl.col("address_norm").fill_null(""),
            pl.col("country").fill_null("")
        ], separator=" | ")
    ).to_series().to_list()

    # Save corpus entity_ids for Part 3 (recall evaluation)
    ids_path = os.path.join(args.cache_dir, "corpus_entity_ids.npy")
    if not os.path.exists(ids_path):
        corpus_ids = corpus_df["entity_id"].to_numpy()
        np.save(ids_path, corpus_ids)
        print(f"Saved corpus entity IDs to {ids_path}")

    # Shard the full corpus
    shard_size = (total + args.n_shards - 1) // args.n_shards  # ceiling division
    print(f"Shard size: {shard_size:,} texts per shard ({args.n_shards} shards total)")

    # Check which shards are already done (disk checkpoint)
    shards_needed = []
    for s in range(args.n_shards):
        shard_path = os.path.join(args.cache_dir, f"corpus_shard_{s:02d}.npy")
        start = s * shard_size
        end = min(start + shard_size, total)
        if start >= total:
            break
        if os.path.exists(shard_path):
            print(f"  [SKIP] Shard {s:02d} already exists ({start:,}–{end:,})")
        else:
            shards_needed.append((s, start, end, shard_path))

    if not shards_needed:
        print("All shards already encoded. Nothing to do.")
        return

    print(f"\nLoading '{args.model_name}' on GPU...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model_name, device=device)
    model.eval()

    t_total = time.time()
    with torch.no_grad():
        for s, start, end, shard_path in shards_needed:
            t0 = time.time()
            shard_texts = c_texts[start:end]
            n_shard = end - start
            print(f"\n  Shard {s:02d} | {start:,}–{end:,} ({n_shard:,} texts)...", flush=True)
            encode_shard_to_disk(model, shard_texts, shard_path, args.batch_size)
            elapsed = time.time() - t0
            shard_mb = os.path.getsize(shard_path) / (1024**2)
            print(f"  Shard {s:02d} done in {elapsed:.0f}s | saved {shard_mb:.0f} MB to {shard_path}", flush=True)

    print(f"\n[PART 2a COMPLETED] All shards encoded in {time.time()-t_total:.0f}s")
    print(f"Shards saved to: {args.cache_dir}/corpus_shard_*.npy")
    print("Run Part 2b next to build the FAISS index from shards.")


if __name__ == "__main__":
    main()
