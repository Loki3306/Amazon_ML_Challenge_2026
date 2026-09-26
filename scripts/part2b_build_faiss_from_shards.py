"""
Part 2b: Build FAISS CPU Index from Disk Shards (ZERO GPU REQUIRED)
===================================================================
DESIGN:
  - Loads corpus_shard_*.npy files one by one from disk.
  - Trains IVF centroids from shard 0 (no GPU needed - pure CPU FAISS).
  - Adds all shard embeddings incrementally into FAISS CPU IndexIVFSQ8.
  - Never holds more than 1 shard in RAM at a time.
  - Saves the final .index file to disk cache.

Peak RAM: ~4.2 GB (1 shard ~380 MB + IVFSQ8 index ~3.84 GB).
GPU VRAM: 0 GB (no GPU used at all).
"""

import os
import sys
import gc
import time
import argparse
import numpy as np

try:
    import faiss
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "faiss-cpu"])
    import faiss


def parse_args():
    parser = argparse.ArgumentParser(description="Part 2b: Build FAISS CPU index from disk shards")
    parser.add_argument("--cache-dir", type=str, default="/kaggle/working/reports/retrieval/cache")
    parser.add_argument("--index-type", type=str, choices=["ivfflat", "ivfsq8"], default="ivfsq8")
    parser.add_argument("--n-shards", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(f" PART 2b: BUILD FAISS CPU INDEX FROM DISK SHARDS ({args.index_type.upper()})")
    print(f"  cache_dir={args.cache_dir} | n_shards={args.n_shards}")
    print("=" * 70)

    index_path = os.path.join(args.cache_dir, f"faiss_index_name_address_country_{args.index_type}.index")
    if os.path.exists(index_path):
        print(f"[SKIP] FAISS index already exists at {index_path}")
        print("Delete the file if you want to rebuild.")
        return

    # Discover all shard files
    shard_paths = []
    for s in range(args.n_shards):
        sp = os.path.join(args.cache_dir, f"corpus_shard_{s:02d}.npy")
        if os.path.exists(sp):
            shard_paths.append(sp)
        else:
            break  # stop at first missing shard

    if not shard_paths:
        raise FileNotFoundError(f"No shard files found in {args.cache_dir}. Run Part 2a first.")

    print(f"Found {len(shard_paths)} shard files.")

    # Load shard 0 to get embedding dimension and train IVF centroids
    print(f"Loading shard 0 to determine dimension & train IVF centroids...")
    shard0 = np.load(shard_paths[0])
    dim = shard0.shape[1]
    total_vectors = sum(np.load(sp, mmap_mode='r').shape[0] for sp in shard_paths)
    effective_nlist = min(16384, max(16, total_vectors // 10))

    print(f"  dim={dim} | total_vectors={total_vectors:,} | nlist={effective_nlist}")

    # Build the FAISS CPU index
    quantizer = faiss.IndexFlatIP(dim)
    if args.index_type.lower() == "ivfsq8":
        print(f"Creating FAISS CPU IndexIVFSQ8 (8-bit Quantization)...")
        cpu_index = faiss.IndexIVFScalarQuantizer(
            quantizer, dim, effective_nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
        )
    else:
        print(f"Creating FAISS CPU IndexIVFFlat (FP32 Uncompressed)...")
        cpu_index = faiss.IndexIVFFlat(quantizer, dim, effective_nlist, faiss.METRIC_INNER_PRODUCT)

    faiss.omp_set_num_threads(os.cpu_count() or 4)

    # Train on shard 0 (full 1M sample is good for 16K centroids)
    print(f"Training IVF centroids on {len(shard0):,} vectors from shard 0...")
    t0 = time.time()
    cpu_index.train(shard0.astype(np.float32))
    print(f"Training done in {time.time()-t0:.0f}s")
    del shard0
    gc.collect()

    # Stream all shards into FAISS index
    t_total = time.time()
    vectors_added = 0
    for i, sp in enumerate(shard_paths):
        t0 = time.time()
        shard = np.load(sp)
        print(f"  Adding shard {i:02d} | {len(shard):,} vectors from {sp}...", flush=True)
        cpu_index.add(shard.astype(np.float32))
        vectors_added += len(shard)
        del shard
        gc.collect()
        print(f"    Done in {time.time()-t0:.0f}s | total added: {vectors_added:,}", flush=True)

    print(f"\nAll shards added in {time.time()-t_total:.0f}s")
    print(f"Total vectors in index: {cpu_index.ntotal:,}")
    print(f"Saving FAISS index to disk: {index_path}")
    faiss.write_index(cpu_index, index_path)
    size_gb = os.path.getsize(index_path) / (1024**3)
    print(f"Index saved: {size_gb:.2f} GB")
    print("[PART 2b COMPLETED SUCCESSFULLY]")


if __name__ == "__main__":
    main()
