"""
Part 2b: Build FAISS CPU Index from Disk Shards (ZERO GPU REQUIRED)
===================================================================
DESIGN:
  - Loads corpus_shard_*.npy files one by one from disk.
  - Trains IVF centroids from the first available shard.
  - Adds all shard embeddings incrementally into FAISS CPU IndexIVFSQ8.
  - DELETES each shard from disk immediately after adding it to free space.
  - RESUME MODE: if index already exists on disk, loads it and only adds
    shards that haven't been processed yet (tracks via a progress file).
  - Never holds more than 1 shard in RAM at a time.

Peak RAM: ~4.2 GB (1 shard ~1.5 GB + IVFSQ8 index ~3.84 GB).
GPU VRAM: 0 GB (no GPU used at all).

DISK RECOVERY WORKFLOW (if part2a ran out of disk space):
  1. Run part2b on available shards  → deletes them, frees space
  2. Re-run part2a                   → encodes missing shards
  3. Re-run part2b                   → resumes, adds new shards only
"""

import os
import sys
import gc
import time
import json
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
    parser.add_argument("--delete-shards", action="store_true", default=True,
                        help="Delete each shard file after adding to index (frees disk space). Default: True.")
    parser.add_argument("--keep-shards", dest="delete_shards", action="store_false",
                        help="Keep shard files on disk after adding.")
    return parser.parse_args()


def get_progress_path(cache_dir, index_type):
    return os.path.join(cache_dir, f"part2b_progress_{index_type}.json")


def load_progress(progress_path):
    """Load progress dict: {shards_added: int, vectors_added: int, trained: bool}"""
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            return json.load(f)
    return {"shards_added": 0, "vectors_added": 0, "trained": False}


def save_progress(progress_path, progress):
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


def main():
    args = parse_args()

    print("=" * 70)
    print(f" PART 2b: BUILD FAISS CPU INDEX FROM DISK SHARDS ({args.index_type.upper()})")
    print(f"  cache_dir={args.cache_dir} | n_shards={args.n_shards}")
    print(f"  delete_shards={args.delete_shards}  ← frees disk space as we go")
    print("=" * 70)

    index_path = os.path.join(args.cache_dir, f"faiss_index_name_address_country_{args.index_type}.index")
    progress_path = get_progress_path(args.cache_dir, args.index_type)
    progress = load_progress(progress_path)

    # Discover all shard files that currently exist on disk
    shard_paths = []
    for s in range(args.n_shards):
        sp = os.path.join(args.cache_dir, f"corpus_shard_{s:02d}.npy")
        if os.path.exists(sp):
            shard_paths.append((s, sp))

    if not shard_paths:
        if progress["vectors_added"] > 0:
            print(f"No shard files found, but progress shows {progress['vectors_added']:,} vectors already added.")
            print("If all shards have been processed, the index is complete.")
        else:
            raise FileNotFoundError(f"No shard files found in {args.cache_dir}. Run Part 2a first.")

    print(f"Found {len(shard_paths)} shard files on disk.")
    print(f"Progress: {progress['shards_added']} shards previously added, "
          f"{progress['vectors_added']:,} vectors in index so far.")

    # Load or create FAISS index
    if os.path.exists(index_path) and progress["trained"]:
        print(f"[RESUME] Loading existing FAISS index from {index_path}...")
        cpu_index = faiss.read_index(index_path)
        print(f"  Loaded index with {cpu_index.ntotal:,} vectors.")
        faiss.omp_set_num_threads(os.cpu_count() or 4)
    else:
        # Need to build fresh - use first available shard for dim + training
        first_shard_idx, first_shard_path = shard_paths[0]
        print(f"Loading shard {first_shard_idx:02d} to determine dimension & train IVF centroids...")
        shard0 = np.load(first_shard_path)
        dim = shard0.shape[1]

        # Estimate total vectors across all shards for nlist
        total_est = shard0.shape[0] * args.n_shards
        effective_nlist = min(16384, max(16, total_est // 10))
        print(f"  dim={dim} | estimated_total={total_est:,} | nlist={effective_nlist}")

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

        print(f"Training IVF centroids on {len(shard0):,} vectors from shard {first_shard_idx:02d}...")
        t0 = time.time()
        cpu_index.train(shard0.astype(np.float32))
        print(f"Training done in {time.time()-t0:.0f}s")
        del shard0
        gc.collect()
        progress["trained"] = True
        save_progress(progress_path, progress)

    # Stream all on-disk shards into FAISS index (skip already-added ones)
    t_total = time.time()
    for shard_num, sp in shard_paths:
        if shard_num < progress["shards_added"]:
            print(f"  [SKIP] Shard {shard_num:02d} already added to index.")
            if args.delete_shards and os.path.exists(sp):
                os.remove(sp)
                print(f"    Deleted {sp}")
            continue

        t0 = time.time()
        shard = np.load(sp)
        n_vec = len(shard)
        print(f"  Adding shard {shard_num:02d} | {n_vec:,} vectors...", flush=True)
        cpu_index.add(shard.astype(np.float32))
        del shard
        gc.collect()

        elapsed = time.time() - t0
        progress["shards_added"] = shard_num + 1
        progress["vectors_added"] = cpu_index.ntotal

        # Save index checkpoint after every shard
        faiss.write_index(cpu_index, index_path)
        save_progress(progress_path, progress)

        # Free disk space by deleting this shard
        if args.delete_shards:
            os.remove(sp)
            # Write sentinel so part2a knows this shard was processed and won't re-encode it
            done_path = sp.replace(".npy", ".done")
            with open(done_path, "w") as f:
                f.write(f"processed_by_part2b\nvectors_added={cpu_index.ntotal}\n")
            freed_gb = (n_vec * cpu_index.d * 4) / (1024**3)
            print(f"    Done in {elapsed:.0f}s | total_vectors={cpu_index.ntotal:,} | "
                  f"freed ~{freed_gb:.1f} GB (deleted shard + wrote .done)", flush=True)
        else:
            print(f"    Done in {elapsed:.0f}s | total_vectors={cpu_index.ntotal:,}", flush=True)

    final_size_gb = os.path.getsize(index_path) / (1024**3)
    print(f"\n[PART 2b] All available shards added in {time.time()-t_total:.0f}s")
    print(f"  Total vectors in index: {cpu_index.ntotal:,}")
    print(f"  Index size on disk: {final_size_gb:.2f} GB")

    if cpu_index.ntotal < args.n_shards * 900_000:
        remaining = args.n_shards - progress["shards_added"]
        print(f"\n⚠️  Index incomplete — {remaining} shards still missing.")
        print(f"  Re-run part2a to encode missing shards, then re-run part2b to resume.")
    else:
        print(f"\n[PART 2b COMPLETED SUCCESSFULLY] — index is complete!")
        # Clean up progress file
        if os.path.exists(progress_path):
            os.remove(progress_path)


if __name__ == "__main__":
    main()
