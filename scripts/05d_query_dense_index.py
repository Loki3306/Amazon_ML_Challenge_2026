"""Query the legacy dense IVF-SQ8 index and stream K50 candidates."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import polars as pl
import psutil
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src")))
from business_entity_resolution.regeneration import (  # noqa: E402
    AtomicParquetWriter, atomic_json, file_identity, gpu_snapshot,
)


PREPROCESSING_ID = "concat:name_norm|address_norm|country:v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5D: Query Dense Index")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--index-dir", default="data/dense_index")
    parser.add_argument("--query-cache-dir", default="",
                        help="Optional split/smoke-specific query embedding cache directory")
    parser.add_argument("--output-dir", default="data/candidates")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--query-chunk-size", type=int, default=50_000)
    parser.add_argument("--search-batch-size", type=int, default=4096)
    parser.add_argument("--nlist", type=int, default=16_384)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--training-rows", type=int, default=1_000_000)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def encode_queries(args, source1_path: Path, cache_dir: Path, dimension: int):
    cache_dir.mkdir(parents=True, exist_ok=True)
    query_path = cache_dir / "query_embeddings_fp32.npy"
    mapping_path = cache_dir / "query_mapping.parquet"
    metadata_path = cache_dir / "query_embedding_metadata.json"
    total = pq.ParquetFile(source1_path).metadata.num_rows
    identity = {
        "split": args.split, "model": args.model_name, "embedding_dimension": dimension,
        "preprocessing": PREPROCESSING_ID, "source1": file_identity(source1_path), "rows": total,
    }
    if query_path.is_file() and mapping_path.is_file() and metadata_path.is_file() and not args.force_rebuild:
        if json.loads(metadata_path.read_text(encoding="utf-8")) == identity:
            cached = np.load(query_path, mmap_mode="r")
            if cached.shape == (total, dimension) and pq.ParquetFile(mapping_path).metadata.num_rows == total:
                print("Validated query embedding cache; reusing it.")
                return cached, pl.read_parquet(mapping_path), identity, 0.0, 0, []
        raise RuntimeError("Query embedding cache mismatch; pass --force-rebuild")

    if not torch.cuda.is_available():
        raise RuntimeError("Dense query encoding requires a Kaggle GPU runtime")
    model = SentenceTransformer(args.model_name, device="cuda")
    embeddings = np.lib.format.open_memmap(query_path, mode="w+", dtype=np.float32, shape=(total, dimension))
    mapping_parts = []
    offset = 0
    gpu_samples = []
    started = time.time()
    parquet = pq.ParquetFile(source1_path)
    available = set(parquet.schema_arrow.names)
    country_column = "country" if "country" in available else "country_norm"
    columns = ["entity_id", "name_norm", "address_norm", country_column]
    for batch in parquet.iter_batches(batch_size=args.query_chunk_size, columns=columns):
        frame = pl.from_arrow(batch)
        if country_column != "country":
            frame = frame.rename({country_column: "country"})
        texts = frame.select(pl.concat_str([
            pl.col("name_norm").fill_null(""), pl.col("address_norm").fill_null(""), pl.col("country").fill_null("")
        ], separator=" | ").alias("text"))["text"].to_list()
        encoded = model.encode(
            texts, batch_size=args.batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32, copy=False)
        end = offset + frame.height
        embeddings[offset:end] = encoded
        embeddings.flush()
        mapping_parts.append(frame.select(pl.col("entity_id").cast(pl.Utf8).alias("query_id")))
        print(f"Encoded queries {offset}:{end}")
        sample = gpu_snapshot()
        if sample:
            gpu_samples.append(sample)
        offset = end
    query_mapping = pl.concat(mapping_parts)
    query_mapping.write_parquet(mapping_path, compression="snappy")
    atomic_json(metadata_path, identity)
    runtime = time.time() - started
    peak_gpu = int(torch.cuda.max_memory_allocated())
    del model
    torch.cuda.empty_cache()
    return embeddings, query_mapping, identity, runtime, peak_gpu, gpu_samples


def build_or_load_index(args, embeddings, embedding_identity, index_dir: Path):
    index_path = index_dir / "faiss_ivfsq8.index"
    metadata_path = index_dir / "faiss_index_metadata.json"
    total_rows, dimension = embeddings.shape
    base_identity = {
        "embedding_identity": embedding_identity, "dimension": int(dimension),
        "metric": "inner_product", "index_type": "IndexIVFScalarQuantizer_QT_8bit",
        "nlist": args.nlist, "training_rows": min(args.training_rows, total_rows),
        "training_sample_method": "first_rows_legacy", "corpus_rows": total_rows,
    }
    if index_path.is_file() and metadata_path.is_file() and not args.force_rebuild:
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(saved.get(key) == value for key, value in base_identity.items()):
            index = faiss.read_index(str(index_path))
            if index.ntotal == total_rows and index.d == dimension:
                print("Validated FAISS cache; reusing it.")
                return index, saved, 0.0
        raise RuntimeError("FAISS cache mismatch; pass --force-rebuild")

    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, dimension, args.nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
    )
    training = np.asarray(embeddings[: base_identity["training_rows"]], dtype=np.float32)
    started = time.time()
    trained_on_gpu = False
    if hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0:
        try:
            resources = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(resources, 0, index)
            gpu_index.train(training)
            index = faiss.index_gpu_to_cpu(gpu_index)
            trained_on_gpu = True
            del gpu_index, resources
        except Exception as exc:
            print(f"FAISS GPU training unsupported/unsafe for this index; using CPU: {exc}")
            index.train(training)
    else:
        print("FAISS CPU build detected; training/search will use CPU while encoding remains GPU.")
        index.train(training)
    del training
    for start in range(0, total_rows, 200_000):
        end = min(start + 200_000, total_rows)
        index.add(np.asarray(embeddings[start:end], dtype=np.float32))
        print(f"Added corpus vectors {end}/{total_rows}")
    faiss.write_index(index, str(index_path))
    metadata = {**base_identity, "trained_on_gpu": trained_on_gpu}
    atomic_json(metadata_path, metadata)
    return index, metadata, time.time() - started


def main():
    args = parse_args()
    if args.top_k != 50:
        raise ValueError("Legacy regeneration requires --top-k 50")
    index_dir, output_dir = Path(args.index_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = index_dir / "corpus_embeddings_fp16.npy"
    mapping_path = index_dir / "corpus_mapping.parquet"
    embedding_metadata_path = index_dir / "embedding_metadata.json"
    for required in (embedding_path, mapping_path, embedding_metadata_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing dense artifact {required}; run 05c first")
    embedding_identity = json.loads(embedding_metadata_path.read_text(encoding="utf-8"))
    if embedding_identity["model"] != args.model_name or embedding_identity["split"] != args.split:
        raise RuntimeError("Dense embedding model/split identity mismatch")
    embeddings = np.load(embedding_path, mmap_mode="r")
    dimension = int(embedding_identity["embedding_dimension"])
    if embeddings.ndim != 2 or embeddings.shape[1] != dimension:
        raise RuntimeError("Dense embedding dimension/metadata mismatch")
    corpus_mapping = pl.read_parquet(mapping_path).sort("faiss_row_id")
    if corpus_mapping.height != embeddings.shape[0]:
        raise RuntimeError("Dense embedding/mapping row mismatch")

    source1_path = Path(args.data_dir) / args.split / f"{args.split}_source1.parquet"
    query_cache_dir = Path(args.query_cache_dir) if args.query_cache_dir else index_dir
    query_embeddings, query_mapping, _, encode_runtime, peak_gpu, gpu_samples = encode_queries(
        args, source1_path, query_cache_dir, dimension
    )
    index, index_metadata, index_runtime = build_or_load_index(args, embeddings, embedding_identity, index_dir)
    index.nprobe = args.nprobe
    output_path = output_dir / f"{args.split}_dense_candidates_K{args.top_k}.parquet"
    writer = AtomicParquetWriter(output_path)
    candidate_ids = corpus_mapping["entity_id"].to_numpy()
    candidate_sources = corpus_mapping["source"].to_numpy()
    query_ids = query_mapping["query_id"].to_numpy()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.time()
    try:
        for start in range(0, len(query_ids), args.search_batch_size):
            end = min(start + args.search_batch_size, len(query_ids))
            scores, indices = index.search(np.asarray(query_embeddings[start:end]), args.top_k)
            local_rows, ranks = np.indices(indices.shape)
            valid = indices >= 0
            if not np.all(valid):
                print(f"Rejected {int((~valid).sum())} FAISS -1 results in query batch {start}:{end}")
            flat_indices = indices[valid]
            frame = pl.DataFrame({
                "query_id": query_ids[start:end][local_rows[valid]],
                "candidate_id": candidate_ids[flat_indices],
                "candidate_source": candidate_sources[flat_indices],
                "dense_rank": (ranks[valid] + 1).astype(np.int32),
                "dense_score": scores[valid].astype(np.float32),
                "retrieval_method": np.repeat("dense", int(valid.sum())),
            })
            writer.write(frame)
            peak_rss = max(peak_rss, process.memory_info().rss)
            print(f"Searched/wrote queries {end}/{len(query_ids)}; rows={writer.rows}")
        writer.close()
    except BaseException:
        writer.abort()
        raise
    search_runtime = time.time() - started
    log = {
        "status": "complete", "split": args.split, "model": args.model_name,
        "dimension": dimension, "faiss": index_metadata, "nprobe": args.nprobe, "k": args.top_k,
        "faiss_build": "gpu-capable" if hasattr(faiss, "StandardGpuResources") else "cpu",
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_peak_memory_bytes": peak_gpu, "query_encode_seconds": encode_runtime,
        "index_build_seconds": index_runtime, "search_seconds": search_runtime,
        "faiss_query_throughput": len(query_ids) / max(search_runtime, 1e-9),
        "host_peak_rss_bytes": peak_rss, "rows_processed": len(query_ids),
        "candidate_rows_written": writer.rows, "output": str(output_path),
        "nvidia_smi_samples": gpu_samples,
        "output_size_bytes": output_path.stat().st_size,
        "disk_free_bytes": shutil.disk_usage(output_dir).free,
    }
    atomic_json(index_dir / "dense_run_log.json", log)
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
