"""Build resumable dense corpus embeddings for legacy K50 retrieval."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src")))
from business_entity_resolution.regeneration import atomic_json, file_identity, gpu_snapshot  # noqa: E402


PREPROCESSING_ID = "concat:name_norm|address_norm|country:v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5C: Build Dense GPU Embeddings")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--index-dir", default="data/dense_index")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def iter_corpus_batches(paths: list[Path], batch_size: int):
    offset = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        country_column = "country" if "country" in available else "country_norm"
        columns = ["entity_id", "source", "name_norm", "address_norm", country_column]
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            frame = pl.from_arrow(batch)
            if country_column != "country":
                frame = frame.rename({country_column: "country"})
            yield offset, frame
            offset += frame.height


def main():
    args = parse_args()
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    split_dir = Path(args.data_dir) / args.split
    corpus_paths = [split_dir / f"{args.split}_source2.parquet", split_dir / f"{args.split}_source3.parquet"]
    missing = [str(path) for path in corpus_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing dense corpus inputs:\n" + "\n".join(missing))

    if not torch.cuda.is_available():
        raise RuntimeError("Dense embedding generation requires a Kaggle GPU runtime")
    model = SentenceTransformer(args.model_name, device="cuda")
    dimension = int(model.get_sentence_embedding_dimension())
    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in corpus_paths)
    identity = {
        "split": args.split,
        "model": args.model_name,
        "embedding_dimension": dimension,
        "preprocessing": PREPROCESSING_ID,
        "total_rows": total_rows,
        "inputs": {path.name: file_identity(path) for path in corpus_paths},
    }

    embedding_path = index_dir / "corpus_embeddings_fp16.npy"
    mapping_path = index_dir / "corpus_mapping.parquet"
    metadata_path = index_dir / "embedding_metadata.json"
    checkpoint_path = index_dir / "embedding_checkpoint.json"
    mapping_parts = index_dir / "mapping_parts"
    mapping_parts.mkdir(exist_ok=True)

    if metadata_path.is_file() and embedding_path.is_file() and mapping_path.is_file() and not args.force_rebuild:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing == identity:
            embeddings = np.load(embedding_path, mmap_mode="r")
            mapping_rows = pq.ParquetFile(mapping_path).metadata.num_rows
            if embeddings.shape == (total_rows, dimension) and mapping_rows == total_rows:
                print("Validated dense corpus cache; reuse is safe.")
                return
        raise RuntimeError("Dense cache identity mismatch; pass --force-rebuild to replace it")

    processed_rows = 0
    if checkpoint_path.is_file() and embedding_path.is_file() and not args.force_rebuild:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity:
            raise RuntimeError("Dense checkpoint identity mismatch; pass --force-rebuild")
        processed_rows = int(checkpoint["processed_rows"])
        embeddings = np.lib.format.open_memmap(embedding_path, mode="r+", dtype=np.float16)
        if embeddings.shape != (total_rows, dimension):
            raise RuntimeError("Dense checkpoint embedding shape mismatch")
        print(f"Resuming corpus encoding at {processed_rows}/{total_rows}")
    else:
        embeddings = np.lib.format.open_memmap(embedding_path, mode="w+", dtype=np.float16, shape=(total_rows, dimension))
        for part in mapping_parts.glob("*.parquet"):
            part.unlink()

    started = time.time()
    encoded_this_run = 0
    peak_gpu = 0
    gpu_samples = []
    for offset, frame in iter_corpus_batches(corpus_paths, args.chunk_size):
        end = offset + frame.height
        if end <= processed_rows:
            continue
        if offset < processed_rows:
            frame = frame.slice(processed_rows - offset)
            offset = processed_rows
            end = offset + frame.height
        texts = frame.select(pl.concat_str([
            pl.col("name_norm").fill_null(""), pl.col("address_norm").fill_null(""), pl.col("country").fill_null("")
        ], separator=" | ").alias("text"))["text"].to_list()
        batch_start = time.time()
        encoded = model.encode(
            texts, batch_size=args.batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float16, copy=False)
        embeddings[offset:end] = encoded
        embeddings.flush()
        frame.select("entity_id", "source").with_columns(
            pl.Series("faiss_row_id", np.arange(offset, end, dtype=np.uint64))
        ).write_parquet(mapping_parts / f"mapping_{offset:012d}.parquet", compression="snappy")
        processed_rows = end
        encoded_this_run += frame.height
        peak_gpu = max(peak_gpu, int(torch.cuda.max_memory_allocated()))
        sample = gpu_snapshot()
        if sample:
            gpu_samples.append(sample)
        atomic_json(checkpoint_path, {"identity": identity, "processed_rows": processed_rows})
        elapsed = time.time() - batch_start
        print(f"Encoded {offset}:{end} at {frame.height / max(elapsed, 1e-9):.1f} rows/s; "
              f"GPU peak={peak_gpu / 2**30:.2f} GiB; disk free={shutil.disk_usage(index_dir).free / 2**30:.2f} GiB")
        del texts, encoded, frame
        gc.collect()

    parts = sorted(mapping_parts.glob("mapping_*.parquet"))
    if not parts:
        raise RuntimeError("No mapping fragments exist")
    pl.concat([pl.scan_parquet(path) for path in parts]).sort("faiss_row_id").sink_parquet(
        mapping_path, compression="snappy"
    )
    if pq.ParquetFile(mapping_path).metadata.num_rows != total_rows:
        raise RuntimeError("Final dense mapping row count mismatch")
    atomic_json(metadata_path, identity)
    checkpoint_path.unlink(missing_ok=True)
    runtime = time.time() - started
    print(json.dumps({
        "status": "complete", "device": "cuda", "gpu": torch.cuda.get_device_name(0),
        "rows": total_rows, "dimension": dimension, "encoded_this_run": encoded_this_run,
        "runtime_seconds": runtime, "throughput_rows_per_second": encoded_this_run / max(runtime, 1e-9),
        "peak_gpu_memory_bytes": peak_gpu, "embedding_path": str(embedding_path),
        "mapping_path": str(mapping_path), "nvidia_smi_samples": gpu_samples,
        "disk_free_bytes": shutil.disk_usage(index_dir).free,
    }, indent=2))


if __name__ == "__main__":
    main()
