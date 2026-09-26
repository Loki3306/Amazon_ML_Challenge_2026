"""Execution utilities for reproducible legacy-candidate regeneration.

This module deliberately contains execution and validation plumbing only.  The
retrieval algorithms remain in their historical producer scripts.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import polars as pl
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp


PROCESSED_COLUMNS = {
    "entity_id",
    "source",
    "name_norm",
    "address_norm",
    "country_norm",
}

CANDIDATE_COLUMNS = {
    "exact": {"query_id", "candidate_id", "candidate_source", "match_name", "match_address"},
    "char": {"query_id", "candidate_id", "candidate_source", "char_score", "found_by_char"},
    "word": {"query_id", "candidate_id", "candidate_source", "bm25_score", "found_by_bm25"},
    "structured": {"query_id", "candidate_id", "candidate_source", "block_names", "found_by_block"},
    "dense": {"query_id", "candidate_id", "candidate_source", "dense_rank", "dense_score"},
}


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def file_identity(path: str | Path, hash_bytes: int = 4 * 1024 * 1024) -> dict[str, Any]:
    """Return a stable, bounded-cost identity using size plus head/tail content."""
    path = Path(path).resolve()
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(hash_bytes))
        if stat.st_size > hash_bytes:
            handle.seek(max(0, stat.st_size - hash_bytes))
            digest.update(handle.read(hash_bytes))
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "head_tail_sha256": digest.hexdigest(),
    }


def git_commit(repo_root: str | Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def gpu_snapshot() -> dict[str, int] | None:
    """Return a lightweight NVIDIA utilization/VRAM sample when available."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0]
        utilization, used_mib, total_mib = [int(value.strip()) for value in output.split(",")]
        return {
            "utilization_percent": utilization,
            "memory_used_bytes": used_mib * 2**20,
            "memory_total_bytes": total_mib * 2**20,
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def processed_paths(data_dir: str | Path, split: str) -> dict[str, Path]:
    if split not in {"train", "test"}:
        raise ValueError(f"split must be train or test, got {split!r}")
    base = Path(data_dir) / split
    return {f"source{i}": base / f"{split}_source{i}.parquet" for i in (1, 2, 3)}


def verify_processed_inputs(data_dir: str | Path, split: str) -> dict[str, Any]:
    paths = processed_paths(data_dir, split)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing processed artifacts:\n" + "\n".join(missing))

    report: dict[str, Any] = {"split": split, "files": {}}
    for label, path in paths.items():
        if path.name.startswith(("train_" if split == "test" else "test_")):
            raise RuntimeError(f"train/test path crossover detected: {path}")
        schema = pq.read_schema(path)
        columns = set(schema.names)
        missing_columns = sorted(PROCESSED_COLUMNS - columns)
        if missing_columns:
            raise ValueError(f"{path} lacks required columns: {missing_columns}")
        metadata = pq.ParquetFile(path).metadata
        if metadata.num_rows <= 0:
            raise ValueError(f"processed artifact is empty: {path}")
        report["files"][label] = {
            "path": str(path.resolve()),
            "rows": metadata.num_rows,
            "schema": str(schema),
            "identity": file_identity(path),
        }
    return report


def deterministic_smoke_ids(source1_path: str | Path, count: int, seed: int) -> list[str]:
    """Select Source-1 query IDs by a stable seeded hash; never sample candidate rows."""
    if count <= 0:
        raise ValueError("smoke query count must be positive")
    ids = (
        pl.scan_parquet(source1_path)
        .select(pl.col("entity_id").cast(pl.Utf8))
        .unique()
        .collect()["entity_id"]
        .to_list()
    )
    ranked = sorted(
        ids,
        key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).digest(),
    )
    return ranked[: min(count, len(ranked))]


def prepare_smoke_inputs(
    data_dir: str | Path,
    smoke_data_dir: str | Path,
    split: str,
    query_ids: Iterable[str],
) -> dict[str, str]:
    """Create a smoke data tree with filtered S1 and the complete S2/S3 corpus."""
    source = processed_paths(data_dir, split)
    target_dir = Path(smoke_data_dir) / split
    target_dir.mkdir(parents=True, exist_ok=True)
    wanted = pl.DataFrame({"entity_id": list(query_ids)}).with_columns(pl.col("entity_id").cast(pl.Utf8))
    s1 = pl.read_parquet(source["source1"]).with_columns(pl.col("entity_id").cast(pl.Utf8))
    selected = s1.join(wanted, on="entity_id", how="semi")
    if selected.height != wanted.height:
        raise RuntimeError(f"selected {selected.height} smoke queries, expected {wanted.height}")
    s1_target = target_dir / f"{split}_source1.parquet"
    selected.write_parquet(s1_target, compression="snappy")

    result = {"source1": str(s1_target)}
    for label in ("source2", "source3"):
        target = target_dir / source[label].name
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            target.symlink_to(source[label].resolve())
        except OSError:
            shutil.copy2(source[label], target)
        result[label] = str(target)
    return result


def sparse_matrix_bytes(matrix: sp.spmatrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def sparse_memory_projection(
    corpus: sp.csr_matrix,
    corpus_t: sp.csr_matrix,
    query: sp.csr_matrix,
    query_count: int,
    top_k: int,
    chunk_size: int,
) -> dict[str, int]:
    # Three fixed-width result arrays plus conservative UTF-8/id-frame overhead.
    result_arrays = int(query_count * top_k * (4 + 4 + 4))
    result_frame = int(query_count * top_k * 96)
    bounded_rows = min(query_count, chunk_size) * top_k
    bounded_result = int(bounded_rows * (4 + 4 + 4 + 96))
    sampled_queries = max(1, query.shape[0])
    legacy_full_query = int(sparse_matrix_bytes(query) * query_count / sampled_queries)
    return {
        "corpus_csr_bytes": sparse_matrix_bytes(corpus),
        "corpus_transpose_csr_bytes": sparse_matrix_bytes(corpus_t),
        "query_csr_bytes": sparse_matrix_bytes(query),
        "query_sample_rows": query.shape[0],
        "legacy_full_query_csr_estimate_bytes": legacy_full_query,
        "legacy_global_result_arrays_bytes": result_arrays,
        "legacy_result_frame_estimate_bytes": result_frame,
        "bounded_result_chunk_estimate_bytes": bounded_result,
        "projected_working_set_bytes": (
            sparse_matrix_bytes(corpus)
            + sparse_matrix_bytes(corpus_t)
            + sparse_matrix_bytes(query)
            + bounded_result
        ),
    }


def enforce_memory_budget(projection: dict[str, int], fraction: float = 0.75) -> None:
    available = psutil.virtual_memory().available
    projected = projection["projected_working_set_bytes"]
    print(json.dumps({**projection, "available_host_bytes": available, "limit_fraction": fraction}, indent=2))
    if projected > available * fraction:
        raise MemoryError(
            f"Projected sparse working set {projected / 2**30:.2f} GiB exceeds "
            f"{fraction:.0%} of available host RAM ({available / 2**30:.2f} GiB)."
        )


def _topk_callable(top_k: int):
    try:
        from sparse_dot_topn import sp_matmul_topn

        return lambda a, b: sp_matmul_topn(a, b, top_n=top_k, n_threads=-1, threshold=0.0), "sp_matmul_topn"
    except ImportError:
        try:
            from sparse_dot_topn import awesome_cossim_topn

            return (
                lambda a, b: awesome_cossim_topn(
                    a, b, ntop=top_k, lower_bound=0.0, use_threads=True, n_jobs=-1
                ),
                "awesome_cossim_topn",
            )
        except ImportError as exc:
            raise RuntimeError(
                "sparse_dot_topn is required for bounded, semantics-preserving full regeneration"
            ) from exc


def iter_sparse_topk(
    query_matrix: sp.csr_matrix,
    corpus_matrix_t: sp.csr_matrix,
    top_k: int,
    chunk_size: int,
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield per-query top-K chunks without retaining global result arrays."""
    topk, implementation = _topk_callable(top_k)
    print(f"Top-K implementation: {implementation}; bounded query chunk={chunk_size}")
    for start in range(0, query_matrix.shape[0], chunk_size):
        end = min(start + chunk_size, query_matrix.shape[0])
        result = topk(query_matrix[start:end], corpus_matrix_t).tocoo()
        yield (
            start,
            result.row.astype(np.int32, copy=False),
            result.col.astype(np.int32, copy=False),
            result.data.astype(np.float32, copy=False),
        )


@dataclass
class AtomicParquetWriter:
    path: Path
    compression: str = "snappy"

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_path = self.path.with_suffix(self.path.suffix + ".partial")
        self.writer: pq.ParquetWriter | None = None
        self.rows = 0
        if self.temp_path.exists():
            self.temp_path.unlink()

    def write(self, frame: pl.DataFrame) -> None:
        if frame.is_empty():
            return
        table = frame.to_arrow()
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.temp_path, table.schema, compression=self.compression)
        self.writer.write_table(table)
        self.rows += frame.height

    def close(self) -> None:
        if self.writer is None:
            raise RuntimeError(f"No rows were written for {self.path}")
        self.writer.close()
        self.writer = None
        os.replace(self.temp_path, self.path)

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.temp_path.exists():
            self.temp_path.unlink()


def write_sparse_candidates(
    output_path: str | Path,
    query_matrix: sp.csr_matrix,
    corpus_matrix_t: sp.csr_matrix,
    query_ids: np.ndarray,
    corpus_ids: np.ndarray,
    corpus_sources: np.ndarray,
    top_k: int,
    chunk_size: int,
    score_column: str,
    flag_column: str,
) -> int:
    writer = AtomicParquetWriter(Path(output_path))
    try:
        for start, local_rows, cols, scores in iter_sparse_topk(
            query_matrix, corpus_matrix_t, top_k, chunk_size
        ):
            global_rows = local_rows + start
            frame = pl.DataFrame(
                {
                    "query_id": query_ids[global_rows],
                    "candidate_id": corpus_ids[cols],
                    "candidate_source": corpus_sources[cols],
                    score_column: scores,
                    flag_column: np.ones(len(scores), dtype=bool),
                }
            ).filter(pl.col("query_id") != pl.col("candidate_id"))
            writer.write(frame)
            print(f"  wrote through query {min(start + chunk_size, len(query_ids))}/{len(query_ids)}")
        writer.close()
        return writer.rows
    except BaseException:
        writer.abort()
        raise


def write_sparse_candidates_from_texts(
    output_path: str | Path,
    vectorizer: Any,
    query_texts: list[str],
    corpus_matrix_t: sp.csr_matrix,
    query_ids: np.ndarray,
    corpus_ids: np.ndarray,
    corpus_sources: np.ndarray,
    top_k: int,
    chunk_size: int,
    score_column: str,
    flag_column: str,
) -> int:
    """Transform, retrieve, and write query chunks without global query/results matrices."""
    writer = AtomicParquetWriter(Path(output_path))
    try:
        for start in range(0, len(query_texts), chunk_size):
            end = min(start + chunk_size, len(query_texts))
            query_chunk = vectorizer.transform(query_texts[start:end]).tocsr()
            for _, local_rows, cols, scores in iter_sparse_topk(
                query_chunk, corpus_matrix_t, top_k, chunk_size
            ):
                global_rows = local_rows + start
                frame = pl.DataFrame(
                    {
                        "query_id": query_ids[global_rows],
                        "candidate_id": corpus_ids[cols],
                        "candidate_source": corpus_sources[cols],
                        score_column: scores,
                        flag_column: np.ones(len(scores), dtype=bool),
                    }
                ).filter(pl.col("query_id") != pl.col("candidate_id"))
                writer.write(frame)
            print(f"  transformed/retrieved/wrote queries {end}/{len(query_texts)}")
        writer.close()
        return writer.rows
    except BaseException:
        writer.abort()
        raise


def validate_candidate(path: str | Path, retriever: str) -> dict[str, Any]:
    path = Path(path)
    if retriever not in CANDIDATE_COLUMNS:
        raise ValueError(f"unknown retriever {retriever!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    schema = pq.read_schema(path)
    missing = sorted(CANDIDATE_COLUMNS[retriever] - set(schema.names))
    if missing:
        raise ValueError(f"{path} lacks candidate columns: {missing}")

    lazy = pl.scan_parquet(path)
    row_count = lazy.select(pl.len()).collect().item()
    if row_count <= 0:
        raise ValueError(f"candidate artifact is empty: {path}")
    query_count = lazy.select(pl.col("query_id").n_unique()).collect().item()
    candidate_count = lazy.select(pl.col("candidate_id").n_unique()).collect().item()
    null_pair_keys = lazy.select(
        (pl.col("query_id").is_null() | pl.col("candidate_id").is_null() |
         pl.col("candidate_source").is_null()).sum()
    ).collect().item()
    duplicate_count = (
        lazy.group_by(["query_id", "candidate_id", "candidate_source"])
        .len()
        .filter(pl.col("len") > 1)
        .select((pl.col("len") - 1).sum())
        .collect()
        .item()
        or 0
    )
    if null_pair_keys:
        raise ValueError(f"{path} contains {null_pair_keys} rows with null pair keys")
    if duplicate_count:
        raise ValueError(f"{path} contains {duplicate_count} duplicate canonical pair keys")
    if retriever in {"char", "word", "dense"} and row_count > query_count * 50:
        raise ValueError(
            f"{path} has {row_count} rows, exceeding the plausible K50 ceiling "
            f"of {query_count * 50}"
        )
    distribution = (
        lazy.group_by("query_id")
        .len()
        .select(
            pl.col("len").mean().alias("mean"),
            pl.col("len").quantile(0.50, interpolation="nearest").alias("p50"),
            pl.col("len").quantile(0.90, interpolation="nearest").alias("p90"),
            pl.col("len").quantile(0.95, interpolation="nearest").alias("p95"),
            pl.col("len").quantile(0.99, interpolation="nearest").alias("p99"),
        )
        .collect()
        .to_dicts()[0]
    )
    return {
        "path": str(path.resolve()),
        "file_size": path.stat().st_size,
        "row_count": row_count,
        "query_count": query_count,
        "candidate_count": candidate_count,
        "duplicate_canonical_key_count": int(duplicate_count),
        "null_pair_key_count": int(null_pair_keys),
        "candidates_per_query": distribution,
        "schema": str(schema),
    }


def manifest_matches(
    manifest_path: str | Path,
    output_path: str | Path,
    expected_config: dict[str, Any],
    retriever: str,
    expected_input_identity: dict[str, Any] | None = None,
) -> bool:
    manifest_path, output_path = Path(manifest_path), Path(output_path)
    if not manifest_path.is_file() or not output_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("config") != expected_config:
            return False
        if (expected_input_identity is not None and
                manifest.get("input_processed_data_identity") != expected_input_identity):
            return False
        current = validate_candidate(output_path, retriever)
        return (
            current["row_count"] == manifest.get("validation", {}).get("row_count")
            and current["file_size"] == manifest.get("validation", {}).get("file_size")
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def build_manifest(
    *,
    artifact_type: str,
    split: str,
    retriever: str,
    source_script: str,
    repo_root: str | Path,
    processed_report: dict[str, Any],
    config: dict[str, Any],
    validation: dict[str, Any],
    runtime_seconds: float,
    peak_rss_bytes: int,
    peak_gpu_memory_bytes: int | None = None,
    dense: dict[str, Any] | None = None,
) -> dict[str, Any]:
    versions = {}
    for name in ("polars", "pyarrow", "numpy", "scipy", "sklearn", "sentence_transformers", "faiss", "psutil"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = None
    payload = {
        "artifact_type": artifact_type,
        "split": split,
        "retriever": retriever,
        "source_script": source_script,
        "git_commit": git_commit(repo_root),
        "input_processed_data_identity": {
            key: value["identity"] for key, value in processed_report["files"].items()
        },
        "config": config,
        "validation": validation,
        "runtime_seconds": runtime_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "library_versions": versions,
        "status": "complete",
    }
    if dense:
        payload["dense"] = dense
    return payload


def environment_report(repo_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(output_root)
    gpu = {"available": False, "model": None, "vram_bytes": None, "cuda": None}
    try:
        import torch

        gpu["available"] = torch.cuda.is_available()
        gpu["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu["model"] = props.name
            gpu["vram_bytes"] = props.total_memory
    except ImportError:
        pass
    faiss_info = {"installed": False, "gpu_build": False, "gpus": 0}
    try:
        import faiss

        faiss_info["installed"] = True
        faiss_info["gpu_build"] = hasattr(faiss, "StandardGpuResources")
        if faiss_info["gpu_build"]:
            faiss_info["gpus"] = int(faiss.get_num_gpus())
    except ImportError:
        pass
    return {
        "git_commit": git_commit(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "host_ram_bytes": vm.total,
        "host_ram_available_bytes": vm.available,
        "free_disk_bytes": disk.free,
        "gpu": gpu,
        "faiss": faiss_info,
    }
