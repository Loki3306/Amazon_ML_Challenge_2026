"""Bounded-memory production builder for canonical candidate_v1 shards.

The builder reads immutable legacy retriever Parquets with PyArrow batches,
stages deterministic query partitions, and aggregates one partition at a time.
It never globally materializes all retriever candidates.  The small Python-row
union in :mod:`business_entity_resolution.candidates` remains the semantic
reference implementation; this module provides its vectorized equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any, Final, Literal, Sequence

import polars as pl
import pyarrow.parquet as pq

from .candidates import (
    CANDIDATE_SCHEMA_VERSION,
    CANONICAL_CANDIDATE_COLUMNS,
    CANONICAL_CANDIDATE_KEY,
    CANONICAL_CANDIDATE_SCHEMA,
    DEFAULT_CANDIDATE_SOURCES,
    PROVENANCE_COLUMNS,
)


PARTITION_HASH_VERSION: Final = "polars_hash64_seeded_v1"
BUILDER_VERSION: Final = "candidate_builder_v1"
ARTIFACT_TYPE: Final = "canonical_candidates"
DEFAULT_REQUIRED_RETRIEVERS: Final[tuple[str, ...]] = (
    "exact",
    "dense",
    "word_tfidf",
    "char_tfidf",
    "structured",
)

RetrieverName = Literal[
    "exact", "dense", "word_tfidf", "char_tfidf", "structured"
]


class CandidateBuildError(RuntimeError):
    """Raised when a canonical build or resume invariant is violated."""


@dataclass(frozen=True)
class RetrieverInput:
    """One immutable legacy retriever artifact."""

    retriever: RetrieverName
    path: Path


@dataclass(frozen=True)
class CandidateBuildConfig:
    """Configuration whose fingerprint defines one canonical artifact."""

    split: str
    inputs: tuple[RetrieverInput, ...]
    output_dir: Path
    staging_dir: Path
    partition_count: int = 64
    hash_seed: int = 42
    batch_size: int = 1_000_000
    required_retrievers: tuple[str, ...] = DEFAULT_REQUIRED_RETRIEVERS
    dataset_fingerprint: str | None = None
    smoke_query_ids: tuple[str, ...] = ()
    enforce_disk_safety: bool = True
    disk_safety_fraction: float = 0.80
    staging_size_factor: float = 1.35
    output_size_factor: float = 1.00

    def __post_init__(self) -> None:
        if self.split not in {"train", "test"}:
            raise CandidateBuildError("split must be 'train' or 'test'")
        if self.partition_count < 1:
            raise CandidateBuildError("partition_count must be positive")
        if self.batch_size < 1:
            raise CandidateBuildError("batch_size must be positive")
        if not 0 < self.disk_safety_fraction <= 1:
            raise CandidateBuildError("disk_safety_fraction must be in (0, 1]")
        names = [item.retriever for item in self.inputs]
        if len(names) != len(set(names)):
            raise CandidateBuildError("each retriever may appear only once")
        missing = sorted(set(self.required_retrievers) - set(names))
        if missing:
            raise CandidateBuildError(
                "MISSING REQUIRED RETRIEVER ARTIFACT:\n" + "\n".join(missing)
            )


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def fingerprint_input(path: Path) -> dict[str, Any]:
    """Fingerprint a large input cheaply using resolved path and stat metadata."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise CandidateBuildError(f"input artifact does not exist: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _library_versions() -> dict[str, str]:
    import pyarrow

    return {
        "python": platform.python_version(),
        "polars": pl.__version__,
        "pyarrow": pyarrow.__version__,
    }


def _source_fingerprints() -> dict[str, str]:
    from . import candidates

    return {
        "candidate_builder.py": sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidates.py": sha256(Path(candidates.__file__).read_bytes()).hexdigest(),
    }


def _static_manifest_fields(config: CandidateBuildConfig) -> dict[str, Any]:
    identities = {
        item.retriever: fingerprint_input(item.path) for item in config.inputs
    }
    smoke_hash = _json_hash(sorted(config.smoke_query_ids)) if config.smoke_query_ids else None
    dataset_fingerprint = config.dataset_fingerprint or _json_hash(identities)
    config_payload = {
        "builder_version": BUILDER_VERSION,
        "builder_source_fingerprints": _source_fingerprints(),
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "split": config.split,
        "retriever_inputs": identities,
        "required_retrievers": list(config.required_retrievers),
        "partition_count": config.partition_count,
        "partition_hash_version": PARTITION_HASH_VERSION,
        "partition_hash_library": {"polars": pl.__version__},
        "partition_hash_seed": config.hash_seed,
        "batch_size": config.batch_size,
        "staging_dir": str(config.staging_dir.resolve()),
        "dataset_fingerprint": dataset_fingerprint,
        "smoke_query_ids_hash": smoke_hash,
        "smoke_query_count": len(config.smoke_query_ids),
        "smoke_query_ids": list(config.smoke_query_ids),
    }
    return {
        "artifact_type": ARTIFACT_TYPE,
        **config_payload,
        "config_fingerprint": _json_hash(config_payload),
    }


def _new_manifest(config: CandidateBuildConfig) -> dict[str, Any]:
    return {
        **_static_manifest_fields(config),
        "creation_commit": _git_commit(),
        "library_versions": _library_versions(),
        "created_at_unix": time.time(),
        "staged_batches": {item.retriever: {} for item in config.inputs},
        "completed_partitions": {},
        "input_rows_by_retriever": {item.retriever: 0 for item in config.inputs},
        "staged_rows_by_retriever": {item.retriever: 0 for item in config.inputs},
        "canonical_row_count": 0,
        "duplicate_rows_collapsed": 0,
        "peak_rss_bytes": None,
        "minimum_free_disk_bytes": None,
        "builder_runtime_seconds": 0.0,
        "status": "building",
    }


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_or_create_manifest(
    config: CandidateBuildConfig, manifest_path: Path
) -> dict[str, Any]:
    """Load a compatible resume manifest or create a new one."""

    expected = _static_manifest_fields(config)
    if not manifest_path.exists():
        manifest = _new_manifest(config)
        _atomic_json_write(manifest_path, manifest)
        return manifest

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for field, expected_value in expected.items():
        if manifest.get(field) != expected_value:
            raise CandidateBuildError(
                f"resume manifest mismatch for {field}: "
                f"stored={manifest.get(field)!r}, expected={expected_value!r}"
            )
    return manifest


def _require_legacy_columns(frame: pl.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise CandidateBuildError(f"{context} is missing columns: {missing}")


def _null(dtype: pl.DataType, name: str) -> pl.Expr:
    return pl.lit(None, dtype=dtype).alias(name)


def _false(name: str) -> pl.Expr:
    return pl.lit(False, dtype=pl.Boolean).alias(name)


def _canonical_base_expressions() -> dict[str, pl.Expr]:
    return {
        "found_by_exact": _false("found_by_exact"),
        "found_by_dense": _false("found_by_dense"),
        "found_by_word_tfidf": _false("found_by_word_tfidf"),
        "found_by_char_tfidf": _false("found_by_char_tfidf"),
        "found_by_structured": _false("found_by_structured"),
        "exact_name_match": _false("exact_name_match"),
        "exact_address_match": _false("exact_address_match"),
        "dense_score": _null(pl.Float64, "dense_score"),
        "dense_rank": _null(pl.Int64, "dense_rank"),
        "word_score": _null(pl.Float64, "word_score"),
        "word_rank": _null(pl.Int64, "word_rank"),
        "char_score": _null(pl.Float64, "char_score"),
        "char_rank": _null(pl.Int64, "char_rank"),
        "structured_block_names": _null(pl.Utf8, "structured_block_names"),
        "block_count": pl.lit(0, dtype=pl.Int64).alias("block_count"),
    }


def _normalised_block_list(column: str) -> pl.Expr:
    return pl.col(column).str.split("|").list.unique().list.sort()


def adapt_legacy_batch(frame: pl.DataFrame, retriever: RetrieverName) -> pl.DataFrame:
    """Vectorize one bounded legacy batch into canonical candidate rows."""

    _require_legacy_columns(frame, CANONICAL_CANDIDATE_KEY, f"legacy {retriever} batch")
    expressions = _canonical_base_expressions()

    if retriever == "exact":
        _require_legacy_columns(frame, ("match_name", "match_address"), "exact batch")
        name_match = pl.col("match_name").fill_null(False).cast(pl.Boolean)
        address_match = pl.col("match_address").fill_null(False).cast(pl.Boolean)
        expressions.update(
            found_by_exact=(name_match | address_match).alias("found_by_exact"),
            exact_name_match=name_match.alias("exact_name_match"),
            exact_address_match=address_match.alias("exact_address_match"),
        )
    elif retriever == "dense":
        _require_legacy_columns(frame, ("dense_score", "dense_rank"), "dense batch")
        expressions.update(
            found_by_dense=pl.lit(True).alias("found_by_dense"),
            dense_score=pl.col("dense_score").cast(pl.Float64).alias("dense_score"),
            dense_rank=pl.col("dense_rank").cast(pl.Int64).alias("dense_rank"),
        )
    elif retriever == "word_tfidf":
        score_column = "word_score" if "word_score" in frame.columns else "bm25_score"
        flag_column = (
            "found_by_word_tfidf"
            if "found_by_word_tfidf" in frame.columns
            else "found_by_bm25"
        )
        _require_legacy_columns(frame, (score_column,), "word TF-IDF batch")
        found = (
            pl.col(flag_column).fill_null(False).cast(pl.Boolean)
            if flag_column in frame.columns
            else pl.lit(True)
        )
        expressions.update(
            found_by_word_tfidf=found.alias("found_by_word_tfidf"),
            word_score=pl.col(score_column).cast(pl.Float64).alias("word_score"),
        )
    elif retriever == "char_tfidf":
        _require_legacy_columns(frame, ("char_score",), "character TF-IDF batch")
        found = (
            pl.col("found_by_char").fill_null(False).cast(pl.Boolean)
            if "found_by_char" in frame.columns
            else pl.lit(True)
        )
        expressions.update(
            found_by_char_tfidf=found.alias("found_by_char_tfidf"),
            char_score=pl.col("char_score").cast(pl.Float64).alias("char_score"),
        )
    elif retriever == "structured":
        _require_legacy_columns(frame, ("block_names",), "structured batch")
        found = (
            pl.col("found_by_block").fill_null(False).cast(pl.Boolean)
            if "found_by_block" in frame.columns
            else pl.lit(True)
        )
        blocks = _normalised_block_list("block_names")
        expressions.update(
            found_by_structured=found.alias("found_by_structured"),
            structured_block_names=blocks.list.join("|").alias(
                "structured_block_names"
            ),
            block_count=blocks.list.len().cast(pl.Int64).alias("block_count"),
        )
    else:
        raise CandidateBuildError(f"unknown retriever: {retriever!r}")

    result = frame.select(
        [
            pl.col("query_id").cast(pl.Utf8),
            pl.col("candidate_id").cast(pl.Utf8),
            pl.col("candidate_source").cast(pl.Utf8),
        ]
        + [expressions[column] for column in CANONICAL_CANDIDATE_COLUMNS[3:-1]]
    ).with_columns(
        pl.sum_horizontal([pl.col(column).cast(pl.Int64) for column in PROVENANCE_COLUMNS])
        .cast(pl.Int64)
        .alias("num_retrievers")
    )
    result = enforce_canonical_schema(result)
    validate_candidate_partition(result, require_unique=False)
    return result


def enforce_canonical_schema(frame: pl.DataFrame) -> pl.DataFrame:
    """Select canonical order and cast every candidate_v1 column explicitly."""

    missing = sorted(set(CANONICAL_CANDIDATE_COLUMNS) - set(frame.columns))
    if missing:
        raise CandidateBuildError(f"canonical frame is missing columns: {missing}")
    return frame.select(
        [
            pl.col(column).cast(dtype).alias(column)
            for column, dtype in CANONICAL_CANDIDATE_SCHEMA.items()
        ]
    )


def _has_rows(frame: pl.DataFrame, predicate: pl.Expr) -> bool:
    return bool(frame.select(predicate.any()).item())


def validate_candidate_partition(
    frame: pl.DataFrame,
    *,
    require_unique: bool = True,
    allowed_candidate_sources: Sequence[str] = tuple(DEFAULT_CANDIDATE_SOURCES),
) -> None:
    """Vectorized strict candidate_v1 validation suitable for large shards."""

    if tuple(frame.columns) != CANONICAL_CANDIDATE_COLUMNS:
        raise CandidateBuildError("candidate columns are missing, extra, or out of order")
    dtype_errors = {
        column: (frame.schema[column], expected)
        for column, expected in CANONICAL_CANDIDATE_SCHEMA.items()
        if frame.schema[column] != expected
    }
    if dtype_errors:
        raise CandidateBuildError(f"candidate dtype mismatch: {dtype_errors}")
    if frame.is_empty():
        return

    key_invalid = pl.any_horizontal(
        [pl.col(column).is_null() | (pl.col(column).str.len_chars() == 0) for column in CANONICAL_CANDIDATE_KEY]
    )
    if _has_rows(frame, key_invalid):
        raise CandidateBuildError("candidate shard contains null or empty key values")
    if _has_rows(frame, ~pl.col("candidate_source").is_in(list(allowed_candidate_sources))):
        raise CandidateBuildError("candidate shard contains invalid candidate_source")
    if require_unique and frame.select(list(CANONICAL_CANDIDATE_KEY)).n_unique() != frame.height:
        raise CandidateBuildError("candidate shard contains duplicate canonical keys")

    required_nonnull = PROVENANCE_COLUMNS + (
        "exact_name_match",
        "exact_address_match",
        "block_count",
        "num_retrievers",
    )
    if _has_rows(
        frame, pl.any_horizontal([pl.col(c).is_null() for c in required_nonnull])
    ):
        raise CandidateBuildError(
            "provenance, exact flags, block_count, and num_retrievers must be non-null"
        )
    if _has_rows(frame, pl.col("block_count") < 0):
        raise CandidateBuildError("block_count must be non-negative")

    for prefix, flag, rank_required in (
        ("dense", "found_by_dense", True),
        ("word", "found_by_word_tfidf", False),
        ("char", "found_by_char_tfidf", False),
    ):
        score = pl.col(f"{prefix}_score")
        rank = pl.col(f"{prefix}_rank")
        found = pl.col(flag)
        if _has_rows(frame, score.is_not_null() & ~score.is_finite()):
            raise CandidateBuildError(f"{prefix} score contains non-finite values")
        if _has_rows(frame, rank.is_not_null() & (rank < 1)):
            raise CandidateBuildError(f"{prefix} rank contains non-positive values")
        if _has_rows(frame, ~found & (score.is_not_null() | rank.is_not_null())):
            raise CandidateBuildError(f"{prefix} evidence exists while flag is false")
        required_missing = score.is_null() | (rank.is_null() if rank_required else pl.lit(False))
        if _has_rows(frame, found & required_missing):
            raise CandidateBuildError(f"{prefix} flag is true but required evidence is missing")

    if _has_rows(
        frame,
        pl.col("found_by_exact")
        & ~(pl.col("exact_name_match") | pl.col("exact_address_match")),
    ):
        raise CandidateBuildError("exact flag is true without exact match evidence")
    if _has_rows(
        frame,
        ~pl.col("found_by_exact")
        & (pl.col("exact_name_match") | pl.col("exact_address_match")),
    ):
        raise CandidateBuildError("exact match evidence exists while exact flag is false")

    expected_blocks = _normalised_block_list("structured_block_names")
    if _has_rows(
        frame,
        pl.col("found_by_structured")
        & (
            pl.col("structured_block_names").is_null()
            | (pl.col("block_count") < 1)
            | (expected_blocks.list.len().cast(pl.Int64) != pl.col("block_count"))
            | (expected_blocks.list.join("|") != pl.col("structured_block_names"))
        ),
    ):
        raise CandidateBuildError("structured evidence or block_count is inconsistent")
    if _has_rows(
        frame,
        ~pl.col("found_by_structured")
        & (
            pl.col("structured_block_names").is_not_null()
            | (pl.col("block_count") != 0)
        ),
    ):
        raise CandidateBuildError("structured evidence exists while flag is false")

    expected_retrievers = pl.sum_horizontal(
        [pl.col(column).cast(pl.Int64) for column in PROVENANCE_COLUMNS]
    )
    if _has_rows(frame, expected_retrievers < 1):
        raise CandidateBuildError("canonical candidate has no retrieval provenance")
    if _has_rows(frame, expected_retrievers != pl.col("num_retrievers")):
        raise CandidateBuildError("num_retrievers does not match provenance flags")


def aggregate_candidate_partition(frame: pl.DataFrame) -> pl.DataFrame:
    """Vectorize candidate_v1 union/dedup for one bounded query partition."""

    validate_candidate_partition(frame, require_unique=False)
    if frame.is_empty():
        return enforce_canonical_schema(frame).sort(
            ["query_id", "candidate_source", "candidate_id"]
        )

    keys = list(CANONICAL_CANDIDATE_KEY)
    aggregated = frame.group_by(keys).agg(
        [pl.col(column).any().alias(column) for column in PROVENANCE_COLUMNS]
        + [
            pl.col("exact_name_match").any().alias("exact_name_match"),
            pl.col("exact_address_match").any().alias("exact_address_match"),
            pl.col("dense_score").max().alias("dense_score"),
            pl.col("dense_rank").min().alias("dense_rank"),
            pl.col("word_score").max().alias("word_score"),
            pl.col("word_rank").min().alias("word_rank"),
            pl.col("char_score").max().alias("char_score"),
            pl.col("char_rank").min().alias("char_rank"),
        ]
    )

    block_rows = (
        frame.filter(pl.col("found_by_structured"))
        .select(keys + [_normalised_block_list("structured_block_names").alias("_blocks")])
        .explode("_blocks")
        .filter(pl.col("_blocks").is_not_null() & (pl.col("_blocks") != ""))
        .unique(subset=keys + ["_blocks"])
        .sort(keys + ["_blocks"])
    )
    if block_rows.is_empty():
        aggregated = aggregated.with_columns(
            _null(pl.Utf8, "structured_block_names"),
            pl.lit(0, dtype=pl.Int64).alias("block_count"),
        )
    else:
        block_aggregation = block_rows.group_by(keys, maintain_order=True).agg(
            pl.col("_blocks").str.concat("|").alias("structured_block_names"),
            pl.len().cast(pl.Int64).alias("block_count"),
        )
        aggregated = aggregated.join(block_aggregation, on=keys, how="left").with_columns(
            pl.col("block_count").fill_null(0).cast(pl.Int64)
        )

    aggregated = aggregated.with_columns(
        pl.sum_horizontal([pl.col(column).cast(pl.Int64) for column in PROVENANCE_COLUMNS])
        .cast(pl.Int64)
        .alias("num_retrievers")
    )
    result = enforce_canonical_schema(aggregated).sort(
        ["query_id", "candidate_source", "candidate_id"]
    )
    validate_candidate_partition(result)
    return result


def add_query_partition(
    frame: pl.DataFrame, partition_count: int, hash_seed: int
) -> pl.DataFrame:
    """Add deterministic seeded Polars-hash query partitions."""

    if partition_count < 1:
        raise CandidateBuildError("partition_count must be positive")
    return frame.with_columns(
        (pl.col("query_id").hash(seed=hash_seed) % partition_count)
        .cast(pl.Int32)
        .alias("query_partition")
    )


def _legacy_columns(path: Path, retriever: RetrieverName) -> list[str]:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    columns = list(CANONICAL_CANDIDATE_KEY)
    if retriever == "exact":
        columns += ["match_name", "match_address"]
    elif retriever == "dense":
        columns += ["dense_score", "dense_rank"]
    elif retriever == "word_tfidf":
        columns += ["word_score" if "word_score" in names else "bm25_score"]
        if "found_by_word_tfidf" in names:
            columns += ["found_by_word_tfidf"]
        elif "found_by_bm25" in names:
            columns += ["found_by_bm25"]
    elif retriever == "char_tfidf":
        columns += ["char_score"]
        if "found_by_char" in names:
            columns += ["found_by_char"]
    elif retriever == "structured":
        columns += ["block_names"]
        if "found_by_block" in names:
            columns += ["found_by_block"]
    missing = sorted(set(columns) - names)
    if missing:
        raise CandidateBuildError(f"{retriever} artifact lacks columns: {missing}")
    return columns


def discover_smoke_query_ids(exact_path: Path, count: int = 100) -> tuple[str, ...]:
    """Choose sorted unique IDs from the first exact batches deterministically."""

    if count < 1:
        raise CandidateBuildError("smoke query count must be positive")
    selected: set[str] = set()
    parquet = pq.ParquetFile(exact_path)
    for batch in parquet.iter_batches(batch_size=max(10_000, count * 10), columns=["query_id"]):
        selected.update(value for value in batch.column(0).to_pylist() if value is not None)
        if len(selected) >= count:
            break
    if not selected:
        raise CandidateBuildError("exact artifact contains no query IDs for smoke mode")
    return tuple(sorted(selected)[:count])


def _fragment_record(path: Path, rows: int) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    return {
        "path": str(path.resolve()),
        "rows": rows,
        "size_bytes": path.stat().st_size,
        "schema_fingerprint": _json_hash(str(parquet.schema_arrow)),
    }


def _verify_file_record(record: dict[str, Any], context: str) -> None:
    path = Path(record["path"])
    if not path.is_file():
        raise CandidateBuildError(f"{context} output is missing: {path}")
    if path.stat().st_size != record["size_bytes"]:
        raise CandidateBuildError(f"{context} output size changed: {path}")
    metadata = pq.ParquetFile(path).metadata
    if metadata.num_rows != record["rows"]:
        raise CandidateBuildError(f"{context} output row count changed: {path}")
    schema_fingerprint = _json_hash(str(pq.ParquetFile(path).schema_arrow))
    if schema_fingerprint != record["schema_fingerprint"]:
        raise CandidateBuildError(f"{context} output schema changed: {path}")


def _atomic_parquet_write(frame: pl.DataFrame, final_path: Path) -> dict[str, Any]:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_suffix(".tmp.parquet")
    frame.write_parquet(temporary, compression="zstd")
    metadata = pq.ParquetFile(temporary).metadata
    if metadata.num_rows != frame.height:
        raise CandidateBuildError(f"temporary Parquet row count mismatch: {temporary}")
    os.replace(temporary, final_path)
    return _fragment_record(final_path, frame.height)


def resource_snapshot(path: Path) -> dict[str, Any]:
    """Return host-memory and disk telemetry without requiring psutil."""

    memory: dict[str, Any] = {"rss_bytes": None, "available_ram_bytes": None}
    try:
        import psutil

        memory = {
            "rss_bytes": psutil.Process().memory_info().rss,
            "available_ram_bytes": psutil.virtual_memory().available,
        }
    except ImportError:
        pass
    disk = shutil.disk_usage(path.resolve().anchor or path)
    return {**memory, "disk_free_bytes": disk.free, "disk_total_bytes": disk.total}


def _update_resource_manifest(
    manifest: dict[str, Any], telemetry: dict[str, Any]
) -> None:
    rss = telemetry.get("rss_bytes")
    if rss is not None:
        previous = manifest.get("peak_rss_bytes")
        manifest["peak_rss_bytes"] = rss if previous is None else max(previous, rss)
    free_disk = telemetry.get("disk_free_bytes")
    if free_disk is not None:
        previous = manifest.get("minimum_free_disk_bytes")
        manifest["minimum_free_disk_bytes"] = (
            free_disk if previous is None else min(previous, free_disk)
        )


def estimate_disk_requirements(config: CandidateBuildConfig) -> dict[str, Any]:
    """Return configurable projections used to guard a full build."""

    input_bytes = sum(item.path.stat().st_size for item in config.inputs)
    staging_bytes = int(input_bytes * config.staging_size_factor)
    output_bytes = int(input_bytes * config.output_size_factor)
    snapshot = resource_snapshot(config.output_dir)
    projected = staging_bytes + output_bytes
    return {
        "input_candidate_bytes": input_bytes,
        "estimated_staging_bytes": staging_bytes,
        "estimated_output_bytes": output_bytes,
        "estimated_peak_new_bytes": projected,
        "free_disk_bytes": snapshot["disk_free_bytes"],
        "disk_safety_fraction": config.disk_safety_fraction,
        "fits_safety_budget": projected
        <= int(snapshot["disk_free_bytes"] * config.disk_safety_fraction),
        "projection_note": (
            "Staging/output factors are configurable conservative projections; "
            "replace them with smoke-run measurements before a full build."
        ),
    }


def _stage_inputs(
    config: CandidateBuildConfig, manifest: dict[str, Any], manifest_path: Path
) -> None:
    smoke_ids = list(config.smoke_query_ids)
    for item in config.inputs:
        parquet = pq.ParquetFile(item.path)
        columns = _legacy_columns(item.path, item.retriever)
        retriever_batches = manifest["staged_batches"].setdefault(item.retriever, {})
        for batch_index, arrow_batch in enumerate(
            parquet.iter_batches(batch_size=config.batch_size, columns=columns)
        ):
            batch_key = f"{batch_index:06d}"
            if batch_key in retriever_batches:
                for record in retriever_batches[batch_key]["fragments"]:
                    _verify_file_record(record, f"staged {item.retriever} batch {batch_key}")
                continue

            legacy = pl.from_arrow(arrow_batch)
            input_rows = legacy.height
            if smoke_ids:
                legacy = legacy.filter(pl.col("query_id").cast(pl.Utf8).is_in(smoke_ids))
            canonical = adapt_legacy_batch(legacy, item.retriever)
            partitioned = add_query_partition(
                canonical, config.partition_count, config.hash_seed
            )

            records: list[dict[str, Any]] = []
            partition_values = sorted(
                partitioned["query_partition"].unique().to_list()
            ) if partitioned.height else []
            for partition in partition_values:
                fragment = partitioned.filter(
                    pl.col("query_partition") == partition
                ).drop("query_partition")
                destination = (
                    config.staging_dir
                    / config.split
                    / item.retriever
                    / f"partition_{partition:03d}"
                    / f"fragment_{batch_index:06d}.parquet"
                )
                records.append(_atomic_parquet_write(fragment, destination))

            retriever_batches[batch_key] = {
                "input_rows": input_rows,
                "selected_rows": canonical.height,
                "fragments": records,
            }
            manifest["input_rows_by_retriever"][item.retriever] += input_rows
            manifest["staged_rows_by_retriever"][item.retriever] += canonical.height
            telemetry = resource_snapshot(config.output_dir)
            _update_resource_manifest(manifest, telemetry)
            _atomic_json_write(manifest_path, manifest)
            print(
                f"staged retriever={item.retriever} batch={batch_index} "
                f"input_rows={input_rows} selected_rows={canonical.height} "
                f"rss={telemetry['rss_bytes']} free_disk={telemetry['disk_free_bytes']}"
            )


def _partition_fragments(
    manifest: dict[str, Any], partition: int
) -> list[Path]:
    """Return only fragments recorded by validated completed staging batches."""

    partition_name = f"partition_{partition:03d}"
    fragments: list[Path] = []
    for batches in manifest["staged_batches"].values():
        for batch in batches.values():
            for record in batch["fragments"]:
                path = Path(record["path"])
                if path.parent.name == partition_name:
                    fragments.append(path)
    return sorted(fragments)


def _build_partitions(
    config: CandidateBuildConfig, manifest: dict[str, Any], manifest_path: Path
) -> None:
    output_split_dir = config.output_dir / config.split
    output_split_dir.mkdir(parents=True, exist_ok=True)

    for partition in range(config.partition_count):
        key = f"{partition:03d}"
        if key in manifest["completed_partitions"]:
            _verify_file_record(
                manifest["completed_partitions"][key], f"canonical partition {key}"
            )
            print(f"skip validated canonical partition {key}")
            continue

        fragments = _partition_fragments(manifest, partition)
        if fragments:
            staged = pl.scan_parquet([str(path) for path in fragments]).collect(
                streaming=True
            )
        else:
            staged = pl.DataFrame(schema=CANONICAL_CANDIDATE_SCHEMA)
        canonical = aggregate_candidate_partition(staged)
        final_path = output_split_dir / f"partition_{partition:03d}.parquet"
        record = _atomic_parquet_write(canonical, final_path)
        # Semantic validation ran on the in-memory partition immediately before
        # writing.  Verify persisted row count/schema without loading a second
        # full copy of the partition into RAM.
        _verify_file_record(record, f"canonical partition {key}")
        manifest["completed_partitions"][key] = record
        manifest["canonical_row_count"] = sum(
            entry["rows"] for entry in manifest["completed_partitions"].values()
        )
        manifest["duplicate_rows_collapsed"] += staged.height - canonical.height
        telemetry = resource_snapshot(config.output_dir)
        _update_resource_manifest(manifest, telemetry)
        _atomic_json_write(manifest_path, manifest)
        print(
            f"completed partition={key} fragments={len(fragments)} "
            f"input_rows={staged.height} canonical_rows={canonical.height} "
            f"rss={telemetry['rss_bytes']} free_disk={telemetry['disk_free_bytes']}"
        )


def build_canonical_candidates(config: CandidateBuildConfig) -> Path:
    """Stage, aggregate, validate, and atomically write candidate_v1 shards."""

    invocation_start = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.staging_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / config.split / "manifest.json"
    stale_fragments = list(
        (config.staging_dir / config.split).glob("**/fragment_*.parquet")
    )
    if not manifest_path.exists() and stale_fragments:
        raise CandidateBuildError(
            "staging fragments exist without a matching resume manifest; "
            "refusing to combine potentially incompatible state"
        )
    manifest = load_or_create_manifest(config, manifest_path)

    estimate = estimate_disk_requirements(config)
    print(json.dumps({"disk_preflight": estimate}, indent=2))
    if (
        config.enforce_disk_safety
        and not config.smoke_query_ids
        and not estimate["fits_safety_budget"]
    ):
        raise CandidateBuildError(
            "projected staging + output exceeds the configured free-disk safety "
            "budget; adjust partition/staging configuration after smoke measurements"
        )

    _stage_inputs(config, manifest, manifest_path)
    _build_partitions(config, manifest, manifest_path)
    expected_collapsed = (
        sum(manifest["staged_rows_by_retriever"].values())
        - manifest["canonical_row_count"]
    )
    if manifest["duplicate_rows_collapsed"] != expected_collapsed:
        raise CandidateBuildError(
            "manifest duplicate-collapse accounting is inconsistent: "
            f"stored={manifest['duplicate_rows_collapsed']}, "
            f"expected={expected_collapsed}"
        )
    manifest["status"] = "complete"
    manifest["completed_at_unix"] = time.time()
    elapsed = time.perf_counter() - invocation_start
    manifest["last_invocation_runtime_seconds"] = elapsed
    manifest["builder_runtime_seconds"] = (
        float(manifest.get("builder_runtime_seconds", 0.0)) + elapsed
    )
    _update_resource_manifest(manifest, resource_snapshot(config.output_dir))
    _atomic_json_write(manifest_path, manifest)
    return manifest_path


def summarize_candidate_artifact(output_dir: Path, split: str) -> dict[str, Any]:
    """Compute descriptive, non-accuracy statistics from completed shards."""

    files = sorted((output_dir / split).glob("partition_*.parquet"))
    if not files:
        raise CandidateBuildError("no canonical partitions found to summarize")
    lazy = pl.scan_parquet([str(path) for path in files])
    summary = lazy.select(
        pl.len().alias("canonical_rows"),
        *[pl.col(column).sum().alias(f"rows_{column}") for column in PROVENANCE_COLUMNS],
    ).collect(streaming=True).to_dicts()[0]
    distribution = (
        lazy.group_by("num_retrievers")
        .agg(pl.len().alias("rows"))
        .sort("num_retrievers")
        .collect(streaming=True)
        .to_dicts()
    )
    counts = (
        lazy.group_by("query_id")
        .agg(pl.len().alias("candidate_count"))
        .select(
            pl.len().alias("queries"),
            pl.col("candidate_count").mean().alias("mean_candidates_per_query"),
            pl.col("candidate_count").quantile(0.50).alias("p50"),
            pl.col("candidate_count").quantile(0.90).alias("p90"),
            pl.col("candidate_count").quantile(0.95).alias("p95"),
            pl.col("candidate_count").quantile(0.99).alias("p99"),
        )
        .collect(streaming=True)
        .to_dicts()[0]
    )
    return {**summary, **counts, "num_retrievers_distribution": distribution}
