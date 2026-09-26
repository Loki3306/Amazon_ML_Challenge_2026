"""Versioned candidate-pair contract and small-frame compatibility helpers.

This module deliberately contains no large-scale I/O.  The adapters and union
function are reference implementations for tests and small deterministic
frames; a later stabilization patch will implement partitioned production
fusion against the same contract.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
import math
from typing import Any, Final, Literal

import polars as pl


CANDIDATE_SCHEMA_VERSION: Final = "candidate_v1"

CANONICAL_CANDIDATE_KEY: Final[tuple[str, str, str]] = (
    "query_id",
    "candidate_id",
    "candidate_source",
)

PROVENANCE_COLUMNS: Final[tuple[str, ...]] = (
    "found_by_exact",
    "found_by_dense",
    "found_by_word_tfidf",
    "found_by_char_tfidf",
    "found_by_structured",
)

# Source values are those emitted by the executable preparation pipeline plus
# short forms commonly used by small fixtures.  Callers may supply a stricter
# or extended collection to validate_candidate_frame.
DEFAULT_CANDIDATE_SOURCES: Final[frozenset[str]] = frozenset(
    {
        "S2",
        "S3",
        "source2",
        "source3",
        "train_source2",
        "train_source3",
        "test_source2",
        "test_source3",
    }
)

# Optional evidence with established executable meaning is retained as part of
# v1.  Labels are intentionally absent: they belong to model training, not the
# candidate contract.
CANONICAL_CANDIDATE_SCHEMA: Final[dict[str, pl.DataType]] = {
    "query_id": pl.Utf8,
    "candidate_id": pl.Utf8,
    "candidate_source": pl.Utf8,
    "found_by_exact": pl.Boolean,
    "found_by_dense": pl.Boolean,
    "found_by_word_tfidf": pl.Boolean,
    "found_by_char_tfidf": pl.Boolean,
    "found_by_structured": pl.Boolean,
    "exact_name_match": pl.Boolean,
    "exact_address_match": pl.Boolean,
    "dense_score": pl.Float64,
    "dense_rank": pl.Int64,
    "word_score": pl.Float64,
    "word_rank": pl.Int64,
    "char_score": pl.Float64,
    "char_rank": pl.Int64,
    # Sorted, distinct legacy block names joined with "|", or null when the
    # pair has no structured evidence.
    "structured_block_names": pl.Utf8,
    "block_count": pl.Int64,
    "num_retrievers": pl.Int64,
}

CANONICAL_CANDIDATE_COLUMNS: Final[tuple[str, ...]] = tuple(
    CANONICAL_CANDIDATE_SCHEMA
)

LegacyRetriever = Literal[
    "exact", "dense", "word_tfidf", "char_tfidf", "structured"
]


class CandidateContractError(ValueError):
    """Raised when candidate data violates the versioned contract."""


def empty_candidate_frame() -> pl.DataFrame:
    """Return an empty frame with the exact candidate_v1 schema."""

    return pl.DataFrame(schema=CANONICAL_CANDIDATE_SCHEMA)


def _base_row(query_id: Any, candidate_id: Any, candidate_source: Any) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "candidate_id": candidate_id,
        "candidate_source": candidate_source,
        "found_by_exact": False,
        "found_by_dense": False,
        "found_by_word_tfidf": False,
        "found_by_char_tfidf": False,
        "found_by_structured": False,
        "exact_name_match": False,
        "exact_address_match": False,
        "dense_score": None,
        "dense_rank": None,
        "word_score": None,
        "word_rank": None,
        "char_score": None,
        "char_rank": None,
        "structured_block_names": None,
        "block_count": 0,
        "num_retrievers": 0,
    }


def _require_columns(frame: pl.DataFrame, required: Collection[str], context: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise CandidateContractError(f"{context} is missing columns: {missing}")


def _optional_bool(row: Mapping[str, Any], column: str, default: bool) -> bool:
    value = row.get(column, default)
    if type(value) is not bool:
        raise CandidateContractError(f"{column} must contain booleans, got {value!r}")
    return value


def _finite_score(value: Any, column: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateContractError(f"{column} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise CandidateContractError(f"{column} must be finite, got {value!r}")
    return result


def _positive_rank(value: Any, column: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CandidateContractError(
            f"{column} must be an integer >= 1, got {value!r}"
        )
    return int(value)


def _parse_block_names(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise CandidateContractError(
            "block_names must be a pipe-delimited string for structured evidence"
        )
    names = tuple(sorted({part.strip() for part in value.split("|") if part.strip()}))
    if not names:
        raise CandidateContractError(
            "structured evidence must contain at least one non-empty block name"
        )
    return names


def _frame_from_rows(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return empty_candidate_frame()
    return (
        pl.DataFrame(rows, schema=CANONICAL_CANDIDATE_SCHEMA)
        .select(CANONICAL_CANDIDATE_COLUMNS)
        .sort(["query_id", "candidate_source", "candidate_id"])
    )


def adapt_legacy_candidates(
    frame: pl.DataFrame,
    retriever: LegacyRetriever,
) -> pl.DataFrame:
    """Map one legacy retriever's small output frame into candidate_v1 rows.

    Word and character legacy outputs expose scores but not ranks, so their
    canonical ranks remain null.  Adaptation never fabricates missing evidence.
    Duplicate keys are permitted here because union_candidate_frames performs
    the explicit deterministic reduction.
    """

    _require_columns(frame, CANONICAL_CANDIDATE_KEY, f"legacy {retriever} frame")
    rows: list[dict[str, Any]] = []

    for legacy in frame.iter_rows(named=True):
        row = _base_row(
            legacy["query_id"], legacy["candidate_id"], legacy["candidate_source"]
        )

        if retriever == "exact":
            _require_columns(frame, ("match_name", "match_address"), "legacy exact frame")
            name_match = _optional_bool(legacy, "match_name", False)
            address_match = _optional_bool(legacy, "match_address", False)
            if not name_match and not address_match:
                raise CandidateContractError(
                    "legacy exact row must match name, address, or both"
                )
            row.update(
                found_by_exact=True,
                exact_name_match=name_match,
                exact_address_match=address_match,
            )

        elif retriever == "dense":
            _require_columns(frame, ("dense_score", "dense_rank"), "legacy dense frame")
            row.update(
                found_by_dense=True,
                dense_score=_finite_score(legacy["dense_score"], "dense_score"),
                dense_rank=_positive_rank(legacy["dense_rank"], "dense_rank"),
            )

        elif retriever == "word_tfidf":
            score_column = "word_score" if "word_score" in frame.columns else "bm25_score"
            flag_column = (
                "found_by_word_tfidf"
                if "found_by_word_tfidf" in frame.columns
                else "found_by_bm25"
            )
            _require_columns(frame, (score_column,), "legacy word TF-IDF frame")
            found = _optional_bool(legacy, flag_column, True)
            row["found_by_word_tfidf"] = found
            if found:
                row["word_score"] = _finite_score(legacy[score_column], score_column)
            elif legacy[score_column] is not None:
                raise CandidateContractError(
                    f"{score_column} exists while {flag_column} is false"
                )

        elif retriever == "char_tfidf":
            _require_columns(frame, ("char_score",), "legacy character TF-IDF frame")
            found = _optional_bool(legacy, "found_by_char", True)
            row["found_by_char_tfidf"] = found
            if found:
                row["char_score"] = _finite_score(legacy["char_score"], "char_score")
            elif legacy["char_score"] is not None:
                raise CandidateContractError(
                    "char_score exists while found_by_char is false"
                )

        elif retriever == "structured":
            _require_columns(frame, ("block_names",), "legacy structured frame")
            found = _optional_bool(legacy, "found_by_block", True)
            row["found_by_structured"] = found
            if found:
                names = _parse_block_names(legacy["block_names"])
                row["structured_block_names"] = "|".join(names)
                row["block_count"] = len(names)
            elif legacy["block_names"] is not None:
                raise CandidateContractError(
                    "block_names exists while found_by_block is false"
                )

        else:  # pragma: no cover - guarded by the public Literal annotation.
            raise CandidateContractError(f"unknown legacy retriever: {retriever!r}")

        row["num_retrievers"] = sum(bool(row[name]) for name in PROVENANCE_COLUMNS)
        rows.append(row)

    adapted = _frame_from_rows(rows)
    validate_candidate_frame(adapted, require_unique=False)
    return adapted


def _best_score(left: Any, right: Any) -> Any:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None


def _best_rank(left: Any, right: Any) -> Any:
    values = [value for value in (left, right) if value is not None]
    return min(values) if values else None


def _merge_block_names(left: Any, right: Any) -> tuple[str | None, int]:
    names: set[str] = set()
    for value in (left, right):
        if value is not None:
            names.update(_parse_block_names(value))
    if not names:
        return None, 0
    ordered = sorted(names)
    return "|".join(ordered), len(ordered)


def union_candidate_frames(frames: Iterable[pl.DataFrame]) -> pl.DataFrame:
    """Deterministically union and deduplicate small canonical frames.

    Scores use max and positive ranks use min, matching the established legacy
    retrieval semantics.  This reference implementation intentionally uses
    Python rows and is not suitable for production-scale candidate artifacts.
    """

    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for frame in frames:
        validate_candidate_frame(frame, require_unique=False)
        for incoming in frame.iter_rows(named=True):
            key = tuple(incoming[column] for column in CANONICAL_CANDIDATE_KEY)
            if key not in merged:
                merged[key] = dict(incoming)
                continue

            current = merged[key]
            for column in PROVENANCE_COLUMNS:
                current[column] = bool(current[column] or incoming[column])
            current["exact_name_match"] = bool(
                current["exact_name_match"] or incoming["exact_name_match"]
            )
            current["exact_address_match"] = bool(
                current["exact_address_match"] or incoming["exact_address_match"]
            )
            for column in ("dense_score", "word_score", "char_score"):
                current[column] = _best_score(current[column], incoming[column])
            for column in ("dense_rank", "word_rank", "char_rank"):
                current[column] = _best_rank(current[column], incoming[column])
            block_names, block_count = _merge_block_names(
                current["structured_block_names"],
                incoming["structured_block_names"],
            )
            current["structured_block_names"] = block_names
            current["block_count"] = block_count
            current["num_retrievers"] = sum(
                bool(current[name]) for name in PROVENANCE_COLUMNS
            )

    result = _frame_from_rows(list(merged.values()))
    validate_candidate_frame(result)
    return result


def validate_candidate_frame(
    frame: pl.DataFrame,
    *,
    schema_version: str | None = None,
    require_unique: bool = True,
    allowed_candidate_sources: Collection[str] | None = None,
    query_ids: Collection[str] | None = None,
    candidate_ids_by_source: Mapping[str, Collection[str]] | None = None,
) -> None:
    """Fail loudly when a frame violates candidate_v1 invariants.

    Optional ID collections enable referential checks for small frames without
    making full-corpus ID materialization part of normal validation.
    """

    if schema_version is not None and schema_version != CANDIDATE_SCHEMA_VERSION:
        raise CandidateContractError(
            f"schema version {schema_version!r} does not match "
            f"{CANDIDATE_SCHEMA_VERSION!r}"
        )

    missing = sorted(set(CANONICAL_CANDIDATE_COLUMNS) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(CANONICAL_CANDIDATE_COLUMNS))
    if missing or extra:
        raise CandidateContractError(
            f"candidate_v1 columns differ; missing={missing}, extra={extra}"
        )
    if tuple(frame.columns) != CANONICAL_CANDIDATE_COLUMNS:
        raise CandidateContractError("candidate_v1 columns are not in canonical order")

    actual_schema = frame.schema
    dtype_errors = {
        column: (actual_schema[column], expected)
        for column, expected in CANONICAL_CANDIDATE_SCHEMA.items()
        if actual_schema[column] != expected
    }
    if dtype_errors:
        raise CandidateContractError(f"candidate_v1 dtype mismatch: {dtype_errors}")

    allowed_sources = set(allowed_candidate_sources or DEFAULT_CANDIDATE_SOURCES)
    known_queries = set(query_ids) if query_ids is not None else None
    known_candidates = (
        {source: set(ids) for source, ids in candidate_ids_by_source.items()}
        if candidate_ids_by_source is not None
        else None
    )
    seen: set[tuple[str, str, str]] = set()

    for index, row in enumerate(frame.iter_rows(named=True)):
        for column in CANONICAL_CANDIDATE_KEY:
            value = row[column]
            if value is None or not isinstance(value, str) or not value:
                raise CandidateContractError(
                    f"row {index}: {column} must be a non-empty string"
                )

        key = tuple(row[column] for column in CANONICAL_CANDIDATE_KEY)
        if require_unique and key in seen:
            raise CandidateContractError(f"duplicate canonical candidate key: {key}")
        seen.add(key)

        source = row["candidate_source"]
        if source not in allowed_sources:
            raise CandidateContractError(
                f"row {index}: invalid candidate_source {source!r}"
            )
        if known_queries is not None and row["query_id"] not in known_queries:
            raise CandidateContractError(
                f"row {index}: unknown Source-1 query_id {row['query_id']!r}"
            )
        if known_candidates is not None:
            if source not in known_candidates:
                raise CandidateContractError(
                    f"row {index}: no candidate ID collection supplied for {source!r}"
                )
            if row["candidate_id"] not in known_candidates[source]:
                raise CandidateContractError(
                    f"row {index}: candidate_id {row['candidate_id']!r} "
                    f"does not belong to {source!r}"
                )

        for column in PROVENANCE_COLUMNS + (
            "exact_name_match",
            "exact_address_match",
        ):
            if type(row[column]) is not bool:
                raise CandidateContractError(
                    f"row {index}: {column} must be a non-null boolean"
                )

        for score_column in ("dense_score", "word_score", "char_score"):
            value = row[score_column]
            if value is not None:
                _finite_score(value, score_column)
        for rank_column in ("dense_rank", "word_rank", "char_rank"):
            value = row[rank_column]
            if value is not None:
                _positive_rank(value, rank_column)

        if row["found_by_dense"]:
            if row["dense_score"] is None or row["dense_rank"] is None:
                raise CandidateContractError(
                    f"row {index}: dense evidence requires finite score and positive rank"
                )
        elif row["dense_score"] is not None or row["dense_rank"] is not None:
            raise CandidateContractError(
                f"row {index}: dense evidence exists while found_by_dense is false"
            )

        for prefix, flag in (
            ("word", "found_by_word_tfidf"),
            ("char", "found_by_char_tfidf"),
        ):
            score = row[f"{prefix}_score"]
            rank = row[f"{prefix}_rank"]
            if row[flag]:
                if score is None:
                    raise CandidateContractError(
                        f"row {index}: {prefix} evidence requires its supplied score"
                    )
                # Legacy word/character retrievers do not expose rank, so a
                # true flag with a null rank is explicitly valid.
            elif score is not None or rank is not None:
                raise CandidateContractError(
                    f"row {index}: {prefix} evidence exists while {flag} is false"
                )

        if row["found_by_exact"]:
            if not row["exact_name_match"] and not row["exact_address_match"]:
                raise CandidateContractError(
                    f"row {index}: exact evidence requires name/address match evidence"
                )
        elif row["exact_name_match"] or row["exact_address_match"]:
            raise CandidateContractError(
                f"row {index}: exact match evidence exists while found_by_exact is false"
            )

        block_count = row["block_count"]
        if isinstance(block_count, bool) or not isinstance(block_count, int) or block_count < 0:
            raise CandidateContractError(
                f"row {index}: block_count must be a non-negative integer"
            )
        if row["found_by_structured"]:
            names = _parse_block_names(row["structured_block_names"])
            if block_count != len(names):
                raise CandidateContractError(
                    f"row {index}: block_count does not match distinct block names"
                )
        elif row["structured_block_names"] is not None or block_count != 0:
            raise CandidateContractError(
                f"row {index}: structured evidence exists while flag is false"
            )

        expected_count = sum(bool(row[name]) for name in PROVENANCE_COLUMNS)
        if expected_count < 1:
            raise CandidateContractError(
                f"row {index}: canonical candidate has no retrieval provenance"
            )
        if row["num_retrievers"] != expected_count:
            raise CandidateContractError(
                f"row {index}: num_retrievers={row['num_retrievers']} "
                f"but expected {expected_count}"
            )
