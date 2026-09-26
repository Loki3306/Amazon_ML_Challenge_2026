"""Small integration tests for the partitioned candidate_v1 builder."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import polars as pl


PACKAGE_SRC = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "business_entity_resolution"
    / "src"
)
sys.path.insert(0, str(PACKAGE_SRC))

from business_entity_resolution.candidate_builder import (  # noqa: E402
    CandidateBuildConfig,
    CandidateBuildError,
    RetrieverInput,
    adapt_legacy_batch,
    add_query_partition,
    aggregate_candidate_partition,
    build_canonical_candidates,
    validate_candidate_partition,
)
from business_entity_resolution.candidates import (  # noqa: E402
    CANONICAL_CANDIDATE_COLUMNS,
    adapt_legacy_candidates,
    union_candidate_frames,
)


def exact_frame(rows):
    return pl.DataFrame(
        rows,
        schema={
            "query_id": pl.Utf8,
            "candidate_id": pl.Utf8,
            "candidate_source": pl.Utf8,
            "match_name": pl.Boolean,
            "match_address": pl.Boolean,
        },
    )


def dense_frame(rows):
    return pl.DataFrame(
        rows,
        schema={
            "query_id": pl.Utf8,
            "candidate_id": pl.Utf8,
            "candidate_source": pl.Utf8,
            "dense_score": pl.Float64,
            "dense_rank": pl.Int64,
        },
    )


class CandidateBuilderTests(unittest.TestCase):
    def test_vectorized_aggregation_matches_a1_reference(self):
        exact = exact_frame(
            [
                ("Q1", "C1", "S2", True, False),
                ("Q2", "C2", "S3", False, True),
            ]
        )
        dense = dense_frame(
            [
                ("Q1", "C1", "S2", 0.8, 3),
                ("Q1", "C3", "S3", 0.7, 4),
            ]
        )
        word = pl.DataFrame(
            {
                "query_id": ["Q1"],
                "candidate_id": ["C1"],
                "candidate_source": ["S2"],
                "found_by_bm25": [True],
                "bm25_score": [2.5],
            }
        )
        char = pl.DataFrame(
            {
                "query_id": ["Q1"],
                "candidate_id": ["C1"],
                "candidate_source": ["S2"],
                "found_by_char": [True],
                "char_score": [0.6],
            }
        )
        structured = pl.DataFrame(
            {
                "query_id": ["Q1"],
                "candidate_id": ["C1"],
                "candidate_source": ["S2"],
                "found_by_block": [True],
                "block_names": ["postal|rare_name"],
            }
        )
        sources = [
            (exact, "exact"),
            (dense, "dense"),
            (word, "word_tfidf"),
            (char, "char_tfidf"),
            (structured, "structured"),
        ]

        reference = union_candidate_frames(
            [adapt_legacy_candidates(frame, name) for frame, name in sources]
        )
        production = aggregate_candidate_partition(
            pl.concat([adapt_legacy_batch(frame, name) for frame, name in sources])
        )
        self.assertTrue(reference.equals(production))

    def test_same_query_never_crosses_partitions(self):
        frame = adapt_legacy_batch(
            exact_frame(
                [
                    ("Q1", "C1", "S2", True, False),
                    ("Q1", "C2", "S3", False, True),
                    ("Q2", "C3", "S2", True, False),
                ]
            ),
            "exact",
        )
        partitioned = add_query_partition(frame, partition_count=7, hash_seed=42)
        per_query = partitioned.group_by("query_id").agg(
            pl.col("query_partition").n_unique().alias("partitions")
        )
        self.assertEqual(per_query["partitions"].to_list(), [1] * per_query.height)

    def test_duplicate_within_dense_keeps_best_score_and_rank(self):
        adapted = adapt_legacy_batch(
            dense_frame(
                [
                    ("Q1", "C1", "S2", 0.4, 8),
                    ("Q1", "C1", "S2", 0.9, 2),
                ]
            ),
            "dense",
        )
        row = aggregate_candidate_partition(adapted).row(0, named=True)
        self.assertEqual(row["dense_score"], 0.9)
        self.assertEqual(row["dense_rank"], 2)

    def test_structured_blocks_union_deterministically(self):
        legacy = pl.DataFrame(
            {
                "query_id": ["Q1", "Q1"],
                "candidate_id": ["C1", "C1"],
                "candidate_source": ["S2", "S2"],
                "found_by_block": [True, True],
                "block_names": ["rare_name|postal", "numeric|postal"],
            }
        )
        row = aggregate_candidate_partition(
            adapt_legacy_batch(legacy, "structured")
        ).row(0, named=True)
        self.assertEqual(row["structured_block_names"], "numeric|postal|rare_name")
        self.assertEqual(row["block_count"], 3)

    def test_missing_retriever_evidence_stays_false_and_null(self):
        row = adapt_legacy_batch(
            exact_frame([("Q1", "C1", "S2", True, False)]), "exact"
        ).row(0, named=True)
        self.assertFalse(row["found_by_dense"])
        self.assertIsNone(row["dense_score"])
        self.assertIsNone(row["dense_rank"])
        self.assertFalse(row["found_by_word_tfidf"])
        self.assertIsNone(row["word_score"])

    def test_different_input_order_produces_identical_output(self):
        legacy = dense_frame(
            [
                ("Q2", "C2", "S3", 0.5, 2),
                ("Q1", "C2", "S2", 0.6, 3),
                ("Q1", "C1", "S2", 0.7, 1),
            ]
        )
        forward = aggregate_candidate_partition(adapt_legacy_batch(legacy, "dense"))
        reverse = aggregate_candidate_partition(
            adapt_legacy_batch(legacy.reverse(), "dense")
        )
        self.assertTrue(forward.equals(reverse))
        self.assertEqual(tuple(forward.columns), CANONICAL_CANDIDATE_COLUMNS)

    def _single_input_config(self, root: Path) -> CandidateBuildConfig:
        legacy_path = root / "train_exact_candidates.parquet"
        exact_frame(
            [
                ("Q1", "C1", "S2", True, False),
                ("Q1", "C2", "S3", False, True),
            ]
        ).write_parquet(legacy_path)
        return CandidateBuildConfig(
            split="train",
            inputs=(RetrieverInput("exact", legacy_path),),
            output_dir=root / "output",
            staging_dir=root / "staging",
            partition_count=2,
            batch_size=1,
            required_retrievers=("exact",),
            enforce_disk_safety=False,
        )

    def test_valid_completed_partitions_are_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._single_input_config(Path(directory))
            manifest_path = build_canonical_candidates(config)
            before = json.loads(manifest_path.read_text(encoding="utf-8"))
            build_canonical_candidates(config)
            after = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                before["completed_partitions"], after["completed_partitions"]
            )

    def test_missing_completed_output_fails_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._single_input_config(Path(directory))
            manifest_path = build_canonical_candidates(config)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            completed = next(iter(manifest["completed_partitions"].values()))
            Path(completed["path"]).unlink()
            with self.assertRaises(CandidateBuildError):
                build_canonical_candidates(config)

    def test_fingerprint_mismatch_fails_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._single_input_config(Path(directory))
            build_canonical_candidates(config)
            changed = CandidateBuildConfig(
                **{**config.__dict__, "partition_count": 3}
            )
            with self.assertRaises(CandidateBuildError):
                build_canonical_candidates(changed)

    def test_temporary_partition_is_not_completion_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._single_input_config(Path(directory))
            temporary = config.output_dir / "train" / "partition_000.tmp.parquet"
            temporary.parent.mkdir(parents=True)
            temporary.write_bytes(b"incomplete")
            manifest_path = build_canonical_candidates(config)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["completed_partitions"]), 2)
            self.assertFalse(temporary.exists())

    def test_vectorized_validation_rejects_duplicate_final_keys(self):
        adapted = adapt_legacy_batch(
            exact_frame(
                [
                    ("Q1", "C1", "S2", True, False),
                    ("Q1", "C1", "S2", True, False),
                ]
            ),
            "exact",
        )
        with self.assertRaises(CandidateBuildError):
            validate_candidate_partition(adapted)

    def test_vectorized_validation_rejects_null_exact_evidence(self):
        adapted = adapt_legacy_batch(
            exact_frame([("Q1", "C1", "S2", True, False)]), "exact"
        ).with_columns(pl.lit(None, dtype=pl.Boolean).alias("exact_name_match"))
        with self.assertRaises(CandidateBuildError):
            validate_candidate_partition(adapted)


if __name__ == "__main__":
    unittest.main()
