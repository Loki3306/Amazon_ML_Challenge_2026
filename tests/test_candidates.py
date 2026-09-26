"""Tiny deterministic tests for the candidate_v1 contract."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import polars as pl


PACKAGE_SRC = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "business_entity_resolution"
    / "src"
)
sys.path.insert(0, str(PACKAGE_SRC))

from business_entity_resolution.candidates import (  # noqa: E402
    CANDIDATE_SCHEMA_VERSION,
    CANONICAL_CANDIDATE_COLUMNS,
    CANONICAL_CANDIDATE_SCHEMA,
    CandidateContractError,
    adapt_legacy_candidates,
    union_candidate_frames,
    validate_candidate_frame,
)


def canonical_row(**updates):
    row = {
        "query_id": "Q1",
        "candidate_id": "C1",
        "candidate_source": "S2",
        "found_by_exact": True,
        "found_by_dense": False,
        "found_by_word_tfidf": False,
        "found_by_char_tfidf": False,
        "found_by_structured": False,
        "exact_name_match": True,
        "exact_address_match": False,
        "dense_score": None,
        "dense_rank": None,
        "word_score": None,
        "word_rank": None,
        "char_score": None,
        "char_rank": None,
        "structured_block_names": None,
        "block_count": 0,
        "num_retrievers": 1,
    }
    row.update(updates)
    return row


def canonical_frame(*rows):
    return pl.DataFrame(list(rows), schema=CANONICAL_CANDIDATE_SCHEMA).select(
        CANONICAL_CANDIDATE_COLUMNS
    )


def legacy_frame(**columns):
    return pl.DataFrame(columns)


class CandidateContractTests(unittest.TestCase):
    def assert_invalid(self, frame, message=None, **kwargs):
        with self.assertRaises(CandidateContractError, msg=message):
            validate_candidate_frame(frame, **kwargs)

    def test_valid_canonical_frame_and_version(self):
        frame = canonical_frame(canonical_row())
        validate_candidate_frame(frame, schema_version=CANDIDATE_SCHEMA_VERSION)

    def test_missing_required_column_is_rejected(self):
        self.assert_invalid(canonical_frame(canonical_row()).drop("candidate_id"))

    def test_non_boolean_provenance_is_rejected(self):
        frame = canonical_frame(canonical_row()).with_columns(
            pl.col("found_by_exact").cast(pl.Int8)
        )
        self.assert_invalid(frame)

    def test_train_test_use_one_schema(self):
        train = canonical_frame(canonical_row(query_id="TRAIN-Q"))
        test = canonical_frame(canonical_row(query_id="TEST-Q"))
        self.assertEqual(train.schema, test.schema)
        self.assertEqual(tuple(train.columns), CANONICAL_CANDIDATE_COLUMNS)

    def test_exact_dense_char_union_deduplicates_and_ors_provenance(self):
        exact = adapt_legacy_candidates(
            legacy_frame(
                query_id=["Q1"], candidate_id=["C9"], candidate_source=["S2"],
                match_name=[True], match_address=[False],
            ),
            "exact",
        )
        dense = adapt_legacy_candidates(
            legacy_frame(
                query_id=["Q1"], candidate_id=["C9"], candidate_source=["S2"],
                dense_score=[0.8], dense_rank=[3], retrieval_method=["dense"],
            ),
            "dense",
        )
        char = adapt_legacy_candidates(
            legacy_frame(
                query_id=["Q1"], candidate_id=["C9"], candidate_source=["S2"],
                char_score=[0.7], found_by_char=[True],
            ),
            "char_tfidf",
        )

        result = union_candidate_frames([exact, dense, char])
        self.assertEqual(result.height, 1)
        row = result.row(0, named=True)
        self.assertTrue(row["found_by_exact"])
        self.assertTrue(row["found_by_dense"])
        self.assertTrue(row["found_by_char_tfidf"])
        self.assertEqual(row["num_retrievers"], 3)

    def test_duplicate_dense_evidence_keeps_max_score_and_min_rank(self):
        first = adapt_legacy_candidates(
            legacy_frame(
                query_id=["Q1"], candidate_id=["C1"], candidate_source=["S2"],
                dense_score=[0.4], dense_rank=[7],
            ),
            "dense",
        )
        second = adapt_legacy_candidates(
            legacy_frame(
                query_id=["Q1"], candidate_id=["C1"], candidate_source=["S2"],
                dense_score=[0.9], dense_rank=[2],
            ),
            "dense",
        )
        row = union_candidate_frames([first, second]).row(0, named=True)
        self.assertEqual(row["dense_score"], 0.9)
        self.assertEqual(row["dense_rank"], 2)
        self.assertEqual(row["num_retrievers"], 1)

    def test_source2_and_source3_are_distinct_pair_identities(self):
        frame = canonical_frame(
            canonical_row(candidate_source="S2"),
            canonical_row(candidate_source="S3"),
        )
        validate_candidate_frame(frame)
        self.assertEqual(union_candidate_frames([frame]).height, 2)

    def test_missing_dense_evidence_is_null_and_never_magic_one(self):
        exact = adapt_legacy_candidates(
            legacy_frame(
                query_id=["Q1"], candidate_id=["C1"], candidate_source=["S2"],
                match_name=[True], match_address=[False],
            ),
            "exact",
        )
        row = exact.row(0, named=True)
        self.assertFalse(row["found_by_dense"])
        self.assertIsNone(row["dense_score"])
        self.assertIsNone(row["dense_rank"])
        self.assertNotEqual((row["dense_score"], row["dense_rank"]), (1, 1))

    def test_contradictory_dense_evidence_is_rejected(self):
        self.assert_invalid(
            canonical_frame(canonical_row(dense_score=1.0, dense_rank=1))
        )

    def test_dense_flag_requires_evidence(self):
        self.assert_invalid(
            canonical_frame(
                canonical_row(
                    found_by_dense=True,
                    num_retrievers=2,
                )
            )
        )

    def test_zero_and_negative_rank_are_rejected(self):
        for rank in (0, -1):
            with self.subTest(rank=rank):
                self.assert_invalid(
                    canonical_frame(
                        canonical_row(
                            found_by_dense=True,
                            dense_score=0.5,
                            dense_rank=rank,
                            num_retrievers=2,
                        )
                    )
                )

    def test_non_finite_scores_are_rejected(self):
        for score in (math.nan, math.inf, -math.inf):
            with self.subTest(score=score):
                self.assert_invalid(
                    canonical_frame(
                        canonical_row(
                            found_by_dense=True,
                            dense_score=score,
                            dense_rank=1,
                            num_retrievers=2,
                        )
                    )
                )

    def test_num_retrievers_mismatch_is_rejected(self):
        self.assert_invalid(canonical_frame(canonical_row(num_retrievers=2)))

    def test_candidate_without_retrieval_provenance_is_rejected(self):
        self.assert_invalid(
            canonical_frame(
                canonical_row(
                    found_by_exact=False,
                    exact_name_match=False,
                    num_retrievers=0,
                )
            )
        )

    def test_null_keys_and_source_are_rejected(self):
        for column in ("query_id", "candidate_id", "candidate_source"):
            with self.subTest(column=column):
                self.assert_invalid(canonical_frame(canonical_row(**{column: None})))

    def test_invalid_candidate_source_is_rejected(self):
        self.assert_invalid(
            canonical_frame(canonical_row(candidate_source="S1"))
        )

    def test_duplicate_finalized_key_is_rejected(self):
        row = canonical_row()
        self.assert_invalid(canonical_frame(row, dict(row)))

    def test_word_legacy_mapping_uses_truthful_canonical_names(self):
        result = adapt_legacy_candidates(
            legacy_frame(
                query_id=["Q1"], candidate_id=["C1"], candidate_source=["S2"],
                found_by_bm25=[True], bm25_score=[2.75],
            ),
            "word_tfidf",
        )
        row = result.row(0, named=True)
        self.assertTrue(row["found_by_word_tfidf"])
        self.assertEqual(row["word_score"], 2.75)
        self.assertIsNone(row["word_rank"])
        self.assertFalse(any("bm25" in name for name in result.columns))

    def test_char_rank_is_truthfully_absent(self):
        result = adapt_legacy_candidates(
            legacy_frame(
                query_id=["Q1"], candidate_id=["C1"], candidate_source=["S2"],
                found_by_char=[True], char_score=[0.5],
            ),
            "char_tfidf",
        )
        self.assertIsNone(result["char_rank"][0])
        validate_candidate_frame(result)

    def test_structured_blocks_are_sorted_distinct_and_counted(self):
        result = adapt_legacy_candidates(
            legacy_frame(
                query_id=["Q1"], candidate_id=["C1"], candidate_source=["S2"],
                found_by_block=[True],
                block_names=["postal|rare_name|postal"],
            ),
            "structured",
        )
        row = result.row(0, named=True)
        self.assertEqual(row["structured_block_names"], "postal|rare_name")
        self.assertEqual(row["block_count"], 2)

    def test_deterministic_output_for_any_input_order(self):
        rows = [
            canonical_row(query_id="Q2", candidate_id="C2", candidate_source="S3"),
            canonical_row(query_id="Q1", candidate_id="C2", candidate_source="S2"),
            canonical_row(query_id="Q1", candidate_id="C1", candidate_source="S2"),
        ]
        forward = union_candidate_frames([canonical_frame(*rows)])
        reverse = union_candidate_frames([canonical_frame(*reversed(rows))])
        self.assertTrue(forward.equals(reverse))
        self.assertEqual(
            forward.select(["query_id", "candidate_source", "candidate_id"]).rows(),
            [("Q1", "S2", "C1"), ("Q1", "S2", "C2"), ("Q2", "S3", "C2")],
        )
        self.assertEqual(tuple(forward.columns), CANONICAL_CANDIDATE_COLUMNS)

    def test_schema_version_mismatch_is_rejected(self):
        self.assert_invalid(
            canonical_frame(canonical_row()), schema_version="candidate_v0"
        )

    def test_optional_referential_validation(self):
        frame = canonical_frame(canonical_row())
        validate_candidate_frame(
            frame,
            query_ids={"Q1"},
            candidate_ids_by_source={"S2": {"C1"}},
        )
        with self.assertRaises(CandidateContractError):
            validate_candidate_frame(frame, query_ids={"OTHER"})


if __name__ == "__main__":
    unittest.main()
