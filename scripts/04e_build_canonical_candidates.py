"""Thin CLI for the partitioned candidate_v1 production builder."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src"
        )
    ),
)

from business_entity_resolution.candidate_builder import (  # noqa: E402
    CandidateBuildConfig,
    RetrieverInput,
    build_canonical_candidates,
    discover_smoke_query_ids,
    summarize_candidate_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STAB-A2: build validated partitioned candidate_v1 artifacts"
    )
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--exact-path", required=True)
    parser.add_argument("--dense-path", required=True)
    parser.add_argument("--word-tfidf-path", required=True)
    parser.add_argument("--char-tfidf-path", required=True)
    parser.add_argument("--structured-path", required=True)
    parser.add_argument(
        "--output-dir", default="artifacts/candidates/candidate_v1"
    )
    parser.add_argument("--staging-dir", default=None)
    parser.add_argument("--partitions", type=int, default=64)
    parser.add_argument("--hash-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--dataset-fingerprint", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-query-count", type=int, default=100)
    parser.add_argument("--disk-safety-fraction", type=float, default=0.80)
    parser.add_argument("--staging-size-factor", type=float, default=1.35)
    parser.add_argument("--output-size-factor", type=float, default=1.00)
    parser.add_argument(
        "--skip-disk-safety",
        action="store_true",
        help="Disable projection guard only after manually verifying free disk",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = (
        RetrieverInput("exact", Path(args.exact_path)),
        RetrieverInput("dense", Path(args.dense_path)),
        RetrieverInput("word_tfidf", Path(args.word_tfidf_path)),
        RetrieverInput("char_tfidf", Path(args.char_tfidf_path)),
        RetrieverInput("structured", Path(args.structured_path)),
    )

    output_dir = Path(args.output_dir)
    if args.smoke:
        output_dir = output_dir.parent / f"{output_dir.name}_smoke"
    staging_dir = (
        Path(args.staging_dir)
        if args.staging_dir
        else output_dir.parent / f"{output_dir.name}_staging"
    )
    smoke_ids = (
        discover_smoke_query_ids(inputs[0].path, args.smoke_query_count)
        if args.smoke
        else ()
    )

    config = CandidateBuildConfig(
        split=args.split,
        inputs=inputs,
        output_dir=output_dir,
        staging_dir=staging_dir,
        partition_count=args.partitions,
        hash_seed=args.hash_seed,
        batch_size=args.batch_size,
        dataset_fingerprint=args.dataset_fingerprint,
        smoke_query_ids=smoke_ids,
        enforce_disk_safety=not args.skip_disk_safety,
        disk_safety_fraction=args.disk_safety_fraction,
        staging_size_factor=args.staging_size_factor,
        output_size_factor=args.output_size_factor,
    )
    manifest_path = build_canonical_candidates(config)
    print(f"manifest={manifest_path}")
    print(json.dumps(summarize_candidate_artifact(output_dir, args.split), indent=2))


if __name__ == "__main__":
    main()
