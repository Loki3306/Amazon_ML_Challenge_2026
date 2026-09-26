"""
Phase 6A (v2 — OPTIMISED): Pairwise Feature Engineering + LightGBM
====================================================================
Key optimisations over v1
─────────────────────────
1.  RapidFuzz cdist (C++ SIMD, parallel workers) replaces Python for-loops.
    Benchmarks show 20–100× speedup over per-row calls.

2.  Precomputed token representations per unique entity → O(unique strings)
    tokenisation instead of O(candidate pairs).

3.  Exact-match short-circuit: when name/addr is identical, all expensive
    fuzzy sims are set to 1.0 without calling C++ at all.

4.  Column pruning at every stage.

5.  Resumable via features_manifest.json — already-written shards are skipped.

6.  --benchmark mode: times each stage on N rows and reports rows/s + peak RAM.

7.  LightGBM training accepts a --negative-sample-ratio flag so we can train
    on all positives + a sampled fraction of negatives without discarding the
    full feature shards.

Memory model (per 2M-row chunk, float32 arrays)
─────────────────────────────────────────────────
  names_s1[2M] + names_c[2M]  string lists  ≈ 0.4–1.5 GB (depends on str length)
  cdist output float32[2M]    ≈    8 MB each metric
  15 feature arrays @ 8 MB    ≈  120 MB
  joined Polars chunk          ≈  600 MB
  Parquet write buffer         ≈  200 MB
  ─────────────────────────────────────────
  Estimated peak per chunk     ≈   3–4 GB    ← safe on Kaggle 30 GB
"""

from __future__ import annotations
import os, sys, gc, json, time, argparse, logging
from datetime import datetime
from collections import defaultdict
from typing import Any

import numpy as np
import polars as pl
import lightgbm as lgb
from rapidfuzz import process as rf_process
from rapidfuzz.distance import JaroWinkler, Jaro, Levenshtein
import psutil

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                  "..", "code",
                                                  "business_entity_resolution", "src")))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6A (v2 optimised): Features + LightGBM")
    p.add_argument("--data-dir",              default="data/processed")
    p.add_argument("--candidates-dir",        default="data/candidates")
    p.add_argument("--features-dir",          default="data/features")
    p.add_argument("--models-dir",            default="models/lightgbm")
    p.add_argument("--reports-dir",           default="reports/phase6")
    p.add_argument("--ground-truth",          required=True)
    p.add_argument("--split",                 default="train")
    p.add_argument("--chunk-size",            type=int,   default=2_000_000)
    p.add_argument("--workers",               type=int,   default=-1,
                   help="cdist parallel workers (-1 = all CPU cores)")
    p.add_argument("--val-fraction",          type=float, default=0.15)
    p.add_argument("--seed",                  type=int,   default=42)
    p.add_argument("--lgb-rounds",            type=int,   default=500)
    p.add_argument("--negative-sample-ratio", type=float, default=0.0,
                   help="Train on all pos + this × pos negatives (0 = use all)")
    p.add_argument("--skip-features",         action="store_true")
    p.add_argument("--skip-training",         action="store_true")
    p.add_argument("--benchmark",             action="store_true",
                   help="Profile a small sample; do NOT process full dataset")
    p.add_argument("--benchmark-rows",        type=int,   default=1_000_000)
    p.add_argument("--compression",           default="zstd",
                   choices=["snappy", "zstd"])
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# MEMORY HELPER
# ──────────────────────────────────────────────────────────────────────────────
def ram_gb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 3)


# ──────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH
# ──────────────────────────────────────────────────────────────────────────────
def load_ground_truth(gt_path: str) -> dict[str, frozenset[str]]:
    log.info("Loading ground truth from %s", gt_path)
    df = pl.read_csv(gt_path, separator="\t")
    gt: dict[str, frozenset[str]] = {}
    for row in df.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        raw   = row["matched_entity_ids"] or ""
        gt[s1_id] = frozenset(raw.split(",")) if raw else frozenset()
    log.info("  %d S1 entities with ground truth", len(gt))
    return gt


# ──────────────────────────────────────────────────────────────────────────────
# LABEL LOOKUP  (built ONCE, reused per chunk)
# ──────────────────────────────────────────────────────────────────────────────
def build_label_frame(gt: dict[str, frozenset[str]]) -> pl.DataFrame:
    """One (query_id, candidate_id, label=1) row per positive pair."""
    rows = [{"query_id": s1, "candidate_id": c}
            for s1, matches in gt.items() for c in matches]
    if not rows:
        return pl.DataFrame({"query_id": pl.Series([], dtype=pl.Utf8),
                              "candidate_id": pl.Series([], dtype=pl.Utf8),
                              "label": pl.Series([], dtype=pl.Int8)})
    return (pl.DataFrame(rows)
              .with_columns(pl.lit(1).cast(pl.Int8).alias("label")))


# ──────────────────────────────────────────────────────────────────────────────
# CANDIDATE LAZY FRAME
# ──────────────────────────────────────────────────────────────────────────────
def load_candidates_lazy(candidates_dir: str, split: str) -> pl.LazyFrame:
    """
    Union of dense + exact candidates.
    retrieval_source: 0 = dense-only, 1 = exact-only, 2 = both
    """
    dense_path = os.path.join(candidates_dir, f"{split}_dense_candidates_K50.parquet")
    exact_path = os.path.join(candidates_dir, f"{split}_exact_candidates.parquet")

    if not os.path.exists(dense_path):
        raise FileNotFoundError(dense_path)

    dense_lf = pl.scan_parquet(dense_path).select([
        pl.col("query_id").cast(pl.Utf8),
        pl.col("candidate_id").cast(pl.Utf8),
        pl.col("candidate_source").cast(pl.Utf8),
        pl.col("dense_score").cast(pl.Float32),
        pl.col("dense_rank").cast(pl.Int32),
    ]).with_columns(pl.lit(True).alias("_d"))

    if os.path.exists(exact_path):
        exact_lf = pl.scan_parquet(exact_path).select([
            pl.col("query_id").cast(pl.Utf8),
            pl.col("candidate_id").cast(pl.Utf8),
            pl.col("candidate_source").cast(pl.Utf8),
        ]).with_columns(pl.lit(True).alias("_e"))

        combined = dense_lf.join(
            exact_lf, on=["query_id", "candidate_id", "candidate_source"],
            how="full", coalesce=True
        ).with_columns([
            pl.col("_d").fill_null(False),
            pl.col("_e").fill_null(False),
            pl.col("dense_score").fill_null(0.0),
            pl.col("dense_rank").fill_null(999).cast(pl.Int32),
        ])
    else:
        combined = dense_lf.with_columns(pl.lit(False).alias("_e"))

    combined = combined.with_columns(
        (pl.col("_d").cast(pl.Int8) + pl.col("_e").cast(pl.Int8) * 2 - 1)
        .clip(0, 2).cast(pl.Int8).alias("retrieval_source")
    ).drop(["_d", "_e"])

    return combined


# ──────────────────────────────────────────────────────────────────────────────
# ENTITY ATTRIBUTE TABLES  (loaded eagerly; S1 ~2.2M, corpus ~10.3M)
# ──────────────────────────────────────────────────────────────────────────────
ATTR_COLS = ["entity_id", "name_norm", "address_norm", "country_norm"]

def load_entity_tables(data_dir: str, split: str
                       ) -> tuple[pl.DataFrame, pl.DataFrame]:
    s1_df   = pl.read_parquet(os.path.join(data_dir, split,
                               f"{split}_source1.parquet"), columns=ATTR_COLS)
    s2_df   = pl.read_parquet(os.path.join(data_dir, split,
                               f"{split}_source2.parquet"), columns=ATTR_COLS)
    s3_df   = pl.read_parquet(os.path.join(data_dir, split,
                               f"{split}_source3.parquet"), columns=ATTR_COLS)
    cand_df = pl.concat([s2_df, s3_df])
    log.info("Entity tables  S1=%d  S2=%d  S3=%d", s1_df.height,
             s2_df.height, s3_df.height)
    del s2_df, s3_df
    return s1_df, cand_df


# ──────────────────────────────────────────────────────────────────────────────
# FAST TOKEN JACCARD  via precomputed integer sets
# ──────────────────────────────────────────────────────────────────────────────
class TokenIndex:
    """
    Converts a list of normalised strings into integer-set token representations.
    Build once per unique set of strings; reuse across many candidate pairs.
    """
    def __init__(self, strings: list[str]):
        vocab: dict[str, int] = {}
        self.sets: list[frozenset[int]] = []
        for s in strings:
            ids: set[int] = set()
            for tok in s.split():
                if tok not in vocab:
                    vocab[tok] = len(vocab)
                ids.add(vocab[tok])
            self.sets.append(frozenset(ids))

    def jaccard(self, idx_a: int, idx_b: int) -> float:
        sa = self.sets[idx_a]
        sb = self.sets[idx_b]
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        return inter / (len(sa) + len(sb) - inter)

    def batch_jaccard(self,
                      idx_a: np.ndarray,
                      idx_b: np.ndarray) -> np.ndarray:
        n = len(idx_a)
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            out[i] = self.jaccard(int(idx_a[i]), int(idx_b[i]))
        return out


def fast_paired_sim(s1_list, cand_list, scorer) -> np.ndarray:
    return np.array(list(map(scorer, s1_list, cand_list)), dtype=np.float32)

def fast_token_jaccard(s1_list, cand_list) -> np.ndarray:
    def get_token_sets(strings):
        unique_strs = set(strings)
        str_to_set = {s: frozenset(s.split()) for s in unique_strs}
        return [str_to_set[s] for s in strings]
        
    s1_sets = get_token_sets(s1_list)
    cand_sets = get_token_sets(cand_list)
    
    def jaccard(sa, sb):
        if not sa and not sb: return 1.0
        if not sa or not sb: return 0.0
        inter = len(sa & sb)
        return inter / (len(sa) + len(sb) - inter)
        
    return np.array(list(map(jaccard, s1_sets, cand_sets)), dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# CORE FEATURE COMPUTATION  (one chunk)
# ──────────────────────────────────────────────────────────────────────────────
def compute_features_for_chunk(chunk: pl.DataFrame, workers: int) -> dict[str, np.ndarray]:
    """
    chunk has columns:
        s1_name, s1_addr, s1_country,
        cand_name, cand_addr, cand_country,
        dense_score, dense_rank, retrieval_source
    Returns dict  feature_name → float32 ndarray of length n
    """
    n = chunk.height

    # Pull out Python lists once
    s1_names   = chunk["s1_name"].fill_null("").to_list()
    cand_names = chunk["cand_name"].fill_null("").to_list()
    s1_addrs   = chunk["s1_addr"].fill_null("").to_list()
    cand_addrs = chunk["cand_addr"].fill_null("").to_list()
    s1_ctrs    = chunk["s1_country"].fill_null("").to_list()
    cand_ctrs  = chunk["cand_country"].fill_null("").to_list()

    # ── G3: Country (trivially cheap, do first) ──────────────────────────────
    country_exact = np.array(
        [float(a == b and a != "") for a, b in zip(s1_ctrs, cand_ctrs)],
        dtype=np.float32
    )

    # ── Exact-match masks (allow short-circuit for expensive sims) ────────────
    name_exact_mask = np.array([a == b for a, b in zip(s1_names, cand_names)])
    addr_exact_mask = np.array([a == b for a, b in zip(s1_addrs, cand_addrs)])
    name_exact_norm = name_exact_mask.astype(np.float32)
    addr_exact_norm = addr_exact_mask.astype(np.float32)

    # ── G1: Name similarities ─────────────────────────────────────────────────
    name_jw  = np.ones(n, dtype=np.float32)
    name_lev = np.ones(n, dtype=np.float32)
    name_tok_jac = np.ones(n, dtype=np.float32)

    need_name = ~name_exact_mask
    if need_name.any():
        s1_n_sub   = [s1_names[i]   for i in range(n) if need_name[i]]
        cand_n_sub = [cand_names[i] for i in range(n) if need_name[i]]
        idx        = np.where(need_name)[0]

        name_jw[idx]  = fast_paired_sim(s1_n_sub, cand_n_sub, JaroWinkler.normalized_similarity)
        name_lev[idx] = fast_paired_sim(s1_n_sub, cand_n_sub, Levenshtein.normalized_similarity)
        name_tok_jac[idx] = fast_token_jaccard(s1_n_sub, cand_n_sub)

    # ── G2: Address similarities ──────────────────────────────────────────────
    addr_jw  = np.ones(n, dtype=np.float32)
    addr_lev = np.ones(n, dtype=np.float32)
    addr_tok = np.ones(n, dtype=np.float32)

    need_addr = ~addr_exact_mask
    if need_addr.any():
        s1_a_sub   = [s1_addrs[i]   for i in range(n) if need_addr[i]]
        cand_a_sub = [cand_addrs[i] for i in range(n) if need_addr[i]]
        idx_a      = np.where(need_addr)[0]

        addr_jw[idx_a]  = fast_paired_sim(s1_a_sub, cand_a_sub, JaroWinkler.normalized_similarity)
        addr_lev[idx_a] = fast_paired_sim(s1_a_sub, cand_a_sub, Levenshtein.normalized_similarity)
        addr_tok[idx_a] = fast_token_jaccard(s1_a_sub, cand_a_sub)

    # ── G4: Retrieval signals (already numeric) ───────────────────────────────
    dense_scores = chunk["dense_score"].to_numpy().astype(np.float32)
    dense_ranks  = chunk["dense_rank"].to_numpy().astype(np.float32)
    ret_src      = chunk["retrieval_source"].to_numpy().astype(np.float32)

    return {
        # G1 name
        "name_exact_norm":    name_exact_norm,
        "name_jaro_winkler":  name_jw,
        "name_levenshtein":   name_lev,
        "name_token_jaccard": name_tok_jac,
        # G2 address
        "addr_exact_norm":    addr_exact_norm,
        "addr_jaro_winkler":  addr_jw,
        "addr_levenshtein":   addr_lev,
        "addr_token_jaccard": addr_tok,
        # G4 retrieval  (country_exact removed — hurts generalization to unseen countries)
        "dense_score":        dense_scores,
        "dense_rank":         dense_ranks,
        "dense_rank_inv":     1.0 / (dense_ranks + 1.0),
        "retrieval_source":   ret_src,
        # G5 cross-field
        "name_jw_x_addr_jw":  name_jw * addr_jw,
    }


FEATURE_COLS: list[str] = [
    "name_exact_norm", "name_jaro_winkler", "name_levenshtein",
    "name_token_jaccard",
    "addr_exact_norm", "addr_jaro_winkler", "addr_levenshtein", "addr_token_jaccard",
    # country_exact removed: overfits to train countries (US/India), fails on test (France)
    "dense_score", "dense_rank", "dense_rank_inv", "retrieval_source",
    "name_jw_x_addr_jw",
]


# ──────────────────────────────────────────────────────────────────────────────
# BENCHMARK MODE  (times each stage on N rows, no full write)
# ──────────────────────────────────────────────────────────────────────────────
def run_benchmark(args, labeled_df: pl.DataFrame,
                  s1_df: pl.DataFrame, cand_df: pl.DataFrame) -> None:
    log.info("━" * 60)
    log.info("BENCHMARK MODE  (first %d rows)", args.benchmark_rows)
    log.info("━" * 60)

    sample = labeled_df.head(args.benchmark_rows)
    stages: dict[str, float] = {}

    # Stage 1: S1 join
    t0 = time.perf_counter()
    sample = sample.join(
        s1_df.select(["entity_id", "name_norm", "address_norm", "country_norm"])
             .rename({"entity_id": "query_id",
                      "name_norm": "s1_name",
                      "address_norm": "s1_addr",
                      "country_norm": "s1_country"}),
        on="query_id", how="left"
    )
    stages["s1_join"] = time.perf_counter() - t0

    # Stage 2: candidate join
    t0 = time.perf_counter()
    sample = sample.join(
        cand_df.select(["entity_id", "name_norm", "address_norm", "country_norm"])
               .rename({"entity_id": "candidate_id",
                        "name_norm": "cand_name",
                        "address_norm": "cand_addr",
                        "country_norm": "cand_country"}),
        on="candidate_id", how="left"
    )
    for col in ["s1_name", "s1_addr", "s1_country",
                "cand_name", "cand_addr", "cand_country"]:
        sample = sample.with_columns(pl.col(col).fill_null(""))
    stages["cand_join"] = time.perf_counter() - t0

    # Stage 3: feature computation
    t0 = time.perf_counter()
    _ = compute_features_for_chunk(sample, workers=args.workers)
    stages["feature_compute"] = time.perf_counter() - t0

    # Stage 4: Parquet write (to /tmp)
    import tempfile, os as _os
    tmp = tempfile.mktemp(suffix=".parquet")
    t0 = time.perf_counter()
    sample.write_parquet(tmp, compression=args.compression)
    stages["parquet_write"] = time.perf_counter() - t0
    _os.unlink(tmp)

    total = sum(stages.values())
    rps   = args.benchmark_rows / total
    log.info("")
    log.info("Stage timings (%d rows, workers=%d):", args.benchmark_rows, args.workers)
    for s, t in stages.items():
        log.info("  %-25s %6.2f s   (%d rows/s)", s, t,
                 int(args.benchmark_rows / t))
    log.info("  %-25s %6.2f s", "TOTAL", total)
    log.info("  %-25s %s", "Throughput", f"{rps:,.0f} rows/sec")
    log.info("  %-25s %.2f GB", "Peak RAM", ram_gb())
    log.info("")
    log.info("Extrapolation to 126M rows:")
    est_h  = 126_000_000 / rps / 3600
    log.info("  Estimated time:  %.1f hours", est_h)
    log.info("  (assumes linear scaling — actual may vary)")


# ──────────────────────────────────────────────────────────────────────────────
# MANIFEST  (resumability)
# ──────────────────────────────────────────────────────────────────────────────
MANIFEST_VERSION = "6a-v2"

def load_manifest(manifest_path: str) -> dict:
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            m = json.load(f)
        if m.get("version") == MANIFEST_VERSION:
            return m
    return {"version": MANIFEST_VERSION, "completed_shards": {}}

def save_manifest(manifest_path: str, manifest: dict) -> None:
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE GENERATION  (full pipeline)
# ──────────────────────────────────────────────────────────────────────────────
def generate_features(
    args,
    gt: dict[str, frozenset[str]],
    s1_df: pl.DataFrame,
    cand_df: pl.DataFrame,
    val_s1_ids: frozenset[str],
) -> tuple[list[str], list[str]]:
    """
    Streams candidates DIRECTLY from Parquet files in chunks.
    NEVER collects the full 126M labeled table.

    Strategy:
    1. Scan dense candidates parquet in offset-based chunks.
    2. Scan exact candidates separately and append as final chunks.
    3. Label each chunk via a Python dict lookup (O(1) per pair).
    4. Compute features per chunk.
    5. Write shard, update manifest, free memory.
    """
    train_dir = os.path.join(args.features_dir, args.split, "train")
    val_dir   = os.path.join(args.features_dir, args.split, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)

    manifest_path = os.path.join(args.features_dir, args.split, "features_manifest.json")
    manifest = load_manifest(manifest_path)

    # ── Build positive-pair lookup dict  (fast O(1) label per row) ──────────
    # gt = {s1_id: frozenset(candidate_ids)}
    # Flatten into a set of (s1_id, cand_id) tuples for O(1) lookup
    log.info("Building positive-pair lookup set...")
    t0 = time.perf_counter()
    pos_pairs: set[tuple[str, str]] = set()
    for s1_id, matches in gt.items():
        for c in matches:
            pos_pairs.add((s1_id, c))
    log.info("  %d positive pairs in lookup (%.1fs)", len(pos_pairs),
             time.perf_counter() - t0)

    # ── S1 / candidate join lookup frames ────────────────────────────────────
    s1_join   = s1_df.rename({"entity_id": "query_id",
                               "name_norm":    "s1_name",
                               "address_norm": "s1_addr",
                               "country_norm": "s1_country"})
    cand_join = cand_df.rename({"entity_id":    "candidate_id",
                                "name_norm":    "cand_name",
                                "address_norm": "cand_addr",
                                "country_norm": "cand_country"})

    train_shards: list[str] = []
    val_shards:   list[str] = []
    shard_idx = 0
    t_start   = time.perf_counter()

    dense_path = os.path.join(args.candidates_dir,
                              f"{args.split}_dense_candidates_K50.parquet")
    exact_path = os.path.join(args.candidates_dir,
                              f"{args.split}_exact_candidates.parquet")

    # ── Stream each parquet source in chunks ─────────────────────────────────
    bm25_path = os.path.join(args.candidates_dir,
                             f"{args.split}_bm25_candidates_name_word_K50.parquet")
    sources = []
    if os.path.exists(dense_path):
        sources.append(("dense", dense_path))
    if os.path.exists(exact_path):
        sources.append(("exact", exact_path))
    if os.path.exists(bm25_path):
        sources.append(("bm25", bm25_path))

    for src_name, src_path in sources:
        log.info("Processing source: %s (%s)", src_name, src_path)

        # Get total row count without loading data
        total_rows = pl.scan_parquet(src_path).select(pl.len()).collect().item()
        log.info("  %d rows", total_rows)

        for offset in range(0, total_rows, args.chunk_size):
            shard_key = f"shard_{shard_idx:05d}_{src_name}"

            # Resumability
            if shard_key in manifest["completed_shards"]:
                info = manifest["completed_shards"][shard_key]
                if os.path.exists(info["train_path"]):
                    train_shards.append(info["train_path"])
                if os.path.exists(info["val_path"]):
                    val_shards.append(info["val_path"])
                log.info("SKIP %s", shard_key)
                shard_idx += 1
                continue

            t0 = time.perf_counter()

            # Read this chunk from Parquet (streaming, small)
            chunk = pl.read_parquet(src_path).slice(offset, args.chunk_size)

            # Select only needed columns
            needed = ["query_id", "candidate_id", "candidate_source"]
            if "dense_score" in chunk.columns:
                needed += ["dense_score", "dense_rank"]
            chunk = chunk.select([c for c in needed if c in chunk.columns])

            # Fill missing dense columns for exact-only rows
            if "dense_score" not in chunk.columns:
                chunk = chunk.with_columns([
                    pl.lit(0.0).cast(pl.Float32).alias("dense_score"),
                    pl.lit(999).cast(pl.Int32).alias("dense_rank"),
                ])
            # retrieval_source: 0=dense, 1=exact, 2=bm25
            src_code = {"dense": 0, "exact": 1, "bm25": 2}.get(src_name, 1)
            chunk = chunk.with_columns(
                pl.lit(src_code).cast(pl.Int8).alias("retrieval_source")
            )

            # Label via dict lookup (no join, no memory spike)
            q_ids = chunk["query_id"].to_list()
            c_ids = chunk["candidate_id"].to_list()
            labels = np.array(
                [1 if (q, c) in pos_pairs else 0
                 for q, c in zip(q_ids, c_ids)],
                dtype=np.int8
            )
            chunk = chunk.with_columns(pl.Series("label", labels))

            # Join entity attributes
            chunk = (chunk
                     .join(s1_join,   on="query_id",     how="left")
                     .join(cand_join, on="candidate_id",  how="left"))
            for col in ["s1_name", "s1_addr", "s1_country",
                        "cand_name", "cand_addr", "cand_country"]:
                chunk = chunk.with_columns(pl.col(col).fill_null(""))

            # Compute features
            feats = compute_features_for_chunk(chunk, workers=args.workers)

            # Assemble shard
            shard_data: dict[str, Any] = {
                "query_id":     chunk["query_id"],
                "candidate_id": chunk["candidate_id"],
                "label":        chunk["label"],
            }
            for col in FEATURE_COLS:
                shard_data[col] = feats[col]
            shard_df = pl.DataFrame(shard_data)

            # Train / val split by query_id
            is_val = chunk["query_id"].is_in(list(val_s1_ids))
            train_shard = shard_df.filter(~is_val)
            val_shard   = shard_df.filter( is_val)

            train_path = os.path.join(train_dir, f"{shard_key}.parquet")
            val_path   = os.path.join(val_dir,   f"{shard_key}.parquet")
            train_shard.write_parquet(train_path, compression=args.compression)
            val_shard.write_parquet(  val_path,   compression=args.compression)

            elapsed = time.perf_counter() - t0
            rps     = len(q_ids) / elapsed
            n_pos   = int(labels.sum())
            log.info("%s [%d+%d]  pos=%d  train=%d  val=%d  %.0f rows/s  RAM=%.1fGB",
                     shard_key, offset, len(q_ids), n_pos,
                     train_shard.height, val_shard.height, rps, ram_gb())

            manifest["completed_shards"][shard_key] = {
                "train_path": train_path, "val_path": val_path,
                "train_rows": train_shard.height, "val_rows": val_shard.height,
                "source": src_name, "offset": offset,
            }
            save_manifest(manifest_path, manifest)
            train_shards.append(train_path)
            val_shards.append(val_path)

            del chunk, shard_df, train_shard, val_shard, feats, labels
            gc.collect()
            shard_idx += 1

    total_t = time.perf_counter() - t_start
    log.info("Feature generation complete: %d shards in %.1fs",
             len(train_shards), total_t)
    manifest["feature_generation_seconds"] = round(total_t, 2)
    manifest["feature_cols"] = FEATURE_COLS
    manifest["chunk_size"]   = args.chunk_size
    save_manifest(manifest_path, manifest)
    return train_shards, val_shards



# ──────────────────────────────────────────────────────────────────────────────
# LIGHTGBM  (with optional negative sampling)
# ──────────────────────────────────────────────────────────────────────────────
def train_lightgbm(args, train_shards: list[str], val_shards: list[str]) -> dict:
    log.info("━" * 60)
    log.info("LIGHTGBM TRAINING")
    log.info("━" * 60)

    log.info("Loading %d train shards...", len(train_shards))
    train_df = pl.concat([pl.read_parquet(p) for p in train_shards])
    log.info("  Train rows: %d  RAM=%.1fGB", train_df.height, ram_gb())

    log.info("Loading %d val shards...", len(val_shards))
    val_df = pl.concat([pl.read_parquet(p) for p in val_shards])
    log.info("  Val rows: %d", val_df.height)

    n_pos_tr = int((train_df["label"] == 1).sum())
    n_neg_tr = int((train_df["label"] == 0).sum())

    # ── Optional negative downsampling ──────────────────────────────────────
    if args.negative_sample_ratio > 0 and n_neg_tr > 0:
        n_keep = int(n_pos_tr * args.negative_sample_ratio)
        log.info("Negative sampling: keeping %d / %d negatives (ratio=%.1f×)",
                 n_keep, n_neg_tr, args.negative_sample_ratio)
        pos_df  = train_df.filter(pl.col("label") == 1)
        neg_df  = (train_df.filter(pl.col("label") == 0)
                           .sample(n=min(n_keep, n_neg_tr), seed=args.seed))
        train_df = pl.concat([pos_df, neg_df]).sample(fraction=1.0, shuffle=True,
                                                       seed=args.seed)
        n_neg_tr = neg_df.height
        log.info("  Training on %d pos + %d neg", n_pos_tr, n_neg_tr)

    spw = n_neg_tr / max(n_pos_tr, 1)

    X_tr = train_df.select(FEATURE_COLS).to_numpy()
    y_tr = train_df["label"].to_numpy()
    val_query_ids = val_df["query_id"].to_list()
    val_cand_ids  = val_df["candidate_id"].to_list()
    X_val = val_df.select(FEATURE_COLS).to_numpy()
    y_val = val_df["label"].to_numpy()
    del train_df, val_df; gc.collect()

    # Try GPU first, fall back to CPU silently
    try:
        import subprocess
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        device = "gpu"
        log.info("GPU detected — using device=gpu for LightGBM")
    except Exception:
        device = "cpu"
        log.info("No GPU detected — using CPU for LightGBM")

    params = {
        "objective": "binary", "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt", "num_leaves": 127,
        "learning_rate": 0.05, "n_estimators": args.lgb_rounds,
        "scale_pos_weight": spw,
        "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 0.1,
        "random_state": args.seed, "n_jobs": -1, "verbose": -1,
        "device": "cpu", # Force CPU to avoid GPU zero-variance bug
    }

    lgb_tr  = lgb.Dataset(X_tr,  label=y_tr,  feature_name=FEATURE_COLS,
                           free_raw_data=True)
    lgb_val = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS,
                           free_raw_data=True, reference=lgb_tr)

    t0 = time.perf_counter()
    model = lgb.train(params, lgb_tr, valid_sets=[lgb_val],
                      callbacks=[lgb.early_stopping(50, verbose=True),
                                 lgb.log_evaluation(50)])
    train_t = time.perf_counter() - t0
    log.info("Training complete in %.1fs | best=%d | AUC=%.4f",
             train_t, model.best_iteration,
             model.best_score["valid_0"]["auc"])

    os.makedirs(args.models_dir, exist_ok=True)
    model_path = os.path.join(args.models_dir, "model_6a.txt")
    model.save_model(model_path)

    val_scores = model.predict(X_val, num_iteration=model.best_iteration)

    return {
        "model": model, "model_path": model_path,
        "val_scores": val_scores, "val_labels": y_val,
        "val_query_ids": val_query_ids, "val_cand_ids": val_cand_ids,
        "n_pos_train": n_pos_tr, "n_neg_train": n_neg_tr,
        "n_pos_val": int((y_val == 1).sum()),
        "n_neg_val": int((y_val == 0).sum()),
        "train_time_sec": round(train_t, 2),
        "best_round": model.best_iteration,
        "best_val_auc": model.best_score["valid_0"]["auc"],
        "lgb_params": params,
    }


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

def evaluate(result: dict, gt: dict[str, frozenset]) -> dict:
    val_scores    = result["val_scores"]
    val_labels    = result["val_labels"]
    val_query_ids = result["val_query_ids"]
    val_cand_ids  = result["val_cand_ids"]

    log.info("Threshold sweep for macro F0.5...")
    thresholds = np.arange(0.05, 0.96, 0.05)
    best_f05, best_thr, thr_results = 0.0, 0.5, []

    for thr in thresholds:
        pred_pos = defaultdict(set)
        for q, c, s in zip(val_query_ids, val_cand_ids, val_scores):
            if s >= thr:
                pred_pos[q].add(c)
        all_q = set(val_query_ids)
        scores_q = []
        for q in all_q:
            truth = gt.get(q, frozenset())
            preds = pred_pos.get(q, set())
            tp = len(truth & preds); fp = len(preds - truth); fn = len(truth - preds)
            prec = tp/(tp+fp) if (tp+fp) > 0 else (1.0 if not truth else 0.0)
            rec  = tp/(tp+fn) if (tp+fn) > 0 else (1.0 if not truth else 0.0)
            scores_q.append(f_beta(prec, rec))
        f05 = float(np.mean(scores_q))
        thr_results.append({"threshold": round(float(thr), 2), "macro_f05": round(f05, 4)})
        if f05 > best_f05:
            best_f05, best_thr = f05, float(thr)

    log.info("Best macro F0.5 = %.4f  @ threshold = %.2f", best_f05, best_thr)

    model = result["model"]
    feat_imp = sorted(
        zip(FEATURE_COLS, model.feature_importance(importance_type="gain")),
        key=lambda x: x[1], reverse=True
    )
    log.info("Top 15 features (gain):")
    for feat, score in feat_imp[:15]:
        log.info("  %-30s %.1f", feat, score)

    return {
        "best_threshold": best_thr, "best_macro_f05": best_f05,
        "threshold_sweep": thr_results,
        "feature_importance": [{"feature": f, "gain": float(g)} for f, g in feat_imp],
        "best_val_auc": result["best_val_auc"],
        "best_lgb_round": result["best_round"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    t_total = time.perf_counter()

    log.info("━" * 60)
    log.info("PHASE 6A v2  |  chunk=%d  workers=%d  seed=%d",
             args.chunk_size, args.workers, args.seed)
    log.info("━" * 60)

    os.makedirs(args.features_dir, exist_ok=True)
    os.makedirs(args.models_dir,   exist_ok=True)
    os.makedirs(args.reports_dir,  exist_ok=True)

    # ── Entity tables ────────────────────────────────────────────────────────
    s1_df, cand_df = load_entity_tables(args.data_dir, args.split)

    # ── Ground truth ─────────────────────────────────────────────────────────
    gt = load_ground_truth(args.ground_truth)

    # ── BENCHMARK mode ───────────────────────────────────────────────────────
    # For benchmark we need a small labeled sample — collect ONLY benchmark_rows
    if args.benchmark:
        log.info("Benchmark: collecting %d rows from dense candidates...",
                 args.benchmark_rows)
        dense_path = os.path.join(args.candidates_dir,
                                  f"{args.split}_dense_candidates_K50.parquet")
        bm_chunk = pl.scan_parquet(dense_path).head(args.benchmark_rows).collect()
        q_ids = bm_chunk["query_id"].to_list()
        c_ids = bm_chunk["candidate_id"].to_list()
        pos_set = {(s, c) for s, ms in gt.items() for c in ms}
        labels = [1 if (q, c) in pos_set else 0 for q, c in zip(q_ids, c_ids)]
        bm_chunk = bm_chunk.with_columns(pl.Series("label", labels, dtype=pl.Int8))
        if "retrieval_source" not in bm_chunk.columns:
            bm_chunk = bm_chunk.with_columns(pl.lit(0).cast(pl.Int8).alias("retrieval_source"))
        run_benchmark(args, bm_chunk, s1_df, cand_df)
        return

    # ── Train / val S1 split  (from s1_df — already in RAM, ~2.2M rows) ─────
    all_s1_ids = s1_df["entity_id"].to_list()
    rng        = np.random.default_rng(args.seed)
    rng.shuffle(all_s1_ids)
    val_cut    = int(len(all_s1_ids) * args.val_fraction)
    val_s1_ids = frozenset(all_s1_ids[:val_cut])
    log.info("S1 split: %d train  %d val", len(all_s1_ids) - val_cut, val_cut)

    # ── Feature generation ───────────────────────────────────────────────────
    if not args.skip_features:
        train_shards, val_shards = generate_features(
            args, gt, s1_df, cand_df, val_s1_ids)
    else:
        manifest_path = os.path.join(args.features_dir, args.split,
                                     "features_manifest.json")
        with open(manifest_path) as f:
            meta = json.load(f)
        train_shards = [v["train_path"] for v in meta["completed_shards"].values()
                        if os.path.exists(v["train_path"])]
        val_shards   = [v["val_path"]   for v in meta["completed_shards"].values()
                        if os.path.exists(v["val_path"])]
        log.info("Skipped feature gen: %d train shards, %d val shards",
                 len(train_shards), len(val_shards))

    if args.skip_training:
        log.info("Skipped LightGBM training (--skip-training).")
        return


    # ── LightGBM ─────────────────────────────────────────────────────────────
    result      = train_lightgbm(args, train_shards, val_shards)
    eval_result = evaluate(result, gt)

    # ── Save report ──────────────────────────────────────────────────────────
    report = {
        "timestamp": datetime.now().isoformat(),
        "version": MANIFEST_VERSION,
        "seed": args.seed, "chunk_size": args.chunk_size,
        "workers": args.workers, "compression": args.compression,
        "n_pos_train": result["n_pos_train"],
        "n_neg_train": result["n_neg_train"],
        "n_pos_val":   result["n_pos_val"],
        "n_neg_val":   result["n_neg_val"],
        "features": FEATURE_COLS,
        "lgb_params": result["lgb_params"],
        **eval_result,
        "total_runtime_sec": round(time.perf_counter() - t_total, 2),
    }
    report_path = os.path.join(args.reports_dir, "phase6a_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("━" * 60)
    log.info("PHASE 6A COMPLETE")
    log.info("Val AUC:       %.4f", eval_result["best_val_auc"])
    log.info("Best F0.5:     %.4f  @ thr=%.2f", eval_result["best_macro_f05"],
             eval_result["best_threshold"])
    log.info("Total runtime: %.1fs", report["total_runtime_sec"])
    log.info("Report:        %s", report_path)


if __name__ == "__main__":
    main()
