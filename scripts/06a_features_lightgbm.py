"""
Phase 6A: Pairwise Feature Engineering + LightGBM
===================================================
Memory budget: ~28 GB Kaggle CPU RAM
Strategy: Polars LazyFrame scan + chunk-by-chunk feature materialisation
Output: data/features/train/shard_NNN.parquet  +  models/lightgbm/model_6a.txt

Schema of processed Parquet (from Phase 2):
    entity_id, source, business_name, business_address, country,
    name_norm, address_norm, country_norm

Candidate Parquet schemas:
  Dense   → query_id, candidate_id, candidate_source, dense_rank, dense_score
  Exact   → query_id, candidate_id, candidate_source, match_name, match_address

Feature groups implemented in 6A (baseline):
    G1: name exact / Jaro-Winkler / Levenshtein ratio / token Jaccard
    G2: address exact / Jaro-Winkler / token Jaccard
    G3: country exact
    G4: retrieval signals (dense_score, dense_rank, retrieval_source)
    G5: cross-field interaction (name_jw * addr_jw)
"""

import os, sys, time, gc, json, argparse, hashlib
from datetime import datetime
import numpy as np
import polars as pl
import lightgbm as lgb
from rapidfuzz import metrics as rf_metrics, process as rf_process
from rapidfuzz.distance import Levenshtein, JaroWinkler, Jaro

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src")))

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Phase 6A: Features + LightGBM")
    p.add_argument("--data-dir",        default="data/processed",   help="Phase-2 canonical Parquet dir")
    p.add_argument("--candidates-dir",  default="data/candidates",  help="Phase-3/5D candidate Parquet dir")
    p.add_argument("--features-dir",    default="data/features",    help="Output shard dir")
    p.add_argument("--models-dir",      default="models/lightgbm",  help="LightGBM model output dir")
    p.add_argument("--reports-dir",     default="reports/phase6",   help="Report output dir")
    p.add_argument("--ground-truth",    required=True,              help="Path to train_ground_truth.tsv")
    p.add_argument("--split",           default="train",            help="Data split to process")
    p.add_argument("--chunk-size",      type=int, default=2_000_000, help="Rows per feature shard")
    p.add_argument("--val-fraction",    type=float, default=0.15,   help="Fraction of S1 entities for validation")
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--lgb-rounds",      type=int, default=500)
    p.add_argument("--skip-features",   action="store_true",        help="Skip feature gen if shards exist")
    p.add_argument("--skip-training",   action="store_true",        help="Skip LightGBM training")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH LOADER  →  {s1_id: frozenset(matched_ids)}
# ──────────────────────────────────────────────────────────────────────────────
def load_ground_truth(gt_path: str) -> dict:
    """Returns dict: s1_entity_id → frozenset of true match entity_ids."""
    print(f"Loading ground truth from {gt_path}...")
    df = pl.read_csv(gt_path, separator="\t")
    gt = {}
    for row in df.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        matches = frozenset(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else frozenset()
        gt[s1_id] = matches
    print(f"  Loaded {len(gt):,} S1 entities with ground truth.")
    return gt


# ──────────────────────────────────────────────────────────────────────────────
# STRING FEATURE HELPERS  (operate on Python strings)
# ──────────────────────────────────────────────────────────────────────────────

def jaro_winkler(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return JaroWinkler.normalized_similarity(a, b)

def jaro_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return Jaro.normalized_similarity(a, b)

def levenshtein_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return Levenshtein.normalized_similarity(a, b)

def token_jaccard(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ──────────────────────────────────────────────────────────────────────────────
# VECTORISED FEATURE BATCH
# Input: two equal-length lists of (name_norm, address_norm, country_norm)
# Returns: dict of feature_name → np.ndarray (float32)
# ──────────────────────────────────────────────────────────────────────────────
def compute_features_batch(
    s1_names, s1_addrs, s1_countries,
    cand_names, cand_addrs, cand_countries,
    dense_scores, dense_ranks, retrieval_sources
) -> dict:
    n = len(s1_names)
    feats = {}

    # G1: Name features
    name_exact_norm  = np.zeros(n, dtype=np.float32)
    name_jw          = np.zeros(n, dtype=np.float32)
    name_jaro        = np.zeros(n, dtype=np.float32)
    name_lev         = np.zeros(n, dtype=np.float32)
    name_tok_jac     = np.zeros(n, dtype=np.float32)

    # G2: Address features
    addr_exact_norm  = np.zeros(n, dtype=np.float32)
    addr_jw          = np.zeros(n, dtype=np.float32)
    addr_lev         = np.zeros(n, dtype=np.float32)
    addr_tok_jac     = np.zeros(n, dtype=np.float32)

    # G3: Country
    country_exact    = np.zeros(n, dtype=np.float32)

    for i in range(n):
        s_n  = s1_names[i]     or ""
        c_n  = cand_names[i]   or ""
        s_a  = s1_addrs[i]     or ""
        c_a  = cand_addrs[i]   or ""
        s_c  = s1_countries[i] or ""
        c_c  = cand_countries[i] or ""

        name_exact_norm[i]  = float(s_n == c_n)
        name_jw[i]          = jaro_winkler(s_n, c_n)
        name_jaro[i]        = jaro_sim(s_n, c_n)
        name_lev[i]         = levenshtein_ratio(s_n, c_n)
        name_tok_jac[i]     = token_jaccard(s_n, c_n)

        addr_exact_norm[i]  = float(s_a == c_a)
        addr_jw[i]          = jaro_winkler(s_a, c_a)
        addr_lev[i]         = levenshtein_ratio(s_a, c_a)
        addr_tok_jac[i]     = token_jaccard(s_a, c_a)

        country_exact[i]    = float(s_c == c_c and s_c != "")

    feats["name_exact_norm"]  = name_exact_norm
    feats["name_jaro_winkler"]= name_jw
    feats["name_jaro"]        = name_jaro
    feats["name_levenshtein"] = name_lev
    feats["name_token_jaccard"] = name_tok_jac

    feats["addr_exact_norm"]  = addr_exact_norm
    feats["addr_jaro_winkler"]= addr_jw
    feats["addr_levenshtein"] = addr_lev
    feats["addr_token_jaccard"] = addr_tok_jac

    feats["country_exact"]    = country_exact

    # G4: Retrieval signals
    feats["dense_score"]      = np.array(dense_scores,      dtype=np.float32)
    feats["dense_rank"]       = np.array(dense_ranks,       dtype=np.float32)
    feats["dense_rank_inv"]   = 1.0 / (np.array(dense_ranks, dtype=np.float32) + 1.0)

    # retrieval_source encoding: dense_only=0, exact_only=1, both=2
    feats["retrieval_source"] = np.array(retrieval_sources, dtype=np.float32)

    # G5: Cross-field interactions
    feats["name_jw_x_addr_jw"] = name_jw * addr_jw

    return feats


FEATURE_COLS = [
    "name_exact_norm", "name_jaro_winkler", "name_jaro", "name_levenshtein",
    "name_token_jaccard",
    "addr_exact_norm", "addr_jaro_winkler", "addr_levenshtein", "addr_token_jaccard",
    "country_exact",
    "dense_score", "dense_rank", "dense_rank_inv", "retrieval_source",
    "name_jw_x_addr_jw",
]


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 6A STEP 1: BUILD HYBRID CANDIDATE TABLE WITH LABELS
# ──────────────────────────────────────────────────────────────────────────────
def build_labeled_candidate_table(candidates_dir: str, gt: dict, split: str) -> pl.LazyFrame:
    """
    Merges dense + exact candidates into a single LazyFrame with:
      query_id, candidate_id, candidate_source,
      dense_score (null if exact-only), dense_rank (null if exact-only),
      match_name (null if dense-only), match_address (null if dense-only),
      retrieval_source (0=dense, 1=exact, 2=both),
      label (int 0/1)
    """
    dense_path = os.path.join(candidates_dir, f"{split}_dense_candidates_K50.parquet")
    exact_path = os.path.join(candidates_dir, f"{split}_exact_candidates.parquet")

    if not os.path.exists(dense_path):
        raise FileNotFoundError(f"Dense candidates not found: {dense_path}")

    print("Scanning dense candidates...")
    dense_lf = pl.scan_parquet(dense_path).select([
        "query_id", "candidate_id", "candidate_source",
        pl.col("dense_score").cast(pl.Float32),
        pl.col("dense_rank").cast(pl.Int32),
    ]).with_columns(pl.lit(0).cast(pl.Int8).alias("_from_dense"))

    if os.path.exists(exact_path):
        print("Scanning exact candidates...")
        exact_lf = pl.scan_parquet(exact_path).select([
            "query_id", "candidate_id", "candidate_source",
            pl.col("match_name").cast(pl.Boolean).fill_null(False),
            pl.col("match_address").cast(pl.Boolean).fill_null(False),
        ]).with_columns(pl.lit(1).cast(pl.Int8).alias("_from_exact"))

        # Outer join dense ← exact to merge retrieval sources
        combined = dense_lf.join(
            exact_lf, on=["query_id", "candidate_id", "candidate_source"], how="full", coalesce=True
        ).fill_null({"_from_dense": 0, "_from_exact": 0, "match_name": False, "match_address": False})
    else:
        combined = dense_lf.with_columns([
            pl.lit(0).cast(pl.Int8).alias("_from_exact"),
            pl.lit(False).alias("match_name"),
            pl.lit(False).alias("match_address"),
        ])

    # Encode retrieval_source: 0=dense-only, 1=exact-only, 2=both
    combined = combined.with_columns(
        (pl.col("_from_dense") + pl.col("_from_exact") * 2 - 1)
        .clip(0, 2)
        .cast(pl.Int8)
        .alias("retrieval_source")
    ).drop(["_from_dense", "_from_exact"])

    # Fill nulls for dense fields that might be missing for exact-only rows
    combined = combined.with_columns([
        pl.col("dense_score").fill_null(0.0),
        pl.col("dense_rank").fill_null(999).cast(pl.Int32),
    ])

    # Attach labels using a known-positive lookup
    # Build a Polars Series-based approach: broadcast gt into a frame
    print("Building positive-pair label lookup...")
    label_rows = []
    for s1_id, match_set in gt.items():
        for cand_id in match_set:
            label_rows.append({"query_id": s1_id, "candidate_id": cand_id})

    if label_rows:
        label_df = pl.DataFrame(label_rows).lazy().with_columns(pl.lit(1).cast(pl.Int8).alias("label"))
    else:
        label_df = pl.DataFrame({"query_id": pl.Series([], dtype=pl.Utf8),
                                  "candidate_id": pl.Series([], dtype=pl.Utf8),
                                  "label": pl.Series([], dtype=pl.Int8)}).lazy()

    combined = combined.join(label_df, on=["query_id", "candidate_id"], how="left").with_columns(
        pl.col("label").fill_null(0).cast(pl.Int8)
    )

    return combined


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 6A STEP 2: FEATURE GENERATION (CHUNKED)
# ──────────────────────────────────────────────────────────────────────────────
def generate_features(args, labeled_lf: pl.LazyFrame, entity_attrs: dict) -> tuple[list, list]:
    """
    Chunks through labeled_lf, joins entity attributes, computes features, writes shards.
    Returns (train_shard_paths, val_shard_paths).
    """
    train_shard_dir = os.path.join(args.features_dir, args.split, "train")
    val_shard_dir   = os.path.join(args.features_dir, args.split, "val")
    os.makedirs(train_shard_dir, exist_ok=True)
    os.makedirs(val_shard_dir,   exist_ok=True)

    # S1→Val split: deterministic hash on query_id
    print(f"\nSplitting S1 entities (val fraction={args.val_fraction}, seed={args.seed})...")
    all_s1_ids = list(entity_attrs["s1"].keys())
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_s1_ids)
    val_cut = int(len(all_s1_ids) * args.val_fraction)
    val_s1_ids   = frozenset(all_s1_ids[:val_cut])
    train_s1_ids = frozenset(all_s1_ids[val_cut:])
    print(f"  Train S1: {len(train_s1_ids):,}  |  Val S1: {len(val_s1_ids):,}")

    # Collect to Python in chunks
    print(f"\nCollecting candidate data in chunks of {args.chunk_size:,}...")
    total_rows = labeled_lf.select(pl.len()).collect().item()
    print(f"  Total candidate pairs: {total_rows:,}")

    s1_attrs_df  = entity_attrs["s1_frame"]   # eagerly loaded (small)
    cand_attrs_df= entity_attrs["cand_frame"]  # eagerly loaded (large-ish)

    train_shards, val_shards = [], []
    shard_idx = 0
    t_feat_start = time.time()

    # Process in chunks using slicing over the collected frame
    # We collect once then iterate slices to avoid multiple LazyFrame materializations
    print("  Collecting full labeled frame (this may take ~60s)...")
    t0 = time.time()
    # Only collect needed columns 
    all_cols = [
        "query_id", "candidate_id", "candidate_source",
        "dense_score", "dense_rank", "retrieval_source", "label"
    ]
    labeled_collected = labeled_lf.select(all_cols).collect()
    print(f"  Collected {labeled_collected.height:,} rows in {time.time()-t0:.1f}s")

    n = labeled_collected.height
    for chunk_start in range(0, n, args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, n)
        chunk = labeled_collected.slice(chunk_start, chunk_end - chunk_start)
        t0 = time.time()

        # Join S1 attributes
        chunk = chunk.join(s1_attrs_df.rename({
            "entity_id": "query_id",
            "name_norm": "s1_name", "address_norm": "s1_addr", "country_norm": "s1_country"
        }), on="query_id", how="left")

        # Join candidate attributes
        chunk = chunk.join(cand_attrs_df.rename({
            "entity_id": "candidate_id",
            "name_norm": "cand_name", "address_norm": "cand_addr", "country_norm": "cand_country"
        }), on="candidate_id", how="left")

        # Fill nulls in text columns
        for col in ["s1_name", "s1_addr", "s1_country", "cand_name", "cand_addr", "cand_country"]:
            chunk = chunk.with_columns(pl.col(col).fill_null(""))

        # Compute features
        feats = compute_features_batch(
            s1_names=chunk["s1_name"].to_list(),
            s1_addrs=chunk["s1_addr"].to_list(),
            s1_countries=chunk["s1_country"].to_list(),
            cand_names=chunk["cand_name"].to_list(),
            cand_addrs=chunk["cand_addr"].to_list(),
            cand_countries=chunk["cand_country"].to_list(),
            dense_scores=chunk["dense_score"].to_list(),
            dense_ranks=chunk["dense_rank"].to_list(),
            retrieval_sources=chunk["retrieval_source"].to_list(),
        )

        # Build shard DataFrame
        shard_data = {
            "query_id":     chunk["query_id"],
            "candidate_id": chunk["candidate_id"],
            "label":        chunk["label"],
        }
        for col in FEATURE_COLS:
            shard_data[col] = feats[col]

        shard_df = pl.DataFrame(shard_data)

        # Split into train / val by S1 entity
        train_mask = chunk["query_id"].is_in(list(train_s1_ids))
        val_mask   = ~train_mask

        train_shard = shard_df.filter(train_mask)
        val_shard   = shard_df.filter(val_mask)

        if train_shard.height > 0:
            path = os.path.join(train_shard_dir, f"shard_{shard_idx:04d}.parquet")
            train_shard.write_parquet(path, compression="snappy")
            train_shards.append(path)

        if val_shard.height > 0:
            path = os.path.join(val_shard_dir, f"shard_{shard_idx:04d}.parquet")
            val_shard.write_parquet(path, compression="snappy")
            val_shards.append(path)

        elapsed = time.time() - t0
        rows_sec = (chunk_end - chunk_start) / elapsed
        print(f"  Shard {shard_idx:04d}: [{chunk_start:>12,}:{chunk_end:>12,}]  "
              f"train={train_shard.height:,}  val={val_shard.height:,}  "
              f"{rows_sec:,.0f} rows/s  ({elapsed:.1f}s)")

        shard_idx += 1
        del chunk, shard_df, train_shard, val_shard
        gc.collect()

    total_feat_time = time.time() - t_feat_start
    print(f"\nFeature generation complete: {shard_idx} shards in {total_feat_time:.1f}s")
    print(f"  Train shards: {len(train_shards)}  Val shards: {len(val_shards)}")

    # Write split metadata for reproducibility
    meta = {
        "split_seed": args.seed,
        "val_fraction": args.val_fraction,
        "train_s1_count": len(train_s1_ids),
        "val_s1_count": len(val_s1_ids),
        "total_candidate_pairs": n,
        "feature_cols": FEATURE_COLS,
        "chunk_size": args.chunk_size,
        "feature_generation_seconds": round(total_feat_time, 2),
        "train_shards": train_shards,
        "val_shards": val_shards,
    }
    meta_path = os.path.join(args.features_dir, args.split, "split_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to {meta_path}")

    return train_shards, val_shards


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 6A STEP 3: LIGHTGBM TRAINING
# ──────────────────────────────────────────────────────────────────────────────
def train_lightgbm(args, train_shards: list, val_shards: list) -> dict:
    print("\n" + "="*60)
    print("LIGHTGBM TRAINING")
    print("="*60)

    print("Loading train shards...")
    t0 = time.time()
    train_df = pl.concat([pl.scan_parquet(p) for p in train_shards]).collect()
    print(f"  Train rows: {train_df.height:,}  (loaded in {time.time()-t0:.1f}s)")

    print("Loading val shards...")
    t0 = time.time()
    val_df   = pl.concat([pl.scan_parquet(p) for p in val_shards]).collect()
    print(f"  Val rows:   {val_df.height:,}  (loaded in {time.time()-t0:.1f}s)")

    # Label stats
    n_pos_train = (train_df["label"] == 1).sum()
    n_neg_train = (train_df["label"] == 0).sum()
    n_pos_val   = (val_df["label"] == 1).sum()
    n_neg_val   = (val_df["label"] == 0).sum()
    print(f"\n  Train: {n_pos_train:,} pos / {n_neg_train:,} neg  "
          f"(ratio 1:{n_neg_train//max(n_pos_train,1)})")
    print(f"  Val:   {n_pos_val:,} pos / {n_neg_val:,} neg")

    # Build LGB datasets
    X_train = train_df.select(FEATURE_COLS).to_numpy()
    y_train = train_df["label"].to_numpy()
    X_val   = val_df.select(FEATURE_COLS).to_numpy()
    y_val   = val_df["label"].to_numpy()

    # Keep query_id in val for entity-level evaluation
    val_query_ids = val_df["query_id"].to_list()
    val_cand_ids  = val_df["candidate_id"].to_list()

    del train_df, val_df
    gc.collect()

    # Class balance via scale_pos_weight
    spw = n_neg_train / max(n_pos_train, 1)
    print(f"\n  scale_pos_weight = {spw:.2f}")

    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS, free_raw_data=True)
    lgb_val   = lgb.Dataset(X_val,   label=y_val,   feature_name=FEATURE_COLS, free_raw_data=True,
                             reference=lgb_train)

    params = {
        "objective":        "binary",
        "metric":           ["binary_logloss", "auc"],
        "boosting_type":    "gbdt",
        "num_leaves":       127,
        "max_depth":        -1,
        "learning_rate":    0.05,
        "n_estimators":     args.lgb_rounds,
        "scale_pos_weight": spw,
        "min_child_samples": 50,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       0.1,
        "random_state":     args.seed,
        "n_jobs":           -1,
        "verbose":          -1,
    }

    print("\nTraining LightGBM...")
    t0 = time.time()
    callbacks = [lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)]
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_val],
        callbacks=callbacks,
    )
    train_time = time.time() - t0
    print(f"  Training complete in {train_time:.1f}s")
    print(f"  Best round: {model.best_iteration}  |  Best val AUC: {model.best_score['valid_0']['auc']:.4f}")

    # Save model
    os.makedirs(args.models_dir, exist_ok=True)
    model_path = os.path.join(args.models_dir, "model_6a.txt")
    model.save_model(model_path)
    print(f"  Model saved to {model_path}")

    # ── Validation scoring ──────────────────────────────────────────
    print("\nScoring validation set...")
    val_scores = model.predict(X_val, num_iteration=model.best_iteration)

    return {
        "model":          model,
        "val_scores":     val_scores,
        "val_labels":     y_val,
        "val_query_ids":  val_query_ids,
        "val_cand_ids":   val_cand_ids,
        "n_pos_train":    int(n_pos_train),
        "n_neg_train":    int(n_neg_train),
        "n_pos_val":      int(n_pos_val),
        "n_neg_val":      int(n_neg_val),
        "train_time_sec": round(train_time, 2),
        "best_round":     model.best_iteration,
        "best_val_auc":   model.best_score['valid_0']['auc'],
        "lgb_params":     params,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 6A STEP 4: EVALUATION (per-S1 macro F0.5)
# ──────────────────────────────────────────────────────────────────────────────
def f_beta(precision, recall, beta=0.5):
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

def evaluate_threshold(val_query_ids, val_cand_ids, val_scores, val_labels, threshold, gt):
    """Computes per-S1 macro F0.5 at a given decision threshold."""
    # Group by query_id
    from collections import defaultdict
    query_preds  = defaultdict(set)
    query_truth  = {}

    for q, c, s, l in zip(val_query_ids, val_cand_ids, val_scores, val_labels):
        if s >= threshold:
            query_preds[q].add(c)
        if q not in query_truth:
            query_truth[q] = gt.get(q, frozenset())

    all_s1 = set(query_truth.keys())
    f05_scores = []
    for q in all_s1:
        truth = query_truth[q]
        preds = query_preds.get(q, set())
        tp = len(truth & preds)
        fp = len(preds - truth)
        fn = len(truth - preds)
        prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not truth else 0.0)
        rec  = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if not truth else 0.0)
        f05_scores.append(f_beta(prec, rec))

    return float(np.mean(f05_scores))

def evaluate(result: dict, gt: dict, args) -> dict:
    """Sweeps thresholds to find best macro F0.5."""
    val_scores    = result["val_scores"]
    val_labels    = result["val_labels"]
    val_query_ids = result["val_query_ids"]
    val_cand_ids  = result["val_cand_ids"]

    print("\nEvaluating macro F0.5 across thresholds...")
    thresholds = np.arange(0.05, 0.96, 0.05)
    best_f05, best_thr = 0.0, 0.5

    thr_results = []
    for thr in thresholds:
        f05 = evaluate_threshold(val_query_ids, val_cand_ids, val_scores, val_labels, thr, gt)
        thr_results.append({"threshold": round(float(thr), 2), "macro_f05": round(f05, 4)})
        if f05 > best_f05:
            best_f05, best_thr = f05, float(thr)

    print(f"  Best macro F0.5 = {best_f05:.4f} at threshold = {best_thr:.2f}")

    # Feature importance
    model = result["model"]
    feat_imp = sorted(
        zip(FEATURE_COLS, model.feature_importance(importance_type="gain")),
        key=lambda x: x[1], reverse=True
    )
    print("\nTop 15 Features (gain):")
    for feat, score in feat_imp[:15]:
        print(f"  {feat:<30s} {score:.1f}")

    return {
        "best_threshold":   best_thr,
        "best_macro_f05":   best_f05,
        "threshold_sweep":  thr_results,
        "feature_importance": [{"feature": f, "gain": float(g)} for f, g in feat_imp],
        "best_val_auc":     result["best_val_auc"],
        "best_lgb_round":   result["best_round"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    run_start = time.time()

    print("="*60)
    print("PHASE 6A: PAIRWISE FEATURE ENGINEERING + LIGHTGBM")
    print("="*60)
    print(f"  Seed:           {args.seed}")
    print(f"  Chunk size:     {args.chunk_size:,}")
    print(f"  Val fraction:   {args.val_fraction}")
    print(f"  LGB rounds:     {args.lgb_rounds}")

    os.makedirs(args.features_dir, exist_ok=True)
    os.makedirs(args.models_dir,   exist_ok=True)
    os.makedirs(args.reports_dir,  exist_ok=True)

    # ── Load ground truth ───────────────────────────────────────────
    gt = load_ground_truth(args.ground_truth)

    # ── Load entity attribute lookup tables (small enough to be eager) ─
    s1_path   = os.path.join(args.data_dir, args.split, f"{args.split}_source1.parquet")
    s2_path   = os.path.join(args.data_dir, args.split, f"{args.split}_source2.parquet")
    s3_path   = os.path.join(args.data_dir, args.split, f"{args.split}_source3.parquet")

    attr_cols = ["entity_id", "name_norm", "address_norm", "country_norm"]
    print("\nLoading entity attribute tables...")
    s1_df   = pl.read_parquet(s1_path,   columns=attr_cols)
    s2_df   = pl.read_parquet(s2_path,   columns=attr_cols)
    s3_df   = pl.read_parquet(s3_path,   columns=attr_cols)
    cand_df = pl.concat([s2_df, s3_df])
    print(f"  S1: {s1_df.height:,}   S2: {s2_df.height:,}   S3: {s3_df.height:,}")

    entity_attrs = {
        "s1":       {row["entity_id"]: row for row in s1_df.iter_rows(named=True)},
        "s1_frame": s1_df,
        "cand_frame": cand_df,
    }
    del s2_df, s3_df

    # ── Build labeled candidate LazyFrame ──────────────────────────
    labeled_lf = build_labeled_candidate_table(args.candidates_dir, gt, args.split)

    # ── Feature generation ─────────────────────────────────────────
    train_shard_dir = os.path.join(args.features_dir, args.split, "train")
    existing_shards = [
        os.path.join(train_shard_dir, f) for f in os.listdir(train_shard_dir)
        if f.endswith(".parquet")
    ] if os.path.exists(train_shard_dir) else []

    if args.skip_features and existing_shards:
        print(f"\nSkipping feature gen (--skip-features). Found {len(existing_shards)} existing shards.")
        with open(os.path.join(args.features_dir, args.split, "split_metadata.json")) as f:
            meta = json.load(f)
        train_shards = meta["train_shards"]
        val_shards   = meta["val_shards"]
    else:
        train_shards, val_shards = generate_features(args, labeled_lf, entity_attrs)

    # ── LightGBM ───────────────────────────────────────────────────
    if args.skip_training:
        print("\nSkipping training (--skip-training).")
        return

    result = train_lightgbm(args, train_shards, val_shards)

    # ── Evaluation ─────────────────────────────────────────────────
    eval_metrics = evaluate(result, gt, args)

    # ── Save full report ───────────────────────────────────────────
    report = {
        "timestamp":            datetime.now().isoformat(),
        "seed":                 args.seed,
        "candidate_pairs":      126_082_540,   # from Phase 5E audit
        "train_positives":      result["n_pos_train"],
        "train_negatives":      result["n_neg_train"],
        "val_positives":        result["n_pos_val"],
        "val_negatives":        result["n_neg_val"],
        "features":             FEATURE_COLS,
        "lgb_params":           result["lgb_params"],
        "best_val_auc":         eval_metrics["best_val_auc"],
        "best_macro_f05":       eval_metrics["best_macro_f05"],
        "best_threshold":       eval_metrics["best_threshold"],
        "threshold_sweep":      eval_metrics["threshold_sweep"],
        "feature_importance":   eval_metrics["feature_importance"],
        "train_time_sec":       result["train_time_sec"],
        "total_runtime_sec":    round(time.time() - run_start, 2),
    }

    report_path = os.path.join(args.reports_dir, "phase6a_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    feat_imp_path = os.path.join(args.reports_dir, "feature_importance.csv")
    with open(feat_imp_path, "w") as f:
        f.write("feature,gain\n")
        for row in eval_metrics["feature_importance"]:
            f.write(f"{row['feature']},{row['gain']}\n")

    print("\n" + "="*60)
    print("PHASE 6A COMPLETE")
    print("="*60)
    print(f"  Val AUC:            {eval_metrics['best_val_auc']:.4f}")
    print(f"  Best macro F0.5:    {eval_metrics['best_macro_f05']:.4f} @ thr={eval_metrics['best_threshold']:.2f}")
    print(f"  Total runtime:      {report['total_runtime_sec']:.1f}s")
    print(f"  Report:             {report_path}")

if __name__ == "__main__":
    main()
