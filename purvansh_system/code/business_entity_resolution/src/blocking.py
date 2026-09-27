"""
Blocking & Candidate Generation Module
Shard-streaming TF-IDF character n-gram cosine retrieval.

Architecture:
    Never holds the full candidate TF-IDF matrix in memory.
    Instead, fits vocabulary on a sample, then streams candidates in
    small shards (300K rows), transforming and multiplying each shard
    against the S1 reference matrix independently, maintaining a
    running top-k per entity across all shards.

    Guaranteed peak RAM: ~4-6 GB regardless of total candidate pool size.
    (vs. previous approach: 15-35 GB for 6.2M US candidates)
"""

import os
import time
import gc
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .preprocessing import clean_text, TSV_DTYPES


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_combined_text(clean_names, addresses):
    """Build combined name + address-prefix strings for TF-IDF vectorization."""
    return [
        str(cn) + " " + " ".join(str(a).split()[:3]).lower()
        for cn, a in zip(clean_names, addresses)
    ]


def _stream_candidate_shards(cand_paths, country, chunksize=300000):
    """
    Generator yielding (entity_ids_array, combined_texts_list) per shard.

    Reads source files sequentially, filters by country, applies
    clean_text, and yields one shard at a time. Never holds more
    than one chunk (~300K rows) in memory.
    """
    for path in cand_paths:
        if not os.path.exists(path):
            continue
        for chunk in pd.read_csv(
            path, sep="\t", chunksize=chunksize,
            dtype=TSV_DTYPES, on_bad_lines="skip"
        ):
            chunk["country"] = chunk["country"].fillna("US")
            c = chunk[chunk["country"] == country]
            if c.empty:
                del chunk
                continue

            eids = c["entity_id"].fillna("").astype(str).values
            names = c["business_name"].fillna("").astype(str).values
            addrs = c["business_address"].fillna("").astype(str).values

            clean_names = [clean_text(str(n), str(eid)) for n, eid in zip(names, eids)]
            texts = _build_combined_text(clean_names, addrs)

            del chunk, c, names, addrs, clean_names
            yield eids, texts


def _merge_topk(existing, new_entries, top_k):
    """
    Merge two lists of (score, cand_id) tuples and keep only the top-k
    by score (descending). Operates on very short lists (len <= 2*top_k),
    so sort is effectively O(1).
    """
    combined = existing + new_entries
    if len(combined) <= top_k:
        return combined
    combined.sort(reverse=True)          # sort by score desc (tuple comparison)
    return combined[:top_k]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def shard_streaming_tfidf_blocking(
    s1_df: pd.DataFrame,
    cand_paths: list,
    country: str,
    top_k: int = 12,
    s1_batch_size: int = 1000,
    shard_chunksize: int = 300000,
    max_features: int = 50000,
    min_df: int = 5,
    max_df: float = 0.30,
    vocab_sample_size: int = 500000,
    checkpoint_file: str = None,
) -> dict:
    """
    Shard-streaming TF-IDF blocking.

    Two-pass architecture:
        Pass 1 — Collect a sample of candidate texts and fit the TF-IDF
                  vocabulary. Only reads a fraction of the files.
        Pass 2 — Stream all candidates in shards of `shard_chunksize` rows.
                  For each shard: transform -> multiply against S1 matrix
                  in sub-batches -> extract & merge per-entity top-k ->
                  delete shard matrix.

    Memory profile (US, 6.2M candidates):
        S1 TF-IDF matrix (60K x 50K sparse)  ~0.3 GB
        One shard matrix  (300K x 50K sparse) ~1.0 GB
        Running top-k dict (60K x 12 tuples)  ~0.05 GB
        Vectorizer object                     ~0.1 GB
        Total peak                            ~4-6 GB

    Args:
        s1_df:             S1 reference DataFrame (must have 'entity_id', 'clean_name', 'business_address').
        cand_paths:        List of file paths for S2/S3 TSV files.
        country:           Country filter string (e.g. 'US', 'India', 'France').
        top_k:             Maximum candidates to retrieve per S1 entity.
        s1_batch_size:     Number of S1 entities per sparse-multiply sub-batch.
        shard_chunksize:   Rows per candidate shard read from disk.
        max_features:      TF-IDF vocabulary size cap.
        min_df:            Minimum document frequency for TF-IDF.
        max_df:            Maximum document frequency for TF-IDF.
        vocab_sample_size: Number of candidate texts to sample for vocabulary fitting.
        checkpoint_file:   Optional path to cache/load blocking results.

    Returns:
        dict: {s1_entity_id: [(cand_entity_id, tfidf_score), ...]}
    """
    # ---- Checkpoint ----
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"  Loading cached blocking results from {checkpoint_file}...", flush=True)
        with open(checkpoint_file, "rb") as f:
            return pickle.load(f)

    if s1_df.empty:
        print("  Notice: Empty S1 partition. Skipping blocking.", flush=True)
        return {}

    t0 = time.time()

    # ==================================================================
    # Pass 1: Collect vocabulary sample (reads only a few chunks)
    # ==================================================================
    print(f"  [Blocking] Pass 1: Collecting vocabulary sample (up to {vocab_sample_size:,})...", flush=True)
    sample_texts = []
    for _, texts in _stream_candidate_shards(cand_paths, country, shard_chunksize):
        sample_texts.extend(texts)
        if len(sample_texts) >= vocab_sample_size:
            sample_texts = sample_texts[:vocab_sample_size]
            break

    if not sample_texts:
        print("  Notice: No candidates found for this country.", flush=True)
        return {}

    print(f"  [Blocking] Collected {len(sample_texts):,} sample texts in {time.time() - t0:.1f}s", flush=True)

    # ---- Fit TF-IDF vectorizer on sample ----
    print(f"  [Blocking] Fitting TF-IDF (features={max_features:,}, ngram=(3,4), "
          f"min_df={min_df}, max_df={max_df})...", flush=True)

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        dtype=np.float32,
        sublinear_tf=True,
    )
    vectorizer.fit(sample_texts)
    actual_vocab = len(vectorizer.vocabulary_)
    del sample_texts
    gc.collect()
    print(f"  [Blocking] Vocabulary size: {actual_vocab:,}", flush=True)

    # ==================================================================
    # Transform S1 references (small matrix, kept in memory)
    # ==================================================================
    s1_ids = s1_df["entity_id"].values
    s1_names = s1_df["clean_name"].fillna("").values
    s1_addrs = s1_df["business_address"].fillna("").values
    s1_texts = _build_combined_text(s1_names, s1_addrs)

    print(f"  [Blocking] Transforming {len(s1_texts):,} S1 references...", flush=True)
    s1_matrix = vectorizer.transform(s1_texts).tocsr()
    del s1_texts, s1_names, s1_addrs
    gc.collect()

    n_s1 = len(s1_ids)

    # ==================================================================
    # Pass 2: Stream candidate shards and build running top-k
    # ==================================================================
    # For each S1 entity, maintain a list of (score, cand_id) of length <= top_k
    running_topk = {}         # sid -> [(score, cid), ...]
    shard_count = 0
    total_candidates = 0
    min_score_threshold = 0.01   # ignore negligible cosine matches

    print(f"  [Blocking] Pass 2: Streaming shards (chunk={shard_chunksize:,}, "
          f"batch={s1_batch_size:,})...", flush=True)

    for shard_eids, shard_texts in _stream_candidate_shards(cand_paths, country, shard_chunksize):
        shard_count += 1
        shard_size = len(shard_eids)
        total_candidates += shard_size

        # Transform this shard
        shard_matrix = vectorizer.transform(shard_texts)     # CSR (shard_size, vocab)
        del shard_texts
        gc.collect()

        # Multiply S1 batches x shard^T
        # CSR @ CSR.T -> CSR (batch x shard_size)
        # scipy handles the transpose efficiently
        for bs in range(0, n_s1, s1_batch_size):
            be = min(bs + s1_batch_size, n_s1)

            scores = (s1_matrix[bs:be] @ shard_matrix.T).tocsr()

            # Walk CSR row pointers for fast top-k extraction
            for r in range(be - bs):
                row_start = scores.indptr[r]
                row_end   = scores.indptr[r + 1]
                if row_start == row_end:
                    continue

                row_data = scores.data[row_start:row_end]
                row_cols = scores.indices[row_start:row_end]

                # Filter negligible matches
                mask = row_data >= min_score_threshold
                if not mask.any():
                    continue
                row_data = row_data[mask]
                row_cols = row_cols[mask]

                # Extract shard-local top-k
                if len(row_data) <= top_k:
                    new_entries = [
                        (float(row_data[j]), shard_eids[row_cols[j]])
                        for j in range(len(row_data))
                    ]
                else:
                    top_idx = np.argpartition(row_data, -top_k)[-top_k:]
                    new_entries = [
                        (float(row_data[j]), shard_eids[row_cols[j]])
                        for j in top_idx
                    ]

                sid = s1_ids[bs + r]

                # Merge with running top-k across previous shards
                existing = running_topk.get(sid, [])
                running_topk[sid] = _merge_topk(existing, new_entries, top_k)

            del scores

        del shard_matrix, shard_eids
        gc.collect()

        elapsed = time.time() - t0
        print(f"    Shard {shard_count}: {total_candidates:,} candidates | "
              f"{elapsed:.1f}s elapsed", flush=True)

    del s1_matrix, vectorizer
    gc.collect()

    # ==================================================================
    # Convert to output format: {sid: [(cid, score), ...]}
    # ==================================================================
    candidates = {}
    for sid, topk_list in running_topk.items():
        # topk_list is [(score, cid), ...], convert to [(cid, score), ...]
        candidates[sid] = [(cid, score) for score, cid in topk_list]
    del running_topk
    gc.collect()

    elapsed = time.time() - t0
    total_pairs = sum(len(v) for v in candidates.values())
    print(f"  [Blocking] Done: {n_s1:,} refs -> {total_pairs:,} pairs "
          f"from {total_candidates:,} candidates in {elapsed:.1f}s", flush=True)

    # ---- Save checkpoint ----
    if checkpoint_file:
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        with open(checkpoint_file, "wb") as f:
            pickle.dump(candidates, f)

    return candidates
