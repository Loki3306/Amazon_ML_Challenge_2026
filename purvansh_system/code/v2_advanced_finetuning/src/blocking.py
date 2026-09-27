"""
Blocking & Candidate Generation Module
Shard-streaming TF-IDF character n-gram cosine retrieval.
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
    Merge two lists of (score, cand_id) tuples and keep only the top-k by score.
    """
    combined = existing + new_entries
    if len(combined) <= top_k:
        return combined
    combined.sort(reverse=True)
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
    """
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"  Loading cached blocking results from {checkpoint_file}...", flush=True)
        with open(checkpoint_file, "rb") as f:
            return pickle.load(f)

    if s1_df.empty:
        print("  Notice: Empty S1 partition. Skipping blocking.", flush=True)
        return {}

    t0 = time.time()

    # Pass 1: Fit TF-IDF on sample
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
    del sample_texts
    gc.collect()

    s1_ids = s1_df["entity_id"].values
    s1_names = s1_df["clean_name"].fillna("").values
    s1_addrs = s1_df["business_address"].fillna("").astype(str).values
    s1_texts = _build_combined_text(s1_names, s1_addrs)

    s1_matrix = vectorizer.transform(s1_texts).tocsr()
    del s1_texts, s1_names, s1_addrs
    gc.collect()

    n_s1 = len(s1_ids)
    running_topk = {}
    shard_count = 0
    total_candidates = 0
    min_score_threshold = 0.01

    print(f"  [Blocking] Pass 2: Streaming shards (chunk={shard_chunksize:,}, batch={s1_batch_size:,})...", flush=True)

    for shard_eids, shard_texts in _stream_candidate_shards(cand_paths, country, shard_chunksize):
        shard_count += 1
        shard_size = len(shard_eids)
        total_candidates += shard_size

        shard_matrix = vectorizer.transform(shard_texts)
        del shard_texts
        gc.collect()

        for bs in range(0, n_s1, s1_batch_size):
            be = min(bs + s1_batch_size, n_s1)
            scores = (s1_matrix[bs:be] @ shard_matrix.T).tocsr()

            for r in range(be - bs):
                row_start = scores.indptr[r]
                row_end   = scores.indptr[r + 1]
                if row_start == row_end:
                    continue

                row_data = scores.data[row_start:row_end]
                row_cols = scores.indices[row_start:row_end]

                mask = row_data >= min_score_threshold
                if not mask.any():
                    continue
                row_data = row_data[mask]
                row_cols = row_cols[mask]

                if len(row_data) <= top_k:
                    new_entries = [(float(row_data[j]), shard_eids[row_cols[j]]) for j in range(len(row_data))]
                else:
                    top_idx = np.argpartition(row_data, -top_k)[-top_k:]
                    new_entries = [(float(row_data[j]), shard_eids[row_cols[j]]) for j in top_idx]

                sid = s1_ids[bs + r]
                existing = running_topk.get(sid, [])
                running_topk[sid] = _merge_topk(existing, new_entries, top_k)

            del scores

        del shard_matrix, shard_eids
        gc.collect()

        elapsed = time.time() - t0
        print(f"    Shard {shard_count}: {total_candidates:,} candidates | {elapsed:.1f}s elapsed", flush=True)

    del s1_matrix, vectorizer
    gc.collect()

    candidates = {}
    for sid, topk_list in running_topk.items():
        candidates[sid] = [(cid, score) for score, cid in topk_list]
    del running_topk
    gc.collect()

    if checkpoint_file:
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        with open(checkpoint_file, "wb") as f:
            pickle.dump(candidates, f)

    return candidates
