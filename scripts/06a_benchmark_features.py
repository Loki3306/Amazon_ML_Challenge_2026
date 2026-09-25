"""
Phase 6A Feature Benchmark
================================
Run targeted benchmarks on the pairwise feature computation to:
1. Identify true memory footprint (fixing the pl.read_parquet bug)
2. Compare workers=1, 2, 4
3. Test precomputed token optimization
4. Profile individual feature times (ablation)
5. Verify correctness against a baseline
"""

import os, time, gc, argparse
import numpy as np
import polars as pl
from concurrent.futures import ThreadPoolExecutor
from rapidfuzz.distance import JaroWinkler, Jaro, Levenshtein
import psutil

def ram_gb():
    return psutil.Process().memory_info().rss / (1024 ** 3)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--candidates-dir", default="data/candidates")
    p.add_argument("--rows", type=int, default=1_000_000)
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Implementations
# ─────────────────────────────────────────────────────────────────────────────

# --- V2 (Current) ---
def v2_sim(s1_list, cand_list, scorer, workers):
    if workers <= 1:
        return np.array([scorer(a, b) for a, b in zip(s1_list, cand_list)], dtype=np.float32)
    n = len(s1_list)
    chunk_size = max(1, n // (workers * 4))
    
    def worker(start, end):
        return np.array([scorer(a, b) for a, b in zip(s1_list[start:end], cand_list[start:end])], dtype=np.float32)

    futures = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(0, n, chunk_size):
            futures.append(ex.submit(worker, i, min(i+chunk_size, n)))
    return np.concatenate([f.result() for f in futures])

class TokenIndex_v2:
    def __init__(self, strings):
        vocab = {}
        self.sets = []
        for s in strings:
            ids = set()
            for tok in s.split():
                if tok not in vocab: vocab[tok] = len(vocab)
                ids.add(vocab[tok])
            self.sets.append(frozenset(ids))

def v2_tok(s1_list, cand_list, workers):
    def worker(s1_sub, cand_sub):
        t1 = TokenIndex_v2(s1_sub)
        t2 = TokenIndex_v2(cand_sub)
        out = np.ones(len(s1_sub), dtype=np.float32)
        for i in range(len(s1_sub)):
            sa, sb = t1.sets[i], t2.sets[i]
            if not sa and not sb: out[i] = 1.0
            elif not sa or not sb: out[i] = 0.0
            else:
                inter = len(sa & sb)
                out[i] = inter / (len(sa) + len(sb) - inter)
        return out
        
    if workers <= 1:
        return worker(s1_list, cand_list)
        
    n = len(s1_list)
    chunk_size = max(1, n // (workers * 4))
    futures = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(0, n, chunk_size):
            futures.append(ex.submit(worker, s1_list[i:min(i+chunk_size, n)], cand_list[i:min(i+chunk_size, n)]))
    return np.concatenate([f.result() for f in futures])


# --- V3 (Optimized Map + Unique Tokens) ---
def v3_sim(s1_list, cand_list, scorer, workers):
    if workers <= 1:
        # map() pushes the loop to C
        return np.array(list(map(scorer, s1_list, cand_list)), dtype=np.float32)
    n = len(s1_list)
    chunk_size = max(1, n // (workers * 4))
    
    def worker(start, end):
        return np.array(list(map(scorer, s1_list[start:end], cand_list[start:end])), dtype=np.float32)

    futures = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(0, n, chunk_size):
            futures.append(ex.submit(worker, i, min(i+chunk_size, n)))
    return np.concatenate([f.result() for f in futures])

def v3_tok(s1_list, cand_list, workers):
    # Pre-tokenize uniques to avoid redundant work (especially for S1 queries!)
    def get_token_sets(strings):
        unique_strs = set(strings)
        vocab = {}
        str_to_set = {}
        for s in unique_strs:
            ids = set()
            for tok in s.split():
                if tok not in vocab: vocab[tok] = len(vocab)
                ids.add(vocab[tok])
            str_to_set[s] = frozenset(ids)
        return [str_to_set[s] for s in strings]

    s1_sets = get_token_sets(s1_list)
    cand_sets = get_token_sets(cand_list)
    
    def jaccard(sa, sb):
        if not sa and not sb: return 1.0
        if not sa or not sb: return 0.0
        inter = len(sa & sb)
        return inter / (len(sa) + len(sb) - inter)
        
    if workers <= 1:
        return np.array(list(map(jaccard, s1_sets, cand_sets)), dtype=np.float32)
        
    n = len(s1_list)
    chunk_size = max(1, n // (workers * 4))
    def worker(start, end):
        return np.array(list(map(jaccard, s1_sets[start:end], cand_sets[start:end])), dtype=np.float32)

    futures = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(0, n, chunk_size):
            futures.append(ex.submit(worker, i, min(i+chunk_size, n)))
    return np.concatenate([f.result() for f in futures])

# ─────────────────────────────────────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    print("="*60)
    print("FEATURE COMPUTATION BENCHMARK")
    print("="*60)
    
    print(f"Loading entity tables... RAM={ram_gb():.2f}GB")
    s1_df = pl.read_parquet(f"{args.data_dir}/train/train_source1.parquet", columns=["entity_id", "name_norm", "address_norm", "country_norm"])
    s2_df = pl.read_parquet(f"{args.data_dir}/train/train_source2.parquet", columns=["entity_id", "name_norm", "address_norm", "country_norm"])
    s3_df = pl.read_parquet(f"{args.data_dir}/train/train_source3.parquet", columns=["entity_id", "name_norm", "address_norm", "country_norm"])
    cand_df = pl.concat([s2_df, s3_df])
    del s2_df, s3_df; gc.collect()
    print(f"Loaded. RAM={ram_gb():.2f}GB")

    print(f"\nLoading {args.rows} candidate pairs (memory-safe)...")
    dense_path = f"{args.candidates_dir}/train_dense_candidates_K50.parquet"
    
    # CRITICAL FIX: read_parquet(n_rows=...) prevents the massive memory spike!
    bm_chunk = pl.read_parquet(dense_path, n_rows=args.rows)
    
    s1_join = s1_df.rename({"entity_id": "query_id", "name_norm": "s1_name", "address_norm": "s1_addr", "country_norm": "s1_country"})
    cand_join = cand_df.rename({"entity_id": "candidate_id", "name_norm": "cand_name", "address_norm": "cand_addr", "country_norm": "cand_country"})
    
    print(f"Joining... RAM={ram_gb():.2f}GB")
    bm_chunk = (bm_chunk
                 .join(s1_join, on="query_id", how="left")
                 .join(cand_join, on="candidate_id", how="left"))
    for col in ["s1_name", "s1_addr", "s1_country", "cand_name", "cand_addr", "cand_country"]:
        bm_chunk = bm_chunk.with_columns(pl.col(col).fill_null(""))
        
    s1_names = bm_chunk["s1_name"].to_list()
    cand_names = bm_chunk["cand_name"].to_list()
    s1_addrs = bm_chunk["s1_addr"].to_list()
    cand_addrs = bm_chunk["cand_addr"].to_list()
    
    # exact masks
    name_exact_mask = np.array([a == b for a, b in zip(s1_names, cand_names)])
    addr_exact_mask = np.array([a == b for a, b in zip(s1_addrs, cand_addrs)])
    need_name = ~name_exact_mask
    need_addr = ~addr_exact_mask
    s1_n_sub = [s1_names[i] for i in range(args.rows) if need_name[i]]
    cand_n_sub = [cand_names[i] for i in range(args.rows) if need_name[i]]
    s1_a_sub = [s1_addrs[i] for i in range(args.rows) if need_addr[i]]
    cand_a_sub = [cand_addrs[i] for i in range(args.rows) if need_addr[i]]
    
    n_sub = len(s1_n_sub)
    a_sub = len(s1_a_sub)
    print(f"Exact match short-circuit left {n_sub} name pairs and {a_sub} addr pairs.")
    print(f"Ready for tests. RAM={ram_gb():.2f}GB\n")

    def measure(name, func, *fargs):
        gc.collect()
        t0 = time.perf_counter()
        res = func(*fargs)
        t = time.perf_counter() - t0
        print(f"  {name:<25}: {t:5.2f}s  |  RAM: {ram_gb():.2f}GB")
        return res, t

    # 1. Implementation comparisons
    print("--- 1. Worker Scaling & Implementation Comparison ---")
    
    # Name JaroWinkler test
    print("Task: Name JaroWinkler")
    v2_1w, _ = measure("V2 (ListComp) W=1", v2_sim, s1_n_sub, cand_n_sub, JaroWinkler.normalized_similarity, 1)
    v2_2w, _ = measure("V2 (ListComp) W=2", v2_sim, s1_n_sub, cand_n_sub, JaroWinkler.normalized_similarity, 2)
    v2_4w, _ = measure("V2 (ListComp) W=4", v2_sim, s1_n_sub, cand_n_sub, JaroWinkler.normalized_similarity, 4)
    v3_1w, _ = measure("V3 (Map) W=1", v3_sim, s1_n_sub, cand_n_sub, JaroWinkler.normalized_similarity, 1)
    v3_2w, _ = measure("V3 (Map) W=2", v3_sim, s1_n_sub, cand_n_sub, JaroWinkler.normalized_similarity, 2)
    v3_4w, _ = measure("V3 (Map) W=4", v3_sim, s1_n_sub, cand_n_sub, JaroWinkler.normalized_similarity, 4)
    assert np.allclose(v2_1w, v3_4w), "Correctness mismatch!"

    # Token Jaccard test
    print("\nTask: Name Token Jaccard")
    t2_1w, _ = measure("V2 (Iterative) W=1", v2_tok, s1_n_sub, cand_n_sub, 1)
    t2_4w, _ = measure("V2 (Iterative) W=4", v2_tok, s1_n_sub, cand_n_sub, 4)
    t3_1w, _ = measure("V3 (Pre-Tokenize) W=1", v3_tok, s1_n_sub, cand_n_sub, 1)
    t3_4w, _ = measure("V3 (Pre-Tokenize) W=4", v3_tok, s1_n_sub, cand_n_sub, 4)
    assert np.allclose(t2_1w, t3_4w), "Correctness mismatch!"

    # 2. Feature Ablation (Runtime Contribution)
    print("\n--- 2. Feature Runtime Ablation (V3 Map, W=4) ---")
    _, t_njw = measure("name_jaro_winkler", v3_sim, s1_n_sub, cand_n_sub, JaroWinkler.normalized_similarity, 4)
    _, t_njar = measure("name_jaro", v3_sim, s1_n_sub, cand_n_sub, Jaro.normalized_similarity, 4)
    _, t_nlev = measure("name_levenshtein", v3_sim, s1_n_sub, cand_n_sub, Levenshtein.normalized_similarity, 4)
    _, t_ntok = measure("name_token_jaccard", v3_tok, s1_n_sub, cand_n_sub, 4)
    
    _, t_ajw = measure("addr_jaro_winkler", v3_sim, s1_a_sub, cand_a_sub, JaroWinkler.normalized_similarity, 4)
    _, t_alev = measure("addr_levenshtein", v3_sim, s1_a_sub, cand_a_sub, Levenshtein.normalized_similarity, 4)
    _, t_atok = measure("addr_token_jaccard", v3_tok, s1_a_sub, cand_a_sub, 4)
    
    total = t_njw + t_njar + t_nlev + t_ntok + t_ajw + t_alev + t_atok
    print(f"\nTotal String Compute Time (V3 W=4): {total:.2f}s")
    print(f"Throughput limit: {args.rows / total:.0f} rows/s")
    print(f"Extrapolated 126M time: {126_000_000 / (args.rows / total) / 3600:.2f} hours")
    
    print("\nCONCLUSION/RECOMMENDATIONS:")
    print("1. Compare the RAM for W=1 vs W=4. W=4 should use negligible extra RAM because GIL drops.")
    print("2. Compare V2 to V3. V3 Pre-Tokenize should be substantially faster.")
    print("3. Look at the ablation times. If name_jaro is just as slow as name_jw but adds no validation value, it's a prime removal candidate.")

if __name__ == "__main__":
    main()
