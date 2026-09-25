import os
import time
import argparse
import polars as pl
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5E: Dense & Hybrid Recall Audit")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to train_ground_truth.tsv")
    parser.add_argument("--candidates-dir", type=str, default="data/candidates", help="Candidates directory")
    parser.add_argument("--k", type=int, default=10, help="The Top-K of the dense file to audit")
    return parser.parse_args()

def evaluate_recall(truth_df: pl.DataFrame, cand_df: pl.DataFrame, name: str):
    """
    Computes pair recall and query recall.
    truth_df: ['query_id', 'candidate_id']
    cand_df: ['query_id', 'candidate_id']
    """
    t0 = time.time()
    # Unique ground truth pairs
    truth_pairs = truth_df.select(["query_id", "candidate_id"]).unique()
    total_true_pairs = truth_pairs.height
    
    # Unique queries in ground truth
    total_queries = truth_df["query_id"].n_unique()
    
    # Ensure candidates are unique pairs
    cand_pairs = cand_df.select(["query_id", "candidate_id"]).unique()
    total_candidates = cand_pairs.height
    
    # Inner join to find matches
    matches = truth_pairs.join(cand_pairs, on=["query_id", "candidate_id"], how="inner")
    matched_pairs = matches.height
    
    # Query recall (queries that got AT LEAST one true candidate)
    queries_with_match = matches["query_id"].n_unique()
    
    pair_recall = matched_pairs / total_true_pairs if total_true_pairs > 0 else 0
    query_recall = queries_with_match / total_queries if total_queries > 0 else 0
    cands_per_query = total_candidates / total_queries if total_queries > 0 else 0
    
    print(f"\n--- {name} Recall ---")
    print(f"Total True Pairs: {total_true_pairs}")
    print(f"Pairs Found:      {matched_pairs} ({pair_recall*100:.2f}%)")
    print(f"Queries w/ Match: {queries_with_match} / {total_queries} ({query_recall*100:.2f}%)")
    print(f"Candidate Volume: {cands_per_query:.2f} cands/query (Total: {total_candidates})")
    print(f"Calculated in {time.time()-t0:.1f}s")
    
    return {
        "pair_recall": pair_recall,
        "query_recall": query_recall,
        "cands_per_query": cands_per_query,
        "total_candidates": total_candidates
    }

def main():
    args = parse_args()
    
    print("==================================================")
    print(f" PHASE 5E: HYBRID RECALL AUDIT (Top-{args.k})")
    print("==================================================")
    
    # 1. Load Ground Truth
    print(f"Loading Ground Truth from {args.ground_truth}...")
    if not os.path.exists(args.ground_truth):
        print(f"ERROR: Ground truth not found at {args.ground_truth}")
        return
        
    if args.ground_truth.endswith('.parquet'):
        gt_df = pl.read_parquet(args.ground_truth)
    else:
        # TSV format is: source1_entity_id \t matched_entity_ids (comma separated)
        gt_df = pl.read_csv(args.ground_truth, separator='\t').rename({
            "source1_entity_id": "query_id", 
            "matched_entity_ids": "candidate_id"
        })
        
        # Split the comma-separated candidate IDs into a list, then explode into rows
        gt_df = gt_df.with_columns(
            pl.col("candidate_id").str.split(",")
        ).explode("candidate_id")
        
    # Standardize types
    gt_df = gt_df.with_columns([
        pl.col("query_id").cast(pl.Utf8),
        pl.col("candidate_id").cast(pl.Utf8)
    ])
    
    # 2. Load Dense Candidates
    dense_path = os.path.join(args.candidates_dir, f"train_dense_candidates_K{args.k}.parquet")
    if not os.path.exists(dense_path):
        print(f"ERROR: Dense candidates not found at {dense_path}")
        return
        
    print(f"Loading Dense Candidates (K={args.k})...")
    dense_df = pl.read_parquet(dense_path, columns=["query_id", "candidate_id"]).with_columns([
        pl.col("query_id").cast(pl.Utf8),
        pl.col("candidate_id").cast(pl.Utf8)
    ])
    
    # 3. Load Exact Candidates (Phase 3)
    exact_path = os.path.join(args.candidates_dir, "train_exact_candidates.parquet")
    has_exact = os.path.exists(exact_path)
    
    if has_exact:
        print("Loading Exact Blocking Candidates...")
        exact_df = pl.read_parquet(exact_path, columns=["query_id", "candidate_id"]).with_columns([
            pl.col("query_id").cast(pl.Utf8),
            pl.col("candidate_id").cast(pl.Utf8)
        ])
    
    # 4. Evaluate
    results = {}
    
    # Dense Alone
    results["Dense"] = evaluate_recall(gt_df, dense_df, "Dense (IVFFlat)")
    
    # Exact Alone
    if has_exact:
        results["Exact"] = evaluate_recall(gt_df, exact_df, "Exact Blocking")
        
        # HYBRID (Exact ∪ Dense)
        print("\nComputing Union (Exact ∪ Dense)...")
        t0 = time.time()
        hybrid_df = pl.concat([exact_df, dense_df]).unique(subset=["query_id", "candidate_id"])
        print(f"Union computed in {time.time()-t0:.1f}s")
        
        results["Hybrid"] = evaluate_recall(gt_df, hybrid_df, "Hybrid (Exact ∪ Dense)")
        
    # Save Report
    report_path = os.path.join(args.candidates_dir, f"recall_audit_K{args.k}.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nAudit saved to {report_path}")

if __name__ == "__main__":
    main()
