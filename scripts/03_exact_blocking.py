import os
import time
import argparse
import polars as pl
from datetime import datetime
import json
import sys

# Ensure imports work when run as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src")))

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3: Exact Blocking via Polars Joins")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Directory containing Canonical Parquet files")
    parser.add_argument("--split", type=str, default="train", help="Which split to run (train or test)")
    parser.add_argument("--output-dir", type=str, default="data/candidates", help="Directory to save candidate pairs")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory to save execution reports")
    return parser.parse_args()

def generate_exact_candidates(data_dir, split, output_dir, artifacts_dir):
    start_time = time.time()
    
    s1_path = os.path.join(data_dir, split, f"{split}_source1.parquet")
    s2_path = os.path.join(data_dir, split, f"{split}_source2.parquet")
    s3_path = os.path.join(data_dir, split, f"{split}_source3.parquet")
    
    if not all(os.path.exists(p) for p in [s1_path, s2_path, s3_path]):
        print(f"Required parquet files not found in {os.path.join(data_dir, split)}. Run Phase 2 first.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    print(f"[{split}] Loading Canonical Parquet files...")
    
    # We only need specific columns for blocking to save memory
    select_cols = ["entity_id", "source", "name_norm", "address_norm"]
    
    s1_lazy = pl.scan_parquet(s1_path).select(select_cols)
    s2_lazy = pl.scan_parquet(s2_path).select(select_cols)
    s3_lazy = pl.scan_parquet(s3_path).select(select_cols)
    
    # Combine S2 and S3 into a single searchable corpus
    corpus_lazy = pl.concat([s2_lazy, s3_lazy])
    
    print(f"[{split}] Performing Exact Name Match Join...")
    # 1. Exact Name Match (excluding empty strings)
    # Filter out empty names to avoid explosive Cartesian products
    s1_valid_names = s1_lazy.filter(pl.col("name_norm") != "")
    corpus_valid_names = corpus_lazy.filter(pl.col("name_norm") != "")
    
    name_matches = s1_valid_names.join(
        corpus_valid_names, 
        on="name_norm", 
        how="inner",
        suffix="_candidate"
    ).select([
        pl.col("entity_id").alias("query_id"),
        pl.col("entity_id_candidate").alias("candidate_id"),
        pl.col("source_candidate").alias("candidate_source"),
        pl.lit(True).alias("match_name")
    ])
    
    print(f"[{split}] Performing Exact Address Match Join...")
    # 2. Exact Address Match (excluding empty strings)
    s1_valid_addrs = s1_lazy.filter(pl.col("address_norm") != "")
    corpus_valid_addrs = corpus_lazy.filter(pl.col("address_norm") != "")
    
    addr_matches = s1_valid_addrs.join(
        corpus_valid_addrs, 
        on="address_norm", 
        how="inner",
        suffix="_candidate"
    ).select([
        pl.col("entity_id").alias("query_id"),
        pl.col("entity_id_candidate").alias("candidate_id"),
        pl.col("source_candidate").alias("candidate_source"),
        pl.lit(True).alias("match_address")
    ])
    
    print(f"[{split}] Consolidating Candidate Sets...")
    # 3. Union and aggregate
    # We want unique pairs (query_id, candidate_id) with boolean flags for why they matched
    # Using outer join on the pair allows us to keep all pairs and merge flags
    
    candidates = name_matches.join(
        addr_matches,
        on=["query_id", "candidate_id", "candidate_source"],
        how="full",
        coalesce=True
    ).fill_null(False)
    
    # Materialize (Polars will execute the graph)
    # We collect in streaming mode (if possible) or chunked. 
    # Exact joins can be large, but Polars handles memory well.
    print(f"[{split}] Executing graph and writing output...")
    output_path = os.path.join(output_dir, f"{split}_exact_candidates.parquet")
    
    # We collect instead of sink because full outer join doesn't always support streaming out of the box in older versions,
    # but let's try sink_parquet first, fallback to collect().write_parquet
    try:
        candidates.sink_parquet(output_path, compression="snappy")
    except Exception as e:
        print("sink_parquet failed (possibly due to full join), falling back to collect()...", e)
        candidates.collect().write_parquet(output_path, compression="snappy")
    
    processing_time = time.time() - start_time
    
    print(f"[{split}] Computing candidate statistics...")
    # Gather stats
    final_candidates = pl.scan_parquet(output_path)
    total_pairs = final_candidates.select(pl.len()).collect().item()
    
    unique_queries_covered = final_candidates.select(pl.col("query_id").n_unique()).collect().item()
    s1_total_rows = s1_lazy.select(pl.len()).collect().item()
    
    stats = {
        "split": split,
        "total_candidate_pairs": total_pairs,
        "unique_queries_with_candidates": unique_queries_covered,
        "total_s1_queries": s1_total_rows,
        "query_coverage_pct": round(unique_queries_covered / s1_total_rows * 100, 2) if s1_total_rows > 0 else 0,
        "avg_candidates_per_query": round(total_pairs / unique_queries_covered, 2) if unique_queries_covered > 0 else 0,
        "processing_time_sec": round(processing_time, 2),
        "output_size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 2)
    }
    
    print(f"[{split}] DONE.")
    print(json.dumps(stats, indent=2))
    return stats

def main():
    args = parse_args()
    print("==================================================")
    print("PHASE 3: EXACT BLOCKING (CANDIDATE GENERATION)")
    print("==================================================")
    
    reports = []
    
    for split in [args.split] if args.split != "both" else ["train", "test"]:
        stats = generate_exact_candidates(args.data_dir, split, args.output_dir, args.artifacts_dir)
        if stats:
            reports.append(stats)
            
    if reports:
        report_path = os.path.join(args.artifacts_dir, f"exact_blocking_report_{args.split}.json")
        with open(report_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "reports": reports
            }, f, indent=2)
        print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    main()
