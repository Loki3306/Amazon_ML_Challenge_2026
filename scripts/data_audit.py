import os
import json
import yaml
import hashlib
import time
from datetime import datetime
import polars as pl

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "dataset.yaml")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

SOURCE_SCHEMA = {
    "entity_id": pl.Utf8,
    "business_name": pl.Utf8,
    "business_address": pl.Utf8,
    "country": pl.Utf8,
}
GT_SCHEMA = {
    "source1_entity_id": pl.Utf8,
    "matched_entity_ids": pl.Utf8,
}

def analyze_file_stats(path):
    abs_path = os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(abs_path):
        return {"status": "missing", "path": path}
    
    size = os.path.getsize(abs_path)
    # Checksum
    h = hashlib.sha256()
    with open(abs_path, 'rb') as f:
        # Read the first 10MB only for speed in this audit, or full file if preferred.
        # We will read the whole file to be thorough.
        while chunk := f.read(10 * 1024 * 1024):
            h.update(chunk)
            
    return {
        "status": "exists",
        "path": path,
        "size_bytes": size,
        "sha256": h.hexdigest()
    }

def profile_source(df_lazy, name):
    print(f"Profiling {name}...")
    
    # Row count
    row_count = df_lazy.select(pl.count()).collect().item()
    
    # Missingness & Uniqueness
    # We do a single collect for column stats
    stats_expr = []
    for col in SOURCE_SCHEMA.keys():
        stats_expr.extend([
            pl.col(col).is_null().sum().alias(f"{col}_null_count"),
            pl.col(col).n_unique().alias(f"{col}_unique_count"),
        ])
    
    stats_df = df_lazy.select(stats_expr).collect().to_dicts()[0]
    
    # Country distribution
    country_dist = df_lazy.group_by("country").count().sort("count", descending=True).collect()
    
    # Length distributions for string columns
    lengths_df = df_lazy.select([
        pl.col("business_name").str.len_bytes().alias("name_len"),
        pl.col("business_address").str.len_bytes().alias("addr_len")
    ]).select([
        pl.col("name_len").mean().alias("name_len_mean"),
        pl.col("name_len").quantile(0.5).alias("name_len_p50"),
        pl.col("name_len").quantile(0.95).alias("name_len_p95"),
        pl.col("addr_len").mean().alias("addr_len_mean"),
        pl.col("addr_len").quantile(0.5).alias("addr_len_p50"),
        pl.col("addr_len").quantile(0.95).alias("addr_len_p95"),
    ]).collect().to_dicts()[0]

    return {
        "row_count": row_count,
        "stats": stats_df,
        "lengths": lengths_df,
        "countries": country_dist.to_dicts()
    }

def analyze_ground_truth(path):
    print("Analyzing ground truth...")
    abs_path = os.path.join(PROJECT_ROOT, path)
    df = pl.read_csv(abs_path, separator="\t", dtypes=GT_SCHEMA, infer_schema_length=0, null_values=[""])
    
    # Calculate cardinality
    # matched_entity_ids is comma separated. If null, 0 matches.
    df = df.with_columns(
        pl.col("matched_entity_ids").fill_null("").str.split(",").list.len().alias("match_count")
    ).with_columns(
        pl.when(pl.col("matched_entity_ids") == "").then(0).otherwise(pl.col("match_count")).alias("match_count")
    )
    
    total = df.height
    counts = df.group_by("match_count").len().sort("match_count")
    
    stats = {
        "total_s1": total,
        "distribution": counts.to_dicts(),
        "mean_matches": df.select(pl.col("match_count").mean()).item(),
        "max_matches": df.select(pl.col("match_count").max()).item()
    }
    return stats

def main():
    start_time = time.time()
    print("Starting Phase 1 Dataset Audit...")
    os.makedirs(os.path.join(PROJECT_ROOT, "scripts"), exist_ok=True)
    
    config = load_config()
    
    manifest = {
        "dataset_version": "1.0",
        "analysis_timestamp": datetime.now().isoformat(),
        "files": {}
    }
    
    profiles = {}
    
    for split in ["train", "test"]:
        profiles[split] = {}
        for source in ["source1", "source2", "source3"]:
            path = config[split][source]
            file_stats = analyze_file_stats(path)
            manifest["files"][f"{split}_{source}"] = file_stats
            
            if file_stats["status"] == "exists":
                df_lazy = pl.scan_csv(os.path.join(PROJECT_ROOT, path), separator="\t", dtypes=SOURCE_SCHEMA, infer_schema_length=0, null_values=[""])
                profiles[split][source] = profile_source(df_lazy, f"{split}_{source}")
                
                # Compute frequencies for source2 and source3 (the corpus)
                if source in ["source2", "source3"]:
                    df_lazy.group_by("business_name").count().sort("count", descending=True).collect().write_parquet(
                        os.path.join(ARTIFACTS_DIR, f"{split}_{source}_name_frequency.parquet")
                    )
                    df_lazy.group_by("business_address").count().sort("count", descending=True).collect().write_parquet(
                        os.path.join(ARTIFACTS_DIR, f"{split}_{source}_address_frequency.parquet")
                    )

    if "ground_truth" in config["train"]:
        gt_path = config["train"]["ground_truth"]
        manifest["files"]["train_ground_truth"] = analyze_file_stats(gt_path)
        gt_stats = analyze_ground_truth(gt_path)
        profiles["ground_truth"] = gt_stats
        
    # Write artifacts
    with open(os.path.join(ARTIFACTS_DIR, "dataset_manifest.yaml"), "w") as f:
        yaml.dump(manifest, f)
        
    with open(os.path.join(ARTIFACTS_DIR, "dataset_profile.json"), "w") as f:
        json.dump(profiles, f, indent=2)
        
    # Generate Markdown Guide
    guide_path = os.path.join(DOCS_DIR, "DATASET_GUIDE.md")
    with open(guide_path, "w") as f:
        f.write("# Dataset Overview\n\n")
        f.write("This is the canonical dataset guide automatically generated by `scripts/data_audit.py`.\n\n")
        f.write("## Ground Truth Distribution\n")
        f.write("```json\n" + json.dumps(profiles.get("ground_truth", {}), indent=2) + "\n```\n\n")
        
        f.write("## Train Profile\n")
        f.write("```json\n" + json.dumps(profiles.get("train", {}), indent=2) + "\n```\n\n")
        
        f.write("## Test Profile\n")
        f.write("```json\n" + json.dumps(profiles.get("test", {}), indent=2) + "\n```\n\n")
        
    print(f"Audit complete in {time.time() - start_time:.2f} seconds.")
    print("Artifacts saved to artifacts/")

if __name__ == "__main__":
    main()
