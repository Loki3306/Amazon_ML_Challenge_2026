import os
import time
import json
import argparse
import yaml
import polars as pl
from datetime import datetime
import sys

# Ensure imports work when run as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src")))

from business_entity_resolution.data.schemas import SOURCE_SCHEMA
from business_entity_resolution.preprocessing.normalization import get_normalization_exprs

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Canonical Data Preparation (Executable on Kaggle)")
    parser.add_argument("--config", type=str, default="config/dataset.yaml", help="Path to dataset YAML config")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save canonical Parquet files")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory to save validation reports")
    return parser.parse_args()

def prepare_and_validate_source(source_name, input_path, output_dir, artifacts_dir):
    start_time = time.time()
    
    if not os.path.exists(input_path):
        print(f"Skipping {source_name}: input file {input_path} not found.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{source_name}.parquet")
    
    print(f"[{source_name}] Ingesting from {input_path}")
    
    # 1. Measure input rows (using lazy frame count for efficiency)
    lazy_df = pl.scan_csv(input_path, separator="\t", dtypes=SOURCE_SCHEMA, infer_schema_length=0, null_values=[""])
    
    # For reporting, we must materialize a small aggregation. 
    # But since we want to validate row counts, we will stream the dataset to Parquet first, 
    # then check the Parquet output.
    
    # Add source column natively
    # Extract just the split and source ID
    lazy_df = lazy_df.with_columns(pl.lit(source_name).alias("source"))
    
    # Apply normalization
    norm_exprs = get_normalization_exprs()
    lazy_df = lazy_df.with_columns(norm_exprs)
    
    # Ensure column order matches canonical expectation
    canonical_columns = [
        "entity_id", "source", "business_name", "business_address", "country",
        "name_norm", "address_norm", "country_norm"
    ]
    lazy_df = lazy_df.select(canonical_columns)
    
    # STREAM TO PARQUET
    # sink_parquet requires streaming to be available for the operations,
    # and select/with_columns are fully streaming-supported in Polars.
    lazy_df.sink_parquet(output_path, compression="snappy")
    
    processing_time = time.time() - start_time
    
    print(f"[{source_name}] Validating output...")
    # 2. Validation
    # We now read the raw and processed to validate
    # Note: scan_csv count might fail if file has bad lines, but we assume it's clean based on Phase 1
    input_rows = pl.scan_csv(input_path, separator="\t", dtypes=SOURCE_SCHEMA, infer_schema_length=0, null_values=[""]).select(pl.len()).collect().item()
    
    out_lazy = pl.scan_parquet(output_path)
    output_rows = out_lazy.select(pl.len()).collect().item()
    
    # Check nulls in normalized columns (should be 0 because we filled with "")
    null_counts = out_lazy.select([
        pl.col("name_norm").is_null().sum().alias("name_norm_nulls"),
        pl.col("address_norm").is_null().sum().alias("address_norm_nulls")
    ]).collect().to_dicts()[0]
    
    # Validation Rules
    assert input_rows == output_rows, f"Row count mismatch: {input_rows} != {output_rows}"
    assert null_counts["name_norm_nulls"] == 0, "Nulls found in name_norm"
    assert null_counts["address_norm_nulls"] == 0, "Nulls found in address_norm"
    
    out_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    stats = {
        "source": source_name,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "rows_dropped": input_rows - output_rows,
        "processing_time_sec": round(processing_time, 2),
        "throughput_rows_per_sec": round(input_rows / processing_time, 2) if processing_time > 0 else 0,
        "output_size_mb": round(out_size_mb, 2),
        "validation_passed": True
    }
    print(f"[{source_name}] DONE. Throughput: {stats['throughput_rows_per_sec']}/sec")
    
    return stats


def main():
    args = parse_args()
    print("==================================================")
    print("PHASE 2: CANONICAL DATA PREPARATION")
    print("==================================================")
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    reports = []
    
    for split in ["train", "test"]:
        if split not in config:
            continue
        for source_key in ["source1", "source2", "source3"]:
            if source_key in config[split]:
                source_name = f"{split}_{source_key}"
                input_path = config[split][source_key]
                
                stats = prepare_and_validate_source(
                    source_name=source_name,
                    input_path=input_path,
                    output_dir=os.path.join(args.output_dir, split),
                    artifacts_dir=args.artifacts_dir
                )
                if stats:
                    reports.append(stats)
                    
    # Save the aggregated report
    report_path = os.path.join(args.artifacts_dir, "prepare_data_report.json")
    os.makedirs(args.artifacts_dir, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "environment": "Kaggle CPU" if "KAGGLE_KERNEL_RUN_TYPE" in os.environ else "Local/Unknown",
            "reports": reports
        }, f, indent=2)
        
    print(f"\nAll preparation finished. Report saved to {report_path}.")

if __name__ == "__main__":
    main()
