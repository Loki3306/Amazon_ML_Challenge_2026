# Phase 2: Canonical Data Preparation (Kaggle Execution)

## Purpose
This script transforms raw TSV files into partitioned/chunked Canonical Parquet files with deterministic, lightweight text normalization. It operates entirely on the CPU using Polars streaming (`sink_parquet`), ensuring it scales to 10M+ rows without memory explosion.

## Kaggle Environment
- **Environment**: Kaggle Notebook (Python)
- **Compute**: CPU (GPU not required for this phase)
- **RAM**: ~30GB (Standard Kaggle CPU kernel)
- **Required Packages**:
  ```bash
  pip install polars pyarrow pyyaml
  ```

## Input Expectations
You must attach the Kaggle dataset to the kernel.
Assuming the dataset is mounted at `/kaggle/input/dataset/`, create or point to a `dataset.yaml` config file reflecting this:

```yaml
train:
  source1: "/kaggle/input/dataset/train/train_source1.tsv"
  source2: "/kaggle/input/dataset/train/train_source2.tsv"
  source3: "/kaggle/input/dataset/train/train_source3.tsv"
test:
  source1: "/kaggle/input/dataset/test/test_source1.tsv"
  source2: "/kaggle/input/dataset/test/test_source2.tsv"
  source3: "/kaggle/input/dataset/test/test_source3.tsv"
```

## Execution Command
Execute the script from the project root:

```bash
python scripts/02_prepare_data.py \
    --config config/dataset.yaml \
    --output-dir /kaggle/working/data/processed \
    --artifacts-dir /kaggle/working/artifacts
```

## Expected Outputs
The script streams the processed data and outputs to `/kaggle/working/data/processed/`:
- `train/train_source1.parquet`
- `train/train_source2.parquet`
...
And generates a validation report:
- `/kaggle/working/artifacts/prepare_data_report.json`

## Disk Requirements
Raw TSVs total ~2.5GB. Snappy-compressed Parquet files will total roughly 1GB - 1.5GB. Kaggle's `/kaggle/working` directory has 20GB of space, which is more than sufficient.

## Artifact Preservation
After execution in the Kaggle Kernel, zip or save the `/kaggle/working/data/processed/` folder as a new Kaggle Dataset so it can be mounted directly in Phase 3+ (avoiding repeating this computation).
