# Amazon ML Challenge 2026 - Business Entity Resolution Challenge

This is the main repository for the Business Entity Resolution Challenge.

## Project Goal
Build a reproducible, compliant, high-quality entity-resolution system whose candidate generation, matching model, decision rules, evaluation, and submission pipeline are empirically justified.

## Data Setup
The dataset is very large and is **not** included in this repository. To run experiments locally, you must manually download and extract the dataset.

1. Download the `student_resource` data from the Kaggle competition page.
2. Create a folder named `data/` in the root of this project.
3. Extract the contents so that your structure looks like this:
   ```text
   data/student_resource/
   ├── dataset/
   │   ├── train/
   │   └── test/
   └── utils/
   ```
*Note: The `data/` directory is explicitly ignored in `.gitignore` to prevent committing massive TSV files.*
