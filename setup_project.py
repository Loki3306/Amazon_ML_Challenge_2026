import os
import textwrap

files_to_create = {
    "README.md": """
    # Amazon ML Challenge 2026 - Business Entity Resolution Challenge
    
    This is the main repository for the Business Entity Resolution Challenge.
    
    ## Project Goal
    Build a reproducible, compliant, high-quality entity-resolution system whose candidate generation, matching model, decision rules, evaluation, and submission pipeline are empirically justified.
    """,
    
    "docs/00_project_overview.md": """
    # Project Overview
    
    This project tackles the **Amazon ML Challenge 2026 — Business Entity Resolution Challenge**.
    We are tasked with determining which records from Source 2 and Source 3 correspond to each deduplicated entity in Source 1.
    
    ## Core Principles
    - **Engineering first**: Correctness, reproducibility, maintainability.
    - **Baseline first**: Always establish a simple baseline before adding complexity.
    - **Data-driven**: Understand the dataset before modeling.
    """,
    
    "docs/01_problem_statement.md": """
    # Problem Statement
    
    The challenge is a **Business Entity Resolution** problem. Business identity data arrives from three independent sources with noisy and inconsistent representations.
    
    - **Source 1** is the deduplicated reference source.
    - We must map each Source 1 entity to zero or more entities in **Source 2** and **Source 3**.
    - This is a one-to-many entity-resolution problem requiring explicit candidate-generation/blocking.
    """,
    
    "docs/02_dataset_specification.md": """
    # Dataset Specification
    
    ## Paths and Formats
    - All data files are TSV format. Always use `sep="\\t"`.
    - Kaggle Dataset Path: `/kaggle/input/datasets/lokeshgile/student-resource-amazonml`
    
    ## Schema
    - `entity_id`: Unique ID. Prefix indicates source (`S1-`, `S2-`, `S3-`).
    - `business_name`: Business name (noisy).
    - `business_address`: Business address (noisy).
    - `country`: Country label (open-set string field, e.g., US, India, France).
    """,
    
    "docs/03_submission_specification.md": """
    # Submission Specification
    
    The final submission must be a ZIP file containing:
    - `output/matching_results.tsv` (zero, one, or more matches per S1 entity)
    - `output/candidate_pairs.tsv` (must contain all final matches)
    - `code/business_entity_resolution/src/` (reproducible code)
    - `code/business_entity_resolution/README.md` and `requirements.txt`
    - `Documentation_template.md`
    
    ## Requirements
    - Every test S1 entity must appear exactly once.
    - No duplicate Source 1 rows.
    - No self-matches (S1 to S1).
    """,
    
    "docs/04_evaluation.md": """
    # Evaluation
    
    The evaluation metric is **F_0.5 (macro-averaged)** across all Source 1 entities.
    
    - **Formula**: `(1.25 * Precision * Recall) / (0.25 * Precision + Recall)`
    - This metric heavily penalizes false positives (false merges).
    - True singletons correctly predicted yield 1.0. Incorrectly predicting a match for a singleton yields 0.0.
    """,
    
    "docs/05_constraints_and_fair_play.md": """
    # Constraints and Fair Play
    
    ## STRICT Rules
    - **No external lookups**: Prohibited use of commercial APIs, internet search, external databases, or geocoding.
    - **Model limitations**: Must be MIT or Apache 2.0 licensed, and must have <= 8 billion parameters.
    - **Environment isolation**: Antigravity is for writing code. Kaggle is the exclusive environment for execution, training, and testing.
    """,
    
    "docs/06_experiment_protocol.md": """
    # Experiment Protocol
    
    Every experiment must be reproducible and documented.
    
    ## Record for each experiment
    - ID, Date, Code version
    - Parameters, features, blocking strategy
    - Validation methodology & metrics
    - Runtime & memory
    - Results & observations
    """,
    
    "docs/07_architecture.md": """
    # Architecture
    
    ## Execution Boundary
    - **Antigravity**: Write code, structure modules, design logic.
    - **Kaggle**: Read dataset, execute experiments, train, infer.
    
    ## Code Organization
    Separation of concerns across: I/O, preprocessing, normalization, blocking, feature generation, matching, evaluation, inference, and submission.
    """,
    
    "docs/08_decisions_and_assumptions.md": """
    # Decisions and Assumptions
    
    ## FACT vs ASSUMPTION
    - Maintain a clear distinction between facts, assumptions, hypotheses, and experimental results.
    - (To be updated as the project progresses)
    """
}

# Create docs
for filepath, content in files_to_create.items():
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(content).strip() + "\n")

# Create python package skeleton
pkg_dir = "code/business_entity_resolution/src/business_entity_resolution"
os.makedirs(pkg_dir, exist_ok=True)

modules = [
    "__init__.py", "config.py", "io.py", "preprocessing.py", 
    "normalization.py", "blocking.py", "features.py", "matching.py", 
    "evaluation.py", "inference.py", "submission.py"
]

for mod in modules:
    with open(os.path.join(pkg_dir, mod), "w", encoding="utf-8") as f:
        f.write(f'"""Module {mod}."""\n')

# Create code README and requirements
with open("code/business_entity_resolution/README.md", "w", encoding="utf-8") as f:
    f.write("# Source Code\n")
with open("code/business_entity_resolution/requirements.txt", "w", encoding="utf-8") as f:
    f.write("pandas\nnumpy\nscikit-learn\n")

print("Project foundation created successfully.")
