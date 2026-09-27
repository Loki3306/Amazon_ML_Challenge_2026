# ML Challenge 2026: Business Entity Resolution Challenge
## Problem Statement & Production Documentation

---

## 1. Overview & Problem Definition

In large-scale commercial platforms, business identity data arrives from multiple independent sources — each contributing partial, noisy fragments of information about the same real-world entities. These fragments share no common identifiers, and the challenge of determining which records refer to the same business is known as **Entity Resolution (ER)**.

Your challenge is to build a Machine Learning solution that, given business records from **3 independent data sources** with noisy and inconsistent fields, determines which records across sources refer to the same real-world business entity.

- **Source 1 (`S1`)** is the deduplicated reference source.
- Your task is to find **all matching records from Source 2 (`S2`) and Source 3 (`S3`)** for each Source 1 entity.
- A Source 1 entity may match **zero, one, or many** records from Source 2 and Source 3.

---

## 2. File Format & Standards

> [!IMPORTANT]
> **All files in this challenge are tab-separated (`.tsv`), and all output submissions must be tab-separated too.**
> Tabs are used because business addresses and ID list columns contain commas.

Always read and write files with an explicit tab separator:

```python
import pandas as pd

# Reading TSV files
df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")

# Writing TSV files
df.to_csv("output/matching_results.tsv", sep="\t", index=False)
```

---

## 3. Data Description

Each source file (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`) contains the following columns:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `entity_id` | String | Unique identifier for the record. The prefix indicates the source: `S1-`, `S2-`, or `S3-`. |
| `business_name` | String | Name of the business entity (may contain abbreviations, legal suffixes, typos, transliterations). |
| `business_address` | String | Address of the business (may contain partial addresses, format variations, missing components, landmark references). |
| `country` | String | Country label for the record (`US`, `India`, `France`). |

> [!WARNING]
> **Open-Set Country Label (`France`):**
> - The **training** data covers `US` and `India`.
> - The **test** set additionally contains a third country, `France`, that does **not** appear in the training data.
> - Treat `country` as an open set of string labels: do **not** hard-code, filter, or one-hot your pipeline strictly to `{US, India}`. Every test entity — `France` included — must appear in your submission.

### Ground Truth File (`train_ground_truth.tsv`)
| Column Name | Type | Description |
| :--- | :--- | :--- |
| `source1_entity_id` | String | The `entity_id` of a Source 1 record. |
| `matched_entity_ids` | String | Comma-separated list of matching `entity_id`s from Source 2 and/or Source 3 (empty when the entity has no matches). |

---

## 4. Expected Noise Patterns

- **Name Variations:**
  - Abbreviations (`Corp` vs. `Corporation`, `Pvt` vs. `Private`, `Ltd` vs. `Limited`).
  - Legal suffix inconsistencies and omissions.
  - DBA / trade names.
  - Punctuation differences (`&` vs. `"and"`).
  - Word-order transpositions (`Apex Nippon` vs. `Nippon Apex`).
  - Typos / OCR errors.

- **Address Variations:**
  - Abbreviations (`Rd` vs. `Road`, `St` vs. `Street`, `Ave` vs. `Avenue`).
  - Transliteration variants.
  - Missing components (missing PIN code, missing state/city).
  - Landmark-based references (e.g., `Near SBI ATM`, `Near Pune Vidhyarthi Gruha`).
  - Municipal numbering formats (`Door No 4-4 C`, `Sr No 1420`).
  - Component reordering / permutations (`State, City, Street` vs. `Street, City, State`).

---

## 5. Dataset Structure & Files

### Training Files (`dataset/train/`)
1. `train_source1.tsv`: Source 1 training records (the deduplicated reference source).
2. `train_source2.tsv`: Source 2 training records.
3. `train_source3.tsv`: Source 3 training records.
4. `train_ground_truth.tsv`: Ground truth matching labels for the training set.

### Test Files (`dataset/test/`)
1. `test_source1.tsv`: Source 1 test records. Generate matches for every entity in this file.
2. `test_source2.tsv`: Source 2 test records.
3. `test_source3.tsv`: Source 3 test records.

---

## 6. Output Format & Validation

Your solution produces **two tab-separated files** placed in the `output/` directory:

### 1. `matching_results.tsv` (Leaderboard Scored File)
Final predicted entity matches:

| Column | Description |
| :--- | :--- |
| `source1_entity_id` | The `entity_id` of a Source 1 record |
| `matched_entity_ids` | Comma-separated list of matching `entity_id`s from Source 2 and/or Source 3 |

**Example:**
```tsv
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

**Formatting Rules:**
- Every Source 1 entity in the test set must have exactly one row.
- Leave `matched_entity_ids` empty for entities with no matches (singletons).
- No duplicate entity IDs within a single ID list.
- ID lists must only contain Source 2 or Source 3 IDs that exist in the test set.

---

### 2. `candidate_pairs.tsv` (Blocking / Candidate Set)
The candidate set produced by your blocking stage *before* your final ML model narrows it down:

| Column | Description |
| :--- | :--- |
| `source1_entity_id` | The `entity_id` of a Source 1 record |
| `candidate_entity_ids` | Comma-separated list of candidate `entity_id`s from Source 2 and/or Source 3 |

**Example:**
```tsv
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	S3-00004
S1-00003	
```

---

### Validation Script
Run the built-in validator before submitting:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 7. Evaluation Metric: Macro $F_{0.5}$ Score

Submissions are evaluated using **$F_{\beta}$ Score ($\beta = 0.5$)** — a precision-heavy metric that penalizes false merges (matching two different businesses) twice as heavily as missed matches.

### Formula
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- **Macro-Averaged:** $F_{0.5}$ is calculated per Source 1 entity, then averaged across **all** Source 1 entities in the evaluation set.
- **Singletons:** Included in the macro-average. A Source 1 entity with no true matches scores **1.0** when you correctly predict an empty list, and **0.0** when you predict any match for it.

---

## 8. Final Submission Package Structure

Every team submits a single zip archive formatted as follows:

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches (leaderboard submission)
│   └── candidate_pairs.tsv         # blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all source code
│       ├── README.md               # instructions to reproduce end-to-end
│       └── requirements.txt        # pinned dependencies
└── Documentation_template.md       # filled methodology document
```

---

## 9. Constraints & Fair Play Rules

1. **Strictly Prohibited: External Data Lookup**
   - **NO** commercial entity resolution APIs.
   - **NO** government database business lookups.
   - **NO** external geocoding APIs.
   - **NO** external data augmentation from internet sources.
   - Any violation results in **immediate disqualification**.
2. **Model Constraint:** Final model must be **MIT / Apache 2.0 licensed** and up to **8 Billion parameters**.
3. **Format Integrity:** Strict adherence to TSV headers, row counts, and ID constraints.

---

## 10. Key Strategies for High Performance

1. **Robust Candidate Generation (Blocking):** Determines your recall ceiling. Partition strictly by `country` and combine TF-IDF / N-gram sparse indexing + phonetic/token blocking.
2. **Precision Optimization:** With $F_{0.5}$, thresholding higher to avoid false positives yields substantial score gains.
3. **Singleton Handling:** Correctly identifying singletons (~5.6% of entities) is worth a full 1.0 on those instances.
4. **Generalization for France:** Ensure address and name preprocessing handles French diacritics and address tokens without hardcoded US/India assumptions.
