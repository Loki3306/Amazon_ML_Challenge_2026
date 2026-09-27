# Submission, Validation & Compliance Guide

---

## 1. Submission Deliverables & Formats

Your final pipeline produces **two tab-separated files**:

```
output/
├── matching_results.tsv   # (Required) Scored on the Leaderboard
└── candidate_pairs.tsv    # (Required) Blocking candidates fed to matching model
```

### 1.1 `matching_results.tsv`
- Header: `source1_entity_id\tmatched_entity_ids`
- Must contain exactly one row for **every** entity in `test_source1.tsv` (1,732,544 rows).
- Comma-separated list of matches with **no spaces, no quotes**.
- Leave second column completely blank if no matches exist (singletons).

**Example:**
```tsv
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

### 1.2 `candidate_pairs.tsv`
- Header: `source1_entity_id\tcandidate_entity_ids`
- Must contain all candidate pairs considered by your blocking pipeline.
- Every matched ID in `matching_results.tsv` **must be a subset** of IDs in `candidate_pairs.tsv`.

---

## 2. Local Submission Validation

Always run the official validation script before uploading to the leaderboard:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### Validation Checklist:
- [x] File is tab-separated (`.tsv`) and UTF-8 encoded.
- [x] Header matches exact casing (`source1_entity_id`, `matched_entity_ids`).
- [x] Exactly 1,732,544 rows (matches `test_source1.tsv` count).
- [x] Only `S2-` and `S3-` prefixes in match columns (no `S1-` self matches).
- [x] No duplicate IDs within any row's comma-separated list.
- [x] No duplicate `source1_entity_id` rows.

---

## 3. Final Submission Package Structure

At competition conclusion, every team must submit a single zip archive:

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── blocking.py
│       │   ├── feature_extraction.py
│       │   ├── train.py
│       │   └── infer.py
│       ├── README.md            # Exact end-to-end instructions
│       └── requirements.txt     # Pinned Python package dependencies
└── Documentation_template.md    # Completed methodology write-up
```

---

## 4. Fair-Play & Rule Compliance Checklist

| Rule | Status | Note |
| :--- | :---: | :--- |
| **No External APIs / Lookups** | **Mandatory** | No Google Maps API, Nominatim, OpenStreetMap, or government entity registry lookups. |
| **Model Size Limit** | **Mandatory** | Max 8 Billion parameters. |
| **Open Source License** | **Mandatory** | Pretrained weights/libraries must be MIT / Apache 2.0. |
| **Reproducibility** | **Mandatory** | The code inside `code/business_entity_resolution/` must regenerate the outputs from scratch. |
