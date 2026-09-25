# Dataset Specification

## Paths and Formats
- All data files are **TSV format**. Always use `sep="\t"` when reading.
- **Kaggle Dataset Path** (Production): `/kaggle/input/datasets/lokeshgile/student-resource-amazonml`
- **Local Dataset Path** (Exploration/Testing): `data/student_resource/`

## Local Structure and File Sizes
```text
data/student_resource/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv (~210 MB)
│   │   ├── train_source2.tsv (~489 MB)
│   │   ├── train_source3.tsv (~503 MB)
│   │   └── train_ground_truth.tsv (~127 MB)
│   └── test/
│       ├── test_source1.tsv (~175 MB)
│       ├── test_source2.tsv (~509 MB)
│       └── test_source3.tsv (~506 MB)
├── utils/
│   └── validate_submission.py
├── Documentation_template.md
└── README.md
```

## Schema
### Source Files (1, 2, 3)
Columns: `entity_id`, `business_name`, `business_address`, `country`
- `entity_id`: Unique ID. Prefix indicates source (`S1-`, `S2-`, `S3-`).
- `business_name`: Business name (noisy).
- `business_address`: Business address (noisy).
- `country`: Country label (open-set string field, e.g., US, India, France).

### Ground Truth File
Columns: `source1_entity_id`, `matched_entity_ids` (comma-separated list of IDs from S2 and S3).
