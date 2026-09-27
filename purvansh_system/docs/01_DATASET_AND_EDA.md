# Dataset Analysis & Exploratory Data Findings

---

## 1. Data File Summary & Scale

The challenge consists of **~24.2 million business records** across 3 data sources and 2 splits (Train & Test).

| File Name | Split | Records (Rows) | File Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| `train_source1.tsv` | Train | **2,206,821** | 200.34 MB | Reference entities (deduplicated) |
| `train_source2.tsv` | Train | **5,034,616** | 466.63 MB | Noisy Source 2 records |
| `train_source3.tsv` | Train | **5,285,603** | 480.37 MB | Noisy Source 3 records |
| `train_ground_truth.tsv` | Train | **2,206,821** | 121.13 MB | Reference S1 $\rightarrow$ matched S2/S3 mapping |
| `test_source1.tsv` | Test | **1,732,544** | 166.91 MB | Target S1 reference records to resolve |
| `test_source2.tsv` | Test | **4,887,273** | 485.86 MB | Candidate Source 2 pool |
| `test_source3.tsv` | Test | **5,082,316** | 482.56 MB | Candidate Source 3 pool |

---

## 2. Country Breakdown & The France Generalization Shift

```
[Training Data: 12,527,040 Total Records]
├── United States (US): 60.0% (~7.51M records)
└── India (India):       40.0% (~5.01M records)

[Test Data: 11,702,133 Total Records]
├── India (India):       46.8% (~5.53M records)
├── United States (US): 38.3% (~4.48M records)
└── France (France):     14.9% (~1.69M records)  <-- Unseen in Training!
```

### Critical Findings on Country:
1. **Intra-Country Strictness:** In the training ground truth, **100% of entity matches are strictly within the same country**. There are 0 cross-country matches. Country partitioning is a **zero-recall-loss hard blocking constraint**.
2. **The "France" Zero-Shot Shift:**
   - France records appear **only** in the test set.
   - **Do not** use country-specific hardcoded lists (e.g. standard US state lists or Indian state lists) as hard barriers for France.
   - Preprocessing must handle French characters (`é`, `è`, `ê`, `à`, `ç`, `ô`, `î`) and French address tokens (`Rue`, `Boulevard`, `Avenue`, `Impasse`, `Cedex`, 5-digit postal codes).

---

## 3. Ground Truth Properties & Distribution

From `train_ground_truth.tsv` (2,206,821 reference entities):

- **Singletons (0 Matches):** `123,247` entities (**5.58%**)
  - In evaluation ($Macro\text{ }F_{0.5}$), correctly leaving a singleton blank yields a score of **1.0**. Matching a wrong entity to a singleton yields **0.0**.
- **Entities with Matches ($\ge 1$):** `2,083,574` entities (**94.42%**)
- **Source Contribution:**
  - Matched S2 Records: **3,693,619** (48.36%)
  - Matched S3 Records: **3,944,746** (51.64%)
- **Match Cardinality:**
  - S2 / S3 records map **1-to-1** to at most one S1 record in ground truth.
  - S1 maps **1-to-many** to {S2, S3}.

### Distribution of Matches per S1 Record:
| Number of Matches | Entity Count | % of S1 Records |
| :---: | :---: | :---: |
| **0** (Singletons) | 123,247 | 5.58% |
| **1** | 119,157 | 5.40% |
| **2** | 375,212 | 17.00% |
| **3** | 530,841 | 24.05% *(Peak)* |
| **4** | 484,115 | 21.94% |
| **5** | 321,957 | 14.59% |
| **6** | 164,868 | 7.47% |
| **7** | 63,968 | 2.90% |
| **8** | 18,680 | 0.85% |
| **9 – 11** | 4,776 | 0.22% *(Max = 11)* |

---

## 4. Missing Values & Field Completeness

| Column | Missing in S1 (Train / Test) | Missing in S2 (Train / Test) | Missing in S3 (Train / Test) |
| :--- | :---: | :---: | :---: |
| `entity_id` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `business_name` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `country` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `business_address` | 0 (0.0%) | **~3.3%** (~169k / ~129k) | **~3.3%** (~176k / ~136k) |

> [!TIP]
> When `business_address` is missing in candidate S2/S3 records, matching must rely on strong name similarity, legal entity stripping, and token phonetic matching.

---

## 5. Noise Patterns & Real Examples from Dataset

```
[Example 1 - Indian Entity with Address Formatting Noise]
S1: Celestial Memorial Trust | No. 35, Brentwood Apartments, Defence Colony, 2Nd Main, Indiranagar, Bangalore, Karnataka
S2: Celestial Memorial       | 35 , BRENTWOOD APARTMENTS, DEFENCE COLONY, 2ND MAIN, INDIRANAGAR, BANGALORE, Karnataka
Notes: Legal entity suffix dropped ("Trust"), address capitalization normalized.

[Example 2 - US Entity with Typo in Name & State Abbreviation]
S1: Gatewood's Iron Works  | 4500 Dewey Avenue, Unit Unit 16, Greece, NY
S3: Gatewood's Ihrno Works | 4500 Dewey Ave, Unit Unit 16, Rochester, New York
Notes: Character transposition typo in name ("Ihrno" vs "Iron"), Street abbreviation ("Ave" vs "Avenue"), City/State variation ("Rochester, New York" vs "Greece, NY").

[Example 3 - Token Inversion in Business Name]
S1: XX Apex Nippon | 5604 Brooklyn Avenue, Kansas City, MO
S2: XX Nippon Apex | BROOKLYN AVENUE, KANSAS CITY, MO
Notes: Name tokens swapped ("Apex Nippon" vs "Nippon Apex"), house number missing in S2.

[Example 4 - Address Inversion]
S1: Peridyn Chain LLC | NY, Brookhaven, 6 Heron Path
S2: Peridyn Chain     | 6-D HERON PATH, CORAM, NY
Notes: S1 starts with State/Town, S2 starts with Street Number/Unit.
```
