# Feature Engineering & Machine Learning Modeling

---

## 1. Feature Taxonomy for Entity Matching

Given a candidate pair $(S_1, S_k)$ where $S_k \in \{S_2, S_3\}$, we compute pairwise similarity signals across **Name**, **Address**, and **Meta-attributes**.

### 1.1 Business Name Similarity Features
- **String Distance Metrics:**
  - `Jaro-Winkler Similarity`: High precision for prefix matches.
  - `Levenshtein Distance / Ratio`: Normal edit distance.
  - `Damerau-Levenshtein`: Handles adjacent character transpositions.
- **Token Overlap & Reordering Metrics:**
  - `Token Set Ratio & Token Sort Ratio` (from `rapidfuzz`): Invariant to word order (e.g. *"XX Apex Nippon"* vs *"XX Nippon Apex"*).
  - `Token Jaccard & Dice Similarity`.
  - `Character 3-gram & 4-gram Cosine Similarity`.
  - `Prefix / Suffix Exact Match` (after stripping legal designations like LLC, Pvt Ltd, Inc, Corp, SA, SAS).
- **Phonetic Encodings:**
  - `Metaphone / Double Metaphone Match`: Invariant to spelling/phonetic typos.

### 1.2 Address Similarity Features
- **Token Overlaps:**
  - `Address Token Jaccard Similarity`.
  - `Address Token Sort Ratio`.
  - `Address Substring Inclusion Flag` (is one address completely contained in another?).
- **Location Components:**
  - `Postal Code Match`: Exact match (binary flag), 3-digit prefix match (regional area).
  - `House / Unit Number Match`: Extracted numeric sequences match.
  - `Street Name Edit Distance`.
  - `City / State Token Overlap`.
  - `Address Missingness Indicator`: Binary flag indicating if candidate address is missing/empty.

### 1.3 Structural & Pairwise Metadata Features
- `Length Difference`: $|len(\text{name}_1) - len(\text{name}_2)|$.
- `Token Count Ratio`: $\frac{\min(|tokens_1|, |tokens_2|)}{\max(|tokens_1|, |tokens_2|)}$.
- `Source Indicator`: Is candidate from `Source 2` or `Source 3`?
- `Blocking Rank / Score`: Cosine score from the TF-IDF / BM25 candidate generation stage.

---

## 2. Model Architecture Selection

```
                  ┌───────────────────────────────┐
                  │   Candidate Pairs (S1, S2/S3) │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │ 30+ Fast Pairwise Features    │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │  GBDT (LightGBM / CatBoost)   │
                  │  * Fast inference on 10M rows │
                  │  * Calibrated probabilities   │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │  Match Probability P(match)   │
                  └───────────────────────────────┘
```

### Why GBDT (LightGBM / CatBoost) over Heavy Transformers for Production Scale?
1. **Inference Speed:** Scoring ~20-50M candidate pairs on Kaggle takes under 2 minutes with LightGBM vs hours/days with cross-encoder transformers.
2. **Tabular Feature Power:** Edit distances, token overlaps, and n-gram similarities are dense, highly non-linear tabular signals where GBDT shines.
3. **Memory Footprint:** LightGBM models are a few megabytes and run purely in RAM without requiring multi-GPU setups.
4. **License & Parameter Limits:** Fits well within the competition constraints ($\le 8\text{B}$ parameters, MIT/Apache 2.0).

---

## 3. Generalization for Unseen Country (France)

Because France is not present in the training set:
- **Language-Agnostic Processing:** Use Unicode normalization (`unicodedata.normalize('NFKD', ...)` to strip accents: `é` $\rightarrow$ `e`).
- **Feature Robustness:** Features based on string similarities (Levenshtein, Jaccard, Token Sort Ratio) are language-invariant and generalize seamlessly to French entities.
- **Do NOT Train on Hardcoded Categoricals:** Do not encode US/Indian state names as explicit one-hot categorical features in the classifier.
