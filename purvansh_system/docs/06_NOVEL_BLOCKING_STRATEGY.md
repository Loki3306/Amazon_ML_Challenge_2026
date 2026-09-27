# Novel Multi-Resolution Blocking & Candidate Generation Strategy

---

## 1. Why Blocking Is the Most Critical Stage

The naive comparison space for this challenge is astronomical:

$$|S_1| \times (|S_2| + |S_3|) = 1{,}732{,}544 \times 9{,}969{,}589 \approx 17.3 \text{ Trillion pairs}$$

No classifier — however fast — can score 17 trillion pairs. **Blocking** reduces this to a manageable candidate set by filtering out obviously non-matching pairs *before* the ML model ever sees them.

> **The Iron Law of Blocking:** Any true match that is not in the candidate set is **permanently lost**. Blocking determines your **Recall Ceiling** — the maximum recall your entire pipeline can ever achieve.

### Design Targets

| Metric | Target | Rationale |
| :--- | :---: | :--- |
| Blocking Recall ($R_{\text{block}}$) | ≥ 98.0% | Preserve nearly all true matches |
| Reduction Ratio ($RR$) | ≥ 99.999% | From 17T → ~25M candidate pairs |
| Avg Candidates per S1 | 15–30 | Tractable for feature computation |

$$R_{\text{block}} = \frac{|\text{True Matches} \cap \text{Candidate Set}|}{|\text{True Matches}|}$$

$$RR = 1 - \frac{|\text{Candidate Pairs}|}{|S_1| \times (|S_2| + |S_3|)}$$

---

## 2. Foundational Step: Country Hard Partitioning

Our EDA on training data showed **100% of ground truth matches are intra-country** — zero cross-country matches. This is the single most powerful blocking constraint:

```python
# Partition all data by country FIRST
for country in ['US', 'India', 'France']:
    s1_country = s1_df[s1_df['country'] == country]
    s2_country = s2_df[s2_df['country'] == country]
    s3_country = s3_df[s3_df['country'] == country]
    candidates = run_blocking(s1_country, s2_country, s3_country)
```

**Immediate Reduction:**
- US: 663K × 3.8M = 2.5T → partitioned
- India: 810K × 4.7M = 3.8T → partitioned
- France: 259K × 1.4M = 0.4T → partitioned

Each partition is processed independently, also solving memory constraints on Kaggle.

---

## 3. Multi-Resolution Blocking Architecture (4 Passes)

We use **4 complementary blocking passes**, each operating at a different resolution. The **union** of all passes forms the final candidate set.

```
┌──────────────────────────────────────────────────────────┐
│              MULTI-RESOLUTION BLOCKING                    │
│                                                          │
│  ┌─────────────────┐  ┌─────────────────┐               │
│  │ Pass 1: Token   │  │ Pass 2: N-Gram  │               │
│  │ Inverted Index  │  │ TF-IDF Top-K    │               │
│  │ (~70% recall)   │  │ (~85% recall)   │               │
│  └────────┬────────┘  └────────┬────────┘               │
│           │                    │                         │
│  ┌────────┴────────┐  ┌───────┴─────────┐               │
│  │ Pass 3: Phone-  │  │ Pass 4: Address │               │
│  │ tic + Postal    │  │ First Fallback  │               │
│  │ (~75% recall)   │  │ (gap filler)    │               │
│  └────────┬────────┘  └───────┬─────────┘               │
│           │                    │                         │
│           └──────────┬─────────┘                         │
│                      ▼                                   │
│            UNION & DEDUPLICATE                           │
│           (~98%+ combined recall)                        │
└──────────────────────────────────────────────────────────┘
```

### Pass 1: Country-Partitioned Exact Token Inverted Index

**Goal:** Catch all pairs that share ≥ 2 significant name tokens.

**Steps:**
1. **Normalize business names:**
   ```python
   import re
   
   LEGAL_SUFFIXES = r'\b(inc|llc|ltd|pvt|corp|co|sa|sas|sarl|gmbh|limited|private|corporation|incorporated|company)\b'
   
   def normalize_name(name):
       name = name.lower().strip()
       name = re.sub(LEGAL_SUFFIXES, '', name)
       name = re.sub(r'[^\w\s]', ' ', name)  # remove punctuation
       name = re.sub(r'\s+', ' ', name).strip()
       return name
   ```
2. **Build inverted index** on S2+S3 records (within each country):
   ```python
   from collections import defaultdict
   
   inverted_index = defaultdict(set)  # token -> set of entity_ids
   token_doc_freq = defaultdict(int)  # token -> document frequency
   
   for eid, name in s2s3_records:
       tokens = normalize_name(name).split()
       for t in tokens:
           inverted_index[t].add(eid)
           token_doc_freq[t] += 1
   ```
3. **Query each S1 entity:** Retrieve S2/S3 entities sharing ≥ 2 **significant** tokens (document frequency < 10,000 to skip stopword-like tokens such as "the", "services", "india"):
   ```python
   MIN_SHARED_TOKENS = 2
   MAX_DOC_FREQ = 10000
   
   for s1_id, s1_name in s1_records:
       tokens = [t for t in normalize_name(s1_name).split() 
                 if token_doc_freq[t] < MAX_DOC_FREQ]
       candidate_counts = defaultdict(int)
       for t in tokens:
           for cid in inverted_index[t]:
               candidate_counts[cid] += 1
       candidates = {cid for cid, cnt in candidate_counts.items() 
                     if cnt >= MIN_SHARED_TOKENS}
   ```

**Characteristics:**
- **Speed:** Pure dictionary lookups — processes 800K S1 entities in ~5 minutes.
- **Expected Recall:** ~70% (misses typos, abbreviation-only differences, name reorderings where tokens are heavily corrupted).

---

### Pass 2: Character N-Gram TF-IDF with Sparse Top-K Retrieval

**Goal:** Catch fuzzy matches where individual characters are corrupted but the overall shape is preserved.

**Steps:**
1. **Build TF-IDF vectors** on character 3-grams of normalized `business_name` + first 3 tokens of `business_address`:
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   import scipy.sparse as sp
   
   def make_text(name, addr):
       addr_tokens = addr.split()[:3] if addr else ''
       return normalize_name(name) + ' ' + ' '.join(addr_tokens).lower()
   
   vectorizer = TfidfVectorizer(
       analyzer='char_wb', ngram_range=(3, 4),
       max_features=500_000, dtype=np.float32,
       sublinear_tf=True
   )
   
   # Fit on S2+S3, transform both S1 and S2+S3
   s2s3_texts = [make_text(n, a) for n, a in s2s3_records]
   s2s3_tfidf = vectorizer.fit_transform(s2s3_texts)  # sparse matrix
   
   s1_texts = [make_text(n, a) for n, a in s1_records]
   s1_tfidf = vectorizer.transform(s1_texts)  # sparse matrix
   ```
2. **Batch sparse matrix multiplication** to get cosine similarities:
   ```python
   BATCH_SIZE = 50_000
   TOP_K = 20
   
   for i in range(0, len(s1_records), BATCH_SIZE):
       batch = s1_tfidf[i:i+BATCH_SIZE]
       scores = batch @ s2s3_tfidf.T  # sparse matmul
       
       # Extract top-K per row
       for row_idx in range(scores.shape[0]):
           row = scores.getrow(row_idx)
           top_indices = row.data.argsort()[-TOP_K:][::-1]
           top_candidates = row.indices[top_indices]
           # Add to candidate set
   ```

**Characteristics:**
- **Speed:** ~15 minutes per country (batch sparse matmul on CPU).
- **Memory:** With `max_features=500K` and `float32`, the S2+S3 matrix for India (~4.7M rows) takes ~8GB. Process one country at a time.
- **Expected Recall:** ~85% (excellent for character-level noise, abbreviations).

---

### Pass 3: Phonetic Encoding + Postal Code Blocking

**Goal:** Catch severe OCR/typo noise where even character n-grams diverge (e.g., `Iron` vs `Ihrno`, `Custom` vs `Custem`).

**Steps:**
1. **Double Metaphone** encoding on the first 2 significant name tokens:
   ```python
   # pip install metaphone
   from metaphone import doublemetaphone
   
   def get_phonetic_key(name):
       tokens = normalize_name(name).split()[:2]
       codes = []
       for t in tokens:
           primary, secondary = doublemetaphone(t)
           codes.append(primary or secondary or t[:4])
       return tuple(sorted(codes))  # sort for order-invariance
   ```
2. **Extract postal codes** (US: 5-digit, India: 6-digit, France: 5-digit):
   ```python
   import re
   
   def extract_postal(address, country):
       if country == 'India':
           m = re.search(r'\b(\d{6})\b', address)
       else:  # US or France
           m = re.search(r'\b(\d{5})\b', address)
       return m.group(1)[:3] if m else 'UNK'  # 3-digit prefix
   ```
3. **Blocking key** = `(country, phonetic_key, postal_prefix)`:
   ```python
   from collections import defaultdict
   
   blocks = defaultdict(list)
   for eid, name, addr, country in s2s3_records:
       key = (country, get_phonetic_key(name), extract_postal(addr, country))
       blocks[key].append(eid)
   
   for s1_id, name, addr, country in s1_records:
       key = (country, get_phonetic_key(name), extract_postal(addr, country))
       candidates = blocks.get(key, [])
   ```

**Characteristics:**
- **Speed:** ~3 minutes per country (pure hash lookups).
- **Expected Recall:** ~75% in isolation (limited by postal code availability — ~3.3% of S2/S3 have no address).

---

### Pass 4: Address-First Fallback Blocking

**Goal:** Catch the remaining gap — entities where the business name is completely different (DBA/trade names, rebranding) but the physical address is the same.

**Trigger:** Only applied to S1 entities that have fewer than 3 total candidates after Passes 1–3.

**Steps:**
1. Build character 3-gram TF-IDF on `business_address` only (within same country).
2. Retrieve top-K=10 nearest S2/S3 entities by address similarity.
3. Add to candidate set.

**Characteristics:**
- Applied to ~5-10% of S1 entities (those with very poor name overlap candidates).
- Catches DBA name changes where address is the only anchor.

---

## 4. Candidate Set Union & Deduplication

```python
final_candidates = {}  # s1_id -> set of candidate_ids

for s1_id in all_s1_ids:
    combined = set()
    combined |= pass1_candidates.get(s1_id, set())
    combined |= pass2_candidates.get(s1_id, set())
    combined |= pass3_candidates.get(s1_id, set())
    combined |= pass4_candidates.get(s1_id, set())
    final_candidates[s1_id] = combined
```

**Expected Statistics:**
| Metric | Value |
| :--- | :--- |
| Total candidate pairs | ~25-40M |
| Avg candidates per S1 | ~15-25 |
| Median candidates per S1 | ~12 |
| P99 candidates per S1 | ~80 |
| Combined Blocking Recall | ≥ 98% |

---

## 5. Memory & Compute Budget on Kaggle

| Operation | RAM Peak | Time | Notes |
| :--- | :--- | :--- | :--- |
| Country partition load | ~4 GB | 2 min | Largest single country (India S2+S3: ~4.7M rows) |
| Pass 1 (Token Index) | ~6 GB | 5 min/country | Dictionary of sets |
| Pass 2 (TF-IDF sparse) | ~12 GB | 15 min/country | Sparse matrix; batch 50K rows |
| Pass 3 (Phonetic) | ~4 GB | 3 min/country | Hash table of phonetic keys |
| Pass 4 (Address fallback) | ~8 GB | 5 min/country | Only for gap entities |
| **Total per country** | **~12 GB peak** | **~28 min** | Sequential passes, free memory between |
| **Total all countries** | **~12 GB peak** | **~90 min** | Process US → India → France sequentially |

---

## 6. Blocking Quality Diagnostics (on Training Data)

Before moving to the classifier stage, validate blocking quality:

```python
def measure_blocking_recall(ground_truth, candidate_set):
    total_true = 0
    found_true = 0
    for s1_id, true_matches in ground_truth.items():
        candidates = candidate_set.get(s1_id, set())
        for mid in true_matches:
            total_true += 1
            if mid in candidates:
                found_true += 1
    recall = found_true / total_true
    return recall

# Target: recall >= 0.98
```

**Per-Pass Contribution Analysis:** Measure which pass found each true match to understand where recall is coming from and identify gaps:
```python
for s1_id, true_matches in ground_truth.items():
    for mid in true_matches:
        found_by = []
        if mid in pass1_candidates.get(s1_id, set()): found_by.append('P1')
        if mid in pass2_candidates.get(s1_id, set()): found_by.append('P2')
        if mid in pass3_candidates.get(s1_id, set()): found_by.append('P3')
        if mid in pass4_candidates.get(s1_id, set()): found_by.append('P4')
        # Log found_by to see overlap patterns
```
