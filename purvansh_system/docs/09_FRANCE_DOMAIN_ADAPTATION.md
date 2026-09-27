# Zero-Shot France Domain Adaptation via Self-Training

---

## 1. The France Challenge

The training data contains only **US** and **India** entities. The test set introduces **France** (~259K S1 entities, ~1.4M S2/S3 candidates) with **zero training examples**. This is a classic **zero-shot domain shift** problem.

### What Changes in France?
| Aspect | US / India (Training) | France (Test Only) |
| :--- | :--- | :--- |
| Language | English / English+Hindi transliteration | French |
| Business Name Tokens | Inc, LLC, Corp, Pvt, Ltd | SA, SAS, SARL, EURL, SCI |
| Address Format | Street, City, State ZIP | Numéro Rue, Code Postal Ville |
| Diacritics | Rare | Common (é, è, ê, à, ç, ô, î, ù, â) |
| Postal Codes | US: 5-digit, India: 6-digit | 5-digit (e.g. 75001 for Paris) |
| Address Keywords | Road, Street, Avenue, Drive, Lane | Rue, Boulevard, Avenue, Impasse, Allée, Place, Chemin, Passage, Cours |
| Common Abbreviations | St, Rd, Ave, Dr, Blvd | Bd, Av, Pl, Imp, All, Ch |

### Why Most Teams Will Fail on France
- Legal suffix stripping hardcoded for English (Inc, LLC) won't remove SA, SAS, SARL.
- Address regex patterns designed for US ZIP codes or Indian PIN codes may fail on French postal codes.
- Phonetic encodings (Soundex, Metaphone) were designed for English and perform poorly on French names like "Boulangerie Château de Montmartre".

---

## 2. Language-Agnostic Preprocessing (Baseline Defense)

Before any self-training, ensure ALL preprocessing is language-agnostic:

### 2.1 Unicode Normalization for Diacritics
```python
import unicodedata

def strip_accents(text):
    """Convert accented characters to ASCII equivalents.
    é → e, ç → c, ô → o, etc.
    """
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

# Example:
# "Boulangerie Château" → "Boulangerie Chateau"
# "Société Générale" → "Societe Generale"
```

### 2.2 Universal Legal Suffix Removal
```python
import re

# Cover English, French, German, and generic legal suffixes
LEGAL_SUFFIXES = r'\b(inc|incorporated|llc|ltd|limited|corp|corporation|co|company|' \
                 r'pvt|private|sa|sas|sarl|eurl|sci|scp|snc|se|' \
                 r'gmbh|ag|ug|ohg|kg|' \
                 r'plc|llp|lp|nv|bv)\b'

def strip_legal_suffix(name):
    name = name.lower()
    name = strip_accents(name)
    name = re.sub(LEGAL_SUFFIXES, '', name)
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name
```

### 2.3 Universal Postal Code Extraction
```python
def extract_postal_code(address, country):
    """Extract postal/ZIP codes for any country."""
    if not address:
        return None
    if country == 'India':
        m = re.search(r'\b(\d{6})\b', address)
    elif country == 'France':
        m = re.search(r'\b(\d{5})\b', address)  # e.g., 75001
    else:  # US
        m = re.search(r'\b(\d{5})(?:-\d{4})?\b', address)
    return m.group(1) if m else None
```

---

## 3. Novel Self-Training Pipeline for France

Self-training (also called pseudo-labeling or bootstrapping) iteratively uses a model's own high-confidence predictions as training labels for the unlabeled domain.

### 3.1 Algorithm

```
SELF-TRAINING ALGORITHM FOR FRANCE DOMAIN ADAPTATION

Input:
  - D_train: Labeled training pairs from US + India
  - D_france: Unlabeled candidate pairs from France (from blocking)
  - M₀: Initial LightGBM model trained on D_train
  - τ_pos = 0.92  (high-confidence match threshold)
  - τ_neg = 0.08  (high-confidence non-match threshold)
  - max_iterations = 3

Algorithm:
  M ← M₀
  for iteration = 1 to max_iterations:
      1. Score all French candidate pairs with M:
         P_france ← M.predict_proba(D_france)
      
      2. Select high-confidence pseudo-labels:
         PL_pos ← {(s1, s2s3) ∈ D_france | P_france > τ_pos}
         PL_neg ← {(s1, s2s3) ∈ D_france | P_france < τ_neg}
      
      3. Sample negatives to balance:
         PL_neg_sample ← random_sample(PL_neg, size=5 * |PL_pos|)
      
      4. Augmented training set:
         D_augmented ← D_train ∪ PL_pos ∪ PL_neg_sample
      
      5. Retrain model:
         M ← train_lgbm(D_augmented)
      
      6. Log statistics:
         print(f"Iteration {iteration}: {|PL_pos|} pseudo-positives, "
               f"{|PL_neg|} pseudo-negatives added")
      
      7. Check convergence:
         if |PL_pos_new - PL_pos_old| / |PL_pos_old| < 0.02:
             break  # predictions stabilized
  
  return M  # Final France-adapted model
```

### 3.2 Why This Works

1. **Feature Transfer:** The 50+ string similarity features (Levenshtein, Jaccard, Token Sort Ratio, etc.) are **language-agnostic**. A French pair with 95% name similarity and matching postal code has the same feature profile as a US pair — the model can score it correctly.

2. **High-Confidence Selection:** By using strict thresholds (τ_pos=0.92, τ_neg=0.08), we only add pseudo-labels the model is very confident about. This prevents error accumulation (confirmation bias).

3. **Iterative Refinement:** Each iteration adds more French examples, helping the model learn French-specific feature distributions (e.g., that French postal codes behave differently from US ZIPs in the postal match feature).

### 3.3 Expected Pseudo-Label Statistics

| Iteration | Pseudo Positives | Pseudo Negatives | Cumulative French Training Pairs |
| :---: | :---: | :---: | :---: |
| 1 | ~150K | ~2M | ~2.15M |
| 2 | ~185K (+23%) | ~2.3M | ~2.49M |
| 3 | ~192K (+4%) | ~2.4M | ~2.59M (converged) |

---

## 4. Safeguards Against Error Accumulation

Self-training can degrade performance if pseudo-labels are noisy. We employ multiple safeguards:

### 4.1 Conservative Thresholds
- τ_pos = 0.92 (not 0.5!) ensures only near-certain matches are added.
- τ_neg = 0.08 ensures only clearly wrong pairs are negative pseudo-labels.
- The "ambiguous zone" (0.08 < P < 0.92) is never pseudo-labeled.

### 4.2 Sample Weighting
```python
# Give pseudo-labeled samples lower weight than ground-truth samples
sample_weights = np.ones(len(D_augmented))
sample_weights[is_pseudo_label] = 0.5  # half weight for pseudo-labels

lgbm_model.fit(X, y, sample_weight=sample_weights)
```

### 4.3 Validation on Cross-Country Diagnostic
Before deploying self-training on France, validate the approach by simulating it:
1. Train on US only.
2. Self-train on India (pretend India is unlabeled).
3. Compare self-trained India F_0.5 vs. direct India validation F_0.5.
4. If self-training helps on India, it will likely help on France.

### 4.4 Monotonic Improvement Check
After each self-training iteration, re-evaluate on the US+India validation set. If the score drops by more than 0.005, stop — the pseudo-labels are hurting the base model.

---

## 5. Alternative: Feature Distribution Alignment (Advanced)

If self-training alone is insufficient, consider **feature distribution alignment**:

1. Compute the mean and variance of each feature for US training data and France test data.
2. Identify features with large distribution shift (e.g., address token Jaccard may have different distribution because French addresses are structured differently).
3. Apply per-feature z-score normalization calibrated to each country's distribution:
   ```python
   # Per-country feature standardization
   for feature in features:
       for country in ['US', 'India', 'France']:
           mask = df['country'] == country
           mean = df.loc[mask, feature].mean()
           std = df.loc[mask, feature].std()
           df.loc[mask, feature] = (df.loc[mask, feature] - mean) / (std + 1e-8)
   ```
4. This makes the model's decision boundaries more transferable across countries.

---

## 6. France-Specific Threshold
Even after self-training, the optimal decision threshold for France may differ from US/India. Since we have no ground truth for France, use the **India threshold as a proxy** (India's noise patterns are closer to France's multilingual complexity than US's more standardized addresses).

```python
# Threshold selection
tau_US = optimize_threshold(val_US_preds, val_US_truth)      # e.g., 0.68
tau_India = optimize_threshold(val_India_preds, val_India_truth)  # e.g., 0.62
tau_France = tau_India  # Use India as proxy for France
```
