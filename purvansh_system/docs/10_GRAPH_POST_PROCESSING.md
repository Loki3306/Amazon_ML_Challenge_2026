# Graph-Based Post-Processing & Consistency Enforcement

---

## 1. Why Post-Processing Matters

Pairwise classifiers score each (S1, S2/S3) candidate pair independently. This creates three types of inconsistencies that cost F_0.5 points:

1. **Cardinality Violations:** A single S2/S3 entity assigned to multiple S1 entities. In the ground truth, each S2/S3 entity maps to at most one S1 entity (1-to-1 from S2/S3 side). If your model assigns S2-A to both S1-X and S1-Y, at least one assignment is wrong (a false positive that hurts precision).

2. **Transitivity Violations:** If S2-A matches S1-X, and S3-B also matches S1-X, then S2-A and S3-B should be similar to each other. If they are very dissimilar, one of the matches is likely wrong.

3. **Singleton Contamination:** Entities that should be singletons (no matches) receive low-confidence false matches, each scoring 0.0 instead of 1.0 on F_0.5.

---

## 2. Cardinality Enforcement via Maximum Weighted Bipartite Matching

### 2.1 Problem Formulation
Build a bipartite graph G = (S1_nodes, S2S3_nodes, Edges) where:
- Each S1 entity is a node on the left.
- Each S2/S3 entity is a node on the right.
- An edge (S1_i, S2S3_j) exists with weight = P(match) from the LightGBM model.
- Only edges with P(match) > threshold τ are included.

The constraint: each S2/S3 node can be matched to **at most one** S1 node.

### 2.2 Greedy Resolution Algorithm
For a full maximum weighted bipartite matching on millions of nodes, the Hungarian algorithm is too slow (O(n³)). Instead, use a greedy approach:

```python
def resolve_cardinality(predictions, threshold):
    """
    predictions: list of (s1_id, s2s3_id, probability)
    Returns: dict {s1_id: set(matched_s2s3_ids)}
    """
    # Sort all predictions by probability descending
    predictions.sort(key=lambda x: x[2], reverse=True)
    
    assigned_s2s3 = set()  # S2/S3 IDs already assigned
    matches = defaultdict(set)
    
    for s1_id, s2s3_id, prob in predictions:
        if prob < threshold:
            break  # Below threshold, stop
        if s2s3_id not in assigned_s2s3:
            matches[s1_id].add(s2s3_id)
            assigned_s2s3.add(s2s3_id)
    
    return matches
```

### 2.3 Why Greedy Works
- Sorting by descending probability ensures the highest-confidence assignments are made first.
- When a conflict occurs (S2-A wants to match both S1-X and S1-Y), the higher-probability edge wins.
- This is optimal in practice because conflicts are rare (<0.5% of predictions) and the probability gap between the correct and incorrect assignment is usually large.

---

## 3. Singleton Verification & Protection

Singletons (~5.58% of S1 entities) are entities with zero true matches. They score 1.0 when correctly predicted as empty, and 0.0 when any false match is predicted. This makes them disproportionately valuable for F_0.5.

### 3.1 Singleton Detection Heuristics

```python
def detect_singletons(s1_candidates, predictions, tau_singleton=0.30):
    """
    If ALL candidates for an S1 entity have P(match) < tau_singleton,
    predict it as a singleton (empty match list).
    """
    singletons = set()
    for s1_id, candidates in s1_candidates.items():
        max_prob = max(predictions.get((s1_id, cid), 0.0) for cid in candidates)
        if max_prob < tau_singleton:
            singletons.add(s1_id)
    return singletons
```

### 3.2 Optimal τ_singleton Selection
- τ_singleton should be calibrated on the validation set.
- Plot F_0.5 as a function of τ_singleton.
- Typical optimal value: 0.25 – 0.40.
- Setting τ_singleton too high → false singletons (missed matches, hurts recall).
- Setting τ_singleton too low → false matches on true singletons (hurts precision more).

---

## 4. Transitivity Consistency Check (Advanced)

For S1 entities with multiple predicted matches, verify internal consistency:

```python
def transitivity_check(s1_id, matched_ids, feature_cache):
    """
    If S2-A and S3-B are both matched to S1-X, then S2-A and S3-B
    should have high mutual similarity. If they don't, the weaker
    match is likely a false positive.
    """
    if len(matched_ids) <= 1:
        return matched_ids  # Nothing to check
    
    # Compute pairwise name similarity among matched entities
    for i, id_a in enumerate(matched_ids):
        for id_b in matched_ids[i+1:]:
            name_sim = token_sort_ratio(names[id_a], names[id_b])
            if name_sim < 0.30:  # Very dissimilar matches
                # Remove the match with lower P(match) to S1
                weaker = id_a if probs[id_a] < probs[id_b] else id_b
                matched_ids.remove(weaker)
    
    return matched_ids
```

This is O(k²) per S1 entity where k = number of matches (typically 2-5), so it's very fast.

---

## 5. Match Count Regularization

From EDA, 99.8% of S1 entities have ≤ 8 matches (max = 11). If the model predicts 15+ matches for any entity, this is almost certainly noisy:

```python
MAX_MATCHES = 11

for s1_id, matched_set in final_matches.items():
    if len(matched_set) > MAX_MATCHES:
        # Keep only the top-MAX_MATCHES by probability
        sorted_matches = sorted(matched_set, key=lambda m: probs[(s1_id, m)], reverse=True)
        final_matches[s1_id] = set(sorted_matches[:MAX_MATCHES])
```

---

## 6. Complete Post-Processing Pipeline

```
Raw LightGBM Predictions (P(match) for all candidate pairs)
    │
    ▼
┌─────────────────────────────┐
│ Step 1: Apply threshold τ*  │  (from F_0.5 optimization)
│ Keep pairs with P > τ*      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Step 2: Cardinality         │  (greedy maximum weighted matching)
│ Resolve S2/S3 conflicts    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Step 3: Singleton           │  (remove matches from S1 entities
│ Verification                │   where max P < τ_singleton)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Step 4: Match Count Cap     │  (truncate to max 11 matches)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Step 5: Transitivity Check  │  (remove internally inconsistent
│ (Optional, advanced)        │   matches within each S1 cluster)
└──────────────┬──────────────┘
               │
               ▼
         Final Matches
      matching_results.tsv
```

---

## 7. Expected Impact on F_0.5

| Post-Processing Step | Expected F_0.5 Improvement |
| :--- | :---: |
| Cardinality enforcement | +0.003 to +0.008 |
| Singleton verification | +0.005 to +0.015 |
| Match count cap | +0.001 to +0.003 |
| Transitivity check | +0.002 to +0.005 |
| **Combined** | **+0.010 to +0.025** |

These gains compound. On a leaderboard where the gap between rank 1 and rank 10 may be 0.02, post-processing alone can move you up several positions.
