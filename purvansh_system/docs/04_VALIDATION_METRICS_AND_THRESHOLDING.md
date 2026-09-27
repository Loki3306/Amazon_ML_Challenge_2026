# Validation Strategy, Evaluation Metric & Threshold Optimization

---

## 1. Local Validation Strategy

To reliably measure local performance that correlates with both public and private leaderboards:

### Recommended Split: Stratified Entity-Level Holdout
- **No Data Leakage:** Group by `source1_entity_id`. Split S1 reference entities (e.g. 80% train / 20% validation).
- **Country Stratification:** Preserve the 60% US / 40% India ratio in both splits.
- **Held-Out Country Simulation (Optional Diagnostic):** Train only on `US` and validate on `India` (or vice-versa) to verify how well the feature pipeline generalizes to unseen domains like `France`.

---

## 2. Macro $F_{0.5}$ Metric Computation

The competition evaluates submissions using **Macro-Averaged $F_{0.5}$ Score**:

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}} = \frac{(1 + 0.5^2) \cdot P \cdot R}{0.5^2 \cdot P + R}$$

### Evaluation Rules:
1. **Per-Entity Calculation:** Compute $F_{0.5}$ for each individual $S_1$ entity.
2. **Singletons:**
   - True match set = $\emptyset$, Predicted match set = $\emptyset$ $\implies$ **Score = 1.0**
   - True match set = $\emptyset$, Predicted match set $\ne \emptyset$ $\implies$ **Score = 0.0** (False Merge penalty)
   - True match set $\ne \emptyset$, Predicted match set = $\emptyset$ $\implies$ **Score = 0.0**
3. **Macro Average:** Arithmetic mean of all per-entity scores:
   $$\text{Macro } F_{0.5} = \frac{1}{|S_1|} \sum_{i=1}^{|S_1|} F_{0.5}(S_{1, i})$$

### Python Implementation:
```python
def compute_macro_f05(ground_truth_dict, predictions_dict):
    """
    ground_truth_dict: {s1_id: set(matched_ids)}
    predictions_dict:  {s1_id: set(matched_ids)}
    """
    scores = []
    for s1_id, true_set in ground_truth_dict.items():
        pred_set = predictions_dict.get(s1_id, set())
        
        # Singleton handling
        if len(true_set) == 0:
            scores.append(1.0 if len(pred_set) == 0 else 0.0)
            continue
        
        if len(pred_set) == 0:
            scores.append(0.0)
            continue
            
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall == 0:
            scores.append(0.0)
        else:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
            scores.append(f05)
            
    return sum(scores) / len(scores)
```

---

## 3. Threshold Optimization for $F_{0.5}$

Because $F_{0.5}$ weights **precision twice as heavily as recall** ($\beta=0.5$):

- A **standard threshold of 0.50** is usually sub-optimal and yields too many false positives.
- Run a grid search over decision thresholds $\tau \in [0.50, 0.95]$ on the validation set to find $\tau^*$ that maximizes the macro $F_{0.5}$.
- Typical optimal thresholds for $F_{0.5}$ in entity resolution range between **0.65 – 0.82**.

---

## 4. Post-Processing Rules

1. **Max Match Truncation:**
   - In ground truth, $99.8\%$ of entities have $\le 8$ matches. If a model predicts 20+ matches for a single entity, rank them by probability and retain only top high-confidence matches.
2. **Cardinality Assignment / De-duplication:**
   - Candidate records from S2 and S3 map 1-to-1 to S1. If an S2 entity has a high score for multiple S1 entities, assign it to the S1 entity with the highest margin.
