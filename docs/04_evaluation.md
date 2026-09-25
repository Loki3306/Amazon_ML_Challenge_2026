# Evaluation

The evaluation metric is **F_0.5 (macro-averaged)** across all Source 1 entities.

- **Formula**: `(1.25 * Precision * Recall) / (0.25 * Precision + Recall)`
- This metric heavily penalizes false positives (false merges).
- True singletons correctly predicted yield 1.0. Incorrectly predicting a match for a singleton yields 0.0.
