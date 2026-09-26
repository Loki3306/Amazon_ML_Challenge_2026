import numpy as np
from evaluation import evaluate_predictions


def optimize_global_threshold(ground_truth_dict, candidate_scores_dict, thresholds=None):
    """
    candidate_scores_dict: {s1_id: [(target_id, proba), ...]}
    Finds the threshold that maximizes macro F_0.5.
    """
    if thresholds is None:
        thresholds = np.linspace(0.30, 0.90, 31)

    best_threshold = 0.5
    best_metrics = None
    best_f05 = -1.0

    for thresh in thresholds:
        preds = {}
        for s1_id, scores in candidate_scores_dict.items():
            matched = {tid for tid, p in scores if p >= thresh}
            preds[s1_id] = matched

        metrics = evaluate_predictions(ground_truth_dict, preds)
        if metrics['macro_f05'] > best_f05:
            best_f05 = metrics['macro_f05']
            best_threshold = float(thresh)
            best_metrics = metrics

    return best_threshold, best_metrics


def optimize_source_specific_thresholds(ground_truth_dict, candidate_scores_dict,
                                        s2_grid=None, s3_grid=None):
    """
    Finds separate thresholds for S2 and S3 candidates.
    """
    if s2_grid is None:
        s2_grid = np.linspace(0.40, 0.85, 10)
    if s3_grid is None:
        s3_grid = np.linspace(0.40, 0.85, 10)

    best_s2 = 0.5
    best_s3 = 0.5
    best_metrics = None
    best_f05 = -1.0

    for t2 in s2_grid:
        for t3 in s3_grid:
            preds = {}
            for s1_id, scores in candidate_scores_dict.items():
                matched = set()
                for tid, p in scores:
                    if tid.startswith('S2-') and p >= t2:
                        matched.add(tid)
                    elif tid.startswith('S3-') and p >= t3:
                        matched.add(tid)
                preds[s1_id] = matched

            metrics = evaluate_predictions(ground_truth_dict, preds)
            if metrics['macro_f05'] > best_f05:
                best_f05 = metrics['macro_f05']
                best_s2 = float(t2)
                best_s3 = float(t3)
                best_metrics = metrics

    return best_s2, best_s3, best_metrics


def apply_threshold_and_deduplication(candidate_scores_dict, s2_threshold=0.5, s3_threshold=0.5):
    """
    Applies thresholds and enforces that each target (S2/S3) is assigned to at most one S1
    (the S1 with the highest probability score).
    """
    # First gather all (s1_id, target_id, proba) above threshold
    all_pairs = []
    for s1_id, scores in candidate_scores_dict.items():
        for tid, p in scores:
            t = s2_threshold if tid.startswith('S2-') else s3_threshold
            if p >= t:
                all_pairs.append((p, s1_id, tid))

    # Sort descending by score
    all_pairs.sort(key=lambda x: x[0], reverse=True)

    assigned_targets = set()
    result = {s1_id: set() for s1_id in candidate_scores_dict.keys()}

    for p, s1_id, tid in all_pairs:
        if tid not in assigned_targets:
            assigned_targets.add(tid)
            result[s1_id].add(tid)

    return result
