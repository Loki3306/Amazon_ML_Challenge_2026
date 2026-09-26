import numpy as np


def compute_entity_f05(true_set, pred_set):
    """
    Computes F_0.5 score for a single Source 1 entity.
    """
    is_true_empty = len(true_set) == 0
    is_pred_empty = len(pred_set) == 0

    if is_true_empty:
        return 1.0 if is_pred_empty else 0.0

    if is_pred_empty:
        return 0.0

    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0

    f05 = (1.25 * precision * recall) / denom
    return f05


def evaluate_predictions(ground_truth_dict, predictions_dict):
    """
    ground_truth_dict: {s1_id: set(target_ids)}
    predictions_dict: {s1_id: set(target_ids)}

    Returns a comprehensive dict of metrics.
    """
    f05_scores = []
    f1_scores = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_true_links = 0
    total_pred_links = 0

    singleton_count = 0
    singleton_correct = 0
    singleton_false_match = 0

    non_singleton_count = 0

    all_s1_ids = sorted(list(ground_truth_dict.keys()))

    for s1_id in all_s1_ids:
        true_set = ground_truth_dict[s1_id]
        pred_set = predictions_dict.get(s1_id, set())

        f05 = compute_entity_f05(true_set, pred_set)
        f05_scores.append(f05)

        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_true_links += len(true_set)
        total_pred_links += len(pred_set)

        if len(true_set) == 0:
            singleton_count += 1
            if len(pred_set) == 0:
                singleton_correct += 1
            else:
                singleton_false_match += 1
        else:
            non_singleton_count += 1
            if len(pred_set) > 0 and tp > 0:
                p = tp / len(pred_set)
                r = tp / len(true_set)
                f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            else:
                f1 = 0.0
            f1_scores.append(f1)

    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    global_precision = total_tp / total_pred_links if total_pred_links > 0 else 1.0
    global_recall = total_tp / total_true_links if total_true_links > 0 else 0.0

    return {
        'macro_f05': macro_f05,
        'macro_f1': macro_f1,
        'global_precision': global_precision,
        'global_recall': global_recall,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_pred_links': total_pred_links,
        'total_true_links': total_true_links,
        'singleton_count': singleton_count,
        'singleton_correct': singleton_correct,
        'singleton_accuracy': singleton_correct / singleton_count if singleton_count > 0 else 1.0,
        'non_singleton_count': non_singleton_count
    }
