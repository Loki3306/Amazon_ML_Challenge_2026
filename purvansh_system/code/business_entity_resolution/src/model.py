"""
Model Training & Macro F_0.5 Optimization Module
Leak-free GroupKFold cross-validation, LightGBM classifier, and threshold tuning.
"""

import time
import numpy as np
from collections import defaultdict
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from .features import FEATURE_NAMES


def evaluate_macro_f05(gt_dict: dict, pred_dict: dict) -> float:
    """
    Compute official challenge Macro F_0.5 metric:
    - F_0.5 weights precision twice as heavily as recall: (1.25 * P * R) / (0.25 * P + R)
    - Entities with no true matches (singletons) score 1.0 if empty, 0.0 if any match predicted.
    - Score is averaged across all S1 entities in the evaluation set.
    """
    scores = []
    for s1_id, true_set in gt_dict.items():
        pred_set = pred_dict.get(s1_id, set())
        
        # Singleton logic
        if not true_set:
            scores.append(1.0 if not pred_set else 0.0)
            continue
            
        if not pred_set:
            scores.append(0.0)
            continue
            
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if p + r == 0.0:
            scores.append(0.0)
        else:
            f05 = (1.25 * p * r) / (0.25 * p + r)
            scores.append(f05)
            
    return float(np.mean(scores))


def train_lightgbm_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    pair_meta: list,
    gt_map: dict,
    random_state: int = 42
) -> tuple:
    """
    Train a regularized LightGBM binary classifier and optimize decision threshold for Macro F_0.5.
    
    Returns:
        tuple: (trained_model, optimal_threshold, validation_f05, validation_auc)
    """
    t0 = time.time()
    print(f"Starting model training on {len(X):,} labeled pairs (Pos: {int(y.sum()):,}, Neg: {int((1-y).sum()):,})...")
    
    # 5-Fold GroupKFold on S1 entity ID (guarantees zero data leakage)
    gkf = GroupKFold(n_splits=5)
    tr_idx, va_idx = next(gkf.split(X, y, groups=groups))
    
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]
    val_pairs = [pair_meta[i] for i in va_idx]
    val_gt = {sid: gt_map[sid] for sid in set(groups[va_idx])}
    
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_NAMES)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
    
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 80,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1
    }
    
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1200,
        valid_sets=[dtrain, dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(period=200)
        ]
    )
    
    val_probs = model.predict(X_va, num_iteration=model.best_iteration)
    val_auc = float(roc_auc_score(y_va, val_probs))
    print(f"Validation AUC-ROC: {val_auc:.5f} (trained in {time.time() - t0:.1f}s)")
    
    # Fine-grained grid search for optimal Macro F_0.5 decision threshold
    print("Optimizing decision threshold tau* on validation set...")
    best_tau, best_f05 = 0.50, 0.0
    for tau in np.linspace(0.35, 0.85, 51):
        pred_dict = defaultdict(set)
        for (sid, cid), prob in zip(val_pairs, val_probs):
            if prob >= tau:
                pred_dict[sid].add(cid)
        score = evaluate_macro_f05(val_gt, pred_dict)
        if score > best_f05:
            best_f05 = score
            best_tau = float(tau)
            
    print(f"Optimal Threshold tau*: {best_tau:.4f} | Validation Macro F_0.5: {best_f05:.5f}")
    return model, best_tau, best_f05, val_auc
