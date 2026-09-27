"""
Fine-Tuning & Ensemble Blending Engine
Performs XGBoost warm-start booster expansion, LightGBM stacking model training,
probability ensemble fusion, and fine-grained Macro F0.5 threshold optimization.
"""

import time
import gc
import numpy as np
from collections import defaultdict
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


def evaluate_macro_f05(gt_dict: dict, pred_dict: dict) -> float:
    """
    Compute official challenge Macro F_0.5 metric:
    F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    Precision is weighted 2x over recall. Singletons score 1.0 if empty, 0.0 if false positive.
    """
    scores = []
    for s1_id, true_set in gt_dict.items():
        pred_set = pred_dict.get(s1_id, set())
        
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


def fine_tune_boosters(
    boosters: list,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    fine_tune_rounds: int = 150,
    learning_rate: float = 0.01
) -> list:
    """
    Perform warm-start fine-tuning on existing XGBoost boosters.
    Appends refined decision trees trained with a small learning rate.
    """
    print(f"\n  [Fine-Tuning] Performing warm-start expansion on {len(boosters)} XGBoost boosters...", flush=True)
    t0 = time.time()
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": learning_rate,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "seed": 42
    }
    
    refined_boosters = []
    for idx, b in enumerate(boosters):
        print(f"    Fine-tuning booster {idx + 1}/{len(boosters)} (+{fine_tune_rounds} rounds, lr={learning_rate})...", flush=True)
        refined_b = xgb.train(
            params,
            dtrain,
            num_boost_round=fine_tune_rounds,
            evals=[(dtrain, "train"), (dval, "val")],
            xgb_model=b,
            verbose_eval=False
        )
        refined_boosters.append(refined_b)
        
    print(f"  [Fine-Tuning] Completed XGBoost booster expansion in {time.time() - t0:.1f}s", flush=True)
    return refined_boosters


def train_ensemble_and_tune_threshold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    pair_meta: list,
    gt_map: dict,
    pretrained_boosters: list = None,
    alpha_xgb: float = 0.60
) -> tuple:
    """
    Train a LightGBM stacking model, combine probability predictions with pre-trained / fine-tuned XGBoost boosters,
    and perform a dense grid search for the optimal decision threshold tau*.
    
    Args:
        X: Pair feature matrix
        y: Binary target labels (1=match, 0=non-match)
        groups: Group array (S1 entity IDs) for GroupKFold
        pair_meta: List of (s1_id, candidate_id) metadata tuples
        gt_map: Dict of ground truth matches {s1_id: set(matched_ids)}
        pretrained_boosters: Optional list of pre-trained XGBoost boosters
        alpha_xgb: Weight given to XGBoost vs LightGBM (1 - alpha_xgb)
        
    Returns:
        tuple: (ensemble_models_dict, optimal_tau, validation_f05, validation_auc)
    """
    t0 = time.time()
    print(f"\n  [Ensemble Engine] Initializing GroupKFold training on {len(X):,} pairs...", flush=True)
    
    gkf = GroupKFold(n_splits=5)
    tr_idx, va_idx = next(gkf.split(X, y, groups=groups))
    
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]
    val_pairs = [pair_meta[i] for i in va_idx]
    val_gt = {sid: gt_map[sid] for sid in set(groups[va_idx])}
    
    # 1. Fine-tune XGBoost Boosters if provided
    xgb_models = []
    if pretrained_boosters:
        xgb_models = fine_tune_boosters(pretrained_boosters, X_tr, y_tr, X_va, y_va)
        
    # 2. Train LightGBM Stacker
    print("  [Ensemble Engine] Training LightGBM stacking model...", flush=True)
    dtrain_lgb = lgb.Dataset(X_tr, label=y_tr)
    dval_lgb = lgb.Dataset(X_va, label=y_va, reference=dtrain_lgb)
    
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": 9,
        "min_child_samples": 50,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }
    
    lgb_model = lgb.train(
        lgb_params,
        dtrain_lgb,
        num_boost_round=1500,
        valid_sets=[dtrain_lgb, dval_lgb],
        callbacks=[
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(period=300)
        ]
    )
    
    # 3. Compute Validation Ensemble Predictions
    lgb_val_probs = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration)
    
    if xgb_models:
        dval_xgb = xgb.DMatrix(X_va)
        xgb_val_probs_list = [bm.predict(dval_xgb) for bm in xgb_models]
        xgb_val_probs = np.mean(xgb_val_probs_list, axis=0)
        
        # Dual-Model Probability Fusion
        ensemble_val_probs = alpha_xgb * xgb_val_probs + (1.0 - alpha_xgb) * lgb_val_probs
    else:
        ensemble_val_probs = lgb_val_probs
        
    val_auc = float(roc_auc_score(y_va, ensemble_val_probs))
    print(f"  [Validation] Ensemble AUC-ROC Score: {val_auc:.5f} (computed in {time.time() - t0:.1f}s)", flush=True)
    
    # 4. Dense Grid Search for Optimal Macro F0.5 Threshold tau*
    print("  [Validation] Dense grid search for optimal decision threshold tau*...", flush=True)
    best_tau, best_f05 = 0.65, 0.0
    
    for tau in np.linspace(0.40, 0.85, 91):
        pred_dict = defaultdict(set)
        for (sid, cid), prob in zip(val_pairs, ensemble_val_probs):
            if prob >= tau:
                pred_dict[sid].add(cid)
        score = evaluate_macro_f05(val_gt, pred_dict)
        if score > best_f05:
            best_f05 = score
            best_tau = float(tau)
            
    print(f"  [Validation] Optimal Decision Threshold tau*: {best_tau:.4f} | Peak Macro F_0.5: {best_f05:.5f}", flush=True)
    
    models_dict = {
        "lgb_model": lgb_model,
        "xgb_models": xgb_models,
        "alpha_xgb": alpha_xgb
    }
    
    return models_dict, best_tau, best_f05, val_auc
