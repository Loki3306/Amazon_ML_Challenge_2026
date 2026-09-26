import os
import joblib
import numpy as np


class EntityMatcherModel:
    def __init__(self, model_type='xgboost_gpu', **params):
        self.model_type = model_type
        self.params = params
        self.model = None

    def fit(self, X, y, sample_weight=None):
        if self.model_type in ('xgboost_gpu', 'xgb_gpu', 'xgboost', 'xgb'):
            import xgboost as xgb
            device = 'cuda' if ('gpu' in self.model_type or self.params.get('device') == 'cuda') else 'cpu'
            default_params = {
                'tree_method': 'hist',
                'device': device,
                'n_estimators': 500,
                'learning_rate': 0.035,
                'max_depth': 7,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'random_state': 42,
                'eval_metric': 'logloss'
            }
            default_params.update(self.params)
            self.model = xgb.XGBClassifier(**default_params)
            self.model.fit(X, y, sample_weight=sample_weight)
        elif self.model_type == 'lightgbm':
            import lightgbm as lgb
            default_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'n_estimators': 450,
                'learning_rate': 0.035,
                'num_leaves': 63,
                'max_depth': 7,
                'min_child_samples': 30,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            default_params.update(self.params)
            self.model = lgb.LGBMClassifier(**default_params)
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            raise ValueError(f'Unknown model type: {self.model_type}')
        return self

    def predict_proba(self, X):
        if hasattr(self.model, 'get_booster'):
            import xgboost as xgb
            b = self.model.get_booster()
            b.set_param({'device': 'cuda'})
            dmat = xgb.DMatrix(X)
            return b.predict(dmat)
        return self.model.predict_proba(X)[:, 1]

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)
