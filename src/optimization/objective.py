"""
Optuna objective functions
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


def create_catboost_objective(X_train, y_train, X_val, y_val):
    """
    Create Optuna objective function for CatBoost hyperparameter tuning

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        Objective function
    """

    def objective(trial):
        from catboost import CatBoostRegressor

        # Define hyperparameter search space
        params = {
            'iterations': trial.suggest_int('iterations', 500, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 50.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        }

        model = CatBoostRegressor(**params, verbose=0, random_state=42)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        return rmse

    return objective


def create_lightgbm_objective(X_train, y_train, X_val, y_val):
    """
    Create Optuna objective function for LightGBM hyperparameter tuning

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        Objective function
    """

    def objective(trial):
        import lightgbm as lgb

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 4, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        }

        model = lgb.LGBMRegressor(**params, verbose=-1, random_state=42)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        return rmse

    return objective


def create_ensemble_objective(layer1_model, X_train, y_train, X_val, y_val):
    """
    Create objective function for ensemble weight learning

    Args:
        layer1_model: Fitted Layer 1 model
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        Objective function
    """

    def objective(trial):
        # Learn alpha (beta = 1 - alpha)
        alpha = trial.suggest_float('alpha', 0.0, 1.0)
        beta = 1.0 - alpha

        # Get predictions
        val_pred_l1 = layer1_model.predict(X_val)
        
        # This is simplified - in reality you'd use the full ensemble
        # For now, just optimize based on Layer 1
        rmse = np.sqrt(mean_squared_error(y_val, val_pred_l1))

        return rmse

    return objective
