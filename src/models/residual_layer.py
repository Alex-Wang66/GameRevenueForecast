"""
Residual learning layer (second layer of cascade ensemble)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import logging
from .base import BaseModel

logger = logging.getLogger(__name__)


class ResidualLayer(BaseModel):
    """Residual learning layer (Layer 2) - learns to predict residuals"""

    def __init__(self, model_type: str = 'lightgbm', params: dict = None):
        """
        Initialize residual learning model

        Args:
            model_type: 'lightgbm' or 'randomforest'
            params: Hyperparameters
        """
        super().__init__(f'ResidualLayer({model_type})')
        self.model_type = model_type

        if model_type == 'lightgbm':
            # Default LightGBM parameters
            default_params = {
                'n_estimators': 800,
                'learning_rate': 0.015,
                'num_leaves': 50,
                'max_depth': 8,
                'min_child_samples': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'verbose': -1,
            }

            if params:
                default_params.update(params)

            self.params = default_params
            self.model = lgb.LGBMRegressor(**default_params)

        elif model_type == 'randomforest':
            # Default RandomForest parameters
            default_params = {
                'n_estimators': 1500,
                'max_depth': 10,
                'min_samples_leaf': 3,
                'max_features': 0.8,
                'random_state': 42,
                'n_jobs': -1,
            }

            if params:
                default_params.update(params)

            self.params = default_params
            self.model = RandomForestRegressor(**default_params)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Fit residual learning model

        Args:
            X: Features (typically combined with base predictions)
            y: Residuals to predict
            **kwargs: Additional arguments
        """
        logger.info(f"Fitting {self.model_type} residual layer...")

        self.model.fit(X, y)
        self.is_fitted = True

        logger.info(f"✓ {self.model_type} residual layer fitted")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict residuals

        Args:
            X: Features

        Returns:
            Predicted residuals
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        return self.model.predict(X)

    def get_feature_importance(self, feature_names: list = None):
        """Get feature importance"""
        if not self.is_fitted:
            logger.warning("Model not fitted yet")
            return None

        if self.model_type == 'lightgbm':
            return self.model.feature_importances_
        elif self.model_type == 'randomforest':
            return self.model.feature_importances_

        return None
