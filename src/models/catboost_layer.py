"""
CatBoost layer (first layer of cascade ensemble)
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import logging
from .base import BaseModel

logger = logging.getLogger(__name__)


class CatBoostLayer(BaseModel):
    """CatBoost-based base predictor (Layer 1)"""

    def __init__(self, params: dict = None):
        """
        Initialize CatBoost model

        Args:
            params: Hyperparameters for CatBoost
        """
        super().__init__('CatBoost')

        # Default parameters (optimized version)
        default_params = {
            'iterations': 2000,
            'learning_rate': 0.03,
            'depth': 10,
            'subsample': 0.5,
            'l2_leaf_reg': 12,
            'bagging_temperature': 0.2,
            'early_stopping_rounds': 100,
            'verbose': 100,
            'random_state': 42,
        }

        if params:
            default_params.update(params)

        self.params = default_params
        self.model = CatBoostRegressor(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None, **kwargs):
        """
        Fit CatBoost model

        Args:
            X: Training features
            y: Training target
            eval_set: Evaluation set for early stopping
            **kwargs: Additional arguments
        """
        logger.info("Fitting CatBoost layer...")

        fit_params = {
            'X': X,
            'y': y,
        }

        if eval_set is not None:
            fit_params['eval_set'] = eval_set

        self.model.fit(**fit_params)
        self.is_fitted = True

        logger.info("✓ CatBoost layer fitted")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Features

        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        return self.model.predict(X)

    def get_feature_importance(self, feature_names: list = None):
        """Get feature importance from CatBoost"""
        if not self.is_fitted:
            logger.warning("Model not fitted yet")
            return None

        return self.model.get_feature_importance(feature_names=feature_names)
