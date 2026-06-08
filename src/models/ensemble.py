"""
Cascading ensemble with learnable weights
"""

import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize_scalar
from .catboost_layer import CatBoostLayer
from .residual_layer import ResidualLayer

logger = logging.getLogger(__name__)


class CascadingEnsemble:
    """
    Two-layer cascading fusion ensemble with optional learnable weights
    
    Architecture:
    Layer 1: CatBoost base predictor → pred_cat
    Layer 2: ResidualModel(features) → pred_res
    Fusion: final = α * pred_cat + β * pred_res
    """

    def __init__(self, catboost_params: dict = None, residual_params: dict = None,
                 residual_model_type: str = 'lightgbm', fixed_weights: bool = False,
                 alpha: float = 0.5, beta: float = 0.5):
        """
        Initialize cascading ensemble

        Args:
            catboost_params: CatBoost hyperparameters
            residual_params: Residual model hyperparameters
            residual_model_type: 'lightgbm' or 'randomforest'
            fixed_weights: Whether to use fixed weights (True = 1:1, False = learnable)
            alpha: CatBoost weight
            beta: Residual weight
        """
        self.layer1 = CatBoostLayer(catboost_params)
        self.layer2 = ResidualLayer(residual_model_type, residual_params)

        self.fixed_weights = fixed_weights
        self.alpha = alpha
        self.beta = beta

        self.train_pred_layer1 = None
        self.val_pred_layer1 = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
           X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Fit cascading ensemble

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (for weight learning)
            y_val: Validation target (for weight learning)

        Returns:
            Self
        """
        logger.info("Fitting cascading ensemble...")

        # Step 1: Fit Layer 1 (CatBoost)
        logger.info("Fitting Layer 1 (CatBoost)...")
        self.layer1.fit(X_train, y_train)

        # Step 2: Get predictions from Layer 1
        self.train_pred_layer1 = self.layer1.predict(X_train)

        # Step 3: Calculate residuals for Layer 2
        train_residuals = y_train - self.train_pred_layer1
        logger.info("Layer 1 fitted. Training residuals calculated.")

        # Step 4: Fit Layer 2 on residuals
        logger.info("Fitting Layer 2 (Residual Model)...")
        self.layer2.fit(X_train, train_residuals)

        # Step 5: Learn weights if enabled
        if not self.fixed_weights and X_val is not None and y_val is not None:
            logger.info("Learning cascade weights...")
            self._learn_weights(X_train, y_train, X_val, y_val)

        logger.info("✓ Cascading ensemble fitted")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using cascading ensemble

        Args:
            X: Input features

        Returns:
            Final predictions
        """
        if not self.layer1.is_fitted or not self.layer2.is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")

        # Get predictions from both layers
        pred_layer1 = self.layer1.predict(X)
        pred_layer2 = self.layer2.predict(X)

        # Combine with learnable or fixed weights
        final_pred = self.alpha * pred_layer1 + self.beta * pred_layer2

        return final_pred

    def _learn_weights(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray):
        """
        Learn optimal weights α and β using validation set

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        """
        # Get validation predictions
        val_pred_layer1 = self.layer1.predict(X_val)
        val_pred_layer2 = self.layer2.predict(X_val)

        # Define objective function to minimize
        def objective(alpha):
            beta = 1.0 - alpha
            val_pred = alpha * val_pred_layer1 + beta * val_pred_layer2

            # Use MSE as loss
            from sklearn.metrics import mean_squared_error
            return mean_squared_error(y_val, val_pred)

        # Optimize alpha in range [0, 1]
        result = minimize_scalar(objective, bounds=(0, 1), method='bounded')

        self.alpha = result.x
        self.beta = 1.0 - result.x

        logger.info(f"Learned weights: α={self.alpha:.4f}, β={self.beta:.4f}")

    def get_layer1_importance(self, feature_names: list = None):
        """Get Layer 1 (CatBoost) feature importance"""
        return self.layer1.get_feature_importance(feature_names)

    def get_layer2_importance(self, feature_names: list = None):
        """Get Layer 2 (Residual) feature importance"""
        return self.layer2.get_feature_importance(feature_names)

    def set_fixed_weights(self, alpha: float, beta: float):
        """Manually set fixed weights"""
        self.alpha = alpha
        self.beta = beta
        logger.info(f"Set fixed weights: α={alpha:.4f}, β={beta:.4f}")
