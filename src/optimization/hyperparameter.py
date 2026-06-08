"""
Hyperparameter optimization with Optuna
"""

import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import logging
from .objective import create_catboost_objective, create_lightgbm_objective

logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """Optuna-based hyperparameter optimizer"""

    def __init__(self, config: dict = None):
        """
        Initialize Optuna optimizer

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.study = None
        self.best_trial = None
        self.best_params = None

    def optimize_catboost(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         n_trials: int = 100, timeout: int = 3600) -> dict:
        """
        Optimize CatBoost hyperparameters

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            n_trials: Number of trials
            timeout: Timeout in seconds

        Returns:
            Best hyperparameters
        """
        logger.info(f"Optimizing CatBoost with Optuna (n_trials={n_trials})...")

        objective = create_catboost_objective(X_train, y_train, X_val, y_val)

        sampler = TPESampler(n_startup_trials=20)
        self.study = optuna.create_study(direction='minimize', sampler=sampler)

        self.study.optimize(objective, n_trials=n_trials, timeout=timeout,
                           show_progress_bar=True)

        self.best_trial = self.study.best_trial
        self.best_params = self.best_trial.params

        logger.info(f"✓ Best CatBoost RMSE: {self.best_trial.value:.4f}")
        logger.info(f"Best params: {self.best_params}")

        return self.best_params

    def optimize_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         n_trials: int = 100, timeout: int = 3600) -> dict:
        """
        Optimize LightGBM hyperparameters

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            n_trials: Number of trials
            timeout: Timeout in seconds

        Returns:
            Best hyperparameters
        """
        logger.info(f"Optimizing LightGBM with Optuna (n_trials={n_trials})...")

        objective = create_lightgbm_objective(X_train, y_train, X_val, y_val)

        sampler = TPESampler(n_startup_trials=20)
        self.study = optuna.create_study(direction='minimize', sampler=sampler)

        self.study.optimize(objective, n_trials=n_trials, timeout=timeout,
                           show_progress_bar=True)

        self.best_trial = self.study.best_trial
        self.best_params = self.best_trial.params

        logger.info(f"✓ Best LightGBM RMSE: {self.best_trial.value:.4f}")
        logger.info(f"Best params: {self.best_params}")

        return self.best_params

    def get_optimization_history(self):
        """
        Get optimization history

        Returns:
            DataFrame with trial history
        """
        if self.study is None:
            logger.warning("No optimization study found")
            return None

        return self.study.trials_dataframe()

    def visualize_optimization(self, save_path: str = None):
        """
        Visualize optimization progress

        Args:
            save_path: Path to save visualization
        """
        if self.study is None:
            logger.warning("No optimization study found")
            return

        try:
            fig = optuna.visualization.plot_optimization_history(self.study).show()
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Visualization saved to {save_path}")
        except Exception as e:
            logger.warning(f"Could not visualize optimization: {e}")
