"""
Time-series aware cross-validation
"""

import numpy as np
from sklearn.model_selection import TimeSeriesSplit, KFold
import logging

logger = logging.getLogger(__name__)


class TimeSeriesCV:
    """Time series cross-validation splitter"""

    def __init__(self, n_splits: int = 5):
        """
        Initialize TimeSeriesCV

        Args:
            n_splits: Number of CV folds
        """
        self.n_splits = n_splits
        self.splitter = TimeSeriesSplit(n_splits=n_splits)

    def get_splits(self, X: np.ndarray, y: np.ndarray = None):
        """
        Get train/test indices for CV splits

        Args:
            X: Features
            y: Target (optional)

        Returns:
            Generator of (train_idx, test_idx) tuples
        """
        return self.splitter.split(X, y)

    def cross_validate(self, model, X: np.ndarray, y: np.ndarray,
                      metric_func=None) -> dict:
        """
        Perform time-series cross-validation

        Args:
            model: Model with fit/predict methods
            X: Features
            y: Target
            metric_func: Function to compute metric

        Returns:
            Dictionary with CV scores
        """
        scores = []

        for fold, (train_idx, test_idx) in enumerate(self.get_splits(X, y)):
            logger.info(f"Fold {fold+1}/{self.n_splits}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Compute metric
            if metric_func is None:
                from sklearn.metrics import mean_squared_error
                score = np.sqrt(mean_squared_error(y_test, y_pred))
            else:
                score = metric_func(y_test, y_pred)

            scores.append(score)
            logger.info(f"Fold {fold+1} score: {score:.4f}")

        return {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
        }


class StratifiedTimeSeriesCV:
    """K-Fold cross-validation for time series (for non-time-dependent folds)"""

    def __init__(self, n_splits: int = 5, shuffle: bool = False):
        """
        Initialize K-Fold CV

        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle (not recommended for time series)
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=42)

    def get_splits(self, X: np.ndarray, y: np.ndarray = None):
        """Get train/test indices"""
        return self.splitter.split(X, y)

    def cross_validate(self, model, X: np.ndarray, y: np.ndarray,
                      metric_func=None) -> dict:
        """Perform K-Fold cross-validation"""
        scores = []

        for fold, (train_idx, test_idx) in enumerate(self.get_splits(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if metric_func is None:
                from sklearn.metrics import mean_squared_error
                score = np.sqrt(mean_squared_error(y_test, y_pred))
            else:
                score = metric_func(y_test, y_pred)

            scores.append(score)

        return {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
        }
