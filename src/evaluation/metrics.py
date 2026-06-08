"""
Evaluation metrics
"""

import numpy as np
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                            r2_score, mean_absolute_percentage_error)
import logging

logger = logging.getLogger(__name__)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized RMSE (by range)"""
    rmse_val = rmse(y_true, y_pred)
    return rmse_val / (np.max(y_true) - np.min(y_true))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true))


class MetricsCalculator:
    """Calculate evaluation metrics"""

    METRIC_FUNCTIONS = {
        'mse': mean_squared_error,
        'rmse': rmse,
        'mae': mean_absolute_error,
        'mape': mape,
        'smape': smape,
        'nrmse': nrmse,
        'r2': r2_score,
    }

    @classmethod
    def compute_all_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Compute all evaluation metrics

        Args:
            y_true: Ground truth
            y_pred: Predictions

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        for metric_name, metric_func in cls.METRIC_FUNCTIONS.items():
            try:
                metrics[metric_name] = metric_func(y_true, y_pred)
            except Exception as e:
                logger.warning(f"Could not compute {metric_name}: {e}")
                metrics[metric_name] = np.nan

        return metrics

    @classmethod
    def compute_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray,
                       metrics: list = None) -> dict:
        """
        Compute specified metrics

        Args:
            y_true: Ground truth
            y_pred: Predictions
            metrics: List of metric names

        Returns:
            Dictionary with computed metrics
        """
        if metrics is None:
            metrics = ['rmse', 'mae', 'smape', 'r2']

        results = {}

        for metric_name in metrics:
            if metric_name not in cls.METRIC_FUNCTIONS:
                logger.warning(f"Unknown metric: {metric_name}")
                continue

            metric_func = cls.METRIC_FUNCTIONS[metric_name]
            results[metric_name] = metric_func(y_true, y_pred)

        return results

    @classmethod
    def format_metrics(cls, metrics: dict, precision: int = 4) -> str:
        """Format metrics for display"""
        lines = []
        for name, value in metrics.items():
            if np.isnan(value):
                lines.append(f"  {name}: N/A")
            else:
                lines.append(f"  {name}: {value:.{precision}f}")
        return "\n".join(lines)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Comprehensive model evaluation

    Args:
        y_true: Ground truth
        y_pred: Predictions

    Returns:
        Dictionary with all metrics
    """
    logger.info("Evaluating model...")

    metrics = MetricsCalculator.compute_all_metrics(y_true, y_pred)

    logger.info("Model Evaluation Results:")
    logger.info("\n" + MetricsCalculator.format_metrics(metrics))

    return metrics
