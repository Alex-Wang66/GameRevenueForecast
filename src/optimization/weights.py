"""
Ensemble weight optimization
"""

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)


class WeightOptimizer:
    """Optimize cascade ensemble weights"""

    @staticmethod
    def optimize_weights(pred_layer1: np.ndarray, pred_layer2: np.ndarray,
                        y_true: np.ndarray, metric: str = 'mse') -> tuple:
        """
        Optimize weights α and β for ensemble combination

        Args:
            pred_layer1: Predictions from Layer 1
            pred_layer2: Predictions from Layer 2
            y_true: Ground truth
            metric: 'mse', 'mae', 'rmse'

        Returns:
            Tuple of (alpha, beta)
        """
        logger.info("Optimizing ensemble weights...")

        def objective(alpha):
            beta = 1.0 - alpha

            # Combine predictions
            y_pred = alpha * pred_layer1 + beta * pred_layer2

            # Compute loss
            if metric == 'mse':
                loss = mean_squared_error(y_true, y_pred)
            elif metric == 'mae':
                loss = mean_absolute_error(y_true, y_pred)
            elif metric == 'rmse':
                loss = np.sqrt(mean_squared_error(y_true, y_pred))
            else:
                loss = mean_squared_error(y_true, y_pred)

            return loss

        # Optimize alpha in [0, 1]
        result = minimize_scalar(objective, bounds=(0, 1), method='bounded')

        alpha = result.x
        beta = 1.0 - alpha

        logger.info(f"✓ Optimal weights found: α={alpha:.4f}, β={beta:.4f}")
        logger.info(f"Optimal {metric}: {result.fun:.4f}")

        return alpha, beta

    @staticmethod
    def grid_search_weights(pred_layer1: np.ndarray, pred_layer2: np.ndarray,
                           y_true: np.ndarray, grid_size: int = 101,
                           metric: str = 'rmse') -> tuple:
        """
        Grid search for optimal weights (alternative to continuous optimization)

        Args:
            pred_layer1: Layer 1 predictions
            pred_layer2: Layer 2 predictions
            y_true: Ground truth
            grid_size: Number of points in grid
            metric: Metric to optimize

        Returns:
            Tuple of (alpha, beta)
        """
        logger.info(f"Grid search for weights (grid_size={grid_size})...")

        best_loss = float('inf')
        best_alpha = 0.5

        alphas = np.linspace(0, 1, grid_size)

        for alpha in alphas:
            beta = 1.0 - alpha
            y_pred = alpha * pred_layer1 + beta * pred_layer2

            if metric == 'mse':
                loss = mean_squared_error(y_true, y_pred)
            elif metric == 'mae':
                loss = mean_absolute_error(y_true, y_pred)
            elif metric == 'rmse':
                loss = np.sqrt(mean_squared_error(y_true, y_pred))
            else:
                loss = mean_squared_error(y_true, y_pred)

            if loss < best_loss:
                best_loss = loss
                best_alpha = alpha

        best_beta = 1.0 - best_alpha

        logger.info(f"✓ Grid search complete: α={best_alpha:.4f}, β={best_beta:.4f}")
        logger.info(f"Best {metric}: {best_loss:.4f}")

        return best_alpha, best_beta

    @staticmethod
    def analyze_weight_sensitivity(pred_layer1: np.ndarray, pred_layer2: np.ndarray,
                                  y_true: np.ndarray, alphas: np.ndarray = None) -> dict:
        """
        Analyze how ensemble performance varies with weights

        Args:
            pred_layer1: Layer 1 predictions
            pred_layer2: Layer 2 predictions
            y_true: Ground truth
            alphas: Array of alpha values to test

        Returns:
            Dictionary with analysis results
        """
        if alphas is None:
            alphas = np.linspace(0, 1, 51)

        results = {
            'alphas': [],
            'mse': [],
            'rmse': [],
            'mae': [],
        }

        for alpha in alphas:
            beta = 1.0 - alpha
            y_pred = alpha * pred_layer1 + beta * pred_layer2

            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)

            results['alphas'].append(alpha)
            results['mse'].append(mse)
            results['rmse'].append(rmse)
            results['mae'].append(mae)

        logger.info("Weight sensitivity analysis complete")
        return results
