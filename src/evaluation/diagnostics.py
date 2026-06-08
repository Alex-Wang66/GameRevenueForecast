"""
Model diagnostics and residual analysis
"""

import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ResidualAnalyzer:
    """Analyze model residuals"""

    @staticmethod
    def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute prediction residuals"""
        return y_true - y_pred

    @staticmethod
    def residual_statistics(residuals: np.ndarray) -> dict:
        """
        Compute residual statistics

        Args:
            residuals: Model residuals

        Returns:
            Dictionary with statistics
        """
        return {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'median': np.median(residuals),
            'variance': np.var(residuals),
            'skewness': (residuals - np.mean(residuals))**3 / np.std(residuals)**3,
        }

    @staticmethod
    def ljung_box_test(residuals: np.ndarray, lags: int = 10) -> tuple:
        """
        Ljung-Box test for autocorrelation

        Args:
            residuals: Model residuals
            lags: Number of lags to test

        Returns:
            Tuple of (lb_stat, p_value)
        """
        lb_result = acorr_ljungbox(residuals, lags=lags, return_df=True)
        return lb_result['lb_stat'].iloc[-1], lb_result['lb_pvalue'].iloc[-1]

    @classmethod
    def analyze_residuals(cls, y_true: np.ndarray, y_pred: np.ndarray,
                         lags: int = 10) -> dict:
        """
        Complete residual analysis

        Args:
            y_true: Ground truth
            y_pred: Predictions
            lags: Number of lags for diagnostics

        Returns:
            Dictionary with analysis results
        """
        residuals = cls.compute_residuals(y_true, y_pred)

        analysis = {
            'residuals': residuals,
            'statistics': cls.residual_statistics(residuals),
        }

        try:
            lb_stat, p_value = cls.ljung_box_test(residuals, lags)
            analysis['ljung_box'] = {
                'statistic': lb_stat,
                'p_value': p_value,
                'autocorrelated': p_value < 0.05,
            }
        except Exception as e:
            logger.warning(f"Could not compute Ljung-Box test: {e}")

        return analysis


class ErrorAnalyzer:
    """Analyze prediction errors"""

    @staticmethod
    def absolute_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute absolute errors"""
        return np.abs(y_true - y_pred)

    @staticmethod
    def percentage_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute percentage errors"""
        return np.abs((y_true - y_pred) / y_true) * 100

    @staticmethod
    def error_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Analyze error distribution

        Args:
            y_true: Ground truth
            y_pred: Predictions

        Returns:
            Dictionary with error statistics
        """
        abs_errors = ErrorAnalyzer.absolute_errors(y_true, y_pred)
        pct_errors = ErrorAnalyzer.percentage_errors(y_true, y_pred)

        return {
            'abs_error_mean': np.mean(abs_errors),
            'abs_error_std': np.std(abs_errors),
            'abs_error_max': np.max(abs_errors),
            'pct_error_mean': np.mean(pct_errors),
            'pct_error_std': np.std(pct_errors),
            'pct_error_max': np.max(pct_errors),
            'errors_within_10pct': np.sum(pct_errors < 10) / len(pct_errors) * 100,
            'errors_within_20pct': np.sum(pct_errors < 20) / len(pct_errors) * 100,
        }

    @staticmethod
    def identify_worst_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                                   top_n: int = 10) -> pd.DataFrame:
        """
        Identify worst predictions

        Args:
            y_true: Ground truth
            y_pred: Predictions
            top_n: Number of worst predictions to return

        Returns:
            DataFrame with worst predictions
        """
        abs_errors = ErrorAnalyzer.absolute_errors(y_true, y_pred)
        worst_indices = np.argsort(abs_errors)[-top_n:][::-1]

        return pd.DataFrame({
            'true': y_true[worst_indices],
            'pred': y_pred[worst_indices],
            'error': abs_errors[worst_indices],
            'error_pct': (abs_errors / np.abs(y_true))[worst_indices] * 100,
        })
