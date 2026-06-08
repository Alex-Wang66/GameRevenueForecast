"""
Residual analysis and visualization utilities
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ResidualPlotter:
    """Plot residuals and diagnostics"""

    @staticmethod
    def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, 
                      title: str = "Residuals"):
        """Plot residuals over time"""
        try:
            import matplotlib.pyplot as plt

            residuals = y_true - y_pred

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Time series plot
            ax1.plot(residuals, marker='o', linestyle='-', alpha=0.7)
            ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax1.set_title(f'{title} Over Time')
            ax1.set_ylabel('Residual')
            ax1.grid(True, alpha=0.3)

            # Histogram
            ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
            ax2.set_title('Residual Distribution')
            ax2.set_xlabel('Residual Value')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.warning(f"Could not plot residuals: {e}")
            return None

    @staticmethod
    def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray,
                                 title: str = "Actual vs Predicted"):
        """Plot actual vs predicted"""
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.scatter(y_true, y_pred, alpha=0.5)

            # Plot perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')

            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.warning(f"Could not plot: {e}")
            return None

    @staticmethod
    def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray):
        """Plot error distribution"""
        try:
            import matplotlib.pyplot as plt

            abs_errors = np.abs(y_true - y_pred)
            pct_errors = (abs_errors / np.abs(y_true)) * 100

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Absolute error
            ax1.hist(abs_errors, bins=50, edgecolor='black', alpha=0.7)
            ax1.set_title('Absolute Error Distribution')
            ax1.set_xlabel('Absolute Error')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)

            # Percentage error
            ax2.hist(pct_errors, bins=50, edgecolor='black', alpha=0.7)
            ax2.set_title('Percentage Error Distribution')
            ax2.set_xlabel('Error (%)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.warning(f"Could not plot error distribution: {e}")
            return None
