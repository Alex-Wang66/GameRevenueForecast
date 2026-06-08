"""
Visualization utilities
"""

import logging

logger = logging.getLogger(__name__)


class ModelPlotter:
    """Comprehensive model visualization"""

    @staticmethod
    def plot_metrics_comparison(metrics_dict: dict, title: str = "Metrics Comparison"):
        """Plot metrics comparison"""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd

            df = pd.DataFrame(metrics_dict).T
            fig, ax = plt.subplots(figsize=(10, 6))

            df.plot(kind='bar', ax=ax)
            ax.set_title(title)
            ax.set_ylabel('Value')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.warning(f"Could not plot metrics: {e}")
            return None

    @staticmethod
    def plot_feature_importance(importance: dict or list, top_n: int = 20,
                               title: str = "Feature Importance"):
        """Plot feature importance"""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd

            if isinstance(importance, dict):
                df = pd.Series(importance).sort_values(ascending=True).tail(top_n)
            else:
                df = pd.Series(importance).sort_values(ascending=True).tail(top_n)

            fig, ax = plt.subplots(figsize=(10, 8))

            df.plot(kind='barh', ax=ax)
            ax.set_title(title)
            ax.set_xlabel('Importance')
            ax.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.warning(f"Could not plot feature importance: {e}")
            return None
