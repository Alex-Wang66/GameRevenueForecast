"""
Inference pipeline
"""

import numpy as np
import pandas as pd
import logging
from .ensemble import CascadingEnsemble

logger = logging.getLogger(__name__)


class InferencePipeline:
    """End-to-end inference pipeline"""

    def __init__(self, feature_engineer, ensemble: CascadingEnsemble):
        """
        Initialize inference pipeline

        Args:
            feature_engineer: Fitted feature engineer
            ensemble: Fitted cascading ensemble
        """
        self.feature_engineer = feature_engineer
        self.ensemble = ensemble
        self.scaler = feature_engineer.scaler if hasattr(feature_engineer, 'scaler') else None

    def predict(self, df: pd.DataFrame, timestamp_col: str = 'ds',
               inverse_transform: bool = True, transform_func=None) -> np.ndarray:
        """
        Make predictions on new data

        Args:
            df: Input DataFrame with features
            timestamp_col: Timestamp column name
            inverse_transform: Whether to apply inverse transformation
            transform_func: Function to apply inverse transformation (e.g., np.expm1)

        Returns:
            Predictions
        """
        logger.info("Starting inference...")

        # Step 1: Feature engineering
        df_engineered = self.feature_engineer.transform(df, timestamp_col=timestamp_col)
        logger.info("✓ Features engineered")

        # Step 2: Get feature matrix
        X = df_engineered.values

        # Step 3: Make predictions
        predictions = self.ensemble.predict(X)
        logger.info("✓ Predictions made")

        # Step 4: Apply inverse transformation if needed
        if inverse_transform and transform_func is not None:
            predictions = transform_func(predictions)
            logger.info("✓ Inverse transformation applied")

        return predictions

    def predict_with_confidence(self, df: pd.DataFrame, timestamp_col: str = 'ds',
                               n_iterations: int = 10) -> tuple:
        """
        Make predictions with uncertainty estimate (bootstrap-based)

        Args:
            df: Input DataFrame
            timestamp_col: Timestamp column
            n_iterations: Number of bootstrap iterations

        Returns:
            Tuple of (mean_predictions, std_predictions)
        """
        predictions_list = []

        for i in range(n_iterations):
            # Add small random noise to features
            df_noisy = df.copy()
            numeric_cols = df_noisy.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                noise = np.random.normal(0, df_noisy[col].std() * 0.01, len(df_noisy))
                df_noisy[col] = df_noisy[col] + noise

            pred = self.predict(df_noisy, timestamp_col)
            predictions_list.append(pred)

        predictions_array = np.array(predictions_list)
        mean_pred = predictions_array.mean(axis=0)
        std_pred = predictions_array.std(axis=0)

        logger.info(f"Predictions with uncertainty (n={n_iterations}) complete")
        return mean_pred, std_pred

    def batch_predict(self, dataframes: list, timestamp_col: str = 'ds') -> list:
        """
        Make predictions on multiple DataFrames

        Args:
            dataframes: List of DataFrames
            timestamp_col: Timestamp column

        Returns:
            List of predictions
        """
        results = []

        for i, df in enumerate(dataframes):
            logger.info(f"Processing batch {i+1}/{len(dataframes)}...")
            pred = self.predict(df, timestamp_col)
            results.append(pred)

        logger.info("Batch prediction complete")
        return results
