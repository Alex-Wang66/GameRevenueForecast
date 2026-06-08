"""
Feature engineering orchestrator
"""

import pandas as pd
import numpy as np
import logging
from .temporal import TemporalFeatures
from .lag_features import LagFeatures
from .rolling_stats import RollingStats
from .cyclical import CyclicalFeatures
from .holidays import HolidayFeatures
from .interactions import InteractionFeatures
from .transforms import DataTransforms

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Orchestrate all feature engineering steps"""

    def __init__(self, config: dict = None):
        """
        Initialize feature engineer

        Args:
            config: Feature engineering configuration dictionary
        """
        self.config = config or {}
        self.scaler = None

    def engineer_features(self, df: pd.DataFrame, metric_name: str = None,
                         target_col: str = 'y', timestamp_col: str = 'ds',
                         fit_mode: bool = True) -> pd.DataFrame:
        """
        Complete feature engineering pipeline

        Args:
            df: Raw input DataFrame
            metric_name: Name of target metric
            target_col: Target column name
            timestamp_col: Timestamp column name
            fit_mode: Whether to fit scalers (True for training)

        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Starting feature engineering pipeline for {metric_name}")

        # Step 1: Temporal features
        df = TemporalFeatures.extract_all_temporal(df, timestamp_col)
        logger.info("✓ Temporal features extracted")

        # Step 2: Cyclical encoding
        df = CyclicalFeatures.encode_all_cyclical(df)
        logger.info("✓ Cyclical features encoded")

        # Step 3: Holiday features
        df = HolidayFeatures.create_holiday_indicators(df, timestamp_col)
        logger.info("✓ Holiday features created")

        # Step 4: Lag features for target
        lag_windows = self.config.get('lag_windows', [1, 3, 7, 12, 24])
        df = LagFeatures.create_lag_features(df, target_col, lag_windows)
        logger.info("✓ Lag features created")

        # Step 5: Rolling statistics
        rolling_windows = self.config.get('rolling_windows', [3, 7, 12, 24])

        # Get numeric columns for rolling stats (exclude temporal features)
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                       if col not in ['hour', 'day', 'month', 'dayofweek', 'week_of_year']]
        df = RollingStats.create_rolling_for_multiple_columns(df, numeric_cols, rolling_windows)
        logger.info("✓ Rolling statistics created")

        # Step 6: Interaction features
        df = InteractionFeatures.create_all_interactions(df, numeric_cols[:5])
        logger.info("✓ Interaction features created")

        # Step 7: Handle missing values
        df = DataTransforms.handle_missing_patterns(df, method='backward_fill')
        logger.info("✓ Missing values handled")

        # Step 8: Standardization (if configured)
        if self.config.get('standardize', True):
            numeric_cols_for_scaling = df.select_dtypes(include=[np.number]).columns.tolist()

            # Exclude sin/cos features from scaling
            numeric_cols_for_scaling = [col for col in numeric_cols_for_scaling
                                       if not any(x in col for x in ['_sin', '_cos'])]

            if fit_mode:
                df, self.scaler = DataTransforms.standardize_features(
                    df, numeric_cols_for_scaling, fit_scaler=True
                )
            elif self.scaler is not None:
                df, _ = DataTransforms.standardize_features(
                    df, numeric_cols_for_scaling, fit_scaler=False, scaler=self.scaler
                )

            logger.info("✓ Features standardized")

        # Step 9: Drop unnecessary columns
        cols_to_drop = [timestamp_col]
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df

    def fit(self, df: pd.DataFrame, metric_name: str = None, target_col: str = 'y',
           timestamp_col: str = 'ds') -> 'FeatureEngineer':
        """
        Fit feature engineer on training data

        Args:
            df: Training DataFrame
            metric_name: Target metric name
            target_col: Target column
            timestamp_col: Timestamp column

        Returns:
            Self (for chaining)
        """
        self.engineer_features(df, metric_name, target_col, timestamp_col, fit_mode=True)
        return self

    def transform(self, df: pd.DataFrame, target_col: str = 'y',
                 timestamp_col: str = 'ds') -> pd.DataFrame:
        """
        Transform new data using fitted engineer

        Args:
            df: New DataFrame to transform
            target_col: Target column
            timestamp_col: Timestamp column

        Returns:
            Transformed DataFrame
        """
        return self.engineer_features(df, target_col=target_col, timestamp_col=timestamp_col,
                                     fit_mode=False)

    def fit_transform(self, df: pd.DataFrame, metric_name: str = None, target_col: str = 'y',
                     timestamp_col: str = 'ds') -> pd.DataFrame:
        """
        Fit and transform in one step

        Args:
            df: Training DataFrame
            metric_name: Target metric
            target_col: Target column
            timestamp_col: Timestamp column

        Returns:
            Transformed DataFrame
        """
        self.fit(df, metric_name, target_col, timestamp_col)
        return self.engineer_features(df, metric_name, target_col, timestamp_col, fit_mode=False)

    def get_feature_names(self, df: pd.DataFrame) -> list:
        """
        Get list of all engineered feature names

        Args:
            df: Processed DataFrame

        Returns:
            List of feature names
        """
        return [col for col in df.columns if col not in ['y', 'ds']]
