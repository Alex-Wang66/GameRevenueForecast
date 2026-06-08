"""
Temporal feature extraction
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TemporalFeatures:
    """Extract time-based features"""

    @staticmethod
    def extract_basic_temporal(df: pd.DataFrame, timestamp_col: str = 'ds') -> pd.DataFrame:
        """
        Extract basic temporal features: hour, day, month, dayofweek, week_of_year

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with added temporal features
        """
        df = df.copy()

        if timestamp_col not in df.columns:
            logger.warning(f"Timestamp column {timestamp_col} not found")
            return df

        # Ensure datetime type
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Extract features
        df['hour'] = df[timestamp_col].dt.hour
        df['day'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['dayofweek'] = df[timestamp_col].dt.dayofweek
        df['week_of_year'] = df[timestamp_col].dt.isocalendar().week.astype(int)

        logger.info("Extracted basic temporal features")
        return df

    @staticmethod
    def extract_derived_temporal(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract derived temporal features

        Args:
            df: DataFrame with basic temporal features

        Returns:
            DataFrame with added derived features
        """
        df = df.copy()

        if 'hour' not in df.columns or 'dayofweek' not in df.columns:
            logger.warning("Basic temporal features not found")
            return df

        # Weekend indicator
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

        # Time period indicators
        df['is_night'] = ((df['hour'] >= 20) | (df['hour'] <= 1)).astype(int)
        df['is_peak'] = (df['hour'].isin([0, 1])).astype(int)
        df['is_moderate'] = ((df['hour'].between(9, 12)) | (df['hour'].between(19, 23))).astype(int)
        df['is_low'] = ((df['hour'].between(2, 8)) | (df['hour'].between(13, 18))).astype(int)

        # Interaction features
        df['weekend_night'] = df['is_weekend'] * df['is_night']
        df['weekend_peak'] = df['is_weekend'] * df['is_peak']

        logger.info("Extracted derived temporal features")
        return df

    @classmethod
    def extract_all_temporal(cls, df: pd.DataFrame, timestamp_col: str = 'ds') -> pd.DataFrame:
        """
        Extract all temporal features

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with all temporal features
        """
        df = cls.extract_basic_temporal(df, timestamp_col)
        df = cls.extract_derived_temporal(df)
        return df
