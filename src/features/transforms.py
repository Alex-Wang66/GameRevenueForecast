"""
Data transformations
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class DataTransforms:
    """Apply transformations to data"""

    @staticmethod
    def apply_log_transform(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Apply log1p transformation to handle skewed distributions

        Args:
            df: Input DataFrame
            column: Column to transform

        Returns:
            DataFrame with log-transformed column
        """
        df = df.copy()

        if column not in df.columns:
            logger.warning(f"Column {column} not found")
            return df

        df[column] = np.log1p(df[column])
        logger.info(f"Applied log1p transformation to {column}")
        return df

    @staticmethod
    def apply_inverse_log_transform(values: np.ndarray) -> np.ndarray:
        """
        Apply inverse log transformation (expm1)

        Args:
            values: Array of log-transformed values

        Returns:
            Inverse-transformed values
        """
        return np.expm1(values)

    @staticmethod
    def standardize_features(df: pd.DataFrame, columns: list = None,
                            fit_scaler=True, scaler=None) -> tuple:
        """
        Standardize features using StandardScaler

        Args:
            df: Input DataFrame
            columns: Columns to standardize
            fit_scaler: Whether to fit new scaler
            scaler: Existing scaler to use

        Returns:
            Tuple of (DataFrame with scaled features, scaler object)
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if fit_scaler:
            scaler = StandardScaler()
            df[columns] = scaler.fit_transform(df[columns])
            logger.info(f"Fitted StandardScaler on {len(columns)} columns")
        elif scaler is not None:
            df[columns] = scaler.transform(df[columns])
            logger.info(f"Applied existing StandardScaler to {len(columns)} columns")

        return df, scaler

    @staticmethod
    def remove_outliers(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers using z-score

        Args:
            df: Input DataFrame
            column: Column to check
            threshold: Z-score threshold

        Returns:
            DataFrame with outliers removed
        """
        df = df.copy()

        if column not in df.columns:
            return df

        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        df = df[z_scores < threshold]

        logger.info(f"Removed outliers from {column} (threshold={threshold})")
        return df

    @staticmethod
    def handle_missing_patterns(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing values with different methods

        Args:
            df: Input DataFrame
            method: 'interpolate', 'forward_fill', 'backward_fill'

        Returns:
            DataFrame with filled values
        """
        df = df.copy()

        if method == 'interpolate':
            df = df.interpolate(method='linear', limit_direction='both')
        elif method == 'forward_fill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif method == 'backward_fill':
            df = df.fillna(method='bfill').fillna(method='ffill')

        logger.info(f"Applied {method} for missing values")
        return df
