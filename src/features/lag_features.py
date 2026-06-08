"""
Lag feature generation
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LagFeatures:
    """Generate lagged features"""

    @staticmethod
    def create_lag_features(df: pd.DataFrame, column: str, lags: list = None,
                           fill_method: str = 'bfill') -> pd.DataFrame:
        """
        Create lagged features for a specific column

        Args:
            df: Input DataFrame
            column: Column to create lags for
            lags: List of lag values (in hours)
            fill_method: How to fill missing values ('bfill', 'ffill')

        Returns:
            DataFrame with lag features
        """
        if lags is None:
            lags = [1, 3, 7, 12, 24]

        df = df.copy()

        for lag in lags:
            lag_col_name = f'{column}_lag_{lag}h'
            df[lag_col_name] = df[column].shift(lag)

            # Fill missing values
            if fill_method == 'bfill':
                df[lag_col_name] = df[lag_col_name].bfill()
            elif fill_method == 'ffill':
                df[lag_col_name] = df[lag_col_name].ffill()

        logger.info(f"Created {len(lags)} lag features for {column}")
        return df

    @staticmethod
    def create_forward_lag_features(df: pd.DataFrame, column: str, lags: list = None,
                                   fill_method: str = 'bfill') -> pd.DataFrame:
        """
        Create forward-looking lag features (shift with negative values)
        Used in optimized version to look ahead

        Args:
            df: Input DataFrame
            column: Column to create lags for
            lags: List of lag values
            fill_method: How to fill missing values

        Returns:
            DataFrame with forward lag features
        """
        if lags is None:
            lags = [1, 3, 7, 12, 24]

        df = df.copy()

        for lag in lags:
            lag_col_name = f'{column}_lag_{lag}h'
            df[lag_col_name] = df[column].shift(-lag)  # Negative shift = look ahead

            # Fill missing values
            if fill_method == 'bfill':
                df[lag_col_name] = df[lag_col_name].bfill()
            elif fill_method == 'ffill':
                df[lag_col_name] = df[lag_col_name].ffill()

        logger.info(f"Created {len(lags)} forward lag features for {column}")
        return df

    @classmethod
    def create_lags_for_multiple_columns(cls, df: pd.DataFrame, columns: list,
                                        lags: list = None, forward: bool = False) -> pd.DataFrame:
        """
        Create lag features for multiple columns

        Args:
            df: Input DataFrame
            columns: List of columns to create lags for
            lags: List of lag values
            forward: Whether to use forward-looking lags

        Returns:
            DataFrame with lag features for all columns
        """
        if lags is None:
            lags = [1, 3, 7, 12, 24]

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue

            if forward:
                df = cls.create_forward_lag_features(df, col, lags)
            else:
                df = cls.create_lag_features(df, col, lags)

        return df
