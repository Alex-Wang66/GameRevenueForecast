"""
Rolling statistics features
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RollingStats:
    """Generate rolling statistics features"""

    @staticmethod
    def create_rolling_features(df: pd.DataFrame, column: str, windows: list = None,
                               fill_method: str = 'bfill') -> pd.DataFrame:
        """
        Create rolling sum and mean features

        Args:
            df: Input DataFrame
            column: Column to compute rolling stats for
            windows: List of window sizes (in hours)
            fill_method: How to fill missing values

        Returns:
            DataFrame with rolling features
        """
        if windows is None:
            windows = [3, 7, 12, 24]

        df = df.copy()

        for window in windows:
            # Rolling sum
            sum_col_name = f'{column}_rolling_sum_{window}'
            df[sum_col_name] = df[column].rolling(window=window).sum()

            # Rolling mean
            mean_col_name = f'{column}_rolling_mean_{window}'
            df[mean_col_name] = df[column].rolling(window=window).mean()

            # Fill missing values
            if fill_method == 'bfill':
                df[sum_col_name] = df[sum_col_name].bfill()
                df[mean_col_name] = df[mean_col_name].bfill()
            elif fill_method == 'ffill':
                df[sum_col_name] = df[sum_col_name].ffill()
                df[mean_col_name] = df[mean_col_name].ffill()

        logger.info(f"Created rolling features for {column} with {len(windows)} windows")
        return df

    @staticmethod
    def create_rolling_std(df: pd.DataFrame, column: str, windows: list = None,
                          fill_method: str = 'bfill') -> pd.DataFrame:
        """
        Create rolling standard deviation features

        Args:
            df: Input DataFrame
            column: Column to compute rolling std for
            windows: List of window sizes
            fill_method: How to fill missing values

        Returns:
            DataFrame with rolling std features
        """
        if windows is None:
            windows = [3, 7, 12, 24]

        df = df.copy()

        for window in windows:
            std_col_name = f'{column}_rolling_std_{window}'
            df[std_col_name] = df[column].rolling(window=window).std()

            if fill_method == 'bfill':
                df[std_col_name] = df[std_col_name].bfill()
            elif fill_method == 'ffill':
                df[std_col_name] = df[std_col_name].ffill()

        logger.info(f"Created rolling std features for {column}")
        return df

    @classmethod
    def create_rolling_for_multiple_columns(cls, df: pd.DataFrame, columns: list,
                                           windows: list = None) -> pd.DataFrame:
        """
        Create rolling features for multiple columns

        Args:
            df: Input DataFrame
            columns: List of columns
            windows: List of window sizes

        Returns:
            DataFrame with rolling features
        """
        if windows is None:
            windows = [3, 7, 12, 24]

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found")
                continue

            df = cls.create_rolling_features(df, col, windows)

        return df
