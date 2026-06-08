"""
Cyclical feature encoding (sin/cos transformation)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CyclicalFeatures:
    """Encode cyclical features using sin/cos transformation"""

    @staticmethod
    def encode_cyclical(df: pd.DataFrame, column: str, period: int) -> pd.DataFrame:
        """
        Encode cyclical feature using sin/cos transformation

        Args:
            df: Input DataFrame
            column: Column to encode
            period: Period of the cycle (e.g., 24 for hours, 365 for days)

        Returns:
            DataFrame with sin/cos encoded features
        """
        df = df.copy()

        sin_col = f'{column}_sin'
        cos_col = f'{column}_cos'

        df[sin_col] = np.sin(2 * np.pi * df[column] / period)
        df[cos_col] = np.cos(2 * np.pi * df[column] / period)

        logger.info(f"Encoded cyclical feature {column} (period={period})")
        return df

    @staticmethod
    def encode_hour(df: pd.DataFrame) -> pd.DataFrame:
        """Encode hour as cyclical feature"""
        if 'hour' not in df.columns:
            logger.warning("Hour column not found")
            return df

        return CyclicalFeatures.encode_cyclical(df, 'hour', period=24)

    @staticmethod
    def encode_day_of_month(df: pd.DataFrame) -> pd.DataFrame:
        """Encode day of month as cyclical feature"""
        if 'day' not in df.columns:
            logger.warning("Day column not found")
            return df

        return CyclicalFeatures.encode_cyclical(df, 'day', period=31)

    @staticmethod
    def encode_month(df: pd.DataFrame) -> pd.DataFrame:
        """Encode month as cyclical feature"""
        if 'month' not in df.columns:
            logger.warning("Month column not found")
            return df

        return CyclicalFeatures.encode_cyclical(df, 'month', period=12)

    @staticmethod
    def encode_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
        """Encode day of week as cyclical feature"""
        if 'dayofweek' not in df.columns:
            logger.warning("Dayofweek column not found")
            return df

        return CyclicalFeatures.encode_cyclical(df, 'dayofweek', period=7)

    @staticmethod
    def encode_week_of_year(df: pd.DataFrame) -> pd.DataFrame:
        """Encode week of year as cyclical feature"""
        if 'week_of_year' not in df.columns:
            logger.warning("Week_of_year column not found")
            return df

        return CyclicalFeatures.encode_cyclical(df, 'week_of_year', period=52)

    @classmethod
    def encode_all_cyclical(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode all cyclical temporal features

        Args:
            df: Input DataFrame with temporal features

        Returns:
            DataFrame with sin/cos encoded features
        """
        df = cls.encode_hour(df)
        df = cls.encode_day_of_month(df)
        df = cls.encode_month(df)
        df = cls.encode_day_of_week(df)
        df = cls.encode_week_of_year(df)

        logger.info("Encoded all cyclical features")
        return df
