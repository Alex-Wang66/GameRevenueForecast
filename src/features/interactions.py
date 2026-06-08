"""
Interaction features
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class InteractionFeatures:
    """Create feature interactions"""

    @staticmethod
    def create_interaction(df: pd.DataFrame, col1: str, col2: str,
                          interaction_name: str = None) -> pd.DataFrame:
        """
        Create interaction between two features

        Args:
            df: Input DataFrame
            col1: First column
            col2: Second column
            interaction_name: Name for interaction feature

        Returns:
            DataFrame with interaction feature
        """
        df = df.copy()

        if col1 not in df.columns or col2 not in df.columns:
            logger.warning(f"Columns {col1} or {col2} not found")
            return df

        if interaction_name is None:
            interaction_name = f'{col1}_{col2}'

        df[interaction_name] = df[col1] * df[col2]
        return df

    @staticmethod
    def create_temporal_interactions(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal feature interactions

        Args:
            df: Input DataFrame with temporal features

        Returns:
            DataFrame with interaction features
        """
        df = df.copy()

        # Weekend interactions
        if 'is_weekend' in df.columns and 'is_night' in df.columns:
            df['weekend_night'] = df['is_weekend'] * df['is_night']

        if 'is_weekend' in df.columns and 'is_peak' in df.columns:
            df['weekend_peak'] = df['is_weekend'] * df['is_peak']

        # Peak hour interactions
        if 'is_peak' in df.columns and 'month' in df.columns:
            df['peak_month_interaction'] = df['is_peak'] * (df['month'] - 6)

        logger.info("Created temporal interaction features")
        return df

    @staticmethod
    def create_metric_interactions(df: pd.DataFrame, metric_columns: list = None) -> pd.DataFrame:
        """
        Create interactions between metric columns

        Args:
            df: Input DataFrame
            metric_columns: List of numeric columns to create interactions from

        Returns:
            DataFrame with metric interactions
        """
        df = df.copy()

        if metric_columns is None:
            metric_columns = []

        if len(metric_columns) < 2:
            logger.warning("Need at least 2 columns for interactions")
            return df

        # Create pairwise interactions for top columns
        for i, col1 in enumerate(metric_columns[:3]):
            for col2 in metric_columns[i+1:4]:
                if col1 in df.columns and col2 in df.columns:
                    interaction_name = f'{col1}_{col2}_interaction'
                    df[interaction_name] = df[col1] * df[col2]

        logger.info(f"Created metric interactions for {len(metric_columns)} columns")
        return df

    @classmethod
    def create_all_interactions(cls, df: pd.DataFrame, metric_columns: list = None) -> pd.DataFrame:
        """
        Create all interaction features

        Args:
            df: Input DataFrame
            metric_columns: List of metric columns

        Returns:
            DataFrame with all interactions
        """
        df = cls.create_temporal_interactions(df)
        if metric_columns:
            df = cls.create_metric_interactions(df, metric_columns)

        return df
