"""
Data loading and preprocessing module
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate data"""

    def __init__(self, excel_path: str = None):
        """Initialize data loader"""
        self.excel_path = Path(excel_path) if excel_path else None

    def load_excel(self, excel_path: str = None) -> pd.DataFrame:
        """Load data from Excel file"""
        path = Path(excel_path) if excel_path else self.excel_path

        if not path or not path.exists():
            raise FileNotFoundError(f"Excel file not found: {path}")

        logger.info(f"Loading data from {path}")
        df = pd.read_excel(path)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df

    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool:
        """Validate data integrity"""
        required_columns = ['rechour2']

        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return False

        if len(df) == 0:
            logger.error("DataFrame is empty")
            return False

        logger.info("Data validation passed")
        return True


class DataPreprocessor:
    """Preprocess raw data"""

    @staticmethod
    def convert_timestamp(df: pd.DataFrame, timestamp_col: str = 'rechour2') -> pd.DataFrame:
        """Convert timestamp column to datetime"""
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            logger.info(f"Converted {timestamp_col} to datetime")
        return df

    @staticmethod
    def convert_numeric(df: pd.DataFrame, numeric_cols: list = None) -> pd.DataFrame:
        """Convert columns to numeric type"""
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(exclude=['datetime64']).columns

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.info(f"Converted {len(numeric_cols)} columns to numeric")
        return df

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, method: str = 'bfill') -> pd.DataFrame:
        """Handle missing values"""
        if method == 'bfill':
            df = df.bfill()
        elif method == 'ffill':
            df = df.ffill()

        null_count = df.isnull().sum().sum()
        logger.info(f"Handled missing values. Remaining nulls: {null_count}")
        return df

    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
        """Remove duplicate rows"""
        initial_len = len(df)
        df = df.drop_duplicates(subset=subset)
        removed = initial_len - len(df)
        logger.info(f"Removed {removed} duplicate rows")
        return df

    @classmethod
    def preprocess(cls, df: pd.DataFrame, timestamp_col: str = 'rechour2',
                  numeric_cols: list = None, drop_duplicates: bool = True) -> pd.DataFrame:
        """Complete preprocessing pipeline"""
        logger.info("Starting data preprocessing")
        
        # Convert timestamp
        df = cls.convert_timestamp(df, timestamp_col)
        
        # Convert to numeric
        df = cls.convert_numeric(df, numeric_cols)
        
        # Handle missing values
        df = cls.handle_missing_values(df, method='bfill')
        
        # Remove duplicates
        if drop_duplicates:
            df = cls.remove_duplicates(df, subset=[timestamp_col])

        logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        return df


class DataSplitter:
    """Split data for time series"""

    @staticmethod
    def time_series_split(df: pd.DataFrame, timestamp_col: str = 'rechour2',
                         test_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data maintaining time order"""
        # Sort by timestamp
        df = df.sort_values(timestamp_col, ascending=True).reset_index(drop=True)

        # Calculate split point
        split_point = int(len(df) * (1 - test_size))

        train = df.iloc[:split_point].copy()
        test = df.iloc[split_point:].copy()

        logger.info(f"Split data: train={len(train)}, test={len(test)}")
        return train, test
