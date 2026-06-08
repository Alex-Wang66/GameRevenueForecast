"""
Holiday features (Chinese holidays)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class HolidayFeatures:
    """Extract holiday-related features"""

    # 25 Chinese holidays with dates
    CHINESE_HOLIDAYS = {
        'New_Year': '2025-01-01',
        'Spring_Festival': '2025-01-29',
        'Lantern_Festival': '2025-02-12',
        'Valentine': '2025-02-14',
        'Girls_Day': '2025-03-07',
        'Women_Day': '2025-03-08',
        'April_Fools': '2025-04-01',
        'Qingming': '2025-04-04',
        'Youth_Day': '2025-05-04',
        'Labor_Day': '2025-05-01',
        'Children_Day': '2025-06-01',
        'Dragon_Boat': '2025-05-31',
        'Father_Day': '2025-06-15',
        'Party_Day': '2025-07-01',
        'Army_Day': '2025-08-01',
        'Double_Seventh': '2025-08-29',
        'Teachers_Day': '2025-09-10',
        'National_Day': '2025-10-01',
        'Mid_Autumn': '2025-10-06',
        'Double_Ninth': '2025-10-29',
        'Halloween_Night': '2025-10-31',
        'Halloween': '2025-11-01',
        'Thanksgiving': '2025-11-27',
        'Christmas_Eve': '2025-12-24',
        'Christmas': '2025-12-25',
    }

    @staticmethod
    def create_holiday_indicators(df: pd.DataFrame, timestamp_col: str = 'ds',
                                 pre_window: int = 1, post_window: int = 1) -> pd.DataFrame:
        """
        Create holiday indicator features

        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            pre_window: Days before holiday to mark
            post_window: Days after holiday to mark

        Returns:
            DataFrame with holiday indicators
        """
        df = df.copy()

        if timestamp_col not in df.columns:
            logger.warning(f"Timestamp column {timestamp_col} not found")
            return df

        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Create base holiday column
        df['is_holiday'] = 0

        # Convert holiday dates
        for holiday_name, holiday_date_str in HolidayFeatures.CHINESE_HOLIDAYS.items():
            holiday_date = pd.to_datetime(holiday_date_str)

            # Mark holiday and surrounding days
            date_mask = df[timestamp_col].dt.date == holiday_date.date()
            df.loc[date_mask, 'is_holiday'] = 1

            # Mark pre-holiday days
            for day_offset in range(1, pre_window + 1):
                pre_date = (holiday_date - timedelta(days=day_offset)).date()
                pre_mask = df[timestamp_col].dt.date == pre_date
                df.loc[pre_mask, f'pre_{holiday_name}'] = 1

            # Mark post-holiday days
            for day_offset in range(1, post_window + 1):
                post_date = (holiday_date + timedelta(days=day_offset)).date()
                post_mask = df[timestamp_col].dt.date == post_date
                df.loc[post_mask, f'post_{holiday_name}'] = 1

        logger.info("Created holiday indicator features")
        return df

    @staticmethod
    def create_holiday_distance(df: pd.DataFrame, timestamp_col: str = 'ds') -> pd.DataFrame:
        """
        Create feature for days until next holiday

        Args:
            df: Input DataFrame
            timestamp_col: Timestamp column

        Returns:
            DataFrame with holiday distance feature
        """
        df = df.copy()

        if timestamp_col not in df.columns:
            return df

        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Find nearest holiday
        def days_to_next_holiday(date):
            min_distance = float('inf')
            for holiday_date_str in HolidayFeatures.CHINESE_HOLIDAYS.values():
                holiday_date = pd.to_datetime(holiday_date_str)
                distance = (holiday_date - date).days
                if distance >= 0:
                    min_distance = min(min_distance, distance)
            return min_distance if min_distance != float('inf') else 365

        df['days_to_holiday'] = df[timestamp_col].apply(days_to_next_holiday)

        logger.info("Created holiday distance feature")
        return df
