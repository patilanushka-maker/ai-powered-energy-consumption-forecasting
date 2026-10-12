"""Time-series feature engineering module for energy consumption forecasting.

Responsibilities:
- Extract calendar and temporal features from timestamps (hour, day_of_week, day_of_month, month, is_weekend).
- Construct timestamp-aware autoregressive lag features (lag_1, lag_24, lag_168) handling missing hours.
- Construct leakage-free rolling window aggregations (rolling_mean_24) with an explicit completeness policy.
- Enforce strict parameter validation (positive integers for lags and rolling window).
- Enforce unique timestamp integrity (duplicate timestamps rejected with clear error).
- Preserve target and metadata columns without dropping incomplete feature rows.
"""

import logging
from typing import Sequence, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_temporal_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Extract calendar and temporal features from a timestamp column.

    Features generated:
    - hour: Hour of the day (0-23).
    - day_of_week: Day of the week (0=Monday, 6=Sunday).
    - day_of_month: Day of the month (1-31).
    - month: Month of the year (1-12).
    - is_weekend: Binary indicator (1 if Saturday or Sunday, else 0).

    Args:
        df: Input DataFrame containing timestamp_col or a DatetimeIndex.
        timestamp_col: Name of the timestamp column (default: 'timestamp').

    Returns:
        pd.DataFrame: New DataFrame containing original columns plus temporal features,
                      sorted chronologically. Caller DataFrame is not mutated.

    Raises:
        ValueError: If timestamp_col is not found, or contains null/unparseable timestamps,
                    or contains duplicate timestamps.
    """
    if timestamp_col in df.columns:
        ts_raw = df[timestamp_col]
    elif isinstance(df.index, pd.DatetimeIndex):
        ts_raw = df.index.to_series()
    else:
        raise ValueError(
            f"Input DataFrame must contain '{timestamp_col}' column or a DatetimeIndex. "
            f"Found columns: {list(df.columns)}"
        )

    res = df.copy()

    # Empty DataFrame handling
    if len(df) == 0:
        res["hour"] = pd.Series(dtype="int64")
        res["day_of_week"] = pd.Series(dtype="int64")
        res["day_of_month"] = pd.Series(dtype="int64")
        res["month"] = pd.Series(dtype="int64")
        res["is_weekend"] = pd.Series(dtype="int64")
        return res

    ts = pd.to_datetime(ts_raw, errors="coerce")

    # Validate non-null timestamps
    if ts.isna().any():
        null_count = int(ts.isna().sum())
        raise ValueError(
            f"Input contains {null_count} missing or unparseable timestamps in '{timestamp_col}'. "
            "Timestamps must be valid and non-null."
        )

    # Validate unique timestamps policy
    if ts.duplicated().any():
        dup_count = int(ts.duplicated().sum())
        raise ValueError(
            f"Input contains {dup_count} duplicate timestamps. Feature engineering requires "
            "strictly unique timestamps. Please clean and aggregate duplicate observations "
            "upstream in preprocessing before generating features."
        )

    res[timestamp_col] = ts

    # Sort chronologically to preserve time ordering
    res = res.sort_values(timestamp_col).reset_index(drop=True)
    ts_sorted = res[timestamp_col]

    # Calendar feature extraction
    res["hour"] = ts_sorted.dt.hour.astype("int64")
    res["day_of_week"] = ts_sorted.dt.dayofweek.astype("int64")
    res["day_of_month"] = ts_sorted.dt.day.astype("int64")
    res["month"] = ts_sorted.dt.month.astype("int64")
    res["is_weekend"] = (ts_sorted.dt.dayofweek >= 5).astype("int64")

    return res


def create_lag_and_rolling_features(
    df: pd.DataFrame,
    target_col: str = "energy_kwh",
    timestamp_col: str = "timestamp",
    lags: Sequence[int] = (1, 24, 168),
    rolling_window_hours: int = 24,
) -> pd.DataFrame:
    """Generate timestamp-aware autoregressive lag and rolling window features.

    Parameter Validation & Missing Timestamp Policy:
    -------------------------------------------------
    - `lags` must be a non-empty sequence of positive integers (e.g. (1, 24, 168)).
      Zero, negative values, booleans, and floats are strictly rejected.
    - `rolling_window_hours` must be a positive integer (e.g. 24).
      Zero, negative values, booleans, and non-integers are strictly rejected.
    - Timestamps must be valid, non-null, and unique. Duplicate timestamps are rejected
      with a ValueError to require explicit upstream resolution in preprocessing.
    - Lags are calculated using exact temporal intervals (e.g. lag_1 refers to timestamp - 1 hour,
      lag_24 refers to timestamp - 24 hours, lag_168 refers to timestamp - 168 hours).
    - If the exact historical timestamp is unavailable in the series, the lag value is set to NaN.
      Row positions are NEVER shifted naively when time gaps exist.

    Rolling Window & Completeness Policy:
    -------------------------------------
    - Rolling features strictly use historical observations strictly BEFORE the target timestamp T
      (target value at T is never included in its own rolling feature).
    - Policy for incomplete windows: rolling_mean_{window} requires a complete set of contiguous hourly
      observations in the lookback window [T - window, T - 1h]. If any hour is missing or NaN,
      the rolling mean evaluates to NaN to avoid bias. Missing values are never filled with constants.

    Row Preservation Policy:
    ------------------------
    - Rows with NaN lag or rolling features are preserved so that subsequent model training
      stages can apply a documented selection policy. No rows are dropped.

    Args:
        df: Input DataFrame containing timestamp and target columns.
        target_col: Target variable column name (default: 'energy_kwh').
        timestamp_col: Timestamp column name (default: 'timestamp').
        lags: Non-empty sequence of positive integer hourly lags (default: (1, 24, 168)).
        rolling_window_hours: Positive integer window size in hours for preceding rolling mean (default: 24).

    Returns:
        pd.DataFrame: New DataFrame containing original columns, temporal lags, and rolling mean,
                      sorted chronologically. Caller DataFrame is not mutated.

    Raises:
        ValueError: If parameters are invalid, required columns are missing, or timestamps contain
                    nulls or duplicates.
    """
    # 1. Validate rolling_window_hours parameter
    if (
        isinstance(rolling_window_hours, bool)
        or not isinstance(rolling_window_hours, int)
        or rolling_window_hours <= 0
    ):
        raise ValueError(
            f"rolling_window_hours must be a positive integer, got {rolling_window_hours!r} "
            f"(type {type(rolling_window_hours).__name__})."
        )

    # 2. Validate lags parameter
    if (
        not isinstance(lags, (tuple, list))
        or len(lags) == 0
        or any(
            isinstance(lag, bool) or not isinstance(lag, int) or lag <= 0
            for lag in lags
        )
    ):
        raise ValueError(
            f"lags must be a non-empty sequence of positive integers (e.g. (1, 24, 168)), got {lags!r}."
        )

    # 3. Validate required columns
    if timestamp_col not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            f"Input DataFrame must contain '{timestamp_col}' column or a DatetimeIndex."
        )

    if target_col not in df.columns:
        raise ValueError(
            f"Input DataFrame must contain target column '{target_col}'. "
            f"Found columns: {list(df.columns)}"
        )

    res = df.copy()

    # Empty DataFrame handling
    if len(df) == 0:
        for lag in lags:
            res[f"lag_{lag}"] = pd.Series(dtype="float64")
        res[f"rolling_mean_{rolling_window_hours}"] = pd.Series(dtype="float64")
        return res

    # Extract timestamps
    if timestamp_col in res.columns:
        ts_parsed = pd.to_datetime(res[timestamp_col], errors="coerce")
    else:
        ts_parsed = pd.to_datetime(res.index, errors="coerce")

    # Validate non-null timestamps
    if ts_parsed.isna().any():
        null_count = int(ts_parsed.isna().sum())
        raise ValueError(
            f"Input contains {null_count} missing or unparseable timestamps in '{timestamp_col}'. "
            "Timestamps must be valid and non-null."
        )

    # Validate duplicate timestamps policy
    if ts_parsed.duplicated().any():
        dup_count = int(ts_parsed.duplicated().sum())
        raise ValueError(
            f"Input contains {dup_count} duplicate timestamps. Feature engineering requires "
            "strictly unique timestamps. Please clean and aggregate duplicate observations "
            "upstream in preprocessing before generating features."
        )

    res[timestamp_col] = ts_parsed

    # Sort chronologically to preserve ordering
    res = res.sort_values(timestamp_col).reset_index(drop=True)

    # 4. Exact Timestamp-Aware Lag Calculation
    # For each lag L, match row at T with observation at (T - L hours)
    for lag in lags:
        lag_col_name = f"lag_{lag}"
        lookup_df = pd.DataFrame(
            {
                "join_key": res[timestamp_col] + pd.Timedelta(hours=lag),
                lag_col_name: pd.to_numeric(res[target_col], errors="coerce").values,
            }
        )
        lookup_df = lookup_df.drop_duplicates(subset=["join_key"])

        res = res.merge(
            lookup_df,
            left_on=timestamp_col,
            right_on="join_key",
            how="left",
        )
        res = res.drop(columns=["join_key"])

    # 5. Strict Leakage-Free Rolling Feature Calculation on Calendar Hourly Grid
    min_ts = res[timestamp_col].min()
    max_ts = res[timestamp_col].max()
    full_calendar_grid = pd.date_range(start=min_ts, end=max_ts, freq="h")

    target_series = pd.Series(
        data=pd.to_numeric(res[target_col], errors="coerce").values,
        index=res[timestamp_col].values,
    )
    grid_series = target_series.reindex(full_calendar_grid)

    # Shift by 1 hour so observation at T uses strictly observations in [T - window, T - 1]
    # Enforce min_periods=rolling_window_hours to require complete lookback window
    rolling_col_name = f"rolling_mean_{rolling_window_hours}"
    shifted_rolling = (
        grid_series.shift(1)
        .rolling(window=rolling_window_hours, min_periods=rolling_window_hours)
        .mean()
    )

    # Map rolling values back to original timestamps
    res[rolling_col_name] = res[timestamp_col].map(shifted_rolling).astype("float64")

    return res
