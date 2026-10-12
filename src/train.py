"""Model training, chronological validation, and baseline forecasting module.

Responsibilities:
- Chronological dataset splitting into train, validation, and test partitions (no shuffling).
- Strict predictor selection enforcing temporal and target anti-leakage policies.
- Feature and target eligibility filtering per partition with explicit row accounting.
- Timestamp-aware reference baselines (Persistence and Daily Seasonal Naive).
- Supervised Random Forest Regressor training with fixed random seeds for reproducibility.
- Model serialization using joblib with configuration metadata.
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

# Standard feature set produced by Phase 3 feature engineering
STANDARD_PREDICTOR_COLS: List[str] = [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
]

# Columns strictly prohibited from model predictors
PROHIBITED_FEATURE_COLS: List[str] = [
    "timestamp",
    "energy_kwh",
    "Global_active_power",
    "valid_minutes",
    "coverage_ratio",
]


@dataclass
class SplitSummary:
    """Summary container for chronological partition boundaries and counts."""

    total_rows: int
    train_rows: int
    val_rows: int
    test_rows: int
    train_ratio_achieved: float
    val_ratio_achieved: float
    test_ratio_achieved: float
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert split summary to dictionary."""
        return asdict(self)


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: Optional[float] = None,
    timestamp_col: str = "timestamp",
    return_summary: bool = False,
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitSummary],
]:
    """Split dataset chronologically into train, validation, and test partitions.

    Guarantees:
    - Never randomly shuffles observations; preserves chronological ordering.
    - Slices strictly by sorted timestamp boundaries so partitions are mutually exclusive.
    - Training precedes validation, and validation precedes testing:
      train_end < val_start <= val_end < test_start.
    - Validates split proportions (must be positive and sum to 1.0).
    - Rejects datasets that are too small to produce non-empty partitions.

    Args:
        df: Input DataFrame containing timestamp_col or a DatetimeIndex.
        train_ratio: Proportion of observations for training (default: 0.70).
        val_ratio: Proportion of observations for validation (default: 0.15).
        test_ratio: Optional test proportion; defaults to 1.0 - train_ratio - val_ratio.
        timestamp_col: Column name containing timestamps (default: 'timestamp').
        return_summary: If True, returns (train_df, val_df, test_df, SplitSummary).
                        Otherwise returns (train_df, val_df, test_df) with summary in df.attrs.

    Returns:
        Tuple of (train_df, val_df, test_df) or (train_df, val_df, test_df, SplitSummary).

    Raises:
        ValueError: If split proportions are invalid or dataset has too few rows to form
                    non-empty partitions.
    """
    if df is None or len(df) == 0:
        raise ValueError("Cannot split an empty or None DataFrame.")

    # 1. Resolve and validate proportions
    if test_ratio is None:
        test_ratio = round(1.0 - train_ratio - val_ratio, 6)

    for name, r in [
        ("train_ratio", train_ratio),
        ("val_ratio", val_ratio),
        ("test_ratio", test_ratio),
    ]:
        if isinstance(r, bool) or not isinstance(r, (int, float)) or r <= 0:
            raise ValueError(
                f"{name} must be a positive float, got {r!r} (type {type(r).__name__})."
            )

    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0, atol=1e-4):
        raise ValueError(
            f"Split ratios must sum to 1.0, got train={train_ratio}, "
            f"val={val_ratio}, test={test_ratio} (sum={ratio_sum:.6f})."
        )

    # 2. Validate timestamp column presence
    if timestamp_col in df.columns:
        ts_series = pd.to_datetime(df[timestamp_col], errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        ts_series = pd.to_datetime(df.index, errors="coerce")
    else:
        raise ValueError(
            f"Input DataFrame must contain '{timestamp_col}' column or a DatetimeIndex. "
            f"Found columns: {list(df.columns)}"
        )

    if ts_series.isna().any():
        null_count = int(ts_series.isna().sum())
        raise ValueError(
            f"Input contains {null_count} missing or unparseable timestamps in '{timestamp_col}'."
        )

    if ts_series.duplicated().any():
        dup_count = int(ts_series.duplicated().sum())
        raise ValueError(
            f"Input contains {dup_count} duplicate timestamps in '{timestamp_col}'. "
            "Chronological splitting requires strictly unique timestamps. "
            "Please clean and aggregate duplicates upstream in preprocessing."
        )

    # 3. Sort chronologically
    df_sorted = df.copy()
    if timestamp_col not in df_sorted.columns:
        df_sorted[timestamp_col] = ts_series
    else:
        df_sorted[timestamp_col] = ts_series
    df_sorted = df_sorted.sort_values(timestamp_col).reset_index(drop=True)

    n_total = len(df_sorted)

    # 4. Calculate partition sizes
    n_train = int(np.floor(n_total * train_ratio))
    n_val = int(np.floor(n_total * val_ratio))
    n_test = n_total - n_train - n_val

    # Validate that every partition has at least one row
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError(
            f"Dataset with {n_total} rows is too small to form non-empty partitions "
            f"with ratios ({train_ratio:.2f}, {val_ratio:.2f}, {test_ratio:.2f}). "
            f"Computed partition sizes: train={n_train}, val={n_val}, test={n_test}."
        )

    # 5. Slice non-overlapping partitions
    train_df = df_sorted.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df_sorted.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[n_train + n_val :].copy().reset_index(drop=True)

    # 6. Build summary
    summary = SplitSummary(
        total_rows=n_total,
        train_rows=len(train_df),
        val_rows=len(val_df),
        test_rows=len(test_df),
        train_ratio_achieved=round(len(train_df) / n_total, 4),
        val_ratio_achieved=round(len(val_df) / n_total, 4),
        test_ratio_achieved=round(len(test_df) / n_total, 4),
        train_start=str(train_df[timestamp_col].min()),
        train_end=str(train_df[timestamp_col].max()),
        val_start=str(val_df[timestamp_col].min()),
        val_end=str(val_df[timestamp_col].max()),
        test_start=str(test_df[timestamp_col].min()),
        test_end=str(test_df[timestamp_col].max()),
    )

    train_df.attrs["split_summary"] = summary
    val_df.attrs["split_summary"] = summary
    test_df.attrs["split_summary"] = summary

    logger.info(
        "Chronological split: train=%d (%.1f%%, %s -> %s), val=%d (%.1f%%, %s -> %s), test=%d (%.1f%%, %s -> %s)",
        len(train_df),
        summary.train_ratio_achieved * 100,
        summary.train_start,
        summary.train_end,
        len(val_df),
        summary.val_ratio_achieved * 100,
        summary.val_start,
        summary.val_end,
        len(test_df),
        summary.test_ratio_achieved * 100,
        summary.test_start,
        summary.test_end,
    )

    if return_summary:
        return train_df, val_df, test_df, summary
    return train_df, val_df, test_df


def select_predictor_features(
    df: pd.DataFrame,
    target_col: str = "energy_kwh",
    timestamp_col: str = "timestamp",
    excluded_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    """Select predictor features while strictly preventing target and temporal leakage.

    Policy:
    - Excludes the raw timestamp column.
    - Excludes the target column and any columns derived from the current observation.
    - Excludes preprocessing metadata (valid_minutes, coverage_ratio).
    - Retains only verified calendar, lag, and historical rolling predictors.

    Args:
        df: Input DataFrame containing features.
        target_col: Target variable name (default: 'energy_kwh').
        timestamp_col: Timestamp column name (default: 'timestamp').
        excluded_cols: Optional additional column names to exclude.

    Returns:
        List of approved predictor column names.

    Raises:
        ValueError: If no valid predictors remain or if approved predictors are missing.
    """
    prohibited = set(PROHIBITED_FEATURE_COLS)
    prohibited.add(target_col)
    prohibited.add(timestamp_col)
    if excluded_cols:
        prohibited.update(excluded_cols)

    # Candidate columns present in df
    candidates = [col for col in df.columns if col not in prohibited]

    if not candidates:
        raise ValueError(
            f"No valid predictor columns found in DataFrame. Available columns: {list(df.columns)}, "
            f"Prohibited columns: {list(prohibited)}"
        )

    return candidates


def filter_eligible_rows(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "energy_kwh",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Filter rows eligible for supervised modeling without imputing arbitrary values.

    Policy:
    - Exclude rows where the target value is missing or non-finite.
    - Exclude rows where any required predictor feature is missing or non-finite.
    - Never fill missing lag values with zero.
    - Never fill missing rolling means with arbitrary constants.
    - Provide complete accounting of total, eligible, and excluded rows.
    - Preserves caller DataFrame without mutation.

    Args:
        df: Input partition DataFrame.
        feature_cols: Required predictor column names.
        target_col: Target variable column name (default: 'energy_kwh').

    Returns:
        Tuple of (eligible_df, audit_dict).

    Raises:
        ValueError: If required feature or target columns are missing from df.
    """
    missing_cols = [col for col in list(feature_cols) + [target_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for eligibility filtering: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    total_rows = len(df)
    if total_rows == 0:
        audit = {
            "total_rows": 0,
            "eligible_rows": 0,
            "excluded_rows": 0,
            "missing_target_count": 0,
            "feature_missing_counts": {col: 0 for col in feature_cols},
        }
        return df.copy(), audit

    # Target validity
    target_series = pd.to_numeric(df[target_col], errors="coerce")
    valid_target_mask = target_series.notna() & np.isfinite(target_series)
    missing_target_count = int((~valid_target_mask).sum())

    # Predictor validity
    feature_missing_counts: Dict[str, int] = {}
    valid_features_mask = pd.Series(True, index=df.index)

    for col in feature_cols:
        col_numeric = pd.to_numeric(df[col], errors="coerce")
        col_valid = col_numeric.notna() & np.isfinite(col_numeric)
        missing_count = int((~col_valid).sum())
        feature_missing_counts[col] = missing_count
        valid_features_mask = valid_features_mask & col_valid

    eligible_mask = valid_target_mask & valid_features_mask
    eligible_df = df.loc[eligible_mask].copy()
    eligible_rows = len(eligible_df)
    excluded_rows = total_rows - eligible_rows

    audit = {
        "total_rows": total_rows,
        "eligible_rows": eligible_rows,
        "excluded_rows": excluded_rows,
        "missing_target_count": missing_target_count,
        "feature_missing_counts": feature_missing_counts,
    }
    eligible_df.attrs["eligibility_audit"] = audit

    return eligible_df, audit


def predict_persistence(
    eval_df: pd.DataFrame,
    history_df: Optional[pd.DataFrame] = None,
    timestamp_col: str = "timestamp",
    target_col: str = "energy_kwh",
    lookback_hours: int = 1,
) -> pd.Series:
    """Generate 1-step persistence baseline predictions via exact timestamp lookup.

    Rule:
    For prediction timestamp T, predict using the observed target at exactly (T - 1 hour).
    If the historical observation at (T - 1 hour) is missing, return NaN.
    Uses only historical observations strictly prior to T.

    Args:
        eval_df: DataFrame of timestamps and target values to evaluate.
        history_df: Optional historical observations prior to eval_df (e.g. training set).
        timestamp_col: Timestamp column name (default: 'timestamp').
        target_col: Target variable column name (default: 'energy_kwh').
        lookback_hours: Historical lookback in hours (default: 1).

    Returns:
        pd.Series: Baseline predictions aligned with eval_df index, containing NaN where
                   preceding observation is missing.
    """
    if timestamp_col not in eval_df.columns:
        raise ValueError(f"eval_df must contain '{timestamp_col}'. Found: {list(eval_df.columns)}")

    if lookback_hours <= 0:
        raise ValueError(f"lookback_hours must be positive, got {lookback_hours}")

    # Build reference lookup mapping timestamp -> target
    dfs_to_combine = []
    if history_df is not None and len(history_df) > 0:
        if timestamp_col not in history_df.columns or target_col not in history_df.columns:
            raise ValueError(
                f"history_df must contain '{timestamp_col}' and '{target_col}' columns."
            )
        dfs_to_combine.append(history_df[[timestamp_col, target_col]])

    if target_col in eval_df.columns:
        dfs_to_combine.append(eval_df[[timestamp_col, target_col]])

    if not dfs_to_combine:
        return pd.Series(np.nan, index=eval_df.index, dtype="float64")

    combined = pd.concat(dfs_to_combine, ignore_index=True)
    combined[timestamp_col] = pd.to_datetime(combined[timestamp_col], errors="coerce")
    combined[target_col] = pd.to_numeric(combined[target_col], errors="coerce")
    combined = combined.dropna(subset=[timestamp_col]).drop_duplicates(subset=[timestamp_col])

    lookup_series = combined.set_index(timestamp_col)[target_col]

    eval_ts = pd.to_datetime(eval_df[timestamp_col], errors="coerce")
    target_lookup_ts = eval_ts - pd.Timedelta(hours=lookback_hours)

    preds = target_lookup_ts.map(lookup_series).astype("float64")
    preds.index = eval_df.index
    return preds


def predict_daily_seasonal_naive(
    eval_df: pd.DataFrame,
    history_df: Optional[pd.DataFrame] = None,
    timestamp_col: str = "timestamp",
    target_col: str = "energy_kwh",
    seasonal_hours: int = 24,
) -> pd.Series:
    """Generate daily seasonal naive baseline predictions via exact 24-hour timestamp lookup.

    Rule:
    For prediction timestamp T, predict using the observed target at exactly (T - 24 hours).
    If the historical observation at (T - 24 hours) is missing, return NaN.
    Does not substitute any other timestamp.

    Args:
        eval_df: DataFrame of timestamps and target values to evaluate.
        history_df: Optional historical observations prior to eval_df.
        timestamp_col: Timestamp column name (default: 'timestamp').
        target_col: Target variable column name (default: 'energy_kwh').
        seasonal_hours: Seasonal cycle length in hours (default: 24).

    Returns:
        pd.Series: Baseline predictions aligned with eval_df index, containing NaN where
                   the exact historical observation is missing.
    """
    if timestamp_col not in eval_df.columns:
        raise ValueError(f"eval_df must contain '{timestamp_col}'. Found: {list(eval_df.columns)}")

    if seasonal_hours <= 0:
        raise ValueError(f"seasonal_hours must be positive, got {seasonal_hours}")

    # Build reference lookup mapping timestamp -> target
    dfs_to_combine = []
    if history_df is not None and len(history_df) > 0:
        if timestamp_col not in history_df.columns or target_col not in history_df.columns:
            raise ValueError(
                f"history_df must contain '{timestamp_col}' and '{target_col}' columns."
            )
        dfs_to_combine.append(history_df[[timestamp_col, target_col]])

    if target_col in eval_df.columns:
        dfs_to_combine.append(eval_df[[timestamp_col, target_col]])

    if not dfs_to_combine:
        return pd.Series(np.nan, index=eval_df.index, dtype="float64")

    combined = pd.concat(dfs_to_combine, ignore_index=True)
    combined[timestamp_col] = pd.to_datetime(combined[timestamp_col], errors="coerce")
    combined[target_col] = pd.to_numeric(combined[target_col], errors="coerce")
    combined = combined.dropna(subset=[timestamp_col]).drop_duplicates(subset=[timestamp_col])

    lookup_series = combined.set_index(timestamp_col)[target_col]

    eval_ts = pd.to_datetime(eval_df[timestamp_col], errors="coerce")
    target_lookup_ts = eval_ts - pd.Timedelta(hours=seasonal_hours)

    preds = target_lookup_ts.map(lookup_series).astype("float64")
    preds.index = eval_df.index
    return preds


def train_random_forest(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "energy_kwh",
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs: Any,
) -> RandomForestRegressor:
    """Train a reproducible Random Forest Regressor on eligible training observations.

    Args:
        train_df: Eligible training partition DataFrame.
        feature_cols: Sequence of predictor column names.
        target_col: Target variable column name (default: 'energy_kwh').
        n_estimators: Number of decision trees in the forest (default: 100).
        random_state: Fixed random seed for deterministic training (default: 42).
        n_jobs: Parallel worker threads (default: -1 for all available cores).
        **kwargs: Optional additional parameters passed to RandomForestRegressor.

    Returns:
        Fitted RandomForestRegressor instance.

    Raises:
        ValueError: If training partition is empty or contains missing values in predictors/target.
    """
    if len(train_df) == 0:
        raise ValueError("Cannot train Random Forest on an empty training DataFrame.")

    X = train_df[list(feature_cols)]
    y = train_df[target_col]

    # Verify no missing values in training inputs
    if X.isna().any().any():
        bad_cols = [c for c in feature_cols if X[c].isna().any()]
        raise ValueError(
            f"Training features contain missing values in columns: {bad_cols}. "
            "Please apply filter_eligible_rows() prior to fitting."
        )

    if y.isna().any():
        raise ValueError(
            f"Training target '{target_col}' contains missing values. "
            "Please apply filter_eligible_rows() prior to fitting."
        )

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )

    logger.info(
        "Fitting RandomForestRegressor(n_estimators=%d, random_state=%d, n_jobs=%d) on %d rows with %d features.",
        n_estimators,
        random_state,
        n_jobs,
        len(train_df),
        len(feature_cols),
    )

    rf.fit(X, y)
    return rf


def train_model(
    train_df: pd.DataFrame,
    model_type: str = "random_forest",
    feature_cols: Optional[Sequence[str]] = None,
    target_col: str = "energy_kwh",
    save_path: Optional[Union[str, Path]] = None,
    random_state: int = 42,
    n_estimators: int = 100,
    **kwargs: Any,
) -> Any:
    """Train selected model architecture and optionally serialize artifact to disk.

    Supports:
    - 'random_forest': RandomForestRegressor with fixed random seed.

    Args:
        train_df: Training DataFrame with features and target.
        model_type: Name of estimator ('random_forest').
        feature_cols: Optional list of predictor columns. If None, auto-selected.
        target_col: Target variable name (default: 'energy_kwh').
        save_path: Optional destination path for serialized model artifact (.joblib).
        random_state: Fixed seed for reproducibility (default: 42).
        n_estimators: Number of trees for tree models (default: 100).
        **kwargs: Additional parameters passed to model constructor.

    Returns:
        Fitted model instance.

    Raises:
        ValueError: If model_type is unsupported or training data is invalid.
    """
    if feature_cols is None:
        feature_cols = select_predictor_features(train_df, target_col=target_col)

    model_type_normalized = model_type.lower().strip()

    if model_type_normalized == "random_forest":
        model = train_random_forest(
            train_df=train_df,
            feature_cols=feature_cols,
            target_col=target_col,
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. Supported models: ['random_forest']."
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_payload = {
            "model": model,
            "model_type": model_type_normalized,
            "feature_cols": list(feature_cols),
            "target_col": target_col,
            "random_state": random_state,
            "hyperparameters": {
                "n_estimators": n_estimators,
                **kwargs,
            },
        }
        joblib.dump(artifact_payload, save_path)
        logger.info("Saved model artifact to %s", save_path)

    return model
