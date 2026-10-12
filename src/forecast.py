"""Forecasting and inference execution module (Phase 4B).

Responsibilities:
- Validate and load serialized forecasting model artifacts (e.g., Random Forest).
- Validate, clean, and extract required contiguous historical observation windows.
- Perform recursive multi-step autoregressive rollout across horizons H in [1, 168].
- Maintain strict feature ordering and prevent lookahead data leakage.
- Generate structured forecast predictions with timestamps, steps, and audit metadata.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum historical hours required for 168-hour lag feature
MIN_REQUIRED_HISTORY_HOURS: int = 168
MAX_FORECAST_HORIZON_HOURS: int = 168
MIN_FORECAST_HORIZON_HOURS: int = 1


def load_forecasting_model(model_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate serialized forecasting model artifact.

    Args:
        model_path: Path to serialized model joblib artifact.

    Returns:
        Dict[str, Any]: Model payload containing estimator and metadata.

    Raises:
        FileNotFoundError: If model_path does not exist on disk.
        ValueError: If artifact payload is corrupt, not a dict, or missing required keys.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Forecasting model artifact not found at '{path.resolve()}'. "
            "Please run 'python main.py --stage train' first to produce the artifact."
        )

    try:
        payload = joblib.load(path)
    except Exception as exc:
        raise ValueError(
            f"Failed to deserialize model artifact at '{path}': {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid model artifact format at '{path}'. Expected dict, got {type(payload).__name__}."
        )

    required_keys = {"model", "feature_cols", "target_col"}
    missing_keys = required_keys - set(payload.keys())
    if missing_keys:
        raise ValueError(
            f"Model artifact at '{path}' is missing required keys: {sorted(list(missing_keys))}."
        )

    model = payload["model"]
    if not hasattr(model, "predict"):
        raise ValueError(
            f"Loaded model object {type(model).__name__} does not implement 'predict' method."
        )

    feature_cols = payload["feature_cols"]
    if not isinstance(feature_cols, (list, tuple)) or len(feature_cols) == 0:
        raise ValueError(
            f"Model artifact has invalid feature_cols specification: {feature_cols}."
        )

    logger.info(
        "Successfully loaded model artifact from '%s' (%s, %d features).",
        path,
        payload.get("model_type", "unknown"),
        len(feature_cols),
    )
    return payload


def validate_and_prepare_history(
    history: Union[pd.DataFrame, str, Path],
    target_col: str = "energy_kwh",
    min_required_hours: int = MIN_REQUIRED_HISTORY_HOURS,
) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Validate historical observations and extract a contiguous ending window.

    The forecasting engine requires at least `min_required_hours` contiguous hourly
    observations ending at the forecast origin to compute lag_168, lag_24, lag_1,
    and rolling_mean_24 without data gaps or imputation.

    Args:
        history: DataFrame or file path to processed historical CSV.
        target_col: Name of consumption target column (default: 'energy_kwh').
        min_required_hours: Minimum contiguous lookback required (default: 168).

    Returns:
        Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
            - Contiguous historical window DataFrame with standardized ['timestamp', target_col].
            - History start timestamp (T_origin - 167h).
            - History end timestamp / forecast origin (T_origin).

    Raises:
        FileNotFoundError: If history file path does not exist.
        ValueError: If history has invalid schema, empty rows, null/duplicate timestamps,
                    non-finite targets, or missing timestamps/gaps in the ending window.
    """
    if isinstance(history, (str, Path)):
        hist_path = Path(history)
        if not hist_path.is_file():
            raise FileNotFoundError(f"Historical data file not found at '{hist_path.resolve()}'.")
        df = pd.read_csv(hist_path)
    elif isinstance(history, pd.DataFrame):
        df = history.copy()
    else:
        raise ValueError(
            f"Invalid history argument: expected DataFrame, str, or Path, got {type(history).__name__}."
        )

    if len(df) == 0:
        raise ValueError("Historical dataset is empty.")

    # Timestamp column normalization (accept 'timestamp' or 'datetime')
    if "timestamp" in df.columns:
        ts_col = "timestamp"
    elif "datetime" in df.columns:
        df["timestamp"] = df["datetime"]
        ts_col = "timestamp"
    elif isinstance(df.index, pd.DatetimeIndex):
        df["timestamp"] = df.index
        ts_col = "timestamp"
    else:
        raise ValueError(
            f"Historical data must contain 'timestamp' or 'datetime' column. Found: {list(df.columns)}"
        )

    if target_col not in df.columns:
        raise ValueError(
            f"Historical data must contain target column '{target_col}'. Found: {list(df.columns)}"
        )

    # Validate timestamps
    try:
        df["timestamp"] = pd.to_datetime(df[ts_col])
    except Exception as exc:
        raise ValueError(f"Failed to parse timestamps in historical data: {exc}") from exc

    if df["timestamp"].isna().any():
        raise ValueError("Historical data contains null or unparseable timestamps.")

    if df["timestamp"].duplicated().any():
        dup_count = df["timestamp"].duplicated().sum()
        raise ValueError(f"Historical data contains {dup_count} duplicate timestamp(s).")

    # Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Check hourly alignment
    non_hourly = (df["timestamp"].dt.minute != 0) | (df["timestamp"].dt.second != 0)
    if non_hourly.any():
        bad_sample = df.loc[non_hourly, "timestamp"].iloc[0]
        raise ValueError(
            f"Historical timestamps must align to top-of-hour boundaries. Found non-hourly timestamp: {bad_sample}"
        )

    if len(df) < min_required_hours:
        raise ValueError(
            f"Historical data has {len(df)} rows, but at least {min_required_hours} contiguous hours "
            f"are required to compute the 168-hour lag feature."
        )

    # Determine forecast origin from the latest historical timestamp
    forecast_origin: pd.Timestamp = df["timestamp"].iloc[-1]
    expected_start: pd.Timestamp = forecast_origin - pd.Timedelta(hours=min_required_hours - 1)

    # Extract required ending window
    window_mask = (df["timestamp"] >= expected_start) & (df["timestamp"] <= forecast_origin)
    window_df = df.loc[window_mask, ["timestamp", target_col]].copy().reset_index(drop=True)

    if len(window_df) != min_required_hours:
        raise ValueError(
            f"Required {min_required_hours}-hour historical window from {expected_start} "
            f"to {forecast_origin} contains missing timestamps or recording gaps. "
            f"Found {len(window_df)} intervals instead of expected {min_required_hours}."
        )

    # Verify contiguous 1-hour spacing
    hourly_diffs = window_df["timestamp"].diff().iloc[1:]
    expected_step = pd.Timedelta(hours=1)
    if not (hourly_diffs == expected_step).all():
        gap_idx = window_df.index[window_df["timestamp"].diff() > expected_step].tolist()[0]
        gap_before = window_df.loc[gap_idx - 1, "timestamp"]
        gap_after = window_df.loc[gap_idx, "timestamp"]
        raise ValueError(
            f"Gap detected in required historical window between {gap_before} and {gap_after}."
        )

    # Validate target values in window
    target_series = window_df[target_col]
    if target_series.isna().any():
        raise ValueError(
            f"Historical '{target_col}' contains missing or NaN values within the required lookback window."
        )

    if not np.isfinite(target_series.to_numpy(dtype="float64")).all():
        raise ValueError(
            f"Historical '{target_col}' contains non-finite (infinite) values within the required lookback window."
        )

    logger.info(
        "Extracted contiguous historical window: %s to %s (%d hours). Forecast origin: %s",
        expected_start,
        forecast_origin,
        len(window_df),
        forecast_origin,
    )
    return window_df, expected_start, forecast_origin


def generate_forecast(
    model_or_path: Union[str, Path, Dict[str, Any]],
    history: Union[pd.DataFrame, str, Path],
    horizon_hours: int = 24,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate multi-hour recursive forward forecast from historical energy consumption.

    Rollout Procedure:
    1. Validates the model artifact and extracts stored feature ordering.
    2. Validates history and extracts the contiguous ending window up to forecast origin T_origin.
    3. Maintains a rolling memory buffer containing actual historical values and previous step predictions.
    4. For each forward step s in [1, horizon_hours]:
       - Derives calendar features from future timestamp T_s = T_origin + s * 1h.
       - Retrieves lag_1, lag_24, and lag_168 from the recursive buffer.
       - Computes rolling_mean_24 across the 24 hours immediately preceding T_s (excluding T_s).
       - Constructs the feature vector adhering strictly to the model's feature column ordering.
       - Generates point prediction y_hat, verifies it is finite, and appends it to the buffer.
    5. Formats results with columns ['timestamp', 'forecast_energy_kwh', 'step'] and metadata attributes.

    Args:
        model_or_path: Path to serialized model artifact or already loaded artifact payload dict.
        history: Processed historical observations DataFrame or CSV path.
        horizon_hours: Number of future hours to forecast (1 to 168 inclusive, default: 24).
        output_path: Optional path to save forecast CSV.

    Returns:
        pd.DataFrame: Forecast results containing ['timestamp', 'forecast_energy_kwh', 'step']
                      with metadata in df.attrs.

    Raises:
        ValueError: If horizon_hours is invalid, features cannot be resolved, or non-finite predictions occur.
    """
    if isinstance(horizon_hours, bool) or not isinstance(horizon_hours, int):
        raise ValueError(
            f"horizon_hours must be an integer, got {type(horizon_hours).__name__}."
        )

    if not (MIN_FORECAST_HORIZON_HOURS <= horizon_hours <= MAX_FORECAST_HORIZON_HOURS):
        raise ValueError(
            f"horizon_hours must be between {MIN_FORECAST_HORIZON_HOURS} and "
            f"{MAX_FORECAST_HORIZON_HOURS} inclusive, got {horizon_hours}."
        )

    # 1. Load model artifact
    if isinstance(model_or_path, (str, Path)):
        payload = load_forecasting_model(model_or_path)
    elif isinstance(model_or_path, dict):
        payload = model_or_path
        required_keys = {"model", "feature_cols", "target_col"}
        if not required_keys.issubset(payload.keys()):
            raise ValueError(f"Model payload dict missing required keys: {required_keys - set(payload.keys())}")
        if not hasattr(payload["model"], "predict"):
            raise ValueError(f"Model object {type(payload['model']).__name__} does not have 'predict' method.")
    else:
        raise ValueError(
            f"Invalid model_or_path argument: expected Path or dict, got {type(model_or_path).__name__}."
        )

    model = payload["model"]
    feature_cols: List[str] = list(payload["feature_cols"])
    target_col: str = str(payload.get("target_col", "energy_kwh"))
    model_type: str = str(payload.get("model_type", "unknown"))

    # 2. Prepare historical window
    window_df, history_start, forecast_origin = validate_and_prepare_history(
        history=history,
        target_col=target_col,
        min_required_hours=MIN_REQUIRED_HISTORY_HOURS,
    )

    # 3. Initialize recursive buffer with verified actual values
    buffer: Dict[pd.Timestamp, float] = {
        ts: float(val) for ts, val in zip(window_df["timestamp"], window_df[target_col])
    }

    records: List[Dict[str, Any]] = []

    # 4. Recursive rollout
    logger.info(
        "Starting recursive forecast rollout: origin=%s, horizon=%d hours.",
        forecast_origin,
        horizon_hours,
    )

    for step in range(1, horizon_hours + 1):
        step_ts: pd.Timestamp = forecast_origin + pd.Timedelta(hours=step)

        # Calendar features derived from step_ts
        cal_features = {
            "hour": int(step_ts.hour),
            "day_of_week": int(step_ts.dayofweek),
            "day_of_month": int(step_ts.day),
            "month": int(step_ts.month),
            "is_weekend": int(1 if step_ts.dayofweek in [5, 6] else 0),
        }

        # Lag features from buffer
        ts_lag1 = step_ts - pd.Timedelta(hours=1)
        ts_lag24 = step_ts - pd.Timedelta(hours=24)
        ts_lag168 = step_ts - pd.Timedelta(hours=168)

        if ts_lag1 not in buffer:
            raise ValueError(f"Step {step} ({step_ts}): Missing required lag_1 value at {ts_lag1}.")
        if ts_lag24 not in buffer:
            raise ValueError(f"Step {step} ({step_ts}): Missing required lag_24 value at {ts_lag24}.")
        if ts_lag168 not in buffer:
            raise ValueError(f"Step {step} ({step_ts}): Missing required lag_168 value at {ts_lag168}.")

        lag_1 = float(buffer[ts_lag1])
        lag_24 = float(buffer[ts_lag24])
        lag_168 = float(buffer[ts_lag168])

        # Rolling 24-hour mean from preceding 24 hours strictly prior to step_ts
        preceding_vals = []
        for offset in range(1, 25):
            ref_ts = step_ts - pd.Timedelta(hours=offset)
            if ref_ts not in buffer:
                raise ValueError(
                    f"Step {step} ({step_ts}): Missing required rolling window value at {ref_ts}."
                )
            preceding_vals.append(buffer[ref_ts])

        rolling_mean_24 = float(np.mean(preceding_vals))

        step_features = {
            **cal_features,
            "lag_1": lag_1,
            "lag_24": lag_24,
            "lag_168": lag_168,
            "rolling_mean_24": rolling_mean_24,
        }

        # Check all required features exist
        for col in feature_cols:
            if col not in step_features:
                raise ValueError(
                    f"Step {step}: Feature '{col}' required by model was not computed. "
                    f"Available: {list(step_features.keys())}"
                )

        # Build feature DataFrame adhering strictly to model's feature column order
        X_step = pd.DataFrame(
            [[step_features[col] for col in feature_cols]],
            columns=feature_cols,
        )

        # Validate feature finiteness before calling model.predict
        feature_array = X_step.to_numpy(dtype="float64")
        if not np.isfinite(feature_array).all():
            raise ValueError(
                f"Step {step} ({step_ts}): One or more input features are non-finite: {step_features}"
            )

        # Predict
        pred_raw = model.predict(X_step)
        pred_val = float(pred_raw[0])

        if not np.isfinite(pred_val):
            raise ValueError(
                f"Step {step} ({step_ts}): Model generated non-finite prediction: {pred_val}"
            )

        if pred_val < 0.0:
            logger.warning(
                "Step %d (%s): Predicted negative consumption (%.4f kWh). "
                "Preserving prediction without silent clipping.",
                step,
                step_ts,
                pred_val,
            )

        # Append to recursive history buffer for subsequent steps
        buffer[step_ts] = pred_val

        records.append(
            {
                "timestamp": step_ts,
                "forecast_energy_kwh": round(pred_val, 6),
                "step": step,
            }
        )

    forecast_df = pd.DataFrame(records)
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
    forecast_df["forecast_energy_kwh"] = forecast_df["forecast_energy_kwh"].astype("float64")
    forecast_df["step"] = forecast_df["step"].astype("int64")

    # Attach provenance attributes
    forecast_df.attrs["forecast_origin"] = str(forecast_origin)
    forecast_df.attrs["history_start"] = str(history_start)
    forecast_df.attrs["history_end"] = str(forecast_origin)
    forecast_df.attrs["horizon_hours"] = int(horizon_hours)
    forecast_df.attrs["model_type"] = model_type

    # Optionally persist to disk
    if output_path is not None:
        save_path = Path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        forecast_df.to_csv(save_path, index=False)
        logger.info(
            "Saved %d-hour forecast to '%s'.",
            len(forecast_df),
            save_path.resolve(),
        )

    return forecast_df
