"""Helper utilities and data ingestion routines for the Streamlit dashboard.

Responsibilities:
- Safe, validated loading of processed hourly consumption datasets.
- Safe loading and parsing of Phase 4A and Phase 4B evaluation/backtest metric reports.
- Date range filtering and summary statistics calculation.
- Preparation of unified datasets for seamless historical + forecast visualization.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_processed_data(
    file_path: Union[str, Path] = Path("data/processed/hourly_energy.csv")
) -> pd.DataFrame:
    """Load and validate processed hourly energy consumption dataset.

    Args:
        file_path: Path to hourly processed CSV.

    Returns:
        pd.DataFrame: Clean DataFrame containing ['timestamp', 'energy_kwh'] sorted chronologically.

    Raises:
        FileNotFoundError: If the processed CSV does not exist.
        ValueError: If required columns are missing or file is empty.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed dataset not found at '{path.resolve()}'. "
            "Please run 'python main.py --stage preprocess' to generate it."
        )

    df = pd.read_csv(path)
    if len(df) == 0:
        raise ValueError("Processed dataset is empty.")

    ts_col = "timestamp" if "timestamp" in df.columns else ("datetime" if "datetime" in df.columns else None)
    if ts_col is None:
        raise ValueError(f"Processed dataset missing 'timestamp' column. Found: {list(df.columns)}")

    if "energy_kwh" not in df.columns:
        raise ValueError(f"Processed dataset missing 'energy_kwh' column. Found: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df[ts_col])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_metrics_report(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate JSON metrics artifact.

    Args:
        file_path: Path to metrics JSON artifact.

    Returns:
        Dict[str, Any]: Parsed JSON report dictionary.

    Raises:
        FileNotFoundError: If the metrics file does not exist.
        ValueError: If JSON deserialization fails.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Metrics report not found at '{path.resolve()}'. "
            "Please run the appropriate training or backtesting pipeline stage."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise ValueError(f"Failed to parse metrics report at '{path}': {exc}") from exc


def filter_data_by_date_range(
    df: pd.DataFrame,
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
) -> pd.DataFrame:
    """Filter DataFrame between start_date and end_date (inclusive of full days).

    Args:
        df: Input DataFrame containing 'timestamp' column.
        start_date: Start date boundary.
        end_date: End date boundary.

    Returns:
        pd.DataFrame: Filtered rows sorted by timestamp.
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp' column.")

    start_ts = pd.Timestamp(start_date).floor("D")
    end_ts = pd.Timestamp(end_date).ceil("D")

    if start_ts > end_ts:
        raise ValueError(f"Start date ({start_ts}) cannot be after end date ({end_ts}).")

    mask = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
    return df.loc[mask].copy().reset_index(drop=True)


def compute_summary_statistics(
    df: pd.DataFrame,
    col: str = "energy_kwh",
) -> Dict[str, float]:
    """Calculate key summary statistics for energy consumption.

    Args:
        df: Input DataFrame.
        col: Column name to aggregate.

    Returns:
        Dict[str, float]: Dictionary containing total, mean, median, min, max, std.
    """
    if col not in df.columns or len(df) == 0:
        return {
            "count": 0.0,
            "total_kwh": 0.0,
            "mean_kwh": 0.0,
            "median_kwh": 0.0,
            "min_kwh": 0.0,
            "max_kwh": 0.0,
            "std_kwh": 0.0,
        }

    series = df[col].dropna()
    if len(series) == 0:
        return {
            "count": 0.0,
            "total_kwh": 0.0,
            "mean_kwh": 0.0,
            "median_kwh": 0.0,
            "min_kwh": 0.0,
            "max_kwh": 0.0,
            "std_kwh": 0.0,
        }

    return {
        "count": float(len(series)),
        "total_kwh": float(round(series.sum(), 4)),
        "mean_kwh": float(round(series.mean(), 4)),
        "median_kwh": float(round(series.median(), 4)),
        "min_kwh": float(round(series.min(), 4)),
        "max_kwh": float(round(series.max(), 4)),
        "std_kwh": float(round(series.std() if len(series) > 1 else 0.0, 4)),
    }


def prepare_forecast_plot_data(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    lookback_hours: int = 48,
) -> pd.DataFrame:
    """Combine trailing actual history and forecasted trajectory into a unified plot DataFrame.

    Ensures seamless connection between the final actual point and the forecast curve.

    Args:
        history_df: Historical DataFrame with ['timestamp', 'energy_kwh'].
        forecast_df: Forecast DataFrame with ['timestamp', 'forecast_energy_kwh', 'step'].
        lookback_hours: Number of trailing historical hours to include for context.

    Returns:
        pd.DataFrame: Merged DataFrame with columns ['timestamp', 'actual_energy_kwh', 'forecast_energy_kwh'].
    """
    if len(history_df) == 0:
        raise ValueError("history_df cannot be empty.")
    if len(forecast_df) == 0:
        raise ValueError("forecast_df cannot be empty.")

    # Slice recent history
    recent_hist = history_df.tail(lookback_hours).copy()
    recent_hist = recent_hist[["timestamp", "energy_kwh"]].rename(
        columns={"energy_kwh": "actual_energy_kwh"}
    )
    recent_hist["forecast_energy_kwh"] = np.nan

    # Format forecast slice
    fc_slice = forecast_df[["timestamp", "forecast_energy_kwh"]].copy()
    fc_slice["actual_energy_kwh"] = np.nan

    # Connect the origin: copy the last actual value as starting anchor for the forecast line
    last_actual_ts = recent_hist["timestamp"].iloc[-1]
    last_actual_val = recent_hist["actual_energy_kwh"].iloc[-1]
    recent_hist.loc[recent_hist.index[-1], "forecast_energy_kwh"] = last_actual_val

    combined = pd.concat([recent_hist, fc_slice], ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined


# =========================================================================
# Phase 6C: Live Monitoring & Prediction Dashboard Helpers
# =========================================================================

def format_measurement_kpi(measurement: Any) -> Dict[str, str]:
    """Format latest EnergyMeasurement for KPI metric card display.

    Args:
        measurement: Optional EnergyMeasurement instance or None.

    Returns:
        Dict[str, str]: Formatted strings for timestamp, energy, status, and completeness.
    """
    if measurement is None:
        return {
            "timestamp_str": "Waiting for stream start...",
            "energy_str": "N/A",
            "status_str": "Uninitialized",
            "coverage_str": "N/A",
        }

    ts_str = str(measurement.timestamp)

    if not measurement.is_valid or measurement.energy_kwh is None or np.isnan(measurement.energy_kwh):
        return {
            "timestamp_str": ts_str,
            "energy_str": "Missing / Gap",
            "status_str": "Unrecorded Interval (No Zero Fill)",
            "coverage_str": "0%",
        }

    energy_val = float(measurement.energy_kwh)
    cov_ratio = float(measurement.metadata.get("coverage_ratio", 1.0))

    return {
        "timestamp_str": ts_str,
        "energy_str": f"{energy_val:.3f} kWh",
        "status_str": "Recorded (Valid)",
        "coverage_str": f"{cov_ratio * 100:.1f}%",
    }


def format_prediction_kpi(prediction: Any) -> Dict[str, str]:
    """Format latest LivePredictionRecord for KPI metric card display.

    Args:
        prediction: Optional LivePredictionRecord instance or None.

    Returns:
        Dict[str, str]: Formatted strings for target timestamp, predicted value, and status.
    """
    if prediction is None:
        return {
            "origin_str": "N/A",
            "target_str": "N/A",
            "predicted_str": "Awaiting simulation step...",
            "status_str": "Pending",
        }

    origin_str = str(prediction.origin_timestamp)
    target_str = str(prediction.target_timestamp)

    if prediction.status != "SUCCESS" or prediction.predicted_energy_kwh is None:
        return {
            "origin_str": origin_str,
            "target_str": target_str,
            "predicted_str": "Withheld (Lookback Gapped)",
            "status_str": prediction.status,
        }

    pred_val = float(prediction.predicted_energy_kwh)
    return {
        "origin_str": origin_str,
        "target_str": target_str,
        "predicted_str": f"{pred_val:.3f} kWh",
        "status_str": "Generated (Step 1)",
    }


def format_live_metrics_summary(metrics: Dict[str, Any]) -> Dict[str, str]:
    """Format live evaluation scorecard metrics.

    Args:
        metrics: Dictionary returned by LivePredictionEngine.get_evaluation_metrics().

    Returns:
        Dict[str, str]: Clean display strings for MAE, RMSE, MBE, count, and confidence badge.
    """
    count = metrics.get("matched_pairs_count", 0)

    if count == 0:
        return {
            "mae_str": "N/A",
            "rmse_str": "N/A",
            "mbe_str": "N/A",
            "count_str": "0 verified pairs",
            "confidence_badge": "No Data",
            "note_str": "Awaiting arrival of predicted target timestamps.",
        }

    mae = metrics.get("mae_kwh")
    rmse = metrics.get("rmse_kwh")
    mbe = metrics.get("mean_bias_error_kwh")
    is_sufficient = metrics.get("is_statistically_sufficient", False)

    mae_str = f"{mae:.4f} kWh" if mae is not None else "N/A"
    rmse_str = f"{rmse:.4f} kWh" if rmse is not None else "N/A"
    mbe_str = f"{mbe:+.4f} kWh" if mbe is not None else "N/A"
    badge = "Preliminary Sample" if not is_sufficient else "Evaluated"

    return {
        "mae_str": mae_str,
        "rmse_str": rmse_str,
        "mbe_str": mbe_str,
        "count_str": f"{count} verified pair(s)",
        "confidence_badge": badge,
        "note_str": metrics.get("note", ""),
    }


def prepare_live_monitoring_chart_data(
    history_df: pd.DataFrame,
    evaluation_history: Any,
    next_prediction: Any = None,
    lookback_hours: int = 48,
) -> pd.DataFrame:
    """Prepare multi-series DataFrame for live telemetry line chart.

    Columns:
    - 'Actual Observed (kWh)': trailing actuals from history_df. Missing intervals remain NaN (never zero).
    - 'Past Model Prediction (kWh)': past predictions matched against historical actuals at target timestamp.
    - 'Next-Hour Forecast (kWh)': single forward point at next_prediction.target_timestamp.

    Args:
        history_df: Trailing historical measurements DataFrame with ['timestamp', 'energy_kwh'].
        evaluation_history: List of LiveEvaluationRecord objects.
        next_prediction: Optional latest LivePredictionRecord.
        lookback_hours: Number of trailing hours of actuals to include (default: 48).

    Returns:
        pd.DataFrame: Merged DataFrame indexed by timestamp.
    """
    if len(history_df) == 0:
        empty = pd.DataFrame(
            columns=["Actual Observed (kWh)", "Past Model Prediction (kWh)", "Next-Hour Forecast (kWh)"]
        )
        return empty

    # 1. Base actual slice
    recent_actual = history_df.tail(lookback_hours)[["timestamp", "energy_kwh"]].copy()
    recent_actual = recent_actual.rename(columns={"energy_kwh": "Actual Observed (kWh)"})
    recent_actual["Past Model Prediction (kWh)"] = np.nan
    recent_actual["Next-Hour Forecast (kWh)"] = np.nan

    # 2. Overlay past predictions from evaluation records
    if evaluation_history and len(evaluation_history) > 0:
        eval_dict = {
            rec.target_timestamp: float(rec.predicted_energy_kwh) for rec in evaluation_history
        }
        for idx, row in recent_actual.iterrows():
            ts = row["timestamp"]
            if ts in eval_dict:
                recent_actual.loc[idx, "Past Model Prediction (kWh)"] = eval_dict[ts]

    # 3. Append next-hour forecast point if present
    if (
        next_prediction is not None
        and next_prediction.status == "SUCCESS"
        and next_prediction.predicted_energy_kwh is not None
    ):
        target_ts = next_prediction.target_timestamp
        pred_val = float(next_prediction.predicted_energy_kwh)

        next_row = pd.DataFrame(
            [
                {
                    "timestamp": target_ts,
                    "Actual Observed (kWh)": np.nan,
                    "Past Model Prediction (kWh)": np.nan,
                    "Next-Hour Forecast (kWh)": pred_val,
                }
            ]
        )
        combined = pd.concat([recent_actual, next_row], ignore_index=True)
    else:
        combined = recent_actual

    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined


def format_event_log_table(
    prediction_history: Any,
    evaluation_history: Any,
    max_rows: int = 50,
) -> pd.DataFrame:
    """Format chronological event log table for Streamlit display.

    Combines prediction attempts with subsequent evaluation records, sorted with
    most recent event first.

    Args:
        prediction_history: List of LivePredictionRecord objects.
        evaluation_history: List of LiveEvaluationRecord objects.
        max_rows: Maximum rows to return (default: 50).

    Returns:
        pd.DataFrame: Tabular log ready for st.dataframe.
    """
    if not prediction_history or len(prediction_history) == 0:
        return pd.DataFrame(
            columns=[
                "Origin (T_origin)",
                "Target (T_target)",
                "Predicted (kWh)",
                "Actual (kWh)",
                "Abs Error (kWh)",
                "Status",
            ]
        )

    # Index evaluations by target timestamp for O(1) matching
    eval_map = {}
    if evaluation_history:
        for ev in evaluation_history:
            eval_map[ev.target_timestamp] = ev

    rows = []
    for pred in reversed(prediction_history):
        target_ts = pred.target_timestamp
        actual_val = "Pending"
        err_val = "Pending"

        if target_ts in eval_map:
            ev = eval_map[target_ts]
            actual_val = f"{ev.actual_energy_kwh:.3f}"
            err_val = f"{ev.absolute_error_kwh:.3f}"

        pred_val_str = (
            f"{pred.predicted_energy_kwh:.3f}"
            if (pred.predicted_energy_kwh is not None and not np.isnan(pred.predicted_energy_kwh))
            else "N/A"
        )

        rows.append(
            {
                "Origin (T_origin)": str(pred.origin_timestamp),
                "Target (T_target)": str(pred.target_timestamp),
                "Predicted (kWh)": pred_val_str,
                "Actual (kWh)": actual_val,
                "Abs Error (kWh)": err_val,
                "Status": pred.status,
            }
        )

        if len(rows) >= max_rows:
            break

    return pd.DataFrame(rows)

