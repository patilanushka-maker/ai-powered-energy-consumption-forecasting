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
