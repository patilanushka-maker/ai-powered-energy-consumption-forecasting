"""Data loading module for the AI-Powered Energy Consumption Forecasting project.

Responsibilities:
- Ingest raw UCI Household Power Consumption dataset from a specified filesystem path.
- Parse semicolon (;) delimiters and treat '?' as missing value markers.
- Validate presence of required columns ('Date', 'Time', 'Global_active_power').
- Unify 'Date' and 'Time' into a parsed 'timestamp' column (%d/%m/%Y %H:%M:%S).
- Convert 'Global_active_power' to numeric float values with coercion of invalid entries.
- Expose data audit details regarding invalid timestamps and missing power readings.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: List[str] = ["Date", "Time", "Global_active_power"]


def load_raw_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """Load raw household electric power consumption dataset from disk.

    Args:
        data_path: Path to the raw text/CSV file.

    Returns:
        pd.DataFrame: DataFrame containing original columns plus a parsed 'timestamp' column.
                     df.attrs['data_loader_audit'] contains loading statistics.

    Raises:
        FileNotFoundError: If the specified data_path does not exist on disk.
        ValueError: If file is empty or required columns are missing.
    """
    path = Path(data_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"Raw dataset file not found at: '{path.resolve()}'. "
            "Please ensure the UCI dataset is placed in the expected directory."
        )

    if path.stat().st_size == 0:
        raise ValueError(f"Dataset file at '{path.resolve()}' is empty (0 bytes).")

    logger.info("Reading raw dataset from %s", path)

    # Read CSV with semicolon separator and '?' as NA marker
    try:
        df = pd.read_csv(
            path,
            sep=";",
            na_values=["?"],
            low_memory=False,
            keep_default_na=True,
        )
    except Exception as exc:
        raise ValueError(f"Failed to parse raw dataset file at '{path}': {exc}") from exc

    # Validate presence of required schema columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in dataset: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    # Combine Date and Time and parse into unified timestamp
    # Source format is dd/mm/yyyy for Date and HH:MM:SS for Time
    combined_dt = (
        df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip()
    )
    df["timestamp"] = pd.to_datetime(
        combined_dt,
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    invalid_timestamps = int(df["timestamp"].isna().sum())
    if invalid_timestamps > 0:
        logger.warning(
            "Found %d rows with unparseable or missing timestamps.",
            invalid_timestamps,
        )

    # Convert Global_active_power to numeric
    original_power_na = int(df["Global_active_power"].isna().sum())
    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"],
        errors="coerce",
    )
    total_power_na = int(df["Global_active_power"].isna().sum())
    coerced_power_na = total_power_na - original_power_na

    if total_power_na > 0:
        logger.info(
            "Global_active_power summary: %d total missing/invalid readings (%d coerced from non-numeric).",
            total_power_na,
            coerced_power_na,
        )

    # Attach audit metadata to DataFrame
    audit: Dict[str, Union[int, str]] = {
        "raw_row_count": len(df),
        "invalid_timestamps": invalid_timestamps,
        "missing_power_readings": total_power_na,
        "coerced_power_readings": coerced_power_na,
        "source_path": str(path),
    }
    df.attrs["data_loader_audit"] = audit

    return df
