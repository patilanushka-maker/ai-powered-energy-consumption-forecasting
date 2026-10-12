"""Data preprocessing, deduplication, and hourly resampling module.

Responsibilities:
- Validate input schemas and coverage parameters.
- Clean and parse timestamp sequences.
- Detect duplicate timestamps and resolve conflicts deterministically.
- Filter invalid, missing, or non-finite electrical measurements.
- Aggregate minute observations into hourly energy consumption (kWh).
- Enforce explicit hourly completeness policies via min_hourly_coverage.
- Provide comprehensive preprocessing audit statistics.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_MINUTES_PER_HOUR: int = 60


@dataclass
class PreprocessingAudit:
    """Audit summary container for preprocessing operations."""

    original_rows: int
    invalid_timestamps: int
    invalid_power_readings: int
    duplicate_timestamps: int
    conflicting_duplicates: int
    accepted_intervals: int
    excluded_intervals: int
    earliest_timestamp: Optional[str]
    latest_timestamp: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit statistics to dictionary."""
        return asdict(self)


def clean_and_resample_hourly(
    df: pd.DataFrame,
    min_hourly_coverage: float = 0.90,
    return_audit: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, PreprocessingAudit]]:
    """Clean minute-level observations and resample to hourly energy consumption.

    Hourly Energy Calculation & Missing Data Policy:
    -------------------------------------------------
    - Source variable 'Global_active_power' is minute-averaged power in kilowatts (kW).
    - In an hour with 60 fully observed minutes, total hourly energy in kWh is:
        energy_kwh = sum(power_kw / 60) == mean(power_kw) * 1.0 hour.
    - For hours meeting or exceeding `min_hourly_coverage` (valid_minutes >= 60 * min_hourly_coverage),
      unobserved minutes within that hour are assumed to be missing at random and represented
      by the hour's observed mean power:
        energy_kwh = mean(valid_power_kw) * 1.0 hour.
      This prevents unobserved minutes from being falsely treated as zero consumption.
    - Hours with coverage below `min_hourly_coverage` are excluded entirely from the output.
    - Completely unobserved hours are excluded and never reported as zero consumption.

    Duplicate Resolution Policy:
    -----------------------------
    - When multiple observations share the exact same minute timestamp:
      1. They are flagged as duplicates. If their power readings differ, they are flagged as conflicting.
      2. Duplicate records are deterministically aggregated using the mean of valid power readings
         for that minute. This produces a single consolidated minute record and prevents duplicates
         from artificially inflating the coverage minute count.

    Args:
        df: Input DataFrame containing 'Global_active_power' and 'timestamp' column or DatetimeIndex.
        min_hourly_coverage: Minimum required fraction of valid minutes in an hour (0.0 to 1.0).
        return_audit: If True, returns a tuple of (DataFrame, PreprocessingAudit).
                      Otherwise returns DataFrame with audit attached in df.attrs['audit'].

    Returns:
        pd.DataFrame or Tuple[pd.DataFrame, PreprocessingAudit]:
            Processed hourly DataFrame sorted by timestamp with columns:
            ['timestamp', 'energy_kwh', 'valid_minutes', 'coverage_ratio'].

    Raises:
        ValueError: If min_hourly_coverage is outside [0.0, 1.0] or required columns are missing.
    """
    # 1. Validate coverage threshold parameter
    if not (0.0 <= min_hourly_coverage <= 1.0):
        raise ValueError(
            f"min_hourly_coverage must be between 0.0 and 1.0, got {min_hourly_coverage}"
        )

    # 2. Validate input columns
    if "Global_active_power" not in df.columns:
        raise ValueError(
            "Input DataFrame must contain 'Global_active_power' column. "
            f"Found: {list(df.columns)}"
        )

    # Extract or validate timestamp series
    if "timestamp" in df.columns:
        ts_series = df["timestamp"]
    elif isinstance(df.index, pd.DatetimeIndex):
        ts_series = pd.Series(df.index, index=df.index, name="timestamp")
    else:
        raise ValueError(
            "Input DataFrame must contain a 'timestamp' column or a DatetimeIndex."
        )

    original_rows = len(df)

    # Handle empty DataFrame gracefully
    empty_schema_df = pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "energy_kwh": pd.Series(dtype="float64"),
            "valid_minutes": pd.Series(dtype="int64"),
            "coverage_ratio": pd.Series(dtype="float64"),
        }
    )

    if original_rows == 0:
        audit = PreprocessingAudit(
            original_rows=0,
            invalid_timestamps=0,
            invalid_power_readings=0,
            duplicate_timestamps=0,
            conflicting_duplicates=0,
            accepted_intervals=0,
            excluded_intervals=0,
            earliest_timestamp=None,
            latest_timestamp=None,
        )
        empty_schema_df.attrs["audit"] = audit
        if return_audit:
            return empty_schema_df, audit
        return empty_schema_df

    # 3. Parse and inspect timestamps
    parsed_timestamps = pd.to_datetime(ts_series, errors="coerce")
    invalid_ts_mask = parsed_timestamps.isna()
    invalid_timestamps = int(invalid_ts_mask.sum())

    working_df = pd.DataFrame(
        {
            "timestamp": parsed_timestamps,
            "raw_power": df["Global_active_power"].values,
        }
    )

    # Filter out rows with invalid timestamps
    valid_ts_df = working_df.loc[~invalid_ts_mask].copy()

    if valid_ts_df.empty:
        audit = PreprocessingAudit(
            original_rows=original_rows,
            invalid_timestamps=invalid_timestamps,
            invalid_power_readings=0,
            duplicate_timestamps=0,
            conflicting_duplicates=0,
            accepted_intervals=0,
            excluded_intervals=0,
            earliest_timestamp=None,
            latest_timestamp=None,
        )
        empty_schema_df.attrs["audit"] = audit
        if return_audit:
            return empty_schema_df, audit
        return empty_schema_df

    # 4. Convert power to numeric and flag invalid readings
    valid_ts_df["power_kw"] = pd.to_numeric(valid_ts_df["raw_power"], errors="coerce")
    invalid_power_mask = (
        valid_ts_df["power_kw"].isna()
        | (~np.isfinite(valid_ts_df["power_kw"]))
        | (valid_ts_df["power_kw"] < 0)
    )
    invalid_power_readings = int(invalid_power_mask.sum())

    # Set invalid power readings to NaN for aggregation
    valid_ts_df.loc[invalid_power_mask, "power_kw"] = np.nan

    # 5. Detect and resolve duplicate timestamps
    # Identify duplicate timestamp occurrences
    ts_counts = valid_ts_df["timestamp"].value_counts()
    duplicate_ts_values = ts_counts[ts_counts > 1].index
    duplicate_timestamps = len(duplicate_ts_values)
    conflicting_duplicates = 0

    if duplicate_timestamps > 0:
        # Check conflicts among duplicates
        dup_subset = valid_ts_df[valid_ts_df["timestamp"].isin(duplicate_ts_values)]
        grouped_dup = dup_subset.groupby("timestamp")["power_kw"]
        for _, group in grouped_dup:
            valid_vals = group.dropna().unique()
            if len(valid_vals) > 1:
                conflicting_duplicates += 1

        logger.info(
            "Found %d duplicate timestamps (%d with conflicting power values). "
            "Resolving by averaging valid readings per timestamp.",
            duplicate_timestamps,
            conflicting_duplicates,
        )

    # Deterministic duplicate resolution:
    # Group by timestamp and compute mean of valid power readings for each unique minute
    # (mean of all NaNs returns NaN via min_count=1)
    deduped_minutes = (
        valid_ts_df.groupby("timestamp")["power_kw"]
        .agg(lambda s: s.dropna().mean() if s.notna().any() else np.nan)
        .reset_index()
    )

    # 6. Hourly Aggregation & Coverage Assessment
    deduped_minutes["hour_start"] = deduped_minutes["timestamp"].dt.floor("h")

    # Group by hour_start to calculate coverage and energy
    hourly_records = []
    earliest_ts = deduped_minutes["timestamp"].min()
    latest_ts = deduped_minutes["timestamp"].max()

    earliest_hour = earliest_ts.floor("h")
    latest_hour = latest_ts.floor("h")

    # Generate full hourly grid between earliest and latest observed hour
    full_hourly_index = pd.date_range(
        start=earliest_hour, end=latest_hour, freq="h", name="hour_start"
    )
    total_expected_intervals = len(full_hourly_index)

    # Group valid unique minutes by hour
    grouped_hours = deduped_minutes.groupby("hour_start")["power_kw"]

    for hour in full_hourly_index:
        if hour in grouped_hours.groups:
            power_series = grouped_hours.get_group(hour)
            valid_readings = power_series.dropna()
            valid_minutes = len(valid_readings)
        else:
            # Completely missing hour in observation span
            valid_minutes = 0
            valid_readings = pd.Series(dtype="float64")

        coverage_ratio = valid_minutes / float(EXPECTED_MINUTES_PER_HOUR)

        if coverage_ratio >= min_hourly_coverage and valid_minutes > 0:
            # Hour meets coverage threshold -> calculate estimated energy in kWh
            mean_power = float(valid_readings.mean())
            # energy_kwh = mean_power * 1.0 hour
            energy_kwh = round(mean_power, 6)
            hourly_records.append(
                {
                    "timestamp": hour,
                    "energy_kwh": energy_kwh,
                    "valid_minutes": valid_minutes,
                    "coverage_ratio": round(coverage_ratio, 6),
                }
            )

    if hourly_records:
        result_df = pd.DataFrame(hourly_records)
        result_df = result_df.sort_values("timestamp").reset_index(drop=True)
        # Ensure correct data types
        result_df["timestamp"] = pd.to_datetime(result_df["timestamp"])
        result_df["energy_kwh"] = result_df["energy_kwh"].astype("float64")
        result_df["valid_minutes"] = result_df["valid_minutes"].astype("int64")
        result_df["coverage_ratio"] = result_df["coverage_ratio"].astype("float64")
    else:
        result_df = empty_schema_df.copy()

    accepted_intervals = len(result_df)
    excluded_intervals = total_expected_intervals - accepted_intervals

    audit = PreprocessingAudit(
        original_rows=original_rows,
        invalid_timestamps=invalid_timestamps,
        invalid_power_readings=invalid_power_readings,
        duplicate_timestamps=duplicate_timestamps,
        conflicting_duplicates=conflicting_duplicates,
        accepted_intervals=accepted_intervals,
        excluded_intervals=excluded_intervals,
        earliest_timestamp=earliest_ts.isoformat() if pd.notna(earliest_ts) else None,
        latest_timestamp=latest_ts.isoformat() if pd.notna(latest_ts) else None,
    )

    result_df.attrs["audit"] = audit

    logger.info(
        "Preprocessing complete: %d rows in, %d accepted hours, %d excluded hours.",
        original_rows,
        accepted_intervals,
        excluded_intervals,
    )

    if return_audit:
        return result_df, audit
    return result_df
