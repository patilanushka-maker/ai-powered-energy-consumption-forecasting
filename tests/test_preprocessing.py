"""Comprehensive unit tests for data preprocessing and hourly resampling.

Validates all 16 required behavioral specifications using isolated synthetic DataFrames:
1. Valid timestamp parsing
2. Numeric power conversion
3. Missing power handling
4. Invalid string power handling
5. Duplicate timestamp detection
6. Deterministic conflicting duplicate resolution
7. Fully observed hour energy calculation
8. Partially observed hour coverage calculation
9. Below-threshold hour exclusion
10. Completely missing hour exclusion (not converted to zero)
11. Chronological output ordering
12. Output schema and data types
13. Coverage parameter validation
14. Missing required columns error handling
15. Empty input behavior
16. Non-finite (inf/-inf) numeric readings handling
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import PreprocessingAudit, clean_and_resample_hourly


def test_invalid_coverage_parameters():
    """Verify that min_hourly_coverage outside [0.0, 1.0] raises ValueError."""
    dummy_df = pd.DataFrame(
        {"timestamp": ["2026-01-01 00:00:00"], "Global_active_power": [1.0]}
    )
    with pytest.raises(ValueError, match="min_hourly_coverage"):
        clean_and_resample_hourly(dummy_df, min_hourly_coverage=-0.1)

    with pytest.raises(ValueError, match="min_hourly_coverage"):
        clean_and_resample_hourly(dummy_df, min_hourly_coverage=1.05)


def test_missing_required_columns():
    """Verify that missing required columns raise descriptive ValueError."""
    df_missing_power = pd.DataFrame({"timestamp": ["2026-01-01 00:00:00"]})
    with pytest.raises(ValueError, match="Global_active_power"):
        clean_and_resample_hourly(df_missing_power)

    df_missing_ts = pd.DataFrame({"Global_active_power": [1.0]})
    with pytest.raises(ValueError, match="timestamp"):
        clean_and_resample_hourly(df_missing_ts)


def test_empty_input_behavior():
    """Verify that empty DataFrame returns expected schema and zeroed audit."""
    empty_df = pd.DataFrame(columns=["timestamp", "Global_active_power"])
    result, audit = clean_and_resample_hourly(empty_df, return_audit=True)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    assert list(result.columns) == [
        "timestamp",
        "energy_kwh",
        "valid_minutes",
        "coverage_ratio",
    ]
    assert audit.original_rows == 0
    assert audit.accepted_intervals == 0
    assert audit.excluded_intervals == 0


def test_valid_timestamp_and_numeric_power_conversion():
    """Verify correct parsing of valid string timestamps and numeric power readings."""
    timestamps = pd.date_range("2026-01-01 10:00:00", periods=60, freq="1min")
    df = pd.DataFrame(
        {
            "timestamp": [str(ts) for ts in timestamps],
            "Global_active_power": ["2.5" for _ in range(60)],
        }
    )
    result = clean_and_resample_hourly(df, min_hourly_coverage=0.9)
    assert len(result) == 1
    assert result.loc[0, "timestamp"] == pd.Timestamp("2026-01-01 10:00:00")
    assert result.loc[0, "energy_kwh"] == pytest.approx(2.5)


def test_missing_and_invalid_numeric_readings():
    """Verify handling of missing (NaN) and invalid string power readings."""
    timestamps = pd.date_range("2026-01-01 10:00:00", periods=60, freq="1min")
    powers = [2.0] * 50 + [np.nan] * 5 + ["corrupted"] * 3 + [-1.5] * 2  # 50 valid
    df = pd.DataFrame({"timestamp": timestamps, "Global_active_power": powers})

    result, audit = clean_and_resample_hourly(
        df, min_hourly_coverage=0.80, return_audit=True
    )
    # 50 valid out of 60 -> coverage 50/60 = 0.833 >= 0.80 -> accepted
    assert len(result) == 1
    assert result.loc[0, "valid_minutes"] == 50
    assert result.loc[0, "coverage_ratio"] == pytest.approx(50 / 60.0)
    assert result.loc[0, "energy_kwh"] == pytest.approx(2.0)
    assert audit.invalid_power_readings == 10  # 5 NaN + 3 non-numeric + 2 negative


def test_non_finite_numeric_readings():
    """Verify that infinite power values (inf, -inf) are treated as invalid."""
    timestamps = pd.date_range("2026-01-01 10:00:00", periods=60, freq="1min")
    powers = [1.5] * 58 + [np.inf, -np.inf]
    df = pd.DataFrame({"timestamp": timestamps, "Global_active_power": powers})

    result, audit = clean_and_resample_hourly(
        df, min_hourly_coverage=0.90, return_audit=True
    )
    assert len(result) == 1
    assert result.loc[0, "valid_minutes"] == 58
    assert audit.invalid_power_readings == 2


def test_invalid_timestamps_handling():
    """Verify handling of invalid or unparseable timestamps."""
    timestamps = [
        "2026-01-01 10:00:00",
        "invalid_date_format",
        None,
    ] + [f"2026-01-01 10:{i:02d}:00" for i in range(3, 60)]
    powers = [2.0] * len(timestamps)
    df = pd.DataFrame({"timestamp": timestamps, "Global_active_power": powers})

    result, audit = clean_and_resample_hourly(
        df, min_hourly_coverage=0.90, return_audit=True
    )
    assert audit.invalid_timestamps == 2
    assert result.loc[0, "valid_minutes"] == 58


def test_duplicate_timestamp_detection():
    """Verify detection of duplicate timestamps and that duplicates do not inflate coverage."""
    # 59 unique minutes plus 1 duplicate of minute 10:00:00
    timestamps = [pd.Timestamp("2026-01-01 10:00:00")] + list(
        pd.date_range("2026-01-01 10:00:00", periods=59, freq="1min")
    )
    powers = [3.0] * 60  # total 60 rows, but only 59 unique minutes
    df = pd.DataFrame({"timestamp": timestamps, "Global_active_power": powers})

    result, audit = clean_and_resample_hourly(
        df, min_hourly_coverage=0.90, return_audit=True
    )
    assert audit.duplicate_timestamps == 1
    assert audit.conflicting_duplicates == 0
    # Must report 59 valid unique minutes, not 60
    assert result.loc[0, "valid_minutes"] == 59
    assert result.loc[0, "coverage_ratio"] == pytest.approx(59 / 60.0)


def test_deterministic_handling_of_conflicting_duplicates():
    """Verify conflicting duplicates are tracked and resolved deterministically via mean."""
    # Timestamp 10:00:00 has two conflicting readings: 2.0 kW and 4.0 kW
    ts_conflict = pd.Timestamp("2026-01-01 10:00:00")
    unique_ts = pd.date_range("2026-01-01 10:01:00", periods=59, freq="1min")

    timestamps = [ts_conflict, ts_conflict] + list(unique_ts)
    powers = [2.0, 4.0] + [3.0] * 59
    df = pd.DataFrame({"timestamp": timestamps, "Global_active_power": powers})

    result, audit = clean_and_resample_hourly(
        df, min_hourly_coverage=0.90, return_audit=True
    )
    assert audit.duplicate_timestamps == 1
    assert audit.conflicting_duplicates == 1
    assert result.loc[0, "valid_minutes"] == 60
    # Expected energy: 59 minutes of 3.0, and 1 minute of mean(2.0, 4.0) = 3.0 -> overall mean is 3.0
    assert result.loc[0, "energy_kwh"] == pytest.approx(3.0)


def test_fully_observed_hour_energy_calculation():
    """Verify energy calculation for fully observed hour (sum of power/60 == mean*1.0)."""
    timestamps = pd.date_range("2026-01-01 14:00:00", periods=60, freq="1min")
    # Alternating 2.0 kW and 4.0 kW (mean = 3.0 kW)
    powers = [2.0 if i % 2 == 0 else 4.0 for i in range(60)]
    df = pd.DataFrame({"timestamp": timestamps, "Global_active_power": powers})

    result = clean_and_resample_hourly(df, min_hourly_coverage=0.90)
    assert len(result) == 1
    assert result.loc[0, "valid_minutes"] == 60
    assert result.loc[0, "coverage_ratio"] == pytest.approx(1.0)
    assert result.loc[0, "energy_kwh"] == pytest.approx(3.0)


def test_partially_observed_hour_coverage_and_energy():
    """Verify coverage and energy calculation for partially observed hour above threshold."""
    # 54 minutes observed with 4.0 kW, 6 minutes missing -> coverage = 54/60 = 0.90
    timestamps = pd.date_range("2026-01-01 15:00:00", periods=54, freq="1min")
    powers = [4.0] * 54
    df = pd.DataFrame({"timestamp": timestamps, "Global_active_power": powers})

    result, audit = clean_and_resample_hourly(
        df, min_hourly_coverage=0.90, return_audit=True
    )
    assert len(result) == 1
    assert result.loc[0, "valid_minutes"] == 54
    assert result.loc[0, "coverage_ratio"] == pytest.approx(0.90)
    # Energy is represented by observed mean over the hour = 4.0 kWh
    assert result.loc[0, "energy_kwh"] == pytest.approx(4.0)
    assert audit.accepted_intervals == 1
    assert audit.excluded_intervals == 0


def test_hour_below_coverage_threshold_excluded():
    """Verify that an hour below the configured coverage threshold is excluded."""
    # 40 minutes observed (coverage = 40/60 = 0.667)
    timestamps = pd.date_range("2026-01-01 16:00:00", periods=40, freq="1min")
    powers = [3.5] * 40
    df = pd.DataFrame({"timestamp": timestamps, "Global_active_power": powers})

    result, audit = clean_and_resample_hourly(
        df, min_hourly_coverage=0.90, return_audit=True
    )
    assert len(result) == 0
    assert audit.accepted_intervals == 0
    assert audit.excluded_intervals == 1


def test_completely_missing_hour_not_converted_to_zero():
    """Verify that completely missing intermediate hours are excluded, not converted to 0 kWh."""
    # Hour 10:00:00 (60 minutes observed at 2.0 kW)
    ts_h10 = pd.date_range("2026-01-01 10:00:00", periods=60, freq="1min")
    # Hour 12:00:00 (60 minutes observed at 5.0 kW)
    ts_h12 = pd.date_range("2026-01-01 12:00:00", periods=60, freq="1min")
    # Hour 11:00:00 is COMPLETELY MISSING (no rows)

    timestamps = list(ts_h10) + list(ts_h12)
    powers = [2.0] * 60 + [5.0] * 60
    df = pd.DataFrame({"timestamp": timestamps, "Global_active_power": powers})

    result, audit = clean_and_resample_hourly(
        df, min_hourly_coverage=0.90, return_audit=True
    )
    assert len(result) == 2
    # Result must contain 10:00 and 12:00
    assert list(result["timestamp"]) == [
        pd.Timestamp("2026-01-01 10:00:00"),
        pd.Timestamp("2026-01-01 12:00:00"),
    ]
    # No zero-consumption row for 11:00:00
    assert 0.0 not in result["energy_kwh"].values
    # Excluded intervals in the span (10:00, 11:00, 12:00) must equal 1
    assert audit.accepted_intervals == 2
    assert audit.excluded_intervals == 1


def test_chronological_output_ordering():
    """Verify that output is strictly sorted chronologically even if input is scrambled."""
    ts_h12 = pd.date_range("2026-01-01 12:00:00", periods=60, freq="1min")
    ts_h10 = pd.date_range("2026-01-01 10:00:00", periods=60, freq="1min")
    # Input in reverse order
    timestamps = list(ts_h12) + list(ts_h10)
    powers = [3.0] * 120
    df = pd.DataFrame({"timestamp": timestamps, "Global_active_power": powers})

    result = clean_and_resample_hourly(df, min_hourly_coverage=0.90)
    assert len(result) == 2
    assert result.loc[0, "timestamp"] == pd.Timestamp("2026-01-01 10:00:00")
    assert result.loc[1, "timestamp"] == pd.Timestamp("2026-01-01 12:00:00")
    assert result["timestamp"].is_monotonic_increasing


def test_output_schema_and_dtypes():
    """Verify exact column names and datatypes of output DataFrame."""
    timestamps = pd.date_range("2026-01-01 10:00:00", periods=60, freq="1min")
    df = pd.DataFrame({"timestamp": timestamps, "Global_active_power": [1.8] * 60})

    result = clean_and_resample_hourly(df, min_hourly_coverage=0.90)
    assert list(result.columns) == [
        "timestamp",
        "energy_kwh",
        "valid_minutes",
        "coverage_ratio",
    ]
    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
    assert pd.api.types.is_float_dtype(result["energy_kwh"])
    assert pd.api.types.is_integer_dtype(result["valid_minutes"])
    assert pd.api.types.is_float_dtype(result["coverage_ratio"])


def test_datetime_index_support():
    """Verify that input with DatetimeIndex (and without explicit timestamp column) works."""
    dt_index = pd.date_range("2026-01-01 10:00:00", periods=60, freq="1min")
    df = pd.DataFrame({"Global_active_power": [2.2] * 60}, index=dt_index)

    result = clean_and_resample_hourly(df, min_hourly_coverage=0.90)
    assert len(result) == 1
    assert result.loc[0, "timestamp"] == pd.Timestamp("2026-01-01 10:00:00")
    assert result.loc[0, "energy_kwh"] == pytest.approx(2.2)
