"""Comprehensive unit tests for time-series feature engineering.

Validates all behavioral specifications using isolated synthetic DataFrames:
1. Correct hour extraction
2. Correct weekday extraction
3. Correct weekend indicator
4. Correct month extraction
5. Correct 1-hour lag (lag_1)
6. Correct 24-hour lag (lag_24)
7. Correct 168-hour lag (lag_168)
8. Missing exact historical timestamp results in missing lag (NaN)
9. Irregular timestamps do not cause naive row-based lag
10. Rolling averages use only historical observations strictly before T
11. Current target value is excluded from rolling average
12. Rolling features respect completeness policy (gap causes NaN)
13. Output timestamps remain sorted
14. Original input DataFrame is not mutated
15. Missing required columns produce clear errors
16. Empty input behavior is tested
17. Feature functions preserve the target column
18. Missing lag values are not silently replaced with zero
19. Rejection of zero, negative, floating-point, boolean, and empty lags
20. Rejection of zero, negative, floating-point, and boolean rolling window sizes
21. Strict rejection of duplicate timestamps
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    create_lag_and_rolling_features,
    extract_temporal_features,
)


def test_missing_required_columns_temporal_features():
    """Verify missing timestamp column raises clear ValueError in temporal features."""
    df_missing_ts = pd.DataFrame({"energy_kwh": [1.0, 2.0]})
    with pytest.raises(ValueError, match="timestamp"):
        extract_temporal_features(df_missing_ts)


def test_missing_required_columns_lag_rolling():
    """Verify missing target or timestamp column raises clear ValueError in lag/rolling features."""
    df_missing_target = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=3, freq="h")}
    )
    with pytest.raises(ValueError, match="target column 'energy_kwh'"):
        create_lag_and_rolling_features(df_missing_target)

    df_missing_ts = pd.DataFrame({"energy_kwh": [1.0, 2.0]})
    with pytest.raises(ValueError, match="timestamp"):
        create_lag_and_rolling_features(df_missing_ts)


def test_empty_input_behavior():
    """Verify that empty DataFrames return expected schema without errors."""
    empty_df = pd.DataFrame(columns=["timestamp", "energy_kwh"])

    temporal_res = extract_temporal_features(empty_df)
    assert len(temporal_res) == 0
    assert {"hour", "day_of_week", "day_of_month", "month", "is_weekend"}.issubset(
        temporal_res.columns
    )

    lag_res = create_lag_and_rolling_features(empty_df)
    assert len(lag_res) == 0
    assert {"lag_1", "lag_24", "lag_168", "rolling_mean_24"}.issubset(lag_res.columns)


def test_temporal_features_extraction():
    """Verify hour, weekday, day of month, month, and weekend indicator extraction."""
    timestamps = [
        pd.Timestamp("2026-01-02 15:00:00"),
        pd.Timestamp("2026-01-03 08:00:00"),
        pd.Timestamp("2026-01-04 12:00:00"),
    ]
    df = pd.DataFrame({"timestamp": timestamps, "energy_kwh": [1.5, 2.0, 2.5]})

    res = extract_temporal_features(df)

    assert res.loc[0, "hour"] == 15
    assert res.loc[0, "day_of_week"] == 4
    assert res.loc[0, "day_of_month"] == 2
    assert res.loc[0, "month"] == 1
    assert res.loc[0, "is_weekend"] == 0

    assert res.loc[1, "hour"] == 8
    assert res.loc[1, "day_of_week"] == 5
    assert res.loc[1, "day_of_month"] == 3
    assert res.loc[1, "is_weekend"] == 1

    assert res.loc[2, "hour"] == 12
    assert res.loc[2, "day_of_week"] == 6
    assert res.loc[2, "day_of_month"] == 4
    assert res.loc[2, "is_weekend"] == 1


def test_correct_one_hour_lag():
    """Verify lag_1 reflects consumption exactly 1 hour prior."""
    timestamps = pd.date_range("2026-01-01 10:00:00", periods=5, freq="h")
    energies = [2.0, 3.5, 4.0, 1.8, 5.2]
    df = pd.DataFrame({"timestamp": timestamps, "energy_kwh": energies})

    res = create_lag_and_rolling_features(df, lags=(1,))

    assert pd.isna(res.loc[0, "lag_1"])
    assert res.loc[1, "lag_1"] == pytest.approx(2.0)
    assert res.loc[2, "lag_1"] == pytest.approx(3.5)
    assert res.loc[3, "lag_1"] == pytest.approx(4.0)
    assert res.loc[4, "lag_1"] == pytest.approx(1.8)


def test_correct_24_hour_and_168_hour_lags():
    """Verify lag_24 and lag_168 reflect observations exactly 24h and 168h prior."""
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=200, freq="h")
    energies = [i * 0.1 for i in range(200)]
    df = pd.DataFrame({"timestamp": timestamps, "energy_kwh": energies})

    res = create_lag_and_rolling_features(df, lags=(24, 168))

    assert pd.isna(res.loc[23, "lag_24"])
    assert res.loc[24, "lag_24"] == pytest.approx(0.0)
    assert res.loc[25, "lag_24"] == pytest.approx(0.1)

    assert pd.isna(res.loc[167, "lag_168"])
    assert res.loc[168, "lag_168"] == pytest.approx(0.0)
    assert res.loc[169, "lag_168"] == pytest.approx(0.1)


def test_missing_exact_historical_timestamp_yields_nan():
    """Verify missing historical timestamps produce NaN and are not replaced with 0 or wrong values."""
    timestamps = [
        pd.Timestamp("2026-01-01 10:00:00"),
        pd.Timestamp("2026-01-01 12:00:00"),
    ]
    df = pd.DataFrame({"timestamp": timestamps, "energy_kwh": [3.0, 4.0]})

    res = create_lag_and_rolling_features(df, lags=(1,))

    assert pd.isna(res.loc[1, "lag_1"])


def test_irregular_timestamps_do_not_cause_wrong_row_lag():
    """Verify irregular timestamps do not naively pick the preceding row."""
    timestamps = [
        pd.Timestamp("2026-01-01 01:00:00"),
        pd.Timestamp("2026-01-01 05:00:00"),
        pd.Timestamp("2026-01-01 06:00:00"),
    ]
    energies = [10.0, 20.0, 30.0]
    df = pd.DataFrame({"timestamp": timestamps, "energy_kwh": energies})

    res = create_lag_and_rolling_features(df, lags=(1,))

    assert pd.isna(res.loc[1, "lag_1"])
    assert res.loc[2, "lag_1"] == pytest.approx(20.0)


def test_rolling_average_excludes_current_target():
    """Verify rolling average strictly excludes current timestamp target value."""
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=25, freq="h")
    energies = [5.0] * 24 + [100.0]
    df = pd.DataFrame({"timestamp": timestamps, "energy_kwh": energies})

    res = create_lag_and_rolling_features(df, rolling_window_hours=24)

    assert res.loc[24, "rolling_mean_24"] == pytest.approx(5.0)


def test_rolling_average_completeness_policy():
    """Verify rolling average produces NaN if lookback window has gaps (completeness policy)."""
    timestamps = list(pd.date_range("2026-01-01 00:00:00", periods=10, freq="h")) + list(
        pd.date_range("2026-01-01 11:00:00", periods=14, freq="h")
    )
    energies = [4.0] * len(timestamps)
    df = pd.DataFrame({"timestamp": timestamps, "energy_kwh": energies})

    res = create_lag_and_rolling_features(df, rolling_window_hours=24)

    last_idx = len(res) - 1
    assert pd.isna(res.loc[last_idx, "rolling_mean_24"])


def test_rolling_average_recovers_after_complete_window():
    """Verify rolling average becomes valid again once 24 continuous hours are observed."""
    timestamps = [pd.Timestamp("2026-01-01 00:00:00")] + list(
        pd.date_range("2026-01-01 06:00:00", periods=30, freq="h")
    )
    energies = [3.0] * len(timestamps)
    df = pd.DataFrame({"timestamp": timestamps, "energy_kwh": energies})

    res = create_lag_and_rolling_features(df, rolling_window_hours=24)

    assert res.loc[25, "rolling_mean_24"] == pytest.approx(3.0)


def test_chronological_ordering_preserved():
    """Verify output rows are sorted chronologically even if input is shuffled."""
    timestamps = [
        pd.Timestamp("2026-01-01 12:00:00"),
        pd.Timestamp("2026-01-01 10:00:00"),
        pd.Timestamp("2026-01-01 11:00:00"),
    ]
    df = pd.DataFrame({"timestamp": timestamps, "energy_kwh": [1.0, 2.0, 3.0]})

    temporal_res = extract_temporal_features(df)
    assert temporal_res["timestamp"].is_monotonic_increasing

    lag_res = create_lag_and_rolling_features(df, lags=(1,))
    assert lag_res["timestamp"].is_monotonic_increasing


def test_input_dataframe_not_mutated():
    """Verify that caller DataFrame is not modified in-place."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
            "energy_kwh": [1.0, 2.0, 3.0],
        }
    )
    original_cols = list(df.columns)
    original_values = df.copy()

    extract_temporal_features(df)
    assert list(df.columns) == original_cols
    pd.testing.assert_frame_equal(df, original_values)

    create_lag_and_rolling_features(df)
    assert list(df.columns) == original_cols
    pd.testing.assert_frame_equal(df, original_values)


def test_preserves_target_and_metadata_columns():
    """Verify target and metadata columns are preserved untouched."""
    timestamps = pd.date_range("2026-01-01", periods=3, freq="h")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "energy_kwh": [2.5, 3.0, 1.8],
            "valid_minutes": [60, 58, 60],
            "coverage_ratio": [1.0, 0.967, 1.0],
        }
    )

    res = create_lag_and_rolling_features(df)
    assert "energy_kwh" in res.columns
    assert "valid_minutes" in res.columns
    assert "coverage_ratio" in res.columns
    assert list(res["energy_kwh"]) == [2.5, 3.0, 1.8]
    assert list(res["valid_minutes"]) == [60, 58, 60]


# ---------------------------------------------------------------------------
# Validation Fix Tests: Lag parameter validation
# ---------------------------------------------------------------------------


def test_lag_parameter_validation_negative_and_zero():
    """Verify zero and negative lags are rejected with ValueError."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
            "energy_kwh": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="lags must be a non-empty sequence of positive integers"):
        create_lag_and_rolling_features(df, lags=(-1,))

    with pytest.raises(ValueError, match="lags must be a non-empty sequence of positive integers"):
        create_lag_and_rolling_features(df, lags=(0,))

    with pytest.raises(ValueError, match="lags must be a non-empty sequence of positive integers"):
        create_lag_and_rolling_features(df, lags=(1, -24))


def test_lag_parameter_validation_non_integers_and_booleans():
    """Verify float, boolean, string, and empty lag sequences are rejected."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
            "energy_kwh": [1.0, 2.0, 3.0],
        }
    )
    # Float rejection
    with pytest.raises(ValueError, match="lags must be a non-empty sequence of positive integers"):
        create_lag_and_rolling_features(df, lags=(1.5,))

    # Boolean rejection (Python bool is subclass of int)
    with pytest.raises(ValueError, match="lags must be a non-empty sequence of positive integers"):
        create_lag_and_rolling_features(df, lags=(True,))

    with pytest.raises(ValueError, match="lags must be a non-empty sequence of positive integers"):
        create_lag_and_rolling_features(df, lags=(1, False))

    # Empty sequence rejection
    with pytest.raises(ValueError, match="lags must be a non-empty sequence of positive integers"):
        create_lag_and_rolling_features(df, lags=())

    with pytest.raises(ValueError, match="lags must be a non-empty sequence of positive integers"):
        create_lag_and_rolling_features(df, lags=[])

    # Valid list of integers accepted
    res_list = create_lag_and_rolling_features(df, lags=[1, 2])
    assert "lag_1" in res_list.columns
    assert "lag_2" in res_list.columns


# ---------------------------------------------------------------------------
# Validation Fix Tests: Rolling window parameter validation
# ---------------------------------------------------------------------------


def test_rolling_window_parameter_validation():
    """Verify zero, negative, float, and boolean rolling window sizes are rejected."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
            "energy_kwh": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="rolling_window_hours must be a positive integer"):
        create_lag_and_rolling_features(df, rolling_window_hours=0)

    with pytest.raises(ValueError, match="rolling_window_hours must be a positive integer"):
        create_lag_and_rolling_features(df, rolling_window_hours=-5)

    with pytest.raises(ValueError, match="rolling_window_hours must be a positive integer"):
        create_lag_and_rolling_features(df, rolling_window_hours=24.5)

    with pytest.raises(ValueError, match="rolling_window_hours must be a positive integer"):
        create_lag_and_rolling_features(df, rolling_window_hours=True)

    with pytest.raises(ValueError, match="rolling_window_hours must be a positive integer"):
        create_lag_and_rolling_features(df, rolling_window_hours="24")


# ---------------------------------------------------------------------------
# Validation Fix Tests: Duplicate timestamp rejection
# ---------------------------------------------------------------------------


def test_duplicate_timestamps_rejected():
    """Verify duplicate timestamps are strictly rejected in feature engineering."""
    # Data with duplicate 10:00 timestamp
    dup_df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-01-01 10:00:00"),
                pd.Timestamp("2026-01-01 10:00:00"),
                pd.Timestamp("2026-01-01 11:00:00"),
            ],
            "energy_kwh": [2.0, 3.0, 4.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate timestamps"):
        create_lag_and_rolling_features(dup_df)

    with pytest.raises(ValueError, match="duplicate timestamps"):
        extract_temporal_features(dup_df)


def test_null_timestamps_rejected():
    """Verify null or unparseable timestamps are rejected with ValueError."""
    null_df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01 10:00:00"), None],
            "energy_kwh": [2.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="missing or unparseable timestamps"):
        create_lag_and_rolling_features(null_df)

    with pytest.raises(ValueError, match="missing or unparseable timestamps"):
        extract_temporal_features(null_df)
