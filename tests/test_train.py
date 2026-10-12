"""Unit tests for chronological splitting, baselines, and model training in src/train.py.

Covers:
- Chronological split ordering (no shuffling)
- Non-overlapping partitions
- Invalid split proportions
- Small dataset error handling
- Persistence baseline exact timestamp behavior
- Daily seasonal baseline exact 24-hour lookup
- Missing historical observations handling
- Eligibility filtering and row accounting
- Target and temporal leakage prevention
- Reproducible model training with fixed seed
- Missing dataset handling
"""

from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest

from src.train import (
    STANDARD_PREDICTOR_COLS,
    chronological_split,
    filter_eligible_rows,
    predict_daily_seasonal_naive,
    predict_persistence,
    select_predictor_features,
    train_model,
    train_random_forest,
)


@pytest.fixture
def synthetic_hourly_data():
    """Create a 100-hour continuous synthetic dataset for testing."""
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=100, freq="h")
    np.random.seed(42)
    energy = np.sin(np.linspace(0, 10, 100)) + 2.0 + np.random.normal(0, 0.1, 100)
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "energy_kwh": energy,
            "hour": timestamps.hour,
            "day_of_week": timestamps.dayofweek,
            "day_of_month": timestamps.day,
            "month": timestamps.month,
            "is_weekend": (timestamps.dayofweek >= 5).astype(int),
            "lag_1": pd.Series(energy).shift(1).values,
            "lag_24": pd.Series(energy).shift(24).values,
            "lag_168": np.nan,  # Not enough history
            "rolling_mean_24": pd.Series(energy).shift(1).rolling(24).mean().values,
        }
    )
    return df


def test_chronological_split_ordering(synthetic_hourly_data):
    """Test 1: Split preserves strict chronological ordering (train < val < test)."""
    train_df, val_df, test_df = chronological_split(
        synthetic_hourly_data, train_ratio=0.70, val_ratio=0.15
    )

    assert train_df["timestamp"].is_monotonic_increasing
    assert val_df["timestamp"].is_monotonic_increasing
    assert test_df["timestamp"].is_monotonic_increasing

    # Verify temporal sequence
    assert train_df["timestamp"].max() < val_df["timestamp"].min()
    assert val_df["timestamp"].max() < test_df["timestamp"].min()


def test_chronological_split_non_overlapping_partitions(synthetic_hourly_data):
    """Test 2: Partitions are strictly non-overlapping and partition counts sum to total."""
    train_df, val_df, test_df, summary = chronological_split(
        synthetic_hourly_data, train_ratio=0.70, val_ratio=0.15, return_summary=True
    )

    total_len = len(synthetic_hourly_data)
    assert len(train_df) + len(val_df) + len(test_df) == total_len

    # Assert sets of timestamps are disjoint
    train_ts = set(train_df["timestamp"])
    val_ts = set(val_df["timestamp"])
    test_ts = set(test_df["timestamp"])

    assert train_ts.isdisjoint(val_ts)
    assert val_ts.isdisjoint(test_ts)
    assert train_ts.isdisjoint(test_ts)

    # Check achieved proportions
    assert summary.train_rows == 70
    assert summary.val_rows == 15
    assert summary.test_rows == 15
    assert summary.train_ratio_achieved == 0.70
    assert summary.val_ratio_achieved == 0.15
    assert summary.test_ratio_achieved == 0.15


@pytest.mark.parametrize(
    "train_r,val_r,test_r",
    [
        (-0.1, 0.2, 0.9),
        (0.7, -0.1, 0.4),
        (0.5, 0.5, 0.5),  # Sum != 1.0
        (0.8, 0.3, None),  # Sum > 1.0
        (0.0, 0.5, 0.5),  # Zero ratio
        (True, 0.15, 0.15),  # Boolean type
    ],
)
def test_chronological_split_invalid_proportions(synthetic_hourly_data, train_r, val_r, test_r):
    """Test 3: Invalid split proportions raise descriptive ValueError."""
    with pytest.raises(ValueError, match="must be a positive float|must sum to 1.0"):
        chronological_split(
            synthetic_hourly_data,
            train_ratio=train_r,
            val_ratio=val_r,
            test_ratio=test_r,
        )


def test_chronological_split_small_dataset():
    """Test 4: Datasets too small to form non-empty partitions raise ValueError."""
    # 2 rows cannot split into 3 partitions
    small_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="h"),
            "energy_kwh": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="too small to form non-empty partitions"):
        chronological_split(small_df, train_ratio=0.70, val_ratio=0.15)


def test_persistence_baseline_exact_timestamp_behavior():
    """Test 5: Persistence baseline predicts using exact T - 1h observation."""
    timestamps = pd.to_datetime(
        ["2026-01-01 10:00:00", "2026-01-01 11:00:00", "2026-01-01 12:00:00"]
    )
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "energy_kwh": [1.5, 2.5, 3.5],
        }
    )

    preds = predict_persistence(df, lookback_hours=1)

    # First timestamp has no prior observation in df -> NaN
    assert np.isnan(preds.iloc[0])
    # Second timestamp: 11:00 uses 10:00 value -> 1.5
    assert preds.iloc[1] == 1.5
    # Third timestamp: 12:00 uses 11:00 value -> 2.5
    assert preds.iloc[2] == 2.5


def test_daily_seasonal_baseline_exact_24h_lookup():
    """Test 6: Daily seasonal baseline predicts using exact T - 24h observation."""
    # Create 30 hours of data
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=30, freq="h")
    values = np.arange(30, dtype=float)
    df = pd.DataFrame({"timestamp": timestamps, "energy_kwh": values})

    preds = predict_daily_seasonal_naive(df, seasonal_hours=24)

    # First 24 hours have no 24h-ago observation -> all NaN
    for i in range(24):
        assert np.isnan(preds.iloc[i])

    # Hour 24 (2026-01-02 00:00:00) should look up hour 0 (value 0.0)
    assert preds.iloc[24] == 0.0
    # Hour 29 should look up hour 5 (value 5.0)
    assert preds.iloc[29] == 5.0


def test_baselines_missing_historical_observations():
    """Test 7: Baselines return NaN when historical timestamp is absent or has gaps."""
    # Series with a 3-hour gap
    timestamps = pd.to_datetime(
        ["2026-01-01 10:00:00", "2026-01-01 14:00:00"]  # 14:00 does not have 13:00
    )
    df = pd.DataFrame({"timestamp": timestamps, "energy_kwh": [10.0, 20.0]})

    preds_persist = predict_persistence(df, lookback_hours=1)
    # 14:00 needs 13:00, which is missing -> must be NaN, NOT 10.0
    assert np.isnan(preds_persist.iloc[1])

    # Seasonal naive with missing exact 24-hour observation
    timestamps_seasonal = pd.to_datetime(
        ["2026-01-01 10:00:00", "2026-01-02 11:00:00"]  # 25 hours later, not 24
    )
    df_seasonal = pd.DataFrame({"timestamp": timestamps_seasonal, "energy_kwh": [10.0, 20.0]})
    preds_seasonal = predict_daily_seasonal_naive(df_seasonal, seasonal_hours=24)
    assert np.isnan(preds_seasonal.iloc[1])


def test_leakage_prevention_predictor_selection():
    """Test 8: select_predictor_features strictly excludes target, timestamp, and metadata."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="h"),
            "energy_kwh": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Global_active_power": [1.0, 2.0, 3.0, 4.0, 5.0],
            "valid_minutes": [60, 60, 60, 60, 60],
            "coverage_ratio": [1.0, 1.0, 1.0, 1.0, 1.0],
            "hour": [0, 1, 2, 3, 4],
            "lag_1": [np.nan, 1.0, 2.0, 3.0, 4.0],
        }
    )

    predictors = select_predictor_features(df)

    assert "energy_kwh" not in predictors
    assert "timestamp" not in predictors
    assert "Global_active_power" not in predictors
    assert "valid_minutes" not in predictors
    assert "coverage_ratio" not in predictors
    assert "hour" in predictors
    assert "lag_1" in predictors


def test_filter_eligible_rows_accounting():
    """Test 9: filter_eligible_rows accurately drops missing rows and records reasons."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h"),
            "energy_kwh": [1.0, 2.0, np.nan, 4.0],  # row 2 has missing target
            "hour": [0, 1, 2, 3],
            "lag_1": [np.nan, 1.0, 2.0, 3.0],  # row 0 has missing lag
        }
    )

    feature_cols = ["hour", "lag_1"]
    eligible_df, audit = filter_eligible_rows(df, feature_cols, target_col="energy_kwh")

    # Row 0 missing lag_1, Row 2 missing energy_kwh -> only rows 1 and 3 are eligible
    assert len(eligible_df) == 2
    assert audit["total_rows"] == 4
    assert audit["eligible_rows"] == 2
    assert audit["excluded_rows"] == 2
    assert audit["missing_target_count"] == 1
    assert audit["feature_missing_counts"]["lag_1"] == 1
    assert audit["feature_missing_counts"]["hour"] == 0

    # Ensure caller DataFrame was not mutated
    assert len(df) == 4
    assert df["energy_kwh"].isna().sum() == 1


def test_reproducible_model_training(tmp_path):
    """Test 10: Fixed random seed produces identical predictions and artifact saves."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feat_1": np.random.normal(0, 1, 50),
            "feat_2": np.random.normal(0, 1, 50),
            "energy_kwh": np.random.normal(2, 0.5, 50),
        }
    )

    feature_cols = ["feat_1", "feat_2"]

    # Train model 1
    save_path = tmp_path / "test_model.joblib"
    m1 = train_model(
        X,
        model_type="random_forest",
        feature_cols=feature_cols,
        save_path=save_path,
        random_state=42,
        n_estimators=10,
    )

    # Train model 2 with identical seed
    m2 = train_random_forest(
        X, feature_cols=feature_cols, random_state=42, n_estimators=10
    )

    test_X = X[feature_cols].copy()
    preds_1 = m1.predict(test_X)
    preds_2 = m2.predict(test_X)

    # Must be identical within floating point precision
    np.testing.assert_allclose(preds_1, preds_2, atol=1e-12)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_missing_dataset_cli_error():
    """Test 11: CLI emits clear error and exits with code 1 when dataset files are absent."""
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--stage",
            "train",
            "--raw-data-path",
            "data/raw/non_existent_file.txt",
            "--processed-data-path",
            "data/processed/non_existent_file.csv",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "ERROR: Required dataset is missing." in result.stderr
    assert "UCI Machine Learning Repository" in result.stderr


def test_chronological_split_rejects_duplicate_timestamps():
    """Regression Test: chronological_split rejects duplicate timestamps with clear ValueError."""
    timestamps = pd.to_datetime(
        ["2026-01-01 00:00:00", "2026-01-01 01:00:00", "2026-01-01 01:00:00", "2026-01-01 02:00:00", "2026-01-01 03:00:00"]
    )
    df = pd.DataFrame({"timestamp": timestamps, "energy_kwh": [1.0, 2.0, 2.5, 3.0, 4.0]})

    with pytest.raises(ValueError, match="duplicate timestamps in 'timestamp'"):
        chronological_split(df, train_ratio=0.60, val_ratio=0.20)


def test_chronological_split_rejects_null_timestamps():
    """Regression Test: chronological_split rejects missing/unparseable timestamps."""
    df = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 00:00:00", "invalid_date", "2026-01-01 02:00:00"],
            "energy_kwh": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="missing or unparseable timestamps in 'timestamp'"):
        chronological_split(df, train_ratio=0.50, val_ratio=0.25)


def test_filter_eligible_rows_preserves_original_index_at_beginning():
    """Regression Test: filter_eligible_rows preserves original indices when first row is excluded."""
    # Row 0 has missing lag_1; rows 1, 2, 3 are valid
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h"),
            "energy_kwh": [1.0, 2.0, 3.0, 4.0],
            "lag_1": [np.nan, 1.0, 2.0, 3.0],
        },
        index=[10, 20, 30, 40],  # Non-consecutive custom index
    )

    eligible_df, audit = filter_eligible_rows(df, feature_cols=["lag_1"], target_col="energy_kwh")

    # Indices must be exactly [20, 30, 40], NOT [0, 1, 2]
    assert list(eligible_df.index) == [20, 30, 40]
    assert audit["excluded_rows"] == 1
    assert audit["eligible_rows"] == 3


def test_filter_eligible_rows_preserves_original_index_in_middle():
    """Regression Test: filter_eligible_rows preserves original indices when middle rows are excluded."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="h"),
            "energy_kwh": [1.0, 2.0, np.nan, 4.0, 5.0],  # Middle row 2 has missing target
            "lag_1": [1.0, 2.0, 3.0, np.nan, 5.0],        # Middle row 3 has missing lag
        },
        index=[0, 1, 2, 3, 4],
    )

    eligible_df, audit = filter_eligible_rows(df, feature_cols=["lag_1"], target_col="energy_kwh")

    # Rows 0, 1, and 4 are valid; rows 2 and 3 excluded
    assert list(eligible_df.index) == [0, 1, 4]
    assert audit["excluded_rows"] == 2
    assert audit["eligible_rows"] == 3


def test_prediction_alignment_after_exclusions():
    """Regression Test: Prediction Series aligns correctly with ground truth when rows are excluded."""
    val_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 00:00:00", periods=5, freq="h"),
            "energy_kwh": [10.0, 20.0, 30.0, 40.0, 50.0],
            "hour": [0, 1, 2, 3, 4],
            "lag_1": [np.nan, 10.0, np.nan, 30.0, 40.0],  # Rows 0 and 2 ineligible
        }
    )

    feature_cols = ["hour", "lag_1"]
    val_eligible, _ = filter_eligible_rows(val_df, feature_cols=feature_cols)

    # Eligible indices should be [1, 3, 4]
    assert list(val_eligible.index) == [1, 3, 4]

    # Mock predictions for eligible rows
    mock_model_preds = np.array([200.0, 400.0, 500.0])

    val_rf_preds = pd.Series(np.nan, index=val_df.index, dtype="float64")
    val_rf_preds.loc[val_eligible.index] = mock_model_preds

    # Verify that row 0 and 2 are NaN, and rows 1, 3, 4 have their exact matching predictions
    assert np.isnan(val_rf_preds.loc[0])
    assert val_rf_preds.loc[1] == 200.0
    assert np.isnan(val_rf_preds.loc[2])
    assert val_rf_preds.loc[3] == 400.0
    assert val_rf_preds.loc[4] == 500.0

    # Verify alignment against ground truth
    assert val_df.loc[1, "energy_kwh"] == 20.0
    assert val_df.loc[3, "energy_kwh"] == 40.0
    assert val_df.loc[4, "energy_kwh"] == 50.0


def test_missing_lag_features_from_timestamp_gaps():
    """Regression Test: Missing lags caused by temporal gaps produce NaNs and are excluded without index drift."""
    from src.feature_engineering import create_lag_and_rolling_features

    # Gap of 5 hours between 01:00 and 06:00
    timestamps = pd.to_datetime(
        ["2026-01-01 00:00:00", "2026-01-01 01:00:00", "2026-01-01 06:00:00", "2026-01-01 07:00:00"]
    )
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "energy_kwh": [1.0, 2.0, 6.0, 7.0],
        }
    )

    df_feats = create_lag_and_rolling_features(df, lags=(1,), rolling_window_hours=2)

    # 06:00 cannot find 05:00 -> lag_1 is NaN
    assert pd.isna(df_feats.loc[2, "lag_1"])
    # 07:00 finds 06:00 -> lag_1 is 6.0
    assert df_feats.loc[3, "lag_1"] == 6.0

    eligible_df, audit = filter_eligible_rows(df_feats, feature_cols=["lag_1"])

    # Row 0 (no prior) and row 2 (prior gap) excluded; rows 1 and 3 are eligible
    assert list(eligible_df.index) == [1, 3]
    assert audit["excluded_rows"] == 2

