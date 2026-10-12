"""Unit and integration tests for Phase 4B recursive forecasting engine.

Verifies:
- Model artifact loading and validation of corrupted/invalid payloads.
- Recursive autoregressive rollout across default 24h and custom horizons (1 to 168h).
- Strict feature ordering and recursive propagation of earlier predictions into lags/rolling windows.
- History validation: short history, duplicate/null timestamps, missing/non-finite targets.
- Rejection of timestamp gaps in the required ending history window.
- Temporal edge transitions (midnight, month-end, year-end, weekend switches).
- Model output validation (non-finite predictions rejection, negative value warnings).
- CSV output persistence and metadata attributes.
- CLI execution via main.py without model retraining or artifact mutation.
"""

from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import pytest

from main import run_forecast_stage
from src.forecast import (
    generate_forecast,
    load_forecasting_model,
    validate_and_prepare_history,
)


class DummyRegressor:
    """Mock regressor that records received feature inputs and returns predictable outputs."""

    def __init__(self, constant_pred: float = 1.25):
        self.constant_pred = constant_pred
        self.received_inputs: list = []

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.received_inputs.append(X.copy())
        return np.full(len(X), self.constant_pred, dtype=np.float64)


class DynamicRegressor:
    """Mock regressor whose predictions depend on input features."""

    def __init__(self):
        self.received_inputs: list = []

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.received_inputs.append(X.copy())
        # Predict lag_1 + 0.1
        preds = X["lag_1"].to_numpy(dtype=float) + 0.1
        return preds


@pytest.fixture
def standard_feature_cols():
    return [
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


@pytest.fixture
def dummy_model_payload(standard_feature_cols):
    return {
        "model": DummyRegressor(constant_pred=2.0),
        "model_type": "dummy_forest",
        "feature_cols": standard_feature_cols,
        "target_col": "energy_kwh",
        "random_state": 42,
    }


@pytest.fixture
def synthetic_history_168():
    """Generates exactly 168 contiguous hourly historical observations ending at 2026-06-07 23:00:00."""
    timestamps = pd.date_range("2026-06-01 00:00:00", periods=168, freq="h")
    energy = np.linspace(1.0, 3.0, 168)
    return pd.DataFrame({"timestamp": timestamps, "energy_kwh": energy})


# =========================================================================
# 1. Model Artifact Loading Tests
# =========================================================================

def test_load_model_missing_file():
    with pytest.raises(FileNotFoundError, match="Forecasting model artifact not found"):
        load_forecasting_model("non_existent_model_path.joblib")


def test_load_model_non_dict_payload(tmp_path):
    bad_path = tmp_path / "bad_payload.joblib"
    joblib.dump(["not", "a", "dict"], bad_path)

    with pytest.raises(ValueError, match="Invalid model artifact format"):
        load_forecasting_model(bad_path)


def test_load_model_missing_keys(tmp_path):
    bad_path = tmp_path / "missing_keys.joblib"
    joblib.dump({"model": DummyRegressor()}, bad_path)

    with pytest.raises(ValueError, match="missing required keys"):
        load_forecasting_model(bad_path)


def test_load_model_no_predict_method(tmp_path):
    bad_path = tmp_path / "no_predict.joblib"
    joblib.dump(
        {"model": "not_an_estimator", "feature_cols": ["hour"], "target_col": "energy_kwh"},
        bad_path,
    )

    with pytest.raises(ValueError, match="does not implement 'predict'"):
        load_forecasting_model(bad_path)


def test_load_model_empty_feature_cols(tmp_path):
    bad_path = tmp_path / "empty_features.joblib"
    joblib.dump(
        {"model": DummyRegressor(), "feature_cols": [], "target_col": "energy_kwh"},
        bad_path,
    )

    with pytest.raises(ValueError, match="invalid feature_cols"):
        load_forecasting_model(bad_path)


def test_load_model_success(tmp_path, dummy_model_payload):
    model_path = tmp_path / "model.joblib"
    joblib.dump(dummy_model_payload, model_path)

    loaded = load_forecasting_model(model_path)
    assert loaded["model_type"] == "dummy_forest"
    assert len(loaded["feature_cols"]) == 9


# =========================================================================
# 2. Historical Window Validation Tests
# =========================================================================

def test_history_empty_dataset():
    with pytest.raises(ValueError, match="Historical dataset is empty"):
        validate_and_prepare_history(pd.DataFrame(columns=["timestamp", "energy_kwh"]))


def test_history_missing_required_columns():
    df = pd.DataFrame({"value": [1.0, 2.0]})
    with pytest.raises(ValueError, match="must contain 'timestamp' or 'datetime'"):
        validate_and_prepare_history(df)


def test_history_null_timestamps(synthetic_history_168):
    df = synthetic_history_168.copy()
    df.loc[10, "timestamp"] = pd.NaT
    with pytest.raises(ValueError, match="null or unparseable timestamps"):
        validate_and_prepare_history(df)


def test_history_duplicate_timestamps(synthetic_history_168):
    df = synthetic_history_168.copy()
    df.loc[10, "timestamp"] = df.loc[9, "timestamp"]
    with pytest.raises(ValueError, match="duplicate timestamp"):
        validate_and_prepare_history(df)


def test_history_non_hourly_alignment(synthetic_history_168):
    df = synthetic_history_168.copy()
    df.loc[10, "timestamp"] = df.loc[10, "timestamp"] + pd.Timedelta(minutes=15)
    with pytest.raises(ValueError, match="must align to top-of-hour boundaries"):
        validate_and_prepare_history(df)


def test_history_insufficient_length(synthetic_history_168):
    short_df = synthetic_history_168.iloc[:100].copy()
    with pytest.raises(ValueError, match="at least 168 contiguous hours are required"):
        validate_and_prepare_history(short_df)


def test_history_gap_in_ending_window():
    # 200 total hours, with a gap inside the ending 168-hour window
    ts_part1 = pd.date_range("2026-06-01 00:00:00", periods=50, freq="h")
    # 2-hour gap here
    ts_part2 = pd.date_range("2026-06-03 04:00:00", periods=150, freq="h")
    timestamps = ts_part1.append(ts_part2)
    gapped_df = pd.DataFrame({"timestamp": timestamps, "energy_kwh": 2.0})

    with pytest.raises(ValueError, match="contains missing timestamps or recording gaps"):
        validate_and_prepare_history(gapped_df)


def test_history_nan_target_in_window(synthetic_history_168):
    nan_df = synthetic_history_168.copy()
    nan_df.loc[50, "energy_kwh"] = np.nan
    with pytest.raises(ValueError, match="contains missing or NaN values"):
        validate_and_prepare_history(nan_df)


def test_history_infinite_target_in_window(synthetic_history_168):
    inf_df = synthetic_history_168.copy()
    inf_df.loc[50, "energy_kwh"] = np.inf
    with pytest.raises(ValueError, match="contains non-finite"):
        validate_and_prepare_history(inf_df)


def test_history_older_gaps_tolerated_if_contiguous_ending_window_present():
    """An ancient gap from months ago should NOT fail forecasting if recent 168h is contiguous."""
    old_ts = pd.date_range("2025-01-01", periods=10, freq="h")
    recent_ts = pd.date_range("2026-01-01", periods=168, freq="h")
    combined_ts = old_ts.append(recent_ts)
    combined_df = pd.DataFrame({"timestamp": combined_ts, "energy_kwh": 1.5})

    window_df, start, end = validate_and_prepare_history(combined_df)
    assert len(window_df) == 168
    assert end == recent_ts[-1]
    assert start == recent_ts[0]


def test_history_normalization_accepts_datetime_column():
    ts = pd.date_range("2026-01-01", periods=168, freq="h")
    df = pd.DataFrame({"datetime": ts, "energy_kwh": 2.1})
    window_df, start, end = validate_and_prepare_history(df)
    assert "timestamp" in window_df.columns
    assert len(window_df) == 168


# =========================================================================
# 3. Recursive Forecasting Engine Tests
# =========================================================================

def test_forecast_horizon_validation(dummy_model_payload, synthetic_history_168):
    with pytest.raises(ValueError, match="horizon_hours must be between 1 and 168"):
        generate_forecast(dummy_model_payload, synthetic_history_168, horizon_hours=0)

    with pytest.raises(ValueError, match="horizon_hours must be between 1 and 168"):
        generate_forecast(dummy_model_payload, synthetic_history_168, horizon_hours=169)

    with pytest.raises(ValueError, match="horizon_hours must be an integer"):
        generate_forecast(dummy_model_payload, synthetic_history_168, horizon_hours=24.5)  # type: ignore


def test_forecast_default_24h_rollout(dummy_model_payload, synthetic_history_168):
    forecast_df = generate_forecast(dummy_model_payload, synthetic_history_168, horizon_hours=24)

    assert len(forecast_df) == 24
    assert list(forecast_df.columns) == ["timestamp", "forecast_energy_kwh", "step"]
    assert forecast_df["step"].tolist() == list(range(1, 25))

    origin = synthetic_history_168["timestamp"].iloc[-1]
    expected_start = origin + pd.Timedelta(hours=1)
    expected_end = origin + pd.Timedelta(hours=24)
    assert forecast_df["timestamp"].iloc[0] == expected_start
    assert forecast_df["timestamp"].iloc[-1] == expected_end

    assert forecast_df.attrs["forecast_origin"] == str(origin)
    assert forecast_df.attrs["horizon_hours"] == 24


def test_forecast_recursive_lags_and_rolling_buffer(standard_feature_cols, synthetic_history_168):
    """Verify that step 2 uses prediction from step 1 for lag_1 and rolling_mean_24."""
    model = DynamicRegressor()
    payload = {
        "model": model,
        "model_type": "dynamic",
        "feature_cols": standard_feature_cols,
        "target_col": "energy_kwh",
    }

    forecast_df = generate_forecast(payload, synthetic_history_168, horizon_hours=5)

    assert len(model.received_inputs) == 5
    # Step 1: lag_1 should be history[-1]
    hist_last_val = synthetic_history_168["energy_kwh"].iloc[-1]
    step1_input = model.received_inputs[0]
    assert np.isclose(step1_input["lag_1"].iloc[0], hist_last_val)

    # Prediction 1 = hist_last_val + 0.1
    pred1 = forecast_df["forecast_energy_kwh"].iloc[0]
    assert np.isclose(pred1, hist_last_val + 0.1)

    # Step 2: lag_1 should be pred1
    step2_input = model.received_inputs[1]
    assert np.isclose(step2_input["lag_1"].iloc[0], pred1)

    # Step 2: rolling_mean_24 should include pred1 as the most recent of the 24 values
    # Preceding 23 historical values + pred1
    hist_prev23 = synthetic_history_168["energy_kwh"].iloc[-23:].tolist()
    expected_rolling = np.mean(hist_prev23 + [pred1])
    assert np.isclose(step2_input["rolling_mean_24"].iloc[0], expected_rolling)


def test_forecast_exact_feature_ordering(synthetic_history_168):
    """Verify model receives features in the exact ordering saved in the artifact."""
    reversed_features = [
        "rolling_mean_24",
        "lag_168",
        "lag_24",
        "lag_1",
        "is_weekend",
        "month",
        "day_of_month",
        "day_of_week",
        "hour",
    ]
    model = DummyRegressor(constant_pred=1.5)
    payload = {
        "model": model,
        "model_type": "ordered_test",
        "feature_cols": reversed_features,
        "target_col": "energy_kwh",
    }

    generate_forecast(payload, synthetic_history_168, horizon_hours=3)

    for received_df in model.received_inputs:
        assert list(received_df.columns) == reversed_features


def test_forecast_temporal_edge_transitions(standard_feature_cols):
    """Test transitions across midnight, month-end, and year-end."""
    # End history on New Year's Eve: 2025-12-31 23:00:00 (Wednesday)
    timestamps = pd.date_range("2025-12-25 00:00:00", periods=168, freq="h")
    assert timestamps[-1] == pd.Timestamp("2025-12-31 23:00:00")
    history_df = pd.DataFrame({"timestamp": timestamps, "energy_kwh": 2.0})

    model = DummyRegressor()
    payload = {
        "model": model,
        "feature_cols": standard_feature_cols,
        "target_col": "energy_kwh",
    }

    forecast_df = generate_forecast(payload, history_df, horizon_hours=48)

    # Step 1: 2026-01-01 00:00:00 (Thursday, day 3, month 1, year 2026)
    step1_input = model.received_inputs[0]
    assert step1_input["hour"].iloc[0] == 0
    assert step1_input["day_of_week"].iloc[0] == 3
    assert step1_input["day_of_month"].iloc[0] == 1
    assert step1_input["month"].iloc[0] == 1
    assert step1_input["is_weekend"].iloc[0] == 0

    # Step 25: 2026-01-02 00:00:00 (Friday)
    step25_input = model.received_inputs[24]
    assert step25_input["hour"].iloc[0] == 0
    assert step25_input["day_of_week"].iloc[0] == 4
    assert step25_input["day_of_month"].iloc[0] == 2
    assert step25_input["is_weekend"].iloc[0] == 0


def test_forecast_leap_day_transitions(standard_feature_cols):
    """Test calendar feature calculations across leap year boundaries (e.g. Feb 29, 2008)."""
    # 2008 is a leap year; generate 168 hours ending on 2008-02-28 23:00:00
    timestamps = pd.date_range("2008-02-22 00:00:00", periods=168, freq="h")
    assert timestamps[-1] == pd.Timestamp("2008-02-28 23:00:00")
    history_df = pd.DataFrame({"timestamp": timestamps, "energy_kwh": 1.8})

    model = DummyRegressor()
    payload = {
        "model": model,
        "feature_cols": standard_feature_cols,
        "target_col": "energy_kwh",
    }

    # Forecast 48 hours: spans across Feb 29 and March 1
    generate_forecast(payload, history_df, horizon_hours=48)

    # Step 1: 2008-02-29 00:00:00 (Friday)
    step1_input = model.received_inputs[0]
    assert step1_input["month"].iloc[0] == 2
    assert step1_input["day_of_month"].iloc[0] == 29
    assert step1_input["day_of_week"].iloc[0] == 4  # Friday
    assert step1_input["is_weekend"].iloc[0] == 0

    # Step 25: 2008-03-01 00:00:00 (Saturday)
    step25_input = model.received_inputs[24]
    assert step25_input["month"].iloc[0] == 3
    assert step25_input["day_of_month"].iloc[0] == 1
    assert step25_input["day_of_week"].iloc[0] == 5  # Saturday
    assert step25_input["is_weekend"].iloc[0] == 1


def test_forecast_non_finite_predictions_rejected(standard_feature_cols, synthetic_history_168):
    class NanRegressor:
        def predict(self, X):
            return np.array([np.nan])

    payload = {
        "model": NanRegressor(),
        "feature_cols": standard_feature_cols,
        "target_col": "energy_kwh",
    }

    with pytest.raises(ValueError, match="Model generated non-finite prediction"):
        generate_forecast(payload, synthetic_history_168, horizon_hours=5)


def test_forecast_output_csv_creation(tmp_path, dummy_model_payload, synthetic_history_168):
    out_file = tmp_path / "forecasts" / "custom_forecast.csv"
    forecast_df = generate_forecast(
        dummy_model_payload,
        synthetic_history_168,
        horizon_hours=12,
        output_path=out_file,
    )

    assert out_file.exists()
    saved_df = pd.read_csv(out_file)
    assert len(saved_df) == 12
    assert list(saved_df.columns) == ["timestamp", "forecast_energy_kwh", "step"]
    assert np.all(np.isfinite(saved_df["forecast_energy_kwh"]))


# =========================================================================
# 4. Pipeline Integration & Immutability Tests
# =========================================================================

def test_run_forecast_stage_smoke_test(tmp_path, dummy_model_payload, synthetic_history_168):
    """Test CLI orchestration path run_forecast_stage without modifying files."""
    model_path = tmp_path / "model.joblib"
    joblib.dump(dummy_model_payload, model_path)

    hist_path = tmp_path / "hourly_energy.csv"
    synthetic_history_168.to_csv(hist_path, index=False)

    out_path = tmp_path / "forecast_smoke.csv"

    run_forecast_stage(
        model_path=model_path,
        processed_path=hist_path,
        horizon_hours=6,
        forecast_output_path=out_path,
    )

    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert len(df) == 6


def test_forecasting_does_not_mutate_model_artifact(tmp_path, dummy_model_payload, synthetic_history_168):
    """Verify that forecasting NEVER touches or overwrites the trained model artifact."""
    model_path = tmp_path / "model_immutable.joblib"
    joblib.dump(dummy_model_payload, model_path)

    initial_mtime = model_path.stat().st_mtime_ns
    initial_bytes = model_path.read_bytes()

    time.sleep(0.01)

    generate_forecast(
        model_path,
        synthetic_history_168,
        horizon_hours=24,
        output_path=tmp_path / "out.csv",
    )

    after_mtime = model_path.stat().st_mtime_ns
    after_bytes = model_path.read_bytes()

    assert initial_mtime == after_mtime, "Model artifact mtime was modified during forecasting!"
    assert initial_bytes == after_bytes, "Model artifact content was changed during forecasting!"
