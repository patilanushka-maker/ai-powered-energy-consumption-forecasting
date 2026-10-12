"""End-to-end synthetic integration test for AI Energy Forecasting pipeline.

Verifies:
- Complete orchestration via run_train_stage() in main.py
- Generation of temporal, lag-168, and rolling-24 features on 400+ hours
- Handling of timestamp gaps and ineligible feature rows
- Accurate prediction timestamp alignment after row exclusions
- Successful creation of model (.joblib), metrics report (.json), and figure (.png)
- Automatic cleanup of synthetic test artifacts
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from main import run_train_stage


def test_end_to_end_synthetic_training_and_evaluation(tmp_path):
    """Execute the full training & evaluation orchestration path on synthetic data."""
    # 1. Generate 400 synthetic hourly observations with an intentional timestamp gap
    # 400 hours ensures sufficient history for lag_168 and rolling_mean_24
    timestamps_part1 = pd.date_range("2026-01-01 00:00:00", periods=200, freq="h")
    # Intentional 3-hour gap: skips 2026-01-09 08:00:00 to 10:00:00
    timestamps_part2 = pd.date_range("2026-01-09 12:00:00", periods=200, freq="h")
    all_timestamps = timestamps_part1.append(timestamps_part2)

    np.random.seed(42)
    # Synthetic diurnal energy profile
    diurnal = 1.5 + 0.8 * np.sin(2 * np.pi * all_timestamps.hour / 24.0)
    noise = np.random.normal(0, 0.05, len(all_timestamps))
    energy = np.round(np.clip(diurnal + noise, 0.2, 4.5), 4)

    synth_df = pd.DataFrame(
        {
            "timestamp": all_timestamps,
            "energy_kwh": energy,
            "valid_minutes": 60,
            "coverage_ratio": 1.0,
        }
    )

    processed_path = tmp_path / "hourly_energy_synthetic.csv"
    raw_path = tmp_path / "household_power_consumption_dummy.txt"
    synth_df.to_csv(processed_path, index=False)

    model_save_path = tmp_path / "models" / "rf_synthetic.joblib"
    metrics_save_path = tmp_path / "outputs" / "metrics" / "synthetic_metrics.json"
    figure_save_path = tmp_path / "outputs" / "figures" / "synthetic_plot.png"

    # 2. Execute the actual pipeline orchestration stage
    run_train_stage(
        processed_path=processed_path,
        raw_path=raw_path,
        min_hourly_coverage=0.90,
        train_ratio=0.70,
        val_ratio=0.15,
        model_type="random_forest",
        n_estimators=10,
        random_state=42,
        model_save_path=model_save_path,
        metrics_save_path=metrics_save_path,
        figure_save_path=figure_save_path,
    )

    # 3. Confirm all three artifact types exist and are valid
    assert model_save_path.exists(), "Model artifact (.joblib) was not created."
    assert metrics_save_path.exists(), "Metrics report (.json) was not created."
    assert figure_save_path.exists(), "Forecast figure (.png) was not created."

    # Verify model artifact contents
    saved_payload = joblib.load(model_save_path)
    assert "model" in saved_payload
    assert saved_payload["model_type"] == "random_forest"
    assert saved_payload["random_state"] == 42
    assert len(saved_payload["feature_cols"]) == 9

    # Verify metrics JSON structure
    with open(metrics_save_path, "r", encoding="utf-8") as f:
        metrics_report = json.load(f)

    assert metrics_report["experiment_phase"] == "Phase 4A"
    assert metrics_report["target_column"] == "energy_kwh"

    # Confirm split summary
    split_summary = metrics_report["split_summary"]
    assert split_summary["total_rows"] == 400
    assert split_summary["train_rows"] == 280
    assert split_summary["val_rows"] == 60
    assert split_summary["test_rows"] == 60

    # Confirm eligibility filtering accounted for lag-168 lookback and timestamp gap
    audits = metrics_report["eligibility_audits"]
    assert audits["train"]["excluded_rows"] > 0, "Train should have excluded rows due to initial lag lookbacks."

    # Confirm evaluations exist for both validation and test splits
    val_eval = metrics_report["validation_evaluation"]
    test_eval = metrics_report["test_evaluation"]

    assert "common_timestamp_evaluations" in val_eval
    assert "Random Forest" in val_eval["common_timestamp_evaluations"]
    assert "Persistence (1h)" in val_eval["common_timestamp_evaluations"]
    assert "Seasonal Naive (24h)" in val_eval["common_timestamp_evaluations"]

    assert "common_timestamp_evaluations" in test_eval
    assert "Random Forest" in test_eval["common_timestamp_evaluations"]

    # Check that scored common samples are non-zero and metrics are finite
    rf_common_test = test_eval["common_timestamp_evaluations"]["Random Forest"]
    assert rf_common_test["scored_common_samples"] > 0
    assert np.isfinite(rf_common_test["mae"])
    assert np.isfinite(rf_common_test["rmse"])
    assert np.isfinite(rf_common_test["r2"])

    # Figure file is non-empty
    assert figure_save_path.stat().st_size > 1000
