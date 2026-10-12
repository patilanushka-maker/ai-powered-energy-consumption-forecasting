"""Unit tests for metric computation, evaluation, and reporting in src/evaluate.py.

Covers:
- Metric calculation on known synthetic values (MAE, RMSE, R²)
- Missing prediction accounting
- Fair multi-model comparison on common non-missing timestamps
- Metric computation edge cases (zero variance, NaNs, length mismatch)
- Metrics JSON report export
- Visual forecast plot generation
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    compute_metrics,
    evaluate_and_compare_models,
    evaluate_model_predictions,
    plot_predictions,
    save_metrics_report,
)


def test_compute_metrics_known_synthetic_values():
    """Test: Compute MAE, RMSE, and R2 on hand-crafted synthetic arrays with known analytical values."""
    # y_true = [2.0, 4.0, 6.0, 8.0]
    # y_pred = [1.0, 5.0, 6.0, 10.0]
    # errors = [-1.0, 1.0, 0.0, 2.0]
    # abs_errors = [1.0, 1.0, 0.0, 2.0] -> MAE = 4 / 4 = 1.0
    # sq_errors = [1.0, 1.0, 0.0, 4.0] -> MSE = 6 / 4 = 1.5 -> RMSE = sqrt(1.5) = 1.22474487
    # y_mean = 5.0 -> total_ss = (2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2 = 9 + 1 + 1 + 9 = 20
    # ss_res = 6.0 -> R2 = 1 - (6 / 20) = 0.70
    y_true = np.array([2.0, 4.0, 6.0, 8.0])
    y_pred = np.array([1.0, 5.0, 6.0, 10.0])

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["n_samples"] == 4
    assert pytest.approx(metrics["mae"], abs=1e-5) == 1.0
    assert pytest.approx(metrics["rmse"], abs=1e-5) == np.sqrt(1.5)
    assert pytest.approx(metrics["r2"], abs=1e-5) == 0.70


def test_missing_prediction_accounting():
    """Test: evaluate_model_predictions accurately tracks total, scored, and missing rows."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="h"),
            "energy_kwh": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    # 2 out of 5 predictions are NaN
    preds = np.array([1.0, np.nan, 3.2, np.nan, 4.8])

    res = evaluate_model_predictions(df, preds, model_name="TestModel")

    assert res["model_name"] == "TestModel"
    assert res["total_rows"] == 5
    assert res["scored_samples"] == 3
    assert res["missing_predictions"] == 2
    assert res["coverage_percentage"] == 60.0
    # Scored on elements 0, 2, 4:
    # y_true = [1.0, 3.0, 5.0], y_pred = [1.0, 3.2, 4.8]
    # abs_errors = [0.0, 0.2, 0.2] -> MAE = 0.4 / 3 = 0.133333
    assert pytest.approx(res["mae"], abs=1e-4) == 0.133333


def test_evaluate_and_compare_common_timestamps():
    """Test: evaluate_and_compare_models produces both individual and common timestamp metrics."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h"),
            "energy_kwh": [1.0, 2.0, 3.0, 4.0],
        }
    )

    # Model A is missing index 0
    preds_a = pd.Series([np.nan, 2.1, 2.9, 4.2], index=df.index)
    # Model B is missing index 3
    preds_b = pd.Series([1.2, 1.9, 3.1, np.nan], index=df.index)

    comparison = evaluate_and_compare_models(
        df,
        predictions_dict={"Model A": preds_a, "Model B": preds_b},
    )

    # Individual evaluations
    assert comparison["individual_evaluations"]["Model A"]["scored_samples"] == 3
    assert comparison["individual_evaluations"]["Model B"]["scored_samples"] == 3

    # Common non-missing timestamps: only indices 1 and 2 are non-missing in BOTH models
    assert comparison["common_scored_count"] == 2
    common_a = comparison["common_timestamp_evaluations"]["Model A"]
    common_b = comparison["common_timestamp_evaluations"]["Model B"]

    assert common_a["scored_common_samples"] == 2
    assert common_b["scored_common_samples"] == 2


def test_compute_metrics_edge_cases():
    """Test: compute_metrics handles edge cases gracefully."""
    # Length mismatch raises ValueError
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_metrics([1.0, 2.0], [1.0])

    # All NaNs returns NaN metrics with n_samples=0
    nan_res = compute_metrics([np.nan, np.nan], [np.nan, np.nan])
    assert nan_res["n_samples"] == 0
    assert np.isnan(nan_res["mae"])
    assert np.isnan(nan_res["rmse"])
    assert np.isnan(nan_res["r2"])

    # Constant target has zero variance -> R2 is NaN but MAE/RMSE are valid
    const_res = compute_metrics([3.0, 3.0, 3.0], [3.1, 2.9, 3.0])
    assert const_res["n_samples"] == 3
    assert pytest.approx(const_res["mae"], abs=1e-5) == (0.1 + 0.1 + 0.0) / 3
    assert np.isnan(const_res["r2"])


def test_save_metrics_report(tmp_path):
    """Test: save_metrics_report writes valid JSON to disk."""
    report_file = tmp_path / "metrics.json"
    data = {
        "model": "RandomForest",
        "mae": 0.25,
        "rmse": 0.45,
        "r2": 0.85,
    }

    saved_path = save_metrics_report(data, report_file)

    assert saved_path.exists()
    with open(saved_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["model"] == "RandomForest"
    assert loaded["mae"] == 0.25


def test_plot_predictions(tmp_path):
    """Test: plot_predictions creates a valid non-empty image file."""
    timestamps = pd.date_range("2026-01-01", periods=10, freq="h")
    y_true = np.linspace(1.0, 5.0, 10)
    preds = {"Persistence": y_true + 0.1, "Random Forest": y_true - 0.05}
    fig_path = tmp_path / "test_forecast_plot.png"

    saved_plot = plot_predictions(
        timestamps=timestamps,
        y_true=y_true,
        predictions_dict=preds,
        save_path=fig_path,
        title="Test Forecast Plot",
    )

    assert saved_plot.exists()
    assert saved_plot.stat().st_size > 1000  # Non-trivial image


def test_evaluation_alignment_permuted_series_index():
    """Regression Test: evaluate_model_predictions correctly reindexes permuted Series index."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
            "energy_kwh": [10.0, 20.0, 30.0],
        },
        index=[0, 1, 2],
    )

    # Predictions passed in permuted order: index 2 is 30, index 0 is 10, index 1 is 20
    scrambled_preds = pd.Series([30.0, 10.0, 20.0], index=[2, 0, 1])

    # If positionally evaluated without alignment, errors would be [20, 10, 10] -> MAE > 0
    # With key-based alignment, error is 0.0!
    res = evaluate_model_predictions(df, scrambled_preds, model_name="PermutedModel")

    assert res["mae"] == 0.0
    assert res["rmse"] == 0.0


def test_evaluation_alignment_disjoint_series_index_raises():
    """Regression Test: Disjoint Series index raises clear ValueError."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
            "energy_kwh": [10.0, 20.0, 30.0],
        },
        index=[0, 1, 2],
    )

    # Disjoint index labels [100, 101, 102]
    disjoint_preds = pd.Series([10.0, 20.0, 30.0], index=[100, 101, 102])

    with pytest.raises(ValueError, match="is completely disjoint from evaluation DataFrame index"):
        evaluate_model_predictions(df, disjoint_preds, model_name="DisjointModel")


def test_evaluation_alignment_foreign_keys_raises():
    """Regression Test: Prediction Series containing foreign keys raises ValueError."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
            "energy_kwh": [10.0, 20.0, 30.0],
        },
        index=[0, 1, 2],
    )

    # Has index 0, 1, and foreign key 999
    foreign_preds = pd.Series([10.0, 20.0, 30.0], index=[0, 1, 999])

    with pytest.raises(ValueError, match="contains keys not present in evaluation DataFrame index"):
        evaluate_model_predictions(df, foreign_preds, model_name="ForeignModel")


def test_evaluation_alignment_array_length_mismatch_raises():
    """Regression Test: Prediction array with length mismatch raises ValueError."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h"),
            "energy_kwh": [1.0, 2.0, 3.0, 4.0],
        }
    )

    with pytest.raises(ValueError, match="does not match evaluation DataFrame length"):
        evaluate_model_predictions(df, [1.0, 2.0], model_name="ShortArrayModel")

