"""Unit and integration tests for recursive forecasting backtest utility."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from src.backtest import run_recursive_backtest


class MockDummyRegressor:
    """Deterministic mock regressor for fast backtest unit testing."""

    def __init__(self, offset: float = 0.5):
        self.offset = offset

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Return a deterministic constant
        return np.full(len(X), self.offset, dtype=np.float64)


@pytest.fixture
def mock_payload():
    return {
        "model": MockDummyRegressor(offset=1.5),
        "model_type": "mock_regressor",
        "feature_cols": [
            "hour",
            "day_of_week",
            "day_of_month",
            "month",
            "is_weekend",
            "lag_1",
            "lag_24",
            "lag_168",
            "rolling_mean_24",
        ],
        "target_col": "energy_kwh",
    }


@pytest.fixture
def synthetic_backtest_df():
    """Generates 300 contiguous hourly observations."""
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=300, freq="h")
    # Alternating diurnal energy pattern
    energy = 1.0 + 0.5 * np.sin(2 * np.pi * timestamps.hour / 24.0)
    return pd.DataFrame({"timestamp": timestamps, "energy_kwh": energy})


def test_backtest_synthetic_execution(tmp_path, mock_payload, synthetic_backtest_df):
    """Verify backtest runs across eligible origins and produces structured metrics."""
    out_json = tmp_path / "backtest_metrics.json"

    # Set start_origin after 168 hours of history: 2026-01-08 00:00:00
    start_origin = "2026-01-08 00:00:00"
    end_origin = "2026-01-12 00:00:00"

    report = run_recursive_backtest(
        model_or_path=mock_payload,
        data_or_path=synthetic_backtest_df,
        horizons=(1, 6, 24),
        origin_stride_hours=24,
        start_origin=start_origin,
        end_origin=end_origin,
        output_metrics_path=out_json,
    )

    assert out_json.exists()
    assert report["origin_accounting"]["eligible_origins"] > 0
    assert report["target_units"] == "kWh"
    assert "model_evaluated" in report
    assert "baseline_evaluated" in report
    assert len(report["limitations"]) > 0
    assert "horizon_1h" in report["horizon_metrics"]
    assert "horizon_6h" in report["horizon_metrics"]
    assert "horizon_24h" in report["horizon_metrics"]

    # Verify metrics are finite
    h1 = report["horizon_metrics"]["horizon_1h"]
    assert h1["unit"] == "kWh"
    assert np.isfinite(h1["random_forest"]["mae"])
    assert np.isfinite(h1["persistence"]["mae"])
    assert h1["evaluated_origins"] == report["origin_accounting"]["eligible_origins"]


def test_backtest_skips_gapped_origins(tmp_path, mock_payload, synthetic_backtest_df):
    """Verify backtest skips origins that have gaps in history or target windows."""
    out_json = tmp_path / "backtest_metrics_gapped.json"

    # Introduce a gap in history at row 100
    gapped_df = synthetic_backtest_df.drop(index=[100, 101]).reset_index(drop=True)

    report = run_recursive_backtest(
        model_or_path=mock_payload,
        data_or_path=gapped_df,
        horizons=(1, 6),
        origin_stride_hours=24,
        start_origin="2026-01-08 00:00:00",
        end_origin="2026-01-12 00:00:00",
        output_metrics_path=out_json,
    )

    # Some origins affected by the gap should be skipped
    accounting = report["origin_accounting"]
    assert accounting["skipped_origins_total"] > 0


def test_backtest_invalid_horizon(mock_payload, synthetic_backtest_df):
    with pytest.raises(ValueError, match="outside valid range"):
        run_recursive_backtest(
            model_or_path=mock_payload,
            data_or_path=synthetic_backtest_df,
            horizons=(0, 24),
        )
