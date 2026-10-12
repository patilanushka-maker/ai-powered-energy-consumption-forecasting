"""Unit tests for dashboard utility functions."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from src.dashboard_utils import (
    compute_summary_statistics,
    filter_data_by_date_range,
    load_metrics_report,
    load_processed_data,
    prepare_forecast_plot_data,
)


def test_load_processed_data_success(tmp_path):
    csv_path = tmp_path / "hourly.csv"
    timestamps = pd.date_range("2026-01-01", periods=10, freq="h")
    pd.DataFrame({"timestamp": timestamps, "energy_kwh": 1.2}).to_csv(csv_path, index=False)

    df = load_processed_data(csv_path)
    assert len(df) == 10
    assert "timestamp" in df.columns
    assert "energy_kwh" in df.columns


def test_load_processed_data_missing_file():
    with pytest.raises(FileNotFoundError, match="Processed dataset not found"):
        load_processed_data("non_existent_hourly.csv")


def test_load_metrics_report_success(tmp_path):
    json_path = tmp_path / "metrics.json"
    data = {"metric": 0.35, "units": "kWh"}
    with open(json_path, "w") as f:
        json.dump(data, f)

    loaded = load_metrics_report(json_path)
    assert loaded["metric"] == 0.35


def test_load_metrics_report_missing_file():
    with pytest.raises(FileNotFoundError, match="Metrics report not found"):
        load_metrics_report("non_existent_metrics.json")


def test_filter_data_by_date_range():
    timestamps = pd.date_range("2026-01-01", periods=100, freq="h")
    df = pd.DataFrame({"timestamp": timestamps, "energy_kwh": 1.0})

    filtered = filter_data_by_date_range(df, "2026-01-02", "2026-01-03")
    assert len(filtered) > 0
    assert filtered["timestamp"].min() >= pd.Timestamp("2026-01-02 00:00:00")
    assert filtered["timestamp"].max() <= pd.Timestamp("2026-01-04 00:00:00")


def test_compute_summary_statistics():
    df = pd.DataFrame({"energy_kwh": [1.0, 2.0, 3.0, 4.0, 5.0]})
    stats = compute_summary_statistics(df)
    assert stats["count"] == 5.0
    assert stats["total_kwh"] == 15.0
    assert stats["mean_kwh"] == 3.0
    assert stats["median_kwh"] == 3.0
    assert stats["min_kwh"] == 1.0
    assert stats["max_kwh"] == 5.0


def test_prepare_forecast_plot_data():
    hist_ts = pd.date_range("2026-01-01 00:00:00", periods=50, freq="h")
    hist_df = pd.DataFrame({"timestamp": hist_ts, "energy_kwh": 2.0})

    fc_ts = pd.date_range("2026-01-03 02:00:00", periods=24, freq="h")
    fc_df = pd.DataFrame({"timestamp": fc_ts, "forecast_energy_kwh": 2.5, "step": range(1, 25)})

    combined = prepare_forecast_plot_data(hist_df, fc_df, lookback_hours=10)
    assert "actual_energy_kwh" in combined.columns
    assert "forecast_energy_kwh" in combined.columns
    assert len(combined) == 10 + 24
