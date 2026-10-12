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


# =========================================================================
# Phase 6C: Tests for Live Monitoring & Prediction Helpers
# =========================================================================

from src.dashboard_utils import (
    format_event_log_table,
    format_live_metrics_summary,
    format_measurement_kpi,
    format_prediction_kpi,
    prepare_live_monitoring_chart_data,
)
from src.live_engine import LiveEvaluationRecord, LivePredictionRecord
from src.live_source import EnergyMeasurement


def test_format_measurement_kpi_valid_and_gap():
    """Verify formatting of valid measurements, gaps, and uninitialized state."""
    # 1. Uninitialized None
    res_none = format_measurement_kpi(None)
    assert res_none["energy_str"] == "N/A"
    assert res_none["status_str"] == "Uninitialized"

    # 2. Valid measurement
    m_valid = EnergyMeasurement(
        timestamp=pd.Timestamp("2010-05-01 12:00:00"),
        energy_kwh=1.23456,
        is_valid=True,
        metadata={"coverage_ratio": 1.0},
    )
    res_valid = format_measurement_kpi(m_valid)
    assert "1.235 kWh" in res_valid["energy_str"]
    assert "Recorded" in res_valid["status_str"]
    assert res_valid["coverage_str"] == "100.0%"

    # 3. Gap measurement
    m_gap = EnergyMeasurement(
        timestamp=pd.Timestamp("2010-05-01 13:00:00"),
        energy_kwh=np.nan,
        is_valid=False,
    )
    res_gap = format_measurement_kpi(m_gap)
    assert "Missing" in res_gap["energy_str"]
    assert "No Zero Fill" in res_gap["status_str"]
    assert res_gap["coverage_str"] == "0%"


def test_format_prediction_kpi_success_and_skipped():
    """Verify formatting of successful predictions, skipped records, and None."""
    # 1. None
    res_none = format_prediction_kpi(None)
    assert res_none["predicted_str"] == "Awaiting simulation step..."

    # 2. Success
    p_success = LivePredictionRecord(
        origin_timestamp=pd.Timestamp("2010-05-01 12:00:00"),
        target_timestamp=pd.Timestamp("2010-05-01 13:00:00"),
        predicted_energy_kwh=2.4567,
        status="SUCCESS",
    )
    res_succ = format_prediction_kpi(p_success)
    assert "2.457 kWh" in res_succ["predicted_str"]
    assert "2010-05-01 13:00:00" in res_succ["target_str"]

    # 3. Skipped
    p_skip = LivePredictionRecord(
        origin_timestamp=pd.Timestamp("2010-05-01 12:00:00"),
        target_timestamp=pd.Timestamp("2010-05-01 13:00:00"),
        predicted_energy_kwh=None,
        status="SKIPPED_GAP_IN_HISTORY",
    )
    res_skip = format_prediction_kpi(p_skip)
    assert "Withheld" in res_skip["predicted_str"]
    assert res_skip["status_str"] == "SKIPPED_GAP_IN_HISTORY"


def test_format_live_metrics_summary():
    """Verify formatting of empty and populated evaluation metrics summaries."""
    # 1. Empty metrics
    empty_m = {
        "matched_pairs_count": 0,
        "is_statistically_sufficient": False,
        "mae_kwh": None,
        "rmse_kwh": None,
        "mean_bias_error_kwh": None,
    }
    res_empty = format_live_metrics_summary(empty_m)
    assert res_empty["mae_str"] == "N/A"
    assert res_empty["confidence_badge"] == "No Data"

    # 2. Populated metrics
    pop_m = {
        "matched_pairs_count": 6,
        "is_statistically_sufficient": True,
        "mae_kwh": 0.25,
        "rmse_kwh": 0.35,
        "mean_bias_error_kwh": 0.05,
        "note": "6 verified pairs.",
    }
    res_pop = format_live_metrics_summary(pop_m)
    assert "0.2500 kWh" in res_pop["mae_str"]
    assert "0.3500 kWh" in res_pop["rmse_str"]
    assert "+0.0500 kWh" in res_pop["mbe_str"]
    assert res_pop["confidence_badge"] == "Evaluated"


def test_prepare_live_monitoring_chart_data():
    """Verify live monitoring chart data includes actuals, evaluations, and forecast point."""
    ts = pd.date_range("2010-01-01 00:00:00", periods=5, freq="1h")
    # Hour 2 has NaN (gap)
    hist_df = pd.DataFrame(
        {"timestamp": ts, "energy_kwh": [1.0, 1.2, np.nan, 1.4, 1.5]}
    )

    eval_rec = LiveEvaluationRecord(
        origin_timestamp=ts[2],
        target_timestamp=ts[3],
        predicted_energy_kwh=1.35,
        actual_energy_kwh=1.4,
        absolute_error_kwh=0.05,
        signed_error_kwh=-0.05,
        squared_error_kwh2=0.0025,
        evaluated_at_timestamp=ts[3],
    )

    next_pred = LivePredictionRecord(
        origin_timestamp=ts[4],
        target_timestamp=ts[4] + pd.Timedelta(hours=1),
        predicted_energy_kwh=1.65,
        status="SUCCESS",
    )

    chart_df = prepare_live_monitoring_chart_data(
        history_df=hist_df,
        evaluation_history=[eval_rec],
        next_prediction=next_pred,
        lookback_hours=5,
    )

    assert len(chart_df) == 6  # 5 historical + 1 forecast point
    assert "Actual Observed (kWh)" in chart_df.columns
    assert "Past Model Prediction (kWh)" in chart_df.columns
    assert "Next-Hour Forecast (kWh)" in chart_df.columns

    # Verify gap row is preserved as NaN (NOT converted to 0.0)
    gap_row = chart_df.loc[chart_df["timestamp"] == ts[2]]
    assert np.isnan(gap_row["Actual Observed (kWh)"].iloc[0])

    # Verify past evaluation is overlaid at target timestamp ts[3]
    eval_row = chart_df.loc[chart_df["timestamp"] == ts[3]]
    assert eval_row["Past Model Prediction (kWh)"].iloc[0] == 1.35

    # Verify next prediction is populated at target timestamp
    target_row = chart_df.loc[chart_df["timestamp"] == (ts[4] + pd.Timedelta(hours=1))]
    assert target_row["Next-Hour Forecast (kWh)"].iloc[0] == 1.65


def test_format_event_log_table():
    """Verify event log table formats records with most recent first."""
    p1 = LivePredictionRecord(
        origin_timestamp=pd.Timestamp("2010-01-01 00:00:00"),
        target_timestamp=pd.Timestamp("2010-01-01 01:00:00"),
        predicted_energy_kwh=1.5,
        status="SUCCESS",
    )
    p2 = LivePredictionRecord(
        origin_timestamp=pd.Timestamp("2010-01-01 01:00:00"),
        target_timestamp=pd.Timestamp("2010-01-01 02:00:00"),
        predicted_energy_kwh=1.8,
        status="SUCCESS",
    )
    ev1 = LiveEvaluationRecord(
        origin_timestamp=pd.Timestamp("2010-01-01 00:00:00"),
        target_timestamp=pd.Timestamp("2010-01-01 01:00:00"),
        predicted_energy_kwh=1.5,
        actual_energy_kwh=1.4,
        absolute_error_kwh=0.1,
        signed_error_kwh=0.1,
        squared_error_kwh2=0.01,
        evaluated_at_timestamp=pd.Timestamp("2010-01-01 01:00:00"),
    )

    table = format_event_log_table([p1, p2], [ev1])
    assert len(table) == 2
    # Reverse order: p2 (most recent) is first row
    assert table.iloc[0]["Origin (T_origin)"] == "2010-01-01 01:00:00"
    assert table.iloc[0]["Actual (kWh)"] == "Pending"

    # p1 is second row and has evaluated actual
    assert table.iloc[1]["Origin (T_origin)"] == "2010-01-01 00:00:00"
    assert table.iloc[1]["Actual (kWh)"] == "1.400"
    assert table.iloc[1]["Abs Error (kWh)"] == "0.100"

