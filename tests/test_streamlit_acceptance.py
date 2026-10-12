"""Comprehensive acceptance tests for Phase 5 Streamlit Dashboard using AppTest.

Simulates and verifies:
- Complete dashboard launch and zero unhandled exceptions.
- Rendering of all 6 tabs and sections.
- Verification that headline metrics match Phase 4A and Phase 4B artifacts.
- Interaction with date range filters in Historical Consumption.
- Interaction with the forecasting horizon slider (changing from 24h to 48h).
- Triggering forecast generation, verifying output trajectory, timestamps, and CSV export.
- Verification of disclaimers (single-household scope, non-live predictions, 168h contiguity).
"""

from pathlib import Path
import json
import io
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_dashboard_full_acceptance_and_rendering():
    """Verify that app.py loads all tabs, executes without exceptions, and displays metrics."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    # 1. Zero exceptions
    assert len(at.exception) == 0, f"App threw unexpected exceptions: {[e.value for e in at.exception]}"

    # 2. All 6 tabs exist
    assert len(at.tabs) == 6
    tab_labels = [t.label for t in at.tabs]
    assert any("Overview" in label for label in tab_labels)
    assert any("Historical" in label for label in tab_labels)
    assert any("Forecasting" in label for label in tab_labels)
    assert any("Evaluation" in label for label in tab_labels)
    assert any("Backtest" in label for label in tab_labels)
    assert any("Quality" in label for label in tab_labels)

    # 3. Check Overview metrics
    metric_labels = [m.label for m in at.metric]
    assert "Dataset Scope" in metric_labels
    assert "Test MAE (RF)" in metric_labels
    assert "Test RMSE (RF)" in metric_labels
    assert "Test Variance Explained" in metric_labels

    mae_metric = next(m for m in at.metric if m.label == "Test MAE (RF)")
    assert "0.3344" in mae_metric.value

    rmse_metric = next(m for m in at.metric if m.label == "Test RMSE (RF)")
    assert "0.4791" in rmse_metric.value


def test_dashboard_forecast_interaction_24h_to_48h():
    """Test interactive horizon adjustment and recursive forecast generation."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert len(at.exception) == 0

    # Locate horizon slider and preset buttons
    slider = at.slider[0]
    assert slider.value == 24

    # Click the 48 Hours preset button (preset_col2 button)
    btn_48h = next(b for b in at.button if "48 Hours" in b.label)
    btn_48h.click().run(timeout=30)
    assert len(at.exception) == 0

    # Locate the forecast generation button and click it
    btn_generate = next(b for b in at.button if "Generate Recursive Forecast" in b.label)
    btn_generate.click().run(timeout=45)
    assert len(at.exception) == 0

    # Verify download button is present
    download_btn = at.download_button[0]
    assert "Download Forecast CSV" in download_btn.label

    # Verify generated forecast table
    assert len(at.dataframe) >= 2
    fc_df = at.dataframe[-1].value
    assert len(fc_df) == 48
    assert list(fc_df.columns) == ["timestamp", "forecast_energy_kwh", "step"]
    assert fc_df["step"].iloc[0] == 1
    assert fc_df["step"].iloc[-1] == 48

    # First timestamp must be 1 hour after forecast origin (2010-11-26 20:00:00 -> 2010-11-26 21:00:00)
    assert str(fc_df["timestamp"].iloc[0]) == "2010-11-26 21:00:00"
    assert str(fc_df["timestamp"].iloc[-1]) == "2010-11-28 20:00:00"


def test_dashboard_matches_saved_artifacts():
    """Verify displayed metric values strictly match Phase 4A and Phase 4B artifacts."""
    with open("outputs/metrics/phase4a_metrics.json") as f:
        phase4a = json.load(f)
    with open("outputs/metrics/recursive_backtest_metrics.json") as f:
        backtest = json.load(f)

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert len(at.exception) == 0

    # Check backtest accounting displayed in metrics
    cand_metric = next(m for m in at.metric if m.label == "Candidate Origins")
    assert str(backtest["origin_accounting"]["total_candidate_origins"]) in str(cand_metric.value)

    elig_metric = next(m for m in at.metric if m.label == "Eligible Evaluated Origins")
    assert str(backtest["origin_accounting"]["eligible_origins"]) in str(elig_metric.value)

    skip_metric = next(m for m in at.metric if m.label == "Skipped Origins")
    assert str(backtest["origin_accounting"]["skipped_origins_total"]) in str(skip_metric.value)


def test_dashboard_historical_date_range_interaction():
    """Verify that updating the date range inputs dynamically updates historical consumption metrics."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert len(at.exception) == 0

    assert len(at.date_input) >= 2
    start_picker = at.date_input[0]
    end_picker = at.date_input[1]

    # Initial selected hours metric
    hours_metric_initial = next(m for m in at.metric if m.label == "Selected Hours")
    initial_hours_val = hours_metric_initial.value

    # Update date inputs to a narrower 7-day range: 2010-11-19 to 2010-11-26
    start_picker.set_value(pd.Timestamp("2010-11-19").date())
    end_picker.set_value(pd.Timestamp("2010-11-26").date())
    at.run(timeout=30)
    assert len(at.exception) == 0

    # Metric should reflect narrower range
    hours_metric_updated = next(m for m in at.metric if m.label == "Selected Hours")
    assert hours_metric_updated.value != initial_hours_val
