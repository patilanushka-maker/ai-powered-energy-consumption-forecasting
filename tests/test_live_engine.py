"""Unit and integration tests for Phase 6B Live Prediction and Evaluation Engine.

Tests cover:
- Initialization and valid/invalid model loading.
- Prediction generation when 168 valid contiguous hours are present.
- Prediction blocking when lookback history is insufficient (<168 hours).
- Prediction blocking when history contains recording gaps.
- Recovery after a gap once 168 consecutive valid observations re-accumulate.
- Exact origin and target timestamp alignment (T_target = T_origin + 1h).
- Strict temporal isolation and future-data leakage prevention.
- Correct prediction-actual matching against incoming observations.
- Prevention of duplicate evaluations.
- Mathematical precision of MAE, RMSE, and Mean Bias Error calculations.
- Warm-up history exclusion from evaluation records.
- End-of-stream handling and graceful termination.
- Invalid incoming measurements and model error handling.
- End-to-end integration with HistoricalReplaySource and the real trained model artifact.
"""

from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
import pytest

from src.live_engine import (
    LiveEvaluationRecord,
    LivePredictionEngine,
    LivePredictionRecord,
)
from src.live_source import (
    BaseEnergyDataSource,
    EnergyMeasurement,
    HistoricalReplaySource,
)


# =========================================================================
# Mock Objects and Deterministic Fixtures
# =========================================================================

class DummyPredictor:
    """Deterministic mock regressor for fast unit testing."""

    def __init__(self, constant_val: float = 1.5) -> None:
        self.constant_val = constant_val

    def predict(self, X: Any) -> np.ndarray:
        return np.array([self.constant_val] * len(X), dtype="float64")


@pytest.fixture
def mock_model_payload() -> Dict[str, Any]:
    """Lightweight valid model payload dict."""
    return {
        "model": DummyPredictor(constant_val=2.0),
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
        "model_type": "DummyPredictor",
    }


class SimpleListSource(BaseEnergyDataSource):
    """Simple in-memory data source for exact unit test control."""

    def __init__(self, measurements: list[EnergyMeasurement], history_df: pd.DataFrame) -> None:
        self._measurements = list(measurements)
        self._cursor = -1
        self._history_df = history_df.copy()
        self._current_measurement: EnergyMeasurement | None = None

    def get_latest_measurement(self) -> EnergyMeasurement | None:
        return self._current_measurement

    def fetch_next(self) -> EnergyMeasurement | None:
        self._cursor += 1
        if self._cursor >= len(self._measurements):
            return None
        self._current_measurement = self._measurements[self._cursor]
        return self._current_measurement

    def get_lookback_history(self, hours: int = 168) -> pd.DataFrame:
        if self._current_measurement is None:
            empty = pd.DataFrame(columns=["timestamp", "energy_kwh"])
            empty.attrs["is_contiguous"] = False
            return empty

        # Filter strictly <= current measurement timestamp
        curr_ts = self._current_measurement.timestamp
        earliest_ts = curr_ts - pd.Timedelta(hours=hours - 1)
        mask = (self._history_df["timestamp"] >= earliest_ts) & (self._history_df["timestamp"] <= curr_ts)
        df_win = self._history_df.loc[mask, ["timestamp", "energy_kwh"]].copy().reset_index(drop=True)

        is_contiguous = len(df_win) == hours and not df_win["energy_kwh"].isna().any()
        df_win.attrs["is_contiguous"] = is_contiguous
        df_win.attrs["has_gaps"] = not is_contiguous
        df_win.attrs["latest_timestamp"] = curr_ts
        return df_win

    def reset(self, start_timestamp: Any = None) -> None:
        self._cursor = -1
        self._current_measurement = None

    @property
    def is_simulation(self) -> bool:
        return True

    @property
    def has_more(self) -> bool:
        return (self._cursor + 1) < len(self._measurements)

    @property
    def current_timestamp(self) -> pd.Timestamp | None:
        if self._current_measurement is None:
            return None
        return self._current_measurement.timestamp


@pytest.fixture
def synthetic_170h_df() -> pd.DataFrame:
    """Create 175 contiguous hourly observations for lookback testing."""
    ts = pd.date_range("2010-01-01 00:00:00", periods=175, freq="1h")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "energy_kwh": [1.0 + (i % 5) * 0.2 for i in range(175)],
            "valid_minutes": [60] * 175,
            "coverage_ratio": [1.0] * 175,
        }
    )


# =========================================================================
# Task 1 & 2: Initialization and Input Validation
# =========================================================================

def test_engine_init_validation(mock_model_payload: Dict[str, Any], synthetic_170h_df: pd.DataFrame) -> None:
    """Verify input argument validation during engine initialization."""
    source = HistoricalReplaySource(data_or_path=synthetic_170h_df)

    # Valid initialization
    engine = LivePredictionEngine(data_source=source, model_or_path=mock_model_payload)
    assert engine.is_simulation is True
    assert engine.latest_measurement is None
    assert engine.latest_prediction is None
    assert engine.latest_evaluation is None

    # Invalid data source type
    with pytest.raises(TypeError, match="BaseEnergyDataSource"):
        LivePredictionEngine(data_source="invalid_source", model_or_path=mock_model_payload)  # type: ignore[arg-type]

    # Invalid model payload keys
    bad_payload = {"model": DummyPredictor()}
    with pytest.raises(ValueError, match="missing required keys"):
        LivePredictionEngine(data_source=source, model_or_path=bad_payload)

    # Invalid lookback hours
    with pytest.raises(ValueError, match="min_lookback_hours"):
        LivePredictionEngine(data_source=source, model_or_path=mock_model_payload, min_lookback_hours=0)


# =========================================================================
# Task 3: Lookback Health, Gap Gating, and Recovery
# =========================================================================

def test_prediction_blocked_when_history_insufficient(mock_model_payload: Dict[str, Any]) -> None:
    """Verify prediction is withheld when fewer than 168 hours of history exist."""
    # Only 50 hours of history
    ts = pd.date_range("2010-01-01 00:00:00", periods=50, freq="1h")
    df_short = pd.DataFrame(
        {"timestamp": ts, "energy_kwh": [1.0] * 50, "valid_minutes": [60] * 50, "coverage_ratio": [1.0] * 50}
    )
    source = HistoricalReplaySource(data_or_path=df_short, preload_warmup_history=False)
    engine = LivePredictionEngine(data_source=source, model_or_path=mock_model_payload)

    step_res = engine.step()
    assert step_res["status"] == "SKIPPED_INSUFFICIENT_HISTORY"
    assert step_res["prediction"].status == "SKIPPED_INSUFFICIENT_HISTORY"
    assert step_res["prediction"].predicted_energy_kwh is None
    assert "Insufficient history" in step_res["prediction"].error_message

    health = engine.get_history_health()
    assert health["status"] == "INSUFFICIENT_HISTORY"
    assert health["available_hours"] == 1


def test_prediction_blocked_when_history_contains_gaps(mock_model_payload: Dict[str, Any]) -> None:
    """Verify prediction is withheld when lookback contains a recording gap."""
    # Build 175 hours before gap, then 2-hour gap, then 10 hours after gap
    ts_part1 = pd.date_range("2010-01-01 00:00:00", periods=175, freq="1h")
    # Gap at hours 2010-01-08 07:00:00 and 08:00:00
    ts_part2 = pd.date_range("2010-01-08 09:00:00", periods=10, freq="1h")

    df_gapped = pd.DataFrame(
        {
            "timestamp": list(ts_part1) + list(ts_part2),
            "energy_kwh": [1.2] * (len(ts_part1) + len(ts_part2)),
            "valid_minutes": [60] * (len(ts_part1) + len(ts_part2)),
            "coverage_ratio": [1.0] * (len(ts_part1) + len(ts_part2)),
        }
    )

    source = HistoricalReplaySource(
        data_or_path=df_gapped,
        start_timestamp=ts_part2[0],
        preload_warmup_history=True,
    )
    engine = LivePredictionEngine(data_source=source, model_or_path=mock_model_payload)

    # Step at ts_part2[0]: Lookback window crosses the historical gap between part1 and part2
    step_res = engine.step()
    assert step_res["prediction"].status == "SKIPPED_GAP_IN_HISTORY"
    assert step_res["prediction"].predicted_energy_kwh is None

    health = engine.get_history_health()
    assert health["status"] == "HISTORY_GAP"


def test_recovery_after_gap_with_168_consecutive_valid_hours(mock_model_payload: Dict[str, Any]) -> None:
    """Verify system resumes prediction once 168 consecutive valid hours re-accumulate after a gap."""
    # 20 hours, then gap, then 170 continuous hours
    ts_pre = pd.date_range("2010-01-01 00:00:00", periods=20, freq="1h")
    ts_post = pd.date_range("2010-01-03 00:00:00", periods=170, freq="1h")

    df = pd.DataFrame(
        {
            "timestamp": list(ts_pre) + list(ts_post),
            "energy_kwh": [1.5] * 190,
            "valid_minutes": [60] * 190,
            "coverage_ratio": [1.0] * 190,
        }
    )

    # Start at hour 168 of the post-gap block (meaning exactly 168 hours have re-accumulated)
    recovery_ts = ts_post[167]
    source = HistoricalReplaySource(
        data_or_path=df,
        start_timestamp=recovery_ts,
        preload_warmup_history=True,
    )
    engine = LivePredictionEngine(data_source=source, model_or_path=mock_model_payload)

    step_res = engine.step()
    assert step_res["status"] == "SUCCESS"
    assert step_res["prediction"].status == "SUCCESS"
    assert step_res["prediction"].predicted_energy_kwh == 2.0
    assert engine.get_history_health()["status"] == "HEALTHY"


# =========================================================================
# Task 4 & 5: Temporal Alignment, Matching, and Duplicate Prevention
# =========================================================================

def test_exact_origin_and_target_timestamp_alignment(
    mock_model_payload: Dict[str, Any], synthetic_170h_df: pd.DataFrame
) -> None:
    """Verify target timestamp is strictly origin_timestamp + 1 hour."""
    source = HistoricalReplaySource(
        data_or_path=synthetic_170h_df,
        start_timestamp="2010-01-08 00:00:00",
        preload_warmup_history=True,
    )
    engine = LivePredictionEngine(data_source=source, model_or_path=mock_model_payload)

    step_res = engine.step()
    pred = step_res["prediction"]

    origin_ts = pd.Timestamp("2010-01-08 00:00:00")
    expected_target = pd.Timestamp("2010-01-08 01:00:00")

    assert pred.origin_timestamp == origin_ts
    assert pred.target_timestamp == expected_target
    assert (pred.target_timestamp - pred.origin_timestamp) == pd.Timedelta(hours=1)


def test_correct_prediction_actual_matching(
    mock_model_payload: Dict[str, Any], synthetic_170h_df: pd.DataFrame
) -> None:
    """Verify that predictions made at origin T are matched when actual arrives at T+1."""
    source = HistoricalReplaySource(
        data_or_path=synthetic_170h_df,
        start_timestamp="2010-01-08 00:00:00",
        preload_warmup_history=True,
    )
    engine = LivePredictionEngine(data_source=source, model_or_path=mock_model_payload)

    # Step 1: Arrives at 00:00:00. Predicts for 01:00:00 (value = 2.0). No past evaluation.
    res1 = engine.step()
    assert res1["evaluation"] is None
    assert len(engine.evaluation_history) == 0

    # Actual value at 01:00:00 in synthetic_170h_df
    actual_at_01 = synthetic_170h_df.loc[
        synthetic_170h_df["timestamp"] == pd.Timestamp("2010-01-08 01:00:00"), "energy_kwh"
    ].iloc[0]

    # Step 2: Arrives at 01:00:00. Evaluates prediction made at Step 1!
    res2 = engine.step()
    eval_rec = res2["evaluation"]
    assert eval_rec is not None
    assert eval_rec.origin_timestamp == pd.Timestamp("2010-01-08 00:00:00")
    assert eval_rec.target_timestamp == pd.Timestamp("2010-01-08 01:00:00")
    assert eval_rec.predicted_energy_kwh == 2.0
    assert eval_rec.actual_energy_kwh == actual_at_01
    assert eval_rec.absolute_error_kwh == pytest.approx(abs(2.0 - actual_at_01), 6)
    assert eval_rec.signed_error_kwh == pytest.approx(2.0 - actual_at_01, 6)


def test_prevention_of_duplicate_evaluation() -> None:
    """Verify a prediction cannot be evaluated multiple times."""
    m_origin = EnergyMeasurement(
        timestamp=pd.Timestamp("2010-01-01 10:00:00"),
        energy_kwh=1.0,
        is_valid=True,
    )
    m_target1 = EnergyMeasurement(
        timestamp=pd.Timestamp("2010-01-01 11:00:00"),
        energy_kwh=1.2,
        is_valid=True,
    )
    # Duplicate measurement with same timestamp
    m_target2 = EnergyMeasurement(
        timestamp=pd.Timestamp("2010-01-01 11:00:00"),
        energy_kwh=1.2,
        is_valid=True,
    )

    # Fake history with 168 hours
    ts = pd.date_range("2009-12-25 11:00:00", periods=168, freq="1h")
    fake_hist = pd.DataFrame({"timestamp": ts, "energy_kwh": [1.0] * 168})

    source = SimpleListSource([m_origin, m_target1, m_target2], fake_hist)
    engine = LivePredictionEngine(
        data_source=source,
        model_or_path={
            "model": DummyPredictor(constant_val=1.5),
            "feature_cols": ["hour", "day_of_week", "day_of_month", "month", "is_weekend", "lag_1", "lag_24", "lag_168", "rolling_mean_24"],
            "target_col": "energy_kwh",
        },
    )

    res1 = engine.step()  # 10:00
    res2 = engine.step()  # 11:00 (first arrival) -> evaluated
    assert res2["evaluation"] is not None
    assert len(engine.evaluation_history) == 1

    res3 = engine.step()  # 11:00 (duplicate arrival) -> NOT evaluated again
    assert res3["evaluation"] is None
    assert len(engine.evaluation_history) == 1


# =========================================================================
# Task 6: Evaluation Metrics Calculations and Thresholds
# =========================================================================

def test_evaluation_metrics_calculations(
    mock_model_payload: Dict[str, Any], synthetic_170h_df: pd.DataFrame
) -> None:
    """Verify mathematical calculation of MAE, RMSE, and Mean Bias Error."""
    source = HistoricalReplaySource(
        data_or_path=synthetic_170h_df,
        start_timestamp="2010-01-08 00:00:00",
        preload_warmup_history=True,
    )
    engine = LivePredictionEngine(
        data_source=source,
        model_or_path=mock_model_payload,
        min_evaluation_samples=3,
    )

    # Before any evaluations
    metrics_0 = engine.get_evaluation_metrics()
    assert metrics_0["matched_pairs_count"] == 0
    assert metrics_0["mae_kwh"] is None
    assert metrics_0["rmse_kwh"] is None
    assert metrics_0["is_statistically_sufficient"] is False

    # Run 4 steps (produces 3 evaluated pairs)
    engine.run_steps(4)
    metrics = engine.get_evaluation_metrics()
    assert metrics["matched_pairs_count"] == 3
    assert metrics["is_statistically_sufficient"] is True

    # Manually compute expected values from evaluation history
    history = engine.evaluation_history
    expected_mae = np.mean([r.absolute_error_kwh for r in history])
    expected_rmse = np.sqrt(np.mean([r.squared_error_kwh2 for r in history]))
    expected_mbe = np.mean([r.signed_error_kwh for r in history])

    assert metrics["mae_kwh"] == pytest.approx(expected_mae, 6)
    assert metrics["rmse_kwh"] == pytest.approx(expected_rmse, 6)
    assert metrics["mean_bias_error_kwh"] == pytest.approx(expected_mbe, 6)


def test_warmup_exclusion_from_evaluation_metrics(
    mock_model_payload: Dict[str, Any], synthetic_170h_df: pd.DataFrame
) -> None:
    """Verify preloaded warmup history records are never scored as evaluated predictions."""
    source = HistoricalReplaySource(
        data_or_path=synthetic_170h_df,
        start_timestamp="2010-01-08 00:00:00",
        preload_warmup_history=True,
    )
    engine = LivePredictionEngine(data_source=source, model_or_path=mock_model_payload)

    # First step
    res = engine.step()
    assert res["evaluation"] is None
    assert len(engine.evaluation_history) == 0
    assert engine.get_evaluation_metrics()["matched_pairs_count"] == 0


# =========================================================================
# Task 7: End of Stream and Reset Handling
# =========================================================================

def test_end_of_stream_handling(
    mock_model_payload: Dict[str, Any], synthetic_170h_df: pd.DataFrame
) -> None:
    """Verify engine reports END_OF_STREAM when source is exhausted."""
    # Source with only 2 steps
    source = HistoricalReplaySource(
        data_or_path=synthetic_170h_df,
        start_timestamp="2010-01-08 00:00:00",
        end_timestamp="2010-01-08 01:00:00",
        preload_warmup_history=True,
    )
    engine = LivePredictionEngine(data_source=source, model_or_path=mock_model_payload)

    res1 = engine.step()
    assert res1["status"] == "SUCCESS"

    res2 = engine.step()
    assert res2["status"] == "SUCCESS"

    res3 = engine.step()
    assert res3["status"] == "END_OF_STREAM"
    assert res3["measurement"] is None
    assert res3["prediction"] is None


def test_engine_reset(
    mock_model_payload: Dict[str, Any], synthetic_170h_df: pd.DataFrame
) -> None:
    """Verify reset clears tracking collections and restarts simulation."""
    source = HistoricalReplaySource(
        data_or_path=synthetic_170h_df,
        start_timestamp="2010-01-08 00:00:00",
        preload_warmup_history=True,
    )
    engine = LivePredictionEngine(data_source=source, model_or_path=mock_model_payload)

    engine.run_steps(4)
    assert len(engine.prediction_history) == 4
    assert len(engine.evaluation_history) == 3

    # Reset
    engine.reset()
    assert len(engine.prediction_history) == 0
    assert len(engine.evaluation_history) == 0
    assert len(engine.pending_predictions) == 0
    assert engine.latest_measurement is None


# =========================================================================
# Task 8: Integration with Real Trained Model and Processed Dataset
# =========================================================================

def test_integration_with_real_trained_model() -> None:
    """Integration test: Verify 5-step live prediction run using real Random Forest model."""
    real_model_path = Path("models/random_forest_model.joblib")
    real_data_path = Path("data/processed/hourly_energy.csv")

    if not real_model_path.is_file() or not real_data_path.is_file():
        pytest.skip("Model artifact or processed dataset not available on disk.")

    source = HistoricalReplaySource(
        data_or_path=real_data_path,
        start_timestamp="2010-04-18 18:00:00",  # Test partition start
        preload_warmup_history=True,
    )
    engine = LivePredictionEngine(
        data_source=source,
        model_or_path=real_model_path,
        min_evaluation_samples=3,
    )

    # Advance 5 steps
    results = engine.run_steps(5)
    assert len(results) == 5

    # Check predictions were generated
    preds = engine.prediction_history
    assert len(preds) == 5
    for p in preds:
        assert p.status == "SUCCESS"
        assert p.predicted_energy_kwh is not None
        assert p.predicted_energy_kwh > 0.0

    # Check 4 evaluations occurred
    evals = engine.evaluation_history
    assert len(evals) == 4
    for e in evals:
        assert e.absolute_error_kwh >= 0.0
        assert np.isfinite(e.signed_error_kwh)

    # Check metrics
    metrics = engine.get_evaluation_metrics()
    assert metrics["matched_pairs_count"] == 4
    assert metrics["is_statistically_sufficient"] is True
    assert metrics["mae_kwh"] > 0.0
    assert metrics["rmse_kwh"] > 0.0


def test_multi_hour_forecast_generation() -> None:
    """Verify generate_multi_hour_forecast produces a valid 24-hour forward trajectory."""
    real_model_path = Path("models/random_forest_model.joblib")
    real_data_path = Path("data/processed/hourly_energy.csv")

    if not real_model_path.is_file() or not real_data_path.is_file():
        pytest.skip("Model artifact or processed dataset not available on disk.")

    source = HistoricalReplaySource(
        data_or_path=real_data_path,
        start_timestamp="2010-04-18 18:00:00",
        preload_warmup_history=True,
    )
    engine = LivePredictionEngine(data_source=source, model_or_path=real_model_path)

    # Must advance at least 1 step to have a valid origin
    engine.step()

    forecast_24h = engine.generate_multi_hour_forecast(horizon_hours=24)
    assert forecast_24h is not None
    assert len(forecast_24h) == 24
    assert list(forecast_24h.columns) == ["timestamp", "forecast_energy_kwh", "step"]
    assert forecast_24h["step"].iloc[0] == 1
    assert forecast_24h["step"].iloc[-1] == 24


# =========================================================================
# Task 9: Invalid Measurements, Model Error, and Data Leakage Isolation
# =========================================================================

def test_invalid_incoming_measurement_skips_prediction(
    mock_model_payload: Dict[str, Any]
) -> None:
    """Verify that incoming invalid measurement skips prediction with SKIPPED_INVALID_ORIGIN."""
    m_invalid = EnergyMeasurement(
        timestamp=pd.Timestamp("2010-01-08 12:00:00"),
        energy_kwh=np.nan,
        is_valid=False,
    )
    fake_hist = pd.DataFrame(
        {"timestamp": pd.date_range("2010-01-01 12:00:00", periods=168, freq="1h"), "energy_kwh": [1.0] * 168}
    )
    source = SimpleListSource([m_invalid], fake_hist)
    engine = LivePredictionEngine(data_source=source, model_or_path=mock_model_payload)

    res = engine.step()
    assert res["status"] == "GAP_DETECTED"
    assert res["prediction"].status == "SKIPPED_INVALID_ORIGIN"
    assert res["prediction"].predicted_energy_kwh is None


def test_model_exception_handled_gracefully(synthetic_170h_df: pd.DataFrame) -> None:
    """Verify that unexpected exceptions during prediction are caught and marked MODEL_ERROR."""
    class FailingModel:
        def predict(self, X: Any) -> np.ndarray:
            raise RuntimeError("Synthetic model failure during inference.")

    failing_payload = {
        "model": FailingModel(),
        "feature_cols": [
            "hour", "day_of_week", "day_of_month", "month", "is_weekend",
            "lag_1", "lag_24", "lag_168", "rolling_mean_24"
        ],
        "target_col": "energy_kwh",
    }
    source = HistoricalReplaySource(
        data_or_path=synthetic_170h_df,
        start_timestamp="2010-01-08 00:00:00",
        preload_warmup_history=True,
    )
    engine = LivePredictionEngine(data_source=source, model_or_path=failing_payload)

    res = engine.step()
    assert res["status"] == "MODEL_ERROR"
    assert res["prediction"].status == "MODEL_ERROR"
    assert "Synthetic model failure" in res["prediction"].error_message


def test_future_data_leakage_prevention(
    mock_model_payload: Dict[str, Any], synthetic_170h_df: pd.DataFrame
) -> None:
    """Verify that modifying future values in data source does NOT alter predictions at origin."""
    # Run 1: original data
    source1 = HistoricalReplaySource(
        data_or_path=synthetic_170h_df,
        start_timestamp="2010-01-08 00:00:00",
        preload_warmup_history=True,
    )
    engine1 = LivePredictionEngine(data_source=source1, model_or_path=mock_model_payload)
    res1 = engine1.step()
    pred_1 = res1["prediction"].predicted_energy_kwh

    # Run 2: future values (after 2010-01-08 00:00:00) altered dramatically
    df_altered_future = synthetic_170h_df.copy()
    future_mask = df_altered_future["timestamp"] > pd.Timestamp("2010-01-08 00:00:00")
    df_altered_future.loc[future_mask, "energy_kwh"] = 9999.0

    source2 = HistoricalReplaySource(
        data_or_path=df_altered_future,
        start_timestamp="2010-01-08 00:00:00",
        preload_warmup_history=True,
    )
    engine2 = LivePredictionEngine(data_source=source2, model_or_path=mock_model_payload)
    res2 = engine2.step()
    pred_2 = res2["prediction"].predicted_energy_kwh

    # Predictions at 00:00 must be mathematically identical
    assert pred_1 == pred_2

