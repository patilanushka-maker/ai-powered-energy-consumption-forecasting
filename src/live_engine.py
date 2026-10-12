"""Live prediction and evaluation engine module (Phase 6B).

Responsibilities:
- Coordinate sequential intake of energy consumption measurements from data sources.
- Enforce strict temporal isolation and prevent future-data leakage.
- Validate contiguous 168-hour historical lookback prerequisites before inference.
- Generate next-hour predictions with exact origin and target timestamps.
- Match predictions against subsequently arriving actual measurements without lookahead.
- Prevent duplicate evaluation of the same prediction.
- Calculate mathematically rigorous point errors (absolute error, signed bias error)
  and cumulative metrics (MAE, RMSE, Mean Bias Error).
- Provide health monitoring for data continuity and history availability.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

from src.forecast import generate_forecast, load_forecasting_model
from src.live_source import BaseEnergyDataSource, EnergyMeasurement

logger = logging.getLogger(__name__)

DEFAULT_MIN_LOOKBACK_HOURS: int = 168
DEFAULT_MIN_EVALUATION_SAMPLES: int = 5


@dataclass(frozen=True)
class LivePredictionRecord:
    """Record of an individual model prediction attempt.

    Attributes:
        origin_timestamp: Timestamp when the prediction was generated (T_origin).
        target_timestamp: Timestamp being predicted (T_origin + 1h).
        predicted_energy_kwh: Predicted energy consumption in kWh (None/NaN if skipped).
        status: Status code ('SUCCESS', 'SKIPPED_INSUFFICIENT_HISTORY', 'SKIPPED_GAP_IN_HISTORY', etc.).
        error_message: Optional explanatory message if prediction failed or was skipped.
        step_number: Forecast horizon step number (1 for next-hour).
        metadata: Additional provenance attributes.
    """

    origin_timestamp: pd.Timestamp
    target_timestamp: pd.Timestamp
    predicted_energy_kwh: Optional[float]
    status: str
    error_message: Optional[str] = None
    step_number: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LiveEvaluationRecord:
    """Record of a verified prediction matched against its observed actual reading.

    Attributes:
        origin_timestamp: Timestamp when the prediction was originally generated.
        target_timestamp: Verification timestamp when actual measurement occurred.
        predicted_energy_kwh: Predicted consumption in kWh.
        actual_energy_kwh: Observed actual consumption in kWh.
        absolute_error_kwh: Absolute error |predicted - actual| in kWh.
        signed_error_kwh: Signed residual (predicted - actual) in kWh.
        squared_error_kwh2: Squared error (predicted - actual)^2 in kWh^2.
        evaluated_at_timestamp: Simulation clock timestamp when evaluation occurred.
    """

    origin_timestamp: pd.Timestamp
    target_timestamp: pd.Timestamp
    predicted_energy_kwh: float
    actual_energy_kwh: float
    absolute_error_kwh: float
    signed_error_kwh: float
    squared_error_kwh2: float
    evaluated_at_timestamp: pd.Timestamp


class LivePredictionEngine:
    """Live streaming prediction coordinator and real-time evaluation engine.

    Ensures strict temporal integrity:
    - Predictions strictly use observations at or before the forecast origin T_origin.
    - Forecast evaluations strictly occur after the target measurement arrives in simulation.
    - Gaps in historical data block forecasting until 168 contiguous valid hours re-accumulate.
    - Missing hours are never zero-filled or interpolated.
    """

    def __init__(
        self,
        data_source: BaseEnergyDataSource,
        model_or_path: Union[str, Path, Dict[str, Any]] = Path("models/random_forest_model.joblib"),
        min_lookback_hours: int = DEFAULT_MIN_LOOKBACK_HOURS,
        min_evaluation_samples: int = DEFAULT_MIN_EVALUATION_SAMPLES,
        auto_generate_prediction: bool = True,
    ) -> None:
        """Initialize live prediction engine.

        Args:
            data_source: Compatible energy data-source instance.
            model_or_path: Path to serialized model artifact or loaded payload dict.
            min_lookback_hours: Contiguous lookback hours required by model (default: 168).
            min_evaluation_samples: Minimum matched pairs to consider evaluation preliminary.
            auto_generate_prediction: Whether step() automatically triggers next-hour inference.

        Raises:
            TypeError: If data_source does not inherit from BaseEnergyDataSource.
            FileNotFoundError: If model file does not exist.
            ValueError: If model payload is invalid or min_lookback_hours < 1.
        """
        if not isinstance(data_source, BaseEnergyDataSource):
            raise TypeError(
                f"data_source must implement BaseEnergyDataSource, got {type(data_source).__name__}."
            )

        if min_lookback_hours < 1:
            raise ValueError(f"min_lookback_hours must be at least 1, got {min_lookback_hours}.")

        if min_evaluation_samples < 1:
            raise ValueError(f"min_evaluation_samples must be at least 1, got {min_evaluation_samples}.")

        self._data_source: BaseEnergyDataSource = data_source
        self._min_lookback_hours: int = min_lookback_hours
        self._min_evaluation_samples: int = min_evaluation_samples
        self._auto_generate_prediction: bool = auto_generate_prediction

        # Load and validate model artifact
        if isinstance(model_or_path, (str, Path)):
            self._model_payload: Dict[str, Any] = load_forecasting_model(model_or_path)
        elif isinstance(model_or_path, dict):
            required_keys = {"model", "feature_cols", "target_col"}
            if not required_keys.issubset(model_or_path.keys()):
                raise ValueError(
                    f"Model payload dict missing required keys: {required_keys - set(model_or_path.keys())}"
                )
            if not hasattr(model_or_path["model"], "predict"):
                raise ValueError("Model object in payload must implement 'predict' method.")
            self._model_payload = model_or_path
        else:
            raise ValueError(f"Invalid model_or_path: expected Path or dict, got {type(model_or_path).__name__}.")

        # Tracking state
        self._prediction_history: List[LivePredictionRecord] = []
        self._pending_predictions: Dict[pd.Timestamp, LivePredictionRecord] = {}
        self._evaluated_target_timestamps: Set[pd.Timestamp] = set()
        self._evaluation_history: List[LiveEvaluationRecord] = []

        self._latest_measurement: Optional[EnergyMeasurement] = None
        self._latest_prediction: Optional[LivePredictionRecord] = None
        self._latest_evaluation: Optional[LiveEvaluationRecord] = None

    def reset(self, start_timestamp: Optional[Union[str, pd.Timestamp]] = None) -> None:
        """Reset data-source cursor and clear all internal prediction and evaluation tracking.

        Args:
            start_timestamp: Optional start timestamp to re-initialize data source from.
        """
        self._data_source.reset(start_timestamp=start_timestamp)
        self._prediction_history = []
        self._pending_predictions = {}
        self._evaluated_target_timestamps = set()
        self._evaluation_history = []
        self._latest_measurement = None
        self._latest_prediction = None
        self._latest_evaluation = None

    def step(self) -> Dict[str, Any]:
        """Advance simulation by one measurement tick and process predictions and evaluations.

        Procedure:
        1. Fetches next measurement from data source.
        2. If measurement timestamp matches a pending prediction, evaluates prediction error.
        3. Checks historical lookback health.
        4. If healthy and auto_generate_prediction is True, generates next-hour prediction.

        Returns:
            Dict[str, Any]: Summary dictionary containing:
                - 'status': Step status ('SUCCESS', 'END_OF_STREAM', 'GAP_DETECTED', etc.)
                - 'measurement': Emitted EnergyMeasurement or None
                - 'prediction': Generated LivePredictionRecord or None
                - 'evaluation': Evaluated LiveEvaluationRecord or None
                - 'history_health': Health status dictionary
        """
        measurement = self._data_source.fetch_next()

        if measurement is None:
            return {
                "status": "END_OF_STREAM",
                "measurement": None,
                "prediction": None,
                "evaluation": None,
                "history_health": self.get_history_health(),
            }

        self._latest_measurement = measurement
        step_ts = measurement.timestamp

        # -----------------------------------------------------------------
        # 1. Prediction Matching and Evaluation against Incoming Actual
        # -----------------------------------------------------------------
        evaluation_record: Optional[LiveEvaluationRecord] = None

        if step_ts in self._pending_predictions:
            pending_pred = self._pending_predictions[step_ts]

            # Ensure prediction is never evaluated twice
            if step_ts not in self._evaluated_target_timestamps:
                if measurement.is_valid and np.isfinite(measurement.energy_kwh):
                    actual_val = float(measurement.energy_kwh)
                    pred_val = float(pending_pred.predicted_energy_kwh)  # type: ignore[arg-type]

                    abs_err = abs(pred_val - actual_val)
                    signed_err = pred_val - actual_val
                    sq_err = signed_err**2

                    evaluation_record = LiveEvaluationRecord(
                        origin_timestamp=pending_pred.origin_timestamp,
                        target_timestamp=step_ts,
                        predicted_energy_kwh=round(pred_val, 6),
                        actual_energy_kwh=round(actual_val, 6),
                        absolute_error_kwh=round(abs_err, 6),
                        signed_error_kwh=round(signed_err, 6),
                        squared_error_kwh2=round(sq_err, 8),
                        evaluated_at_timestamp=step_ts,
                    )
                    self._evaluation_history.append(evaluation_record)
                    self._evaluated_target_timestamps.add(step_ts)
                    self._latest_evaluation = evaluation_record
                else:
                    logger.warning(
                        "Prediction targeting %s could not be evaluated: incoming measurement is invalid/gap.",
                        step_ts,
                    )

            # Remove from pending queue regardless of validity to avoid leaks
            del self._pending_predictions[step_ts]

        # -----------------------------------------------------------------
        # 2. Historical Lookback Validation and Next-Hour Prediction
        # -----------------------------------------------------------------
        prediction_record: Optional[LivePredictionRecord] = None

        if self._auto_generate_prediction:
            prediction_record = self._try_generate_next_hour_prediction(measurement)
            self._latest_prediction = prediction_record

        # Determine step overall status
        step_status = "SUCCESS"
        if not measurement.is_valid:
            step_status = "GAP_DETECTED"
        elif prediction_record is not None and prediction_record.status != "SUCCESS":
            step_status = prediction_record.status

        return {
            "status": step_status,
            "measurement": measurement,
            "prediction": prediction_record,
            "evaluation": evaluation_record,
            "history_health": self.get_history_health(),
        }

    def _try_generate_next_hour_prediction(
        self, measurement: EnergyMeasurement
    ) -> LivePredictionRecord:
        """Attempt to generate next-hour prediction from current measurement origin."""
        origin_ts = measurement.timestamp
        target_ts = origin_ts + pd.Timedelta(hours=1)

        # Condition 1: Origin measurement itself must be valid
        if not measurement.is_valid or np.isnan(measurement.energy_kwh):  # type: ignore[arg-type]
            record = LivePredictionRecord(
                origin_timestamp=origin_ts,
                target_timestamp=target_ts,
                predicted_energy_kwh=None,
                status="SKIPPED_INVALID_ORIGIN",
                error_message="Incoming measurement at origin is missing or marked invalid.",
            )
            self._prediction_history.append(record)
            return record

        # Condition 2: Historical lookback must contain observations
        history_df = self._data_source.get_lookback_history(hours=self._min_lookback_hours)

        if len(history_df) == 0:
            record = LivePredictionRecord(
                origin_timestamp=origin_ts,
                target_timestamp=target_ts,
                predicted_energy_kwh=None,
                status="SKIPPED_INSUFFICIENT_HISTORY",
                error_message="No historical observations available.",
                metadata={"available_hours": 0, "required_hours": self._min_lookback_hours},
            )
            self._prediction_history.append(record)
            return record

        has_gaps_flag = bool(history_df.attrs.get("has_gaps", False))
        has_nans = bool(history_df["energy_kwh"].isna().any())

        earliest_required_ts = origin_ts - pd.Timedelta(hours=self._min_lookback_hours - 1)
        earliest_actual_ts = history_df["timestamp"].iloc[0]

        # If earliest actual timestamp is later than required start, not enough calendar time has elapsed
        if earliest_actual_ts > earliest_required_ts:
            record = LivePredictionRecord(
                origin_timestamp=origin_ts,
                target_timestamp=target_ts,
                predicted_energy_kwh=None,
                status="SKIPPED_INSUFFICIENT_HISTORY",
                error_message=(
                    f"Insufficient history: available from {earliest_actual_ts} to {origin_ts} "
                    f"({len(history_df)} hours), requires {self._min_lookback_hours} hours."
                ),
                metadata={"available_hours": len(history_df), "required_hours": self._min_lookback_hours},
            )
            self._prediction_history.append(record)
            return record

        # Condition 3: Lookback window covers calendar span but contains unrecorded gaps or NaNs
        if (
            len(history_df) < self._min_lookback_hours
            or has_gaps_flag
            or has_nans
            or not history_df.attrs.get("is_contiguous", False)
        ):
            record = LivePredictionRecord(
                origin_timestamp=origin_ts,
                target_timestamp=target_ts,
                predicted_energy_kwh=None,
                status="SKIPPED_GAP_IN_HISTORY",
                error_message=(
                    f"Lookback window ending at {origin_ts} contains unrecorded gaps or missing entries. "
                    "Predictions are withheld until 168 contiguous valid hours re-accumulate."
                ),
            )
            self._prediction_history.append(record)
            return record

        # Condition 4: Generate prediction using existing forecasting implementation
        try:
            forecast_df = generate_forecast(
                model_or_path=self._model_payload,
                history=history_df,
                horizon_hours=1,
            )
            pred_kwh = float(forecast_df["forecast_energy_kwh"].iloc[0])

            record = LivePredictionRecord(
                origin_timestamp=origin_ts,
                target_timestamp=target_ts,
                predicted_energy_kwh=pred_kwh,
                status="SUCCESS",
                step_number=1,
                metadata={
                    "model_type": self._model_payload.get("model_type", "RandomForestRegressor"),
                    "lookback_start": str(history_df["timestamp"].iloc[0]),
                    "lookback_end": str(origin_ts),
                },
            )
            self._prediction_history.append(record)
            self._pending_predictions[target_ts] = record
            return record

        except Exception as exc:
            logger.error("Failed to generate prediction at origin %s: %s", origin_ts, exc)
            record = LivePredictionRecord(
                origin_timestamp=origin_ts,
                target_timestamp=target_ts,
                predicted_energy_kwh=None,
                status="MODEL_ERROR",
                error_message=f"Forecasting engine exception: {exc}",
            )
            self._prediction_history.append(record)
            return record

    def run_steps(self, n: int) -> List[Dict[str, Any]]:
        """Advance simulation by n consecutive measurement ticks.

        Args:
            n: Number of steps to advance.

        Returns:
            List[Dict[str, Any]]: List of step result summaries.
        """
        results: List[Dict[str, Any]] = []
        for _ in range(n):
            step_res = self.step()
            results.append(step_res)
            if step_res["status"] == "END_OF_STREAM":
                break
        return results

    def generate_multi_hour_forecast(self, horizon_hours: int = 24) -> Optional[pd.DataFrame]:
        """Generate a multi-hour recursive forward forecast from current simulation origin.

        Args:
            horizon_hours: Horizon in hours (1 to 168, default: 24).

        Returns:
            Optional[pd.DataFrame]: Recursive forecast DataFrame or None if lookback is invalid.

        Raises:
            ValueError: If lookback window is not contiguous or has gaps.
        """
        if self._latest_measurement is None:
            raise ValueError("Simulation not started: no measurements available.")

        history_df = self._data_source.get_lookback_history(hours=self._min_lookback_hours)
        if len(history_df) < self._min_lookback_hours:
            raise ValueError(
                f"Cannot generate {horizon_hours}-hour forecast: only {len(history_df)} hours available "
                f"(requires {self._min_lookback_hours} hours)."
            )

        if not history_df.attrs.get("is_contiguous", False) or history_df["energy_kwh"].isna().any():
            raise ValueError(
                "Cannot generate forecast: lookback window contains historical recording gaps."
            )

        return generate_forecast(
            model_or_path=self._model_payload,
            history=history_df,
            horizon_hours=horizon_hours,
        )

    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """Calculate cumulative evaluation metrics across all matched prediction-actual pairs.

        Returns:
            Dict[str, Any]: Metrics dictionary containing:
                - 'matched_pairs_count': Number of verified pairs
                - 'min_samples_threshold': Minimum count for preliminary confidence
                - 'is_statistically_sufficient': True if count >= threshold
                - 'mae_kwh': Mean Absolute Error (None if count == 0)
                - 'rmse_kwh': Root Mean Squared Error (None if count == 0)
                - 'mean_bias_error_kwh': Mean Signed Error (None if count == 0)
                - 'note': Contextual statement regarding sample size validity
        """
        count = len(self._evaluation_history)

        if count == 0:
            return {
                "matched_pairs_count": 0,
                "min_samples_threshold": self._min_evaluation_samples,
                "is_statistically_sufficient": False,
                "mae_kwh": None,
                "rmse_kwh": None,
                "mean_bias_error_kwh": None,
                "note": "No matched prediction-actual pairs available yet.",
            }

        abs_errors = [rec.absolute_error_kwh for rec in self._evaluation_history]
        sq_errors = [rec.squared_error_kwh2 for rec in self._evaluation_history]
        signed_errors = [rec.signed_error_kwh for rec in self._evaluation_history]

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(sq_errors)))
        mbe = float(np.mean(signed_errors))

        is_sufficient = count >= self._min_evaluation_samples
        if is_sufficient:
            note = (
                f"Calculated across {count} verified pairs. Note: For robust operational validation, "
                "evaluation across multiple diurnal cycles and varying weather conditions is recommended."
            )
        else:
            note = (
                f"Preliminary sample: {count} pair(s). Minimum threshold is {self._min_evaluation_samples}. "
                "Metrics may exhibit high variance until sufficient hours accumulate."
            )

        return {
            "matched_pairs_count": count,
            "min_samples_threshold": self._min_evaluation_samples,
            "is_statistically_sufficient": is_sufficient,
            "mae_kwh": round(mae, 6),
            "rmse_kwh": round(rmse, 6),
            "mean_bias_error_kwh": round(mbe, 6),
            "note": note,
        }

    def get_history_health(self) -> Dict[str, Any]:
        """Audit current lookback history health and contiguity.

        Returns:
            Dict[str, Any]: Health status dictionary containing status, count, and message.
        """
        if self._latest_measurement is None:
            return {
                "status": "UNINITIALIZED",
                "available_hours": 0,
                "required_hours": self._min_lookback_hours,
                "is_contiguous": False,
                "latest_timestamp": None,
                "message": "Data source has not emitted any measurements.",
            }

        history_df = self._data_source.get_lookback_history(hours=self._min_lookback_hours)
        avail_count = len(history_df)
        is_contiguous = bool(history_df.attrs.get("is_contiguous", False))
        has_gaps = bool(history_df.attrs.get("has_gaps", False))

        earliest_required_ts = self._latest_measurement.timestamp - pd.Timedelta(hours=self._min_lookback_hours - 1)
        earliest_actual_ts = history_df["timestamp"].iloc[0] if len(history_df) > 0 else None

        if len(history_df) == 0 or earliest_actual_ts is None or earliest_actual_ts > earliest_required_ts:
            status = "INSUFFICIENT_HISTORY"
            msg = (
                f"Accumulating history: {avail_count}/{self._min_lookback_hours} hours. "
                "Waiting for 168 hours to compute lag_168."
            )
        elif has_gaps or not is_contiguous or avail_count < self._min_lookback_hours:
            status = "HISTORY_GAP"
            msg = "Lookback window contains missing hours or gaps. Forecasts are blocked."
        else:
            status = "HEALTHY"
            msg = f"Lookback window is healthy and contiguous ({self._min_lookback_hours} valid hours)."

        return {
            "status": status,
            "available_hours": avail_count,
            "required_hours": self._min_lookback_hours,
            "is_contiguous": is_contiguous,
            "latest_timestamp": self._latest_measurement.timestamp,
            "message": msg,
        }

    # ---------------------------------------------------------------------
    # Public Properties
    # ---------------------------------------------------------------------

    @property
    def latest_measurement(self) -> Optional[EnergyMeasurement]:
        """Return the most recently received measurement."""
        return self._latest_measurement

    @property
    def latest_prediction(self) -> Optional[LivePredictionRecord]:
        """Return the most recently generated prediction record."""
        return self._latest_prediction

    @property
    def latest_evaluation(self) -> Optional[LiveEvaluationRecord]:
        """Return the most recently evaluated prediction record."""
        return self._latest_evaluation

    @property
    def prediction_history(self) -> List[LivePredictionRecord]:
        """Return chronological history of all prediction attempts."""
        return list(self._prediction_history)

    @property
    def evaluation_history(self) -> List[LiveEvaluationRecord]:
        """Return chronological history of all verified prediction-actual pairs."""
        return list(self._evaluation_history)

    @property
    def pending_predictions(self) -> List[LivePredictionRecord]:
        """Return list of predictions currently awaiting actual observation verification."""
        return list(self._pending_predictions.values())

    @property
    def current_simulation_timestamp(self) -> Optional[pd.Timestamp]:
        """Return the current simulation clock timestamp."""
        if self._latest_measurement is None:
            return None
        return self._latest_measurement.timestamp

    @property
    def is_simulation(self) -> bool:
        """Indicate whether the underlying data source is a simulation."""
        return self._data_source.is_simulation
