"""Live energy data source interfaces and historical replay simulation (Phase 6A).

Responsibilities:
- Define standardized EnergyMeasurement dataclass for live consumption telemetry.
- Define BaseEnergyDataSource abstract interface for data providers.
- Implement HistoricalReplaySource for chronological simulation using processed historical data.
- Explicitly detect and flag recording gaps without zero-imputation or fabricated values.
- Provide reliable lookback-history access with strict lookahead protection.
- Maintain clear distinction between simulated historical replay and live telemetry.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_PATH: Path = Path("data/processed/hourly_energy.csv")
DEFAULT_LOOKBACK_HOURS: int = 168


@dataclass(frozen=True)
class EnergyMeasurement:
    """Standardized measurement payload emitted by data sources.

    Attributes:
        timestamp: Top-of-hour timestamp of the observation.
        energy_kwh: Energy consumption in kilowatt-hours. None or NaN if invalid/gap.
        is_valid: Boolean indicating whether reading is a valid recorded measurement.
        source_type: Identifier of the source ('historical_replay_simulation', 'smart_meter_api', etc.).
        metadata: Additional context dictionary (e.g., valid_minutes, coverage_ratio, is_gap).
    """

    timestamp: pd.Timestamp
    energy_kwh: Optional[float]
    is_valid: bool
    source_type: str = "historical_replay_simulation"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseEnergyDataSource(ABC):
    """Abstract contract for energy consumption telemetry providers."""

    @abstractmethod
    def get_latest_measurement(self) -> Optional[EnergyMeasurement]:
        """Return the most recently received measurement, or None if not started."""
        pass

    @abstractmethod
    def fetch_next(self) -> Optional[EnergyMeasurement]:
        """Advance and return the next measurement in chronological sequence."""
        pass

    @abstractmethod
    def get_lookback_history(self, hours: int = DEFAULT_LOOKBACK_HOURS) -> pd.DataFrame:
        """Return historical measurements up to the current timestamp.

        Args:
            hours: Lookback window length in hours (default: 168).

        Returns:
            pd.DataFrame: DataFrame containing ['timestamp', 'energy_kwh'] for up to
                         the requested hours ending at the latest measurement timestamp.
        """
        pass

    @abstractmethod
    def reset(self, start_timestamp: Optional[Union[str, pd.Timestamp]] = None) -> None:
        """Reset the source state to the beginning or specified timestamp."""
        pass

    @property
    @abstractmethod
    def is_simulation(self) -> bool:
        """Indicate whether the source represents a simulation or genuine live telemetry."""
        pass

    @property
    @abstractmethod
    def has_more(self) -> bool:
        """Return True if subsequent measurements remain to be fetched."""
        pass

    @property
    @abstractmethod
    def current_timestamp(self) -> Optional[pd.Timestamp]:
        """Return the timestamp of the latest emitted measurement, or None."""
        pass


class HistoricalReplaySource(BaseEnergyDataSource):
    """Historical replay simulation source for sequential streaming demonstration.

    Replays observations in forward chronological order from processed hourly datasets.
    Can operate in:
    1. 'hourly_clock' mode (default): Advances strictly 1 hour at a time. If an hour was
       missing or excluded during preprocessing, emits an explicit gap measurement with
       is_valid=False and energy_kwh=np.nan.
    2. 'records' mode: Advances row-by-row across recorded dataset entries, attaching
       gap metadata when timestamp deltas exceed 1 hour.
    """

    def __init__(
        self,
        data_or_path: Union[str, Path, pd.DataFrame] = DEFAULT_PROCESSED_PATH,
        start_timestamp: Optional[Union[str, pd.Timestamp]] = None,
        end_timestamp: Optional[Union[str, pd.Timestamp]] = None,
        preload_warmup_history: bool = True,
        step_mode: str = "hourly_clock",
    ) -> None:
        """Initialize historical replay source.

        Args:
            data_or_path: Path to processed CSV or pre-loaded DataFrame.
            start_timestamp: Earliest timestamp to begin replay. Defaults to earliest
                             recorded timestamp in the dataset.
            end_timestamp: Latest timestamp to conclude replay. Defaults to latest
                           recorded timestamp in the dataset.
            preload_warmup_history: If True and start_timestamp is specified, historical
                                    records prior to start_timestamp are preloaded into
                                    the history buffer to support immediate 168-hour lookback.
            step_mode: 'hourly_clock' (default) or 'records'.

        Raises:
            FileNotFoundError: If data file does not exist.
            ValueError: If dataset is empty, missing required columns, or invalid arguments.
        """
        if step_mode not in ("hourly_clock", "records"):
            raise ValueError(f"Invalid step_mode '{step_mode}'. Expected 'hourly_clock' or 'records'.")

        self._step_mode: str = step_mode
        self._preload_warmup_history: bool = preload_warmup_history

        # 1. Load and validate underlying dataset
        self._df: pd.DataFrame = self._load_dataset(data_or_path)

        # 2. Build fast indexed lookup: timestamp -> dict of row attributes
        self._records_by_timestamp: Dict[pd.Timestamp, Dict[str, Any]] = {}
        for row in self._df.itertuples(index=False):
            ts = getattr(row, "timestamp")
            self._records_by_timestamp[ts] = {
                "energy_kwh": getattr(row, "energy_kwh"),
                "valid_minutes": getattr(row, "valid_minutes", 60),
                "coverage_ratio": getattr(row, "coverage_ratio", 1.0),
            }

        self._dataset_min_ts: pd.Timestamp = self._df["timestamp"].iloc[0]
        self._dataset_max_ts: pd.Timestamp = self._df["timestamp"].iloc[-1]

        # 3. Configure boundaries
        self._initial_start_timestamp: Optional[Union[str, pd.Timestamp]] = start_timestamp
        self._initial_end_timestamp: Optional[Union[str, pd.Timestamp]] = end_timestamp

        # 4. State tracking attributes
        self._start_ts: pd.Timestamp = self._dataset_min_ts
        self._end_ts: pd.Timestamp = self._dataset_max_ts
        self._current_ts: Optional[pd.Timestamp] = None
        self._latest_measurement: Optional[EnergyMeasurement] = None
        self._history_records: List[Dict[str, Any]] = []

        # In 'records' mode, track row index
        self._record_indices: List[pd.Timestamp] = []
        self._record_cursor: int = -1

        # Initialize / reset state
        self.reset(start_timestamp=start_timestamp)

    def _load_dataset(self, data_or_path: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Load, validate, and sort processed hourly consumption dataset."""
        if isinstance(data_or_path, (str, Path)):
            path = Path(data_or_path)
            if not path.is_file():
                raise FileNotFoundError(f"Processed dataset file not found at: '{path.resolve()}'.")
            df = pd.read_csv(path)
        elif isinstance(data_or_path, pd.DataFrame):
            df = data_or_path.copy()
        else:
            raise ValueError(f"Invalid data_or_path: expected DataFrame or path, got {type(data_or_path).__name__}.")

        if len(df) == 0:
            raise ValueError("Processed dataset is empty (0 rows).")

        ts_col = "timestamp" if "timestamp" in df.columns else ("datetime" if "datetime" in df.columns else None)
        if ts_col is None:
            raise ValueError(f"Processed dataset missing 'timestamp' column. Found: {list(df.columns)}")

        if "energy_kwh" not in df.columns:
            raise ValueError(f"Processed dataset missing 'energy_kwh' column. Found: {list(df.columns)}")

        df["timestamp"] = pd.to_datetime(df[ts_col])
        if df["timestamp"].isna().any():
            raise ValueError("Processed dataset contains null or invalid timestamps.")

        # Ensure sorted and reset index
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        # Check hourly alignment
        non_hourly = (df["timestamp"].dt.minute != 0) | (df["timestamp"].dt.second != 0)
        if non_hourly.any():
            bad_sample = df.loc[non_hourly, "timestamp"].iloc[0]
            raise ValueError(f"Timestamps must align to top-of-hour boundaries. Found non-hourly: {bad_sample}")

        return df

    def reset(self, start_timestamp: Optional[Union[str, pd.Timestamp]] = None) -> None:
        """Reset the replay cursor to the beginning or specified start timestamp.

        Args:
            start_timestamp: Optional timestamp to restart from. If None, uses initial configuration.
        """
        chosen_start = start_timestamp if start_timestamp is not None else self._initial_start_timestamp

        if chosen_start is not None:
            parsed_start = pd.to_datetime(chosen_start)
            if parsed_start.minute != 0 or parsed_start.second != 0:
                raise ValueError(f"start_timestamp must align to top-of-hour. Got: {parsed_start}")
            if parsed_start < self._dataset_min_ts or parsed_start > self._dataset_max_ts:
                raise ValueError(
                    f"start_timestamp {parsed_start} is outside dataset range "
                    f"[{self._dataset_min_ts}, self._dataset_max_ts]."
                )
            self._start_ts = parsed_start
        else:
            self._start_ts = self._dataset_min_ts

        if self._initial_end_timestamp is not None:
            parsed_end = pd.to_datetime(self._initial_end_timestamp)
            if parsed_end < self._start_ts:
                raise ValueError(f"end_timestamp {parsed_end} cannot be earlier than start_timestamp {self._start_ts}.")
            self._end_ts = min(parsed_end, self._dataset_max_ts)
        else:
            self._end_ts = self._dataset_max_ts

        self._current_ts = None
        self._latest_measurement = None
        self._history_records = []

        # Preload warmup history prior to start_timestamp if requested
        if self._preload_warmup_history:
            warmup_mask = self._df["timestamp"] < self._start_ts
            warmup_slice = self._df.loc[warmup_mask, ["timestamp", "energy_kwh"]]
            for row in warmup_slice.itertuples(index=False):
                self._history_records.append(
                    {
                        "timestamp": row.timestamp,
                        "energy_kwh": float(row.energy_kwh),
                        "is_valid": True,
                    }
                )

        if self._step_mode == "records":
            mask = (self._df["timestamp"] >= self._start_ts) & (self._df["timestamp"] <= self._end_ts)
            self._record_indices = self._df.loc[mask, "timestamp"].tolist()
            self._record_cursor = -1

    def fetch_next(self) -> Optional[EnergyMeasurement]:
        """Advance and return the next measurement in chronological sequence.

        Returns:
            Optional[EnergyMeasurement]: Next measurement payload, or None if stream is exhausted.
        """
        if not self.has_more:
            return None

        if self._step_mode == "hourly_clock":
            return self._fetch_next_hourly_clock()
        else:
            return self._fetch_next_record()

    def _fetch_next_hourly_clock(self) -> Optional[EnergyMeasurement]:
        """Advance by exactly 1 hour on the chronological clock."""
        if self._current_ts is None:
            next_ts = self._start_ts
        else:
            next_ts = self._current_ts + pd.Timedelta(hours=1)

        if next_ts > self._end_ts:
            return None

        self._current_ts = next_ts

        # Check if measurement exists in processed dataset
        if next_ts in self._records_by_timestamp:
            row_data = self._records_by_timestamp[next_ts]
            val = float(row_data["energy_kwh"])
            measurement = EnergyMeasurement(
                timestamp=next_ts,
                energy_kwh=val,
                is_valid=True,
                source_type="historical_replay_simulation",
                metadata={
                    "is_gap": False,
                    "valid_minutes": int(row_data.get("valid_minutes", 60)),
                    "coverage_ratio": float(row_data.get("coverage_ratio", 1.0)),
                    "dataset_source": "UCI Household Power Consumption (2006-2010)",
                },
            )
            self._history_records.append(
                {
                    "timestamp": next_ts,
                    "energy_kwh": val,
                    "is_valid": True,
                }
            )
        else:
            # Explicit gap detection: unobserved or excluded hour
            measurement = EnergyMeasurement(
                timestamp=next_ts,
                energy_kwh=np.nan,
                is_valid=False,
                source_type="historical_replay_simulation",
                metadata={
                    "is_gap": True,
                    "reason": "missing_or_excluded_interval",
                    "note": "Unrecorded interval in historical dataset; never substituted with zero.",
                    "dataset_source": "UCI Household Power Consumption (2006-2010)",
                },
            )
            self._history_records.append(
                {
                    "timestamp": next_ts,
                    "energy_kwh": np.nan,
                    "is_valid": False,
                }
            )

        self._latest_measurement = measurement
        return measurement

    def _fetch_next_record(self) -> Optional[EnergyMeasurement]:
        """Advance to the next recorded entry in the dataset."""
        self._record_cursor += 1
        if self._record_cursor >= len(self._record_indices):
            return None

        next_ts = self._record_indices[self._record_cursor]
        prev_ts = self._current_ts
        self._current_ts = next_ts

        row_data = self._records_by_timestamp[next_ts]
        val = float(row_data["energy_kwh"])

        metadata: Dict[str, Any] = {
            "valid_minutes": int(row_data.get("valid_minutes", 60)),
            "coverage_ratio": float(row_data.get("coverage_ratio", 1.0)),
            "dataset_source": "UCI Household Power Consumption (2006-2010)",
        }

        # Check if there was a time gap between successive recorded rows
        if prev_ts is not None:
            hours_diff = int((next_ts - prev_ts) / pd.Timedelta(hours=1))
            if hours_diff > 1:
                metadata["gap_detected_before"] = True
                metadata["skipped_gap_hours"] = hours_diff - 1

        measurement = EnergyMeasurement(
            timestamp=next_ts,
            energy_kwh=val,
            is_valid=True,
            source_type="historical_replay_simulation",
            metadata=metadata,
        )

        self._history_records.append(
            {
                "timestamp": next_ts,
                "energy_kwh": val,
                "is_valid": True,
            }
        )

        self._latest_measurement = measurement
        return measurement

    def get_latest_measurement(self) -> Optional[EnergyMeasurement]:
        """Return the most recently received measurement, or None if stream not started."""
        return self._latest_measurement

    def get_lookback_history(self, hours: int = DEFAULT_LOOKBACK_HOURS) -> pd.DataFrame:
        """Return historical observations ending at the current measurement timestamp.

        Enforces strict temporal boundary: observations with timestamp > current_timestamp
        are never included, preventing lookahead leakage.

        Args:
            hours: Lookback window length in hours (default: 168).

        Returns:
            pd.DataFrame: DataFrame containing ['timestamp', 'energy_kwh'].
                         Includes attributes:
                         - df.attrs['is_contiguous']: True if window has zero gaps and exactly requested hours.
                         - df.attrs['has_gaps']: True if missing hours or NaNs exist in the window.
                         - df.attrs['latest_timestamp']: Timestamp of the latest observation.
        """
        if self._current_ts is None or len(self._history_records) == 0:
            empty_df = pd.DataFrame(columns=["timestamp", "energy_kwh"])
            empty_df.attrs["is_contiguous"] = False
            empty_df.attrs["has_gaps"] = False
            empty_df.attrs["latest_timestamp"] = None
            return empty_df

        # Filter history records strictly <= current_ts
        earliest_window_ts = self._current_ts - pd.Timedelta(hours=hours - 1)

        window_records = [
            r for r in self._history_records if earliest_window_ts <= r["timestamp"] <= self._current_ts
        ]

        if len(window_records) == 0:
            empty_df = pd.DataFrame(columns=["timestamp", "energy_kwh"])
            empty_df.attrs["is_contiguous"] = False
            empty_df.attrs["has_gaps"] = False
            empty_df.attrs["latest_timestamp"] = self._current_ts
            return empty_df

        df_out = pd.DataFrame(window_records)[["timestamp", "energy_kwh"]]
        df_out = df_out.sort_values("timestamp").reset_index(drop=True)

        # Check contiguity and gap presence
        has_nans = df_out["energy_kwh"].isna().any()
        is_exact_count = len(df_out) == hours

        # Verify consecutive 1-hour step spacing
        if len(df_out) > 1:
            diffs = df_out["timestamp"].diff().iloc[1:]
            is_step_valid = (diffs == pd.Timedelta(hours=1)).all()
        else:
            is_step_valid = hours == 1

        is_contiguous = is_exact_count and is_step_valid and (not has_nans)

        df_out.attrs["is_contiguous"] = bool(is_contiguous)
        df_out.attrs["has_gaps"] = bool(has_nans or (not is_step_valid) or (len(df_out) < hours))
        df_out.attrs["latest_timestamp"] = self._current_ts
        df_out.attrs["lookback_hours_requested"] = hours
        df_out.attrs["available_hours_count"] = len(df_out)

        return df_out

    @property
    def is_simulation(self) -> bool:
        """Indicate that this data source is a historical simulation."""
        return True

    @property
    def has_more(self) -> bool:
        """Return True if more measurements remain in the configured range."""
        if self._step_mode == "hourly_clock":
            if self._current_ts is None:
                return self._start_ts <= self._end_ts
            return (self._current_ts + pd.Timedelta(hours=1)) <= self._end_ts
        else:
            return (self._record_cursor + 1) < len(self._record_indices)

    @property
    def current_timestamp(self) -> Optional[pd.Timestamp]:
        """Return the timestamp of the latest emitted measurement, or None."""
        return self._current_ts
