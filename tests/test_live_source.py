"""Unit and integration tests for Phase 6A live data-source module.

Tests cover:
- EnergyMeasurement dataclass immutability and default fields.
- BaseEnergyDataSource abstract interface contracts.
- HistoricalReplaySource initialization, validation, and boundary conditions.
- Strict forward chronological ordering.
- Reset functionality and deterministic replay.
- Explicit gap detection without zero-imputation or fabricated values.
- Lookback history extraction, contiguity gating, and gap detection.
- Prevention of future-data leakage through the source interface.
- Stream exhaustion and boundary behavior.
- Warmup history preloading.
- Alternative 'records' step mode.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.live_source import (
    BaseEnergyDataSource,
    EnergyMeasurement,
    HistoricalReplaySource,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def synthetic_hourly_df() -> pd.DataFrame:
    """Create a small clean synthetic hourly DataFrame with 10 contiguous hours."""
    timestamps = pd.date_range("2010-01-01 00:00:00", periods=10, freq="1h")
    data = {
        "timestamp": timestamps,
        "energy_kwh": [1.0, 1.2, 0.8, 1.5, 2.0, 1.8, 1.1, 0.9, 1.3, 1.6],
        "valid_minutes": [60] * 10,
        "coverage_ratio": [1.0] * 10,
    }
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_df_with_gap() -> pd.DataFrame:
    """Create a synthetic DataFrame with an intentional 2-hour recording gap."""
    # Hours 00:00, 01:00, gap at 02:00 and 03:00, then 04:00, 05:00
    timestamps = [
        pd.Timestamp("2010-01-01 00:00:00"),
        pd.Timestamp("2010-01-01 01:00:00"),
        pd.Timestamp("2010-01-01 04:00:00"),
        pd.Timestamp("2010-01-01 05:00:00"),
    ]
    data = {
        "timestamp": timestamps,
        "energy_kwh": [1.0, 1.5, 2.2, 1.8],
        "valid_minutes": [60, 60, 60, 60],
        "coverage_ratio": [1.0, 1.0, 1.0, 1.0],
    }
    return pd.DataFrame(data)


# =========================================================================
# Task 1 & 2: Dataclass and Abstract Interface Tests
# =========================================================================

def test_energy_measurement_dataclass() -> None:
    """Verify EnergyMeasurement fields, defaults, and immutability."""
    ts = pd.Timestamp("2010-05-01 12:00:00")
    m = EnergyMeasurement(
        timestamp=ts,
        energy_kwh=1.45,
        is_valid=True,
        source_type="historical_replay_simulation",
        metadata={"valid_minutes": 60},
    )

    assert m.timestamp == ts
    assert m.energy_kwh == 1.45
    assert m.is_valid is True
    assert m.source_type == "historical_replay_simulation"
    assert m.metadata["valid_minutes"] == 60

    # Immutability check
    with pytest.raises(Exception):
        m.energy_kwh = 2.0  # type: ignore[misc]


def test_base_energy_data_source_is_abstract() -> None:
    """Verify BaseEnergyDataSource cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseEnergyDataSource()  # type: ignore[abstract]


# =========================================================================
# Initialization and Validation Tests
# =========================================================================

def test_historical_replay_source_init_synthetic(synthetic_hourly_df: pd.DataFrame) -> None:
    """Verify initialization with synthetic DataFrame and default state."""
    source = HistoricalReplaySource(
        data_or_path=synthetic_hourly_df,
        step_mode="hourly_clock",
    )
    assert source.is_simulation is True
    assert source.has_more is True
    assert source.current_timestamp is None
    assert source.get_latest_measurement() is None


def test_historical_replay_source_invalid_inputs(synthetic_hourly_df: pd.DataFrame) -> None:
    """Verify proper exceptions on invalid inputs."""
    # Non-existent path
    with pytest.raises(FileNotFoundError):
        HistoricalReplaySource(data_or_path="non_existent_file.csv")

    # Empty DataFrame
    with pytest.raises(ValueError, match="empty"):
        HistoricalReplaySource(data_or_path=pd.DataFrame())

    # Missing required column
    bad_df = synthetic_hourly_df.drop(columns=["energy_kwh"])
    with pytest.raises(ValueError, match="energy_kwh"):
        HistoricalReplaySource(data_or_path=bad_df)

    # Invalid step mode
    with pytest.raises(ValueError, match="step_mode"):
        HistoricalReplaySource(data_or_path=synthetic_hourly_df, step_mode="invalid_mode")

    # Start timestamp out of bounds
    with pytest.raises(ValueError, match="outside dataset range"):
        HistoricalReplaySource(
            data_or_path=synthetic_hourly_df,
            start_timestamp="2025-01-01 00:00:00",
        )

    # Non-hourly start timestamp
    with pytest.raises(ValueError, match="top-of-hour"):
        HistoricalReplaySource(
            data_or_path=synthetic_hourly_df,
            start_timestamp="2010-01-01 00:15:00",
        )

    # End timestamp earlier than start timestamp
    with pytest.raises(ValueError, match="cannot be earlier"):
        HistoricalReplaySource(
            data_or_path=synthetic_hourly_df,
            start_timestamp="2010-01-01 05:00:00",
            end_timestamp="2010-01-01 02:00:00",
        )


def test_historical_replay_source_non_hourly_dataset() -> None:
    """Verify dataset with non-top-of-hour timestamps is rejected."""
    bad_df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2010-01-01 00:15:00")],
            "energy_kwh": [1.0],
        }
    )
    with pytest.raises(ValueError, match="top-of-hour"):
        HistoricalReplaySource(data_or_path=bad_df)


# =========================================================================
# Chronological Ordering and Progression Tests
# =========================================================================

def test_chronological_ordering(synthetic_hourly_df: pd.DataFrame) -> None:
    """Verify successive calls to fetch_next advance monotonically by 1 hour."""
    source = HistoricalReplaySource(data_or_path=synthetic_hourly_df)

    emitted_timestamps = []
    for _ in range(5):
        m = source.fetch_next()
        assert m is not None
        assert m.is_valid is True
        assert source.current_timestamp == m.timestamp
        assert source.get_latest_measurement() == m
        emitted_timestamps.append(m.timestamp)

    # Verify monotonic 1-hour progression
    expected = list(synthetic_hourly_df["timestamp"].iloc[:5])
    assert emitted_timestamps == expected

    for i in range(1, len(emitted_timestamps)):
        assert (emitted_timestamps[i] - emitted_timestamps[i - 1]) == pd.Timedelta(hours=1)


def test_reset_functionality(synthetic_hourly_df: pd.DataFrame) -> None:
    """Verify reset restores the source to initial state and enables deterministic replay."""
    source = HistoricalReplaySource(data_or_path=synthetic_hourly_df)

    # Advance 4 steps
    first_run = [source.fetch_next() for _ in range(4)]
    assert source.current_timestamp == pd.Timestamp("2010-01-01 03:00:00")

    # Reset
    source.reset()
    assert source.current_timestamp is None
    assert source.get_latest_measurement() is None

    # Replay and verify deterministic identity
    second_run = [source.fetch_next() for _ in range(4)]
    assert [m.timestamp for m in first_run] == [m.timestamp for m in second_run]
    assert [m.energy_kwh for m in first_run] == [m.energy_kwh for m in second_run]

    # Reset to a specific new start timestamp
    source.reset(start_timestamp="2010-01-01 05:00:00")
    m_new = source.fetch_next()
    assert m_new is not None
    assert m_new.timestamp == pd.Timestamp("2010-01-01 05:00:00")


# =========================================================================
# Explicit Gap Detection Tests (No Zero Fabrication)
# =========================================================================

def test_gap_detection_explicit_and_no_zero_fabrication(synthetic_df_with_gap: pd.DataFrame) -> None:
    """Verify unrecorded hours are explicitly flagged as invalid gaps without zero-filling."""
    source = HistoricalReplaySource(
        data_or_path=synthetic_df_with_gap,
        step_mode="hourly_clock",
        preload_warmup_history=False,
    )

    # Step 1: 00:00 (recorded)
    m0 = source.fetch_next()
    assert m0 is not None
    assert m0.timestamp == pd.Timestamp("2010-01-01 00:00:00")
    assert m0.is_valid is True
    assert m0.energy_kwh == 1.0
    assert m0.metadata["is_gap"] is False

    # Step 2: 01:00 (recorded)
    m1 = source.fetch_next()
    assert m1 is not None
    assert m1.timestamp == pd.Timestamp("2010-01-01 01:00:00")
    assert m1.is_valid is True
    assert m1.energy_kwh == 1.5

    # Step 3: 02:00 (GAP - not recorded in dataset)
    m2 = source.fetch_next()
    assert m2 is not None
    assert m2.timestamp == pd.Timestamp("2010-01-01 02:00:00")
    assert m2.is_valid is False
    assert np.isnan(m2.energy_kwh)  # Strictly NaN, NEVER 0.0!
    assert m2.energy_kwh != 0.0
    assert m2.metadata["is_gap"] is True
    assert m2.metadata["reason"] == "missing_or_excluded_interval"

    # Step 4: 03:00 (GAP - second missing hour)
    m3 = source.fetch_next()
    assert m3 is not None
    assert m3.timestamp == pd.Timestamp("2010-01-01 03:00:00")
    assert m3.is_valid is False
    assert np.isnan(m3.energy_kwh)
    assert m3.metadata["is_gap"] is True

    # Step 5: 04:00 (recorded again)
    m4 = source.fetch_next()
    assert m4 is not None
    assert m4.timestamp == pd.Timestamp("2010-01-01 04:00:00")
    assert m4.is_valid is True
    assert m4.energy_kwh == 2.2
    assert m4.metadata["is_gap"] is False


def test_gap_detection_with_real_processed_dataset() -> None:
    """Verify gap detection against the real UCI processed dataset around 2007-06-09 18:00."""
    real_csv = Path("data/processed/hourly_energy.csv")
    if not real_csv.is_file():
        pytest.skip("Processed hourly dataset not found.")

    # 2007-06-09 18:00:00 is an actual excluded gap hour in data/processed/hourly_energy.csv
    source = HistoricalReplaySource(
        data_or_path=real_csv,
        start_timestamp="2007-06-09 17:00:00",
        end_timestamp="2007-06-09 19:00:00",
        preload_warmup_history=False,
    )

    # 17:00: valid
    m17 = source.fetch_next()
    assert m17 is not None and m17.is_valid is True
    assert m17.timestamp == pd.Timestamp("2007-06-09 17:00:00")

    # 18:00: GAP
    m18 = source.fetch_next()
    assert m18 is not None
    assert m18.timestamp == pd.Timestamp("2007-06-09 18:00:00")
    assert m18.is_valid is False
    assert np.isnan(m18.energy_kwh)
    assert m18.metadata["is_gap"] is True

    # 19:00: valid
    m19 = source.fetch_next()
    assert m19 is not None and m19.is_valid is True
    assert m19.timestamp == pd.Timestamp("2007-06-09 19:00:00")


# =========================================================================
# Lookback History and Temporal Isolation Tests
# =========================================================================

def test_lookback_history_contiguous(synthetic_hourly_df: pd.DataFrame) -> None:
    """Verify get_lookback_history returns clean contiguous DataFrame when observations exist."""
    source = HistoricalReplaySource(data_or_path=synthetic_hourly_df)

    # Advance 5 hours
    for _ in range(5):
        source.fetch_next()

    # Request 3 hours of lookback
    hist_3 = source.get_lookback_history(hours=3)
    assert len(hist_3) == 3
    assert hist_3.attrs["is_contiguous"] is True
    assert hist_3.attrs["has_gaps"] is False
    assert hist_3["timestamp"].iloc[-1] == pd.Timestamp("2010-01-01 04:00:00")
    assert hist_3["timestamp"].iloc[0] == pd.Timestamp("2010-01-01 02:00:00")


def test_lookback_history_with_gaps_flagged(synthetic_df_with_gap: pd.DataFrame) -> None:
    """Verify get_lookback_history flags gaps and preserves NaN values."""
    source = HistoricalReplaySource(
        data_or_path=synthetic_df_with_gap,
        step_mode="hourly_clock",
        preload_warmup_history=False,
    )

    # Advance through 00:00, 01:00, 02:00 (gap), 03:00 (gap), 04:00 (recorded)
    for _ in range(5):
        source.fetch_next()

    # Request 5 hours lookback ending at 04:00
    hist = source.get_lookback_history(hours=5)
    assert len(hist) == 5
    assert hist.attrs["is_contiguous"] is False
    assert hist.attrs["has_gaps"] is True
    assert hist["energy_kwh"].isna().sum() == 2  # exactly 2 gap hours


def test_lookback_history_strict_temporal_isolation(synthetic_hourly_df: pd.DataFrame) -> None:
    """Verify observations beyond the current timestamp are never leaked into lookback."""
    source = HistoricalReplaySource(data_or_path=synthetic_hourly_df)

    # Advance to hour index 2 (02:00:00)
    for _ in range(3):
        source.fetch_next()

    current_t = source.current_timestamp
    assert current_t == pd.Timestamp("2010-01-01 02:00:00")

    hist = source.get_lookback_history(hours=10)
    # Even though 10 hours exist in dataset, only 3 hours are <= 02:00:00
    assert len(hist) == 3
    assert (hist["timestamp"] <= current_t).all()
    assert (hist["timestamp"] > current_t).sum() == 0


def test_lookback_history_before_streaming_starts(synthetic_hourly_df: pd.DataFrame) -> None:
    """Verify get_lookback_history returns empty DataFrame when source has not started."""
    source = HistoricalReplaySource(data_or_path=synthetic_hourly_df)
    hist = source.get_lookback_history(hours=5)
    assert len(hist) == 0
    assert hist.attrs["is_contiguous"] is False
    assert list(hist.columns) == ["timestamp", "energy_kwh"]


# =========================================================================
# Boundary Conditions and Exhaustion Tests
# =========================================================================

def test_stream_exhaustion(synthetic_hourly_df: pd.DataFrame) -> None:
    """Verify has_more transitions to False and fetch_next returns None when stream ends."""
    source = HistoricalReplaySource(
        data_or_path=synthetic_hourly_df,
        start_timestamp="2010-01-01 00:00:00",
        end_timestamp="2010-01-01 02:00:00",
    )

    # Total 3 hours: 00:00, 01:00, 02:00
    m1 = source.fetch_next()
    assert m1 is not None and source.has_more is True

    m2 = source.fetch_next()
    assert m2 is not None and source.has_more is True

    m3 = source.fetch_next()
    assert m3 is not None and source.has_more is False

    # Beyond stream end
    m4 = source.fetch_next()
    assert m4 is None
    assert source.has_more is False


def test_warmup_history_preloading(synthetic_hourly_df: pd.DataFrame) -> None:
    """Verify preload_warmup_history populates buffer with observations prior to start_timestamp."""
    # Start at hour 5, preload earlier 5 hours
    source = HistoricalReplaySource(
        data_or_path=synthetic_hourly_df,
        start_timestamp="2010-01-01 05:00:00",
        preload_warmup_history=True,
    )

    # Fetch first active measurement at 05:00:00
    m = source.fetch_next()
    assert m is not None
    assert m.timestamp == pd.Timestamp("2010-01-01 05:00:00")

    # Lookback of 6 hours should include 00:00 through 05:00
    hist = source.get_lookback_history(hours=6)
    assert len(hist) == 6
    assert hist.attrs["is_contiguous"] is True
    assert hist["timestamp"].iloc[0] == pd.Timestamp("2010-01-01 00:00:00")
    assert hist["timestamp"].iloc[-1] == pd.Timestamp("2010-01-01 05:00:00")


# =========================================================================
# Alternative Records Step Mode Tests
# =========================================================================

def test_records_step_mode(synthetic_df_with_gap: pd.DataFrame) -> None:
    """Verify 'records' mode steps row-by-row and attaches gap metadata."""
    source = HistoricalReplaySource(
        data_or_path=synthetic_df_with_gap,
        step_mode="records",
        preload_warmup_history=False,
    )

    # Row 1: 00:00
    r1 = source.fetch_next()
    assert r1 is not None and r1.timestamp == pd.Timestamp("2010-01-01 00:00:00")

    # Row 2: 01:00
    r2 = source.fetch_next()
    assert r2 is not None and r2.timestamp == pd.Timestamp("2010-01-01 01:00:00")
    assert "gap_detected_before" not in r2.metadata

    # Row 3: 04:00 (jumps by 3 hours from 01:00)
    r3 = source.fetch_next()
    assert r3 is not None and r3.timestamp == pd.Timestamp("2010-01-01 04:00:00")
    assert r3.metadata.get("gap_detected_before") is True
    assert r3.metadata.get("skipped_gap_hours") == 2
