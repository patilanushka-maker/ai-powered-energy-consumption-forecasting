"""Unit tests for the raw data loader module (src.data_loader).

Uses pytest tmp_path fixtures to test raw file ingestion without network access:
- Semicolon-delimited file parsing
- Date and Time concatenation and timestamp parsing
- Missing value '?' handling
- Invalid numeric power coercion
- Missing required schema columns error
- Nonexistent file path error
- Empty file handling
- Unparseable timestamp handling
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import load_raw_data


def test_missing_file_raises_filenotfound(tmp_path: Path):
    """Verify that a nonexistent file path raises FileNotFoundError."""
    nonexistent = tmp_path / "does_not_exist.txt"
    with pytest.raises(FileNotFoundError, match="Raw dataset file not found"):
        load_raw_data(nonexistent)


def test_empty_file_raises_valueerror(tmp_path: Path):
    """Verify that an empty (0 byte) file raises ValueError."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_raw_data(empty_file)


def test_missing_required_columns(tmp_path: Path):
    """Verify that files missing required columns (e.g. Global_active_power) raise ValueError."""
    bad_header_file = tmp_path / "bad_columns.txt"
    bad_header_file.write_text(
        "Date;Time;Voltage\n16/12/2006;17:24:00;244.80\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        load_raw_data(bad_header_file)


def test_valid_semicolon_file_loading(tmp_path: Path):
    """Verify loading of a valid semicolon-separated UCI sample file."""
    sample_file = tmp_path / "sample_power.txt"
    content = (
        "Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3\n"
        "16/12/2006;17:24:00;4.216;0.418;234.840;18.400;0.000;1.000;17.000\n"
        "16/12/2006;17:25:00;5.360;0.436;233.630;23.000;0.000;1.000;16.000\n"
    )
    sample_file.write_text(content, encoding="utf-8")

    df = load_raw_data(sample_file)

    assert len(df) == 2
    assert "timestamp" in df.columns
    assert df.loc[0, "timestamp"] == pd.Timestamp("2006-12-16 17:24:00")
    assert df.loc[1, "timestamp"] == pd.Timestamp("2006-12-16 17:25:00")
    assert df.loc[0, "Global_active_power"] == pytest.approx(4.216)
    assert df.loc[1, "Global_active_power"] == pytest.approx(5.360)
    assert "data_loader_audit" in df.attrs
    assert df.attrs["data_loader_audit"]["raw_row_count"] == 2
    assert df.attrs["data_loader_audit"]["invalid_timestamps"] == 0
    assert df.attrs["data_loader_audit"]["missing_power_readings"] == 0


def test_question_mark_na_handling(tmp_path: Path):
    """Verify that '?' values are parsed as NA/NaN."""
    sample_file = tmp_path / "sample_na.txt"
    content = (
        "Date;Time;Global_active_power;Voltage\n"
        "16/12/2006;17:24:00;?;234.840\n"
        "16/12/2006;17:25:00;3.120;?\n"
    )
    sample_file.write_text(content, encoding="utf-8")

    df = load_raw_data(sample_file)

    assert pd.isna(df.loc[0, "Global_active_power"])
    assert df.loc[1, "Global_active_power"] == pytest.approx(3.120)
    assert df.attrs["data_loader_audit"]["missing_power_readings"] == 1


def test_invalid_numeric_readings_coercion(tmp_path: Path):
    """Verify that corrupted non-numeric readings are coerced to NaN."""
    sample_file = tmp_path / "sample_corrupt_numeric.txt"
    content = (
        "Date;Time;Global_active_power\n"
        "16/12/2006;17:24:00;corrupted_power\n"
        "16/12/2006;17:25:00;2.500\n"
    )
    sample_file.write_text(content, encoding="utf-8")

    df = load_raw_data(sample_file)

    assert pd.isna(df.loc[0, "Global_active_power"])
    assert df.loc[1, "Global_active_power"] == pytest.approx(2.500)
    assert df.attrs["data_loader_audit"]["missing_power_readings"] == 1
    assert df.attrs["data_loader_audit"]["coerced_power_readings"] == 1


def test_invalid_timestamp_reporting(tmp_path: Path):
    """Verify that malformed dates/times are flagged in audit."""
    sample_file = tmp_path / "sample_corrupt_date.txt"
    content = (
        "Date;Time;Global_active_power\n"
        "99/99/9999;99:99:99;2.100\n"
        "16/12/2006;17:25:00;2.500\n"
    )
    sample_file.write_text(content, encoding="utf-8")

    df = load_raw_data(sample_file)

    assert pd.isna(df.loc[0, "timestamp"])
    assert df.loc[1, "timestamp"] == pd.Timestamp("2006-12-16 17:25:00")
    assert df.attrs["data_loader_audit"]["invalid_timestamps"] == 1
