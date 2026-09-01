from pathlib import Path

import pandas as pd
import pytest

from pravaha_ml.data.historical import (
    REQUIRED_COLUMNS,
    historical_records_to_training_samples,
    load_historical_csv,
)


def build_valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "CASE_001",
                "timestamp": "2026-08-30T12:00:00Z",
                "latitude": 30.10,
                "longitude": 78.20,
                "source": "test_fixture",
                "label": 0,

                "rain_15m_mm": 3.0,
                "rain_30m_mm": 6.0,
                "rain_1h_mm": 12.0,
                "rain_3h_mm": 20.0,
                "rain_6h_mm": 30.0,
                "rain_24h_mm": 50.0,

                "api_mm": 35.0,

                "soil_moisture_percentage": 40.0,
                "soil_saturation": 0.40,
                "moisture_condition": "NORMAL",

                "base_curve_number": 70.0,
                "effective_curve_number": 70.0,

                "runoff_mm": 2.0,
                "runoff_ratio": 0.16,

                "flow_length_m": 2500.0,
                "slope_fraction": 0.08,
                "concentration_time_minutes": 45.0,
            },
            {
                "event_id": "CASE_002",
                "timestamp": "2026-08-30T13:00:00Z",
                "latitude": 30.15,
                "longitude": 78.25,
                "source": "test_fixture",
                "label": 1,

                "rain_15m_mm": 25.0,
                "rain_30m_mm": 45.0,
                "rain_1h_mm": 80.0,
                "rain_3h_mm": 130.0,
                "rain_6h_mm": 180.0,
                "rain_24h_mm": 250.0,

                "api_mm": 220.0,

                "soil_moisture_percentage": 85.0,
                "soil_saturation": 0.85,
                "moisture_condition": "WET",

                "base_curve_number": 80.0,
                "effective_curve_number": 92.0,

                "runoff_mm": 48.0,
                "runoff_ratio": 0.60,

                "flow_length_m": 1800.0,
                "slope_fraction": 0.18,
                "concentration_time_minutes": 20.0,
            },
        ]
    )


def test_required_columns_include_model_features():
    assert "rain_1h_mm" in REQUIRED_COLUMNS
    assert "api_mm" in REQUIRED_COLUMNS
    assert "runoff_mm" in REQUIRED_COLUMNS
    assert (
        "concentration_time_minutes"
        in REQUIRED_COLUMNS
    )


def test_load_historical_csv(
    tmp_path: Path,
):
    dataframe = build_valid_dataframe()

    path = (
        tmp_path
        / "historical.csv"
    )

    dataframe.to_csv(
        path,
        index=False,
    )

    records = load_historical_csv(
        path
    )

    assert len(records) == 2

    assert records[0].label == 0
    assert records[1].label == 1

    assert (
        records[1]
        .features
        .soil_saturation
        == pytest.approx(0.85)
    )


def test_historical_records_convert_to_training_samples(
    tmp_path: Path,
):
    dataframe = build_valid_dataframe()

    path = (
        tmp_path
        / "historical.csv"
    )

    dataframe.to_csv(
        path,
        index=False,
    )

    records = load_historical_csv(
        path
    )

    samples = (
        historical_records_to_training_samples(
            records
        )
    )

    assert len(samples) == 2

    assert {
        sample.label
        for sample in samples
    } == {0, 1}


def test_missing_required_column_rejected(
    tmp_path: Path,
):
    dataframe = (
        build_valid_dataframe()
        .drop(
            columns=[
                "runoff_mm"
            ]
        )
    )

    path = (
        tmp_path
        / "historical.csv"
    )

    dataframe.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        load_historical_csv(
            path
        )


def test_invalid_label_rejected(
    tmp_path: Path,
):
    dataframe = build_valid_dataframe()

    dataframe.loc[
        1,
        "label",
    ] = 2

    path = (
        tmp_path
        / "historical.csv"
    )

    dataframe.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="label must be 0 or 1",
    ):
        load_historical_csv(
            path
        )


def test_invalid_latitude_rejected(
    tmp_path: Path,
):
    dataframe = build_valid_dataframe()

    dataframe.loc[
        0,
        "latitude",
    ] = 120.0

    path = (
        tmp_path
        / "historical.csv"
    )

    dataframe.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="latitude must be between -90 and 90",
    ):
        load_historical_csv(
            path
        )


def test_invalid_soil_saturation_rejected(
    tmp_path: Path,
):
    dataframe = build_valid_dataframe()

    dataframe.loc[
        0,
        "soil_saturation",
    ] = 1.5

    path = (
        tmp_path
        / "historical.csv"
    )

    dataframe.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="soil_saturation must be between 0 and 1",
    ):
        load_historical_csv(
            path
        )


def test_missing_file_rejected(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "does_not_exist.csv"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        load_historical_csv(
            path
        )


def test_empty_dataset_rejected(
    tmp_path: Path,
):
    dataframe = pd.DataFrame(
        columns=REQUIRED_COLUMNS
    )

    path = (
        tmp_path
        / "historical.csv"
    )

    dataframe.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="contains no records",
    ):
        load_historical_csv(
            path
        )