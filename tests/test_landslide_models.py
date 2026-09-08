import pytest

from pravaha_ml.landslide.models import (
    LandslideDataProvenance,
    LandslideInputs,
)


def make_inputs(
    **overrides,
) -> LandslideInputs:
    values = {
        "zone_id": "LS_001",
        "slope_degrees": 25.0,
        "soil_saturation": 0.60,
        "rain_1h_mm": 30.0,
        "rain_24h_mm": 80.0,
        "api_mm": 60.0,
        "historical_landslide_score": 0.30,
        "land_cover_disturbance_score": 0.20,
        "data_confidence": 0.90,
        "historical_inventory_provenance": (
            LandslideDataProvenance.VERIFIED
        ),
    }

    values.update(
        overrides
    )

    return LandslideInputs(
        **values
    )


def test_valid_landslide_inputs():
    inputs = make_inputs()

    assert (
        inputs.zone_id
        == "LS_001"
    )


def test_empty_zone_id_rejected():
    with pytest.raises(
        ValueError,
        match="zone_id cannot be empty",
    ):
        make_inputs(
            zone_id=""
        )


def test_invalid_slope_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "slope_degrees must be between 0 and 90"
        ),
    ):
        make_inputs(
            slope_degrees=95.0
        )


def test_invalid_soil_saturation_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "soil_saturation must be between 0 and 1"
        ),
    ):
        make_inputs(
            soil_saturation=1.20
        )


def test_negative_rainfall_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "rain_1h_mm cannot be negative"
        ),
    ):
        make_inputs(
            rain_1h_mm=-1.0
        )


def test_invalid_historical_score_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "historical_landslide_score must be "
            "between 0 and 1"
        ),
    ):
        make_inputs(
            historical_landslide_score=1.50
        )


def test_invalid_data_confidence_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "data_confidence must be between 0 and 1"
        ),
    ):
        make_inputs(
            data_confidence=-0.10
        )