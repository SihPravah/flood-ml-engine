from datetime import datetime, timezone

import pytest

from pravaha_ml.features.hydrology_features import (
    build_hydrology_features,
)
from pravaha_ml.features.rainfall import RainfallObservation


UTC = timezone.utc


def test_build_hydrology_features():
    observations = [
        RainfallObservation(
            timestamp=datetime(
                2026, 8, 30, 13, 0, tzinfo=UTC
            ),
            rainfall_mm_per_hr=20.0,
        ),
        RainfallObservation(
            timestamp=datetime(
                2026, 8, 30, 13, 30, tzinfo=UTC
            ),
            rainfall_mm_per_hr=40.0,
        ),
    ]

    prediction_time = datetime(
        2026, 8, 30, 14, 0, tzinfo=UTC
    )

    features = build_hydrology_features(
        rainfall_observations=observations,
        prediction_time=prediction_time,
        soil_moisture_percentage=82.0,
        base_curve_number=80.0,
        flow_length_m=2000.0,
        slope_fraction=0.10,
        dry_threshold_percentage=35.0,
        wet_threshold_percentage=70.0,
    )

    assert features.rain_1h_mm == pytest.approx(30.0)

    assert features.soil_saturation == pytest.approx(0.82)

    assert features.moisture_condition == "WET"

    assert features.effective_curve_number > features.base_curve_number

    assert features.runoff_mm >= 0.0

    assert 0.0 <= features.runoff_ratio <= 1.0

    assert features.concentration_time_minutes > 0.0


def test_wetter_soil_increases_runoff():
    observations = [
        RainfallObservation(
            timestamp=datetime(
                2026, 8, 30, 13, 0, tzinfo=UTC
            ),
            rainfall_mm_per_hr=80.0,
        ),
    ]

    prediction_time = datetime(
        2026, 8, 30, 14, 0, tzinfo=UTC
    )

    dry_features = build_hydrology_features(
        rainfall_observations=observations,
        prediction_time=prediction_time,
        soil_moisture_percentage=20.0,
        base_curve_number=80.0,
        flow_length_m=2000.0,
        slope_fraction=0.10,
        dry_threshold_percentage=35.0,
        wet_threshold_percentage=70.0,
    )

    wet_features = build_hydrology_features(
        rainfall_observations=observations,
        prediction_time=prediction_time,
        soil_moisture_percentage=85.0,
        base_curve_number=80.0,
        flow_length_m=2000.0,
        slope_fraction=0.10,
        dry_threshold_percentage=35.0,
        wet_threshold_percentage=70.0,
    )

    assert (
        wet_features.effective_curve_number
        > dry_features.effective_curve_number
    )

    assert wet_features.runoff_mm > dry_features.runoff_mm


def test_steeper_catchment_reduces_concentration_time():
    observations = [
        RainfallObservation(
            timestamp=datetime(
                2026, 8, 30, 13, 0, tzinfo=UTC
            ),
            rainfall_mm_per_hr=50.0,
        ),
    ]

    prediction_time = datetime(
        2026, 8, 30, 14, 0, tzinfo=UTC
    )

    gentle = build_hydrology_features(
        rainfall_observations=observations,
        prediction_time=prediction_time,
        soil_moisture_percentage=60.0,
        base_curve_number=80.0,
        flow_length_m=2000.0,
        slope_fraction=0.05,
        dry_threshold_percentage=35.0,
        wet_threshold_percentage=70.0,
    )

    steep = build_hydrology_features(
        rainfall_observations=observations,
        prediction_time=prediction_time,
        soil_moisture_percentage=60.0,
        base_curve_number=80.0,
        flow_length_m=2000.0,
        slope_fraction=0.20,
        dry_threshold_percentage=35.0,
        wet_threshold_percentage=70.0,
    )

    assert (
        steep.concentration_time_minutes
        < gentle.concentration_time_minutes
    )


def test_invalid_soil_moisture_propagates_validation():
    prediction_time = datetime(
        2026, 8, 30, 14, 0, tzinfo=UTC
    )

    with pytest.raises(
        ValueError,
        match="soil_moisture_percentage must be between 0 and 100",
    ):
        build_hydrology_features(
            rainfall_observations=[],
            prediction_time=prediction_time,
            soil_moisture_percentage=130.0,
            base_curve_number=80.0,
            flow_length_m=2000.0,
            slope_fraction=0.10,
            dry_threshold_percentage=35.0,
            wet_threshold_percentage=70.0,
        )