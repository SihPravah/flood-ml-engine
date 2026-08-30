from datetime import datetime, timezone

import pytest

from pravaha_ml.features.antecedent import (
    build_antecedent_features,
    calculate_api,
)
from pravaha_ml.features.rainfall import RainfallObservation


UTC = timezone.utc


def test_api_single_hour_of_rainfall():
    observations = [
        RainfallObservation(
            timestamp=datetime(
                2026, 8, 30, 13, 0, tzinfo=UTC
            ),
            rainfall_mm_per_hr=20.0,
        ),
    ]

    prediction_time = datetime(
        2026, 8, 30, 14, 0, tzinfo=UTC
    )

    api = calculate_api(
        observations,
        prediction_time,
        decay_factor=0.90,
    )

    # 20 mm/hr for one hour = 20 mm depth.
    assert api == pytest.approx(20.0)


def test_api_accumulates_multiple_intervals():
    observations = [
        RainfallObservation(
            timestamp=datetime(
                2026, 8, 30, 12, 0, tzinfo=UTC
            ),
            rainfall_mm_per_hr=10.0,
        ),
        RainfallObservation(
            timestamp=datetime(
                2026, 8, 30, 13, 0, tzinfo=UTC
            ),
            rainfall_mm_per_hr=20.0,
        ),
    ]

    prediction_time = datetime(
        2026, 8, 30, 14, 0, tzinfo=UTC
    )

    api = calculate_api(
        observations,
        prediction_time,
        decay_factor=0.90,
    )

    # First hour:
    # 10 mm
    #
    # Second hour:
    # 20 mm
    #
    # API:
    # 20 + 0.9 * 10 = 29 mm
    assert api == pytest.approx(29.0)


def test_more_recent_rainfall_has_greater_effect():
    old_rain = [
        RainfallObservation(
            timestamp=datetime(
                2026, 8, 30, 11, 0, tzinfo=UTC
            ),
            rainfall_mm_per_hr=20.0,
        ),
        RainfallObservation(
            timestamp=datetime(
                2026, 8, 30, 12, 0, tzinfo=UTC
            ),
            rainfall_mm_per_hr=0.0,
        ),
        RainfallObservation(
            timestamp=datetime(
                2026, 8, 30, 13, 0, tzinfo=UTC
            ),
            rainfall_mm_per_hr=0.0,
        ),
    ]

    recent_rain = [
        RainfallObservation(
            timestamp=datetime(
                2026, 8, 30, 11, 0, tzinfo=UTC
            ),
            rainfall_mm_per_hr=0.0,
        ),
        RainfallObservation(
            timestamp=datetime(
                2026, 8, 30, 12, 0, tzinfo=UTC
            ),
            rainfall_mm_per_hr=0.0,
        ),
        RainfallObservation(
            timestamp=datetime(
                2026, 8, 30, 13, 0, tzinfo=UTC
            ),
            rainfall_mm_per_hr=20.0,
        ),
    ]

    prediction_time = datetime(
        2026, 8, 30, 14, 0, tzinfo=UTC
    )

    old_api = calculate_api(
        old_rain,
        prediction_time,
        decay_factor=0.8,
    )

    recent_api = calculate_api(
        recent_rain,
        prediction_time,
        decay_factor=0.8,
    )

    assert recent_api > old_api


def test_invalid_decay_factor_rejected():
    prediction_time = datetime(
        2026, 8, 30, 14, 0, tzinfo=UTC
    )

    with pytest.raises(
        ValueError,
        match="decay_factor must be between 0 and 1",
    ):
        calculate_api(
            [],
            prediction_time,
            decay_factor=1.2,
        )


def test_empty_observations_return_zero():
    prediction_time = datetime(
        2026, 8, 30, 14, 0, tzinfo=UTC
    )

    api = calculate_api(
        [],
        prediction_time,
    )

    assert api == 0.0


def test_build_antecedent_features():
    observations = [
        RainfallObservation(
            timestamp=datetime(
                2026, 8, 30, 13, 0, tzinfo=UTC
            ),
            rainfall_mm_per_hr=25.0,
        ),
    ]

    prediction_time = datetime(
        2026, 8, 30, 14, 0, tzinfo=UTC
    )

    features = build_antecedent_features(
        observations,
        prediction_time,
    )

    assert features.api_mm == pytest.approx(25.0)