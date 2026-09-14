from datetime import datetime, timezone

import pytest

from pravaha_ml.features.rainfall import (
    RainfallObservation,
    build_rainfall_features,
)


UTC = timezone.utc


def test_constant_rainfall_for_one_hour():
    observations = [
        RainfallObservation(
            timestamp=datetime(
                2026, 8, 30, 13, 0, tzinfo=UTC
            ),
            rainfall_mm_per_hr=60.0,
        ),
    ]

    prediction_time = datetime(
        2026, 8, 30, 14, 0, tzinfo=UTC
    )

    features = build_rainfall_features(
        observations,
        prediction_time,
    )

    assert features.rain_15m_mm == pytest.approx(15.0)
    assert features.rain_30m_mm == pytest.approx(30.0)
    assert features.rain_1h_mm == pytest.approx(60.0)

    # We only have one hour of observations.
    assert features.rain_3h_mm == pytest.approx(60.0)
    assert features.rain_6h_mm == pytest.approx(60.0)
    assert features.rain_24h_mm == pytest.approx(60.0)


def test_changing_rainfall_intensity():
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

    features = build_rainfall_features(
        observations,
        prediction_time,
    )

    # First 30 min:
    # 20 mm/hr * 0.5 hr = 10 mm
    #
    # Second 30 min:
    # 40 mm/hr * 0.5 hr = 20 mm
    #
    # Total = 30 mm
    assert features.rain_1h_mm == pytest.approx(30.0)

    # Final 30 minutes are at 40 mm/hr.
    assert features.rain_30m_mm == pytest.approx(20.0)

    # Final 15 minutes are also at 40 mm/hr.
    assert features.rain_15m_mm == pytest.approx(10.0)


def test_observations_are_sorted_automatically():
    observations = [
        RainfallObservation(
            timestamp=datetime(
                2026, 8, 30, 13, 30, tzinfo=UTC
            ),
            rainfall_mm_per_hr=40.0,
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

    features = build_rainfall_features(
        observations,
        prediction_time,
    )

    assert features.rain_1h_mm == pytest.approx(30.0)


def test_negative_rainfall_rejected():
    observations = [
        RainfallObservation(
            timestamp=datetime(
                2026, 8, 30, 13, 0, tzinfo=UTC
            ),
            rainfall_mm_per_hr=-10.0,
        ),
    ]

    prediction_time = datetime(
        2026, 8, 30, 14, 0, tzinfo=UTC
    )

    with pytest.raises(
        ValueError,
        match="Rainfall intensity cannot be negative",
    ):
        build_rainfall_features(
            observations,
            prediction_time,
        )


def test_naive_timestamp_rejected():
    observations = [
        RainfallObservation(
            timestamp=datetime(
                2026, 8, 30, 13, 0
            ),
            rainfall_mm_per_hr=20.0,
        ),
    ]

    prediction_time = datetime(
        2026, 8, 30, 14, 0, tzinfo=UTC
    )

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        build_rainfall_features(
            observations,
            prediction_time,
        )


def test_empty_observations_return_zero():
    prediction_time = datetime(
        2026, 8, 30, 14, 0, tzinfo=UTC
    )

    features = build_rainfall_features(
        [],
        prediction_time,
    )

    assert features.rain_15m_mm == 0.0
    assert features.rain_30m_mm == 0.0
    assert features.rain_1h_mm == 0.0
    assert features.rain_3h_mm == 0.0
    assert features.rain_6h_mm == 0.0
    assert features.rain_24h_mm == 0.0