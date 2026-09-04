from datetime import (
    datetime,
    timedelta,
    timezone,
)

import pytest

from pravaha_ml.features.quality_aware_hydrology import (
    TemporalDataQualityError,
    build_quality_aware_hydrology_features,
)
from pravaha_ml.features.rainfall import (
    RainfallObservation,
)
from pravaha_ml.features.temporal_quality import (
    TemporalQualityLevel,
    TemporalQualityPolicy,
)


UTC = timezone.utc


def test_quality_aware_builder_returns_features():
    prediction_time = datetime(
        2026,
        9,
        4,
        12,
        0,
        tzinfo=UTC,
    )

    observations = [
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=20)
            ),
            rainfall_mm_per_hr=30.0,
        ),
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=10)
            ),
            rainfall_mm_per_hr=50.0,
        ),
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=5)
            ),
            rainfall_mm_per_hr=60.0,
        ),
    ]

    result = (
        build_quality_aware_hydrology_features(
            rainfall_observations=observations,
            prediction_time=prediction_time,
            soil_moisture_percentage=82.0,
            base_curve_number=80.0,
            flow_length_m=2000.0,
            slope_fraction=0.10,
            dry_threshold_percentage=35.0,
            wet_threshold_percentage=70.0,
        )
    )

    assert (
        result.temporal_quality.level
        == TemporalQualityLevel.GOOD
    )

    assert (
        result.temporal_quality.can_predict
        is True
    )

    assert result.features.api_mm > 0.0
    assert result.features.runoff_mm >= 0.0

    assert (
        result.features
        .concentration_time_minutes
        > 0.0
    )


def test_degraded_data_is_allowed_but_flagged():
    prediction_time = datetime(
        2026,
        9,
        4,
        12,
        0,
        tzinfo=UTC,
    )

    policy = TemporalQualityPolicy(
        max_gap_minutes=30.0,
        max_staleness_minutes=20.0,
        degraded_gap_fraction=0.70,
        degraded_staleness_fraction=0.70,
    )

    observations = [
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=35)
            ),
            rainfall_mm_per_hr=30.0,
        ),
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=14)
            ),
            rainfall_mm_per_hr=50.0,
        ),
    ]

    result = (
        build_quality_aware_hydrology_features(
            rainfall_observations=observations,
            prediction_time=prediction_time,
            soil_moisture_percentage=70.0,
            base_curve_number=80.0,
            flow_length_m=2000.0,
            slope_fraction=0.10,
            dry_threshold_percentage=35.0,
            wet_threshold_percentage=70.0,
            temporal_quality_policy=policy,
        )
    )

    assert (
        result.temporal_quality.level
        == TemporalQualityLevel.DEGRADED
    )

    assert (
        result.temporal_quality.can_predict
        is True
    )


def test_stale_data_blocks_feature_generation():
    prediction_time = datetime(
        2026,
        9,
        4,
        12,
        0,
        tzinfo=UTC,
    )

    observations = [
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=60)
            ),
            rainfall_mm_per_hr=40.0,
        ),
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=45)
            ),
            rainfall_mm_per_hr=50.0,
        ),
    ]

    with pytest.raises(
        TemporalDataQualityError,
        match=(
            "Temporal rainfall data is not reliable"
        ),
    ):
        build_quality_aware_hydrology_features(
            rainfall_observations=observations,
            prediction_time=prediction_time,
            soil_moisture_percentage=80.0,
            base_curve_number=80.0,
            flow_length_m=2000.0,
            slope_fraction=0.10,
            dry_threshold_percentage=35.0,
            wet_threshold_percentage=70.0,
        )


def test_excessive_gap_blocks_feature_generation():
    prediction_time = datetime(
        2026,
        9,
        4,
        12,
        0,
        tzinfo=UTC,
    )

    policy = TemporalQualityPolicy(
        max_gap_minutes=20.0,
        max_staleness_minutes=15.0,
    )

    observations = [
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=50)
            ),
            rainfall_mm_per_hr=30.0,
        ),
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=10)
            ),
            rainfall_mm_per_hr=50.0,
        ),
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=5)
            ),
            rainfall_mm_per_hr=60.0,
        ),
    ]

    with pytest.raises(
        TemporalDataQualityError,
        match=(
            "Temporal rainfall data is not reliable"
        ),
    ):
        build_quality_aware_hydrology_features(
            rainfall_observations=observations,
            prediction_time=prediction_time,
            soil_moisture_percentage=80.0,
            base_curve_number=80.0,
            flow_length_m=2000.0,
            slope_fraction=0.10,
            dry_threshold_percentage=35.0,
            wet_threshold_percentage=70.0,
            temporal_quality_policy=policy,
        )


def test_time_aware_api_is_used_in_final_features():
    prediction_time = datetime(
        2026,
        9,
        4,
        12,
        0,
        tzinfo=UTC,
    )

    observations = [
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=20)
            ),
            rainfall_mm_per_hr=60.0,
        ),
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=10)
            ),
            rainfall_mm_per_hr=30.0,
        ),
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=5)
            ),
            rainfall_mm_per_hr=15.0,
        ),
    ]

    result = (
        build_quality_aware_hydrology_features(
            rainfall_observations=observations,
            prediction_time=prediction_time,
            soil_moisture_percentage=75.0,
            base_curve_number=80.0,
            flow_length_m=2000.0,
            slope_fraction=0.10,
            dry_threshold_percentage=35.0,
            wet_threshold_percentage=70.0,
            api_decay_factor_per_hour=0.80,
        )
    )

    assert result.features.api_mm > 0.0