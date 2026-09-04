from datetime import (
    datetime,
    timedelta,
    timezone,
)

import pytest

from pravaha_ml.features.rainfall import (
    RainfallObservation,
)
from pravaha_ml.features.temporal_quality import (
    TemporalQualityLevel,
    TemporalQualityPolicy,
    calculate_time_aware_api,
    evaluate_temporal_quality,
)


UTC = timezone.utc


def make_observation(
    minutes_before_prediction: int,
    rainfall_mm_per_hr: float,
    prediction_time: datetime,
) -> RainfallObservation:
    return RainfallObservation(
        timestamp=(
            prediction_time
            - timedelta(
                minutes=minutes_before_prediction
            )
        ),
        rainfall_mm_per_hr=(
            rainfall_mm_per_hr
        ),
    )


def test_good_temporal_quality():
    prediction_time = datetime(
        2026,
        9,
        4,
        12,
        0,
        tzinfo=UTC,
    )

    observations = [
        make_observation(
            20,
            20.0,
            prediction_time,
        ),
        make_observation(
            10,
            30.0,
            prediction_time,
        ),
        make_observation(
            5,
            40.0,
            prediction_time,
        ),
    ]

    report = evaluate_temporal_quality(
        observations=observations,
        prediction_time=prediction_time,
    )

    assert (
        report.level
        == TemporalQualityLevel.GOOD
    )

    assert report.can_predict is True

    assert (
        report.latest_observation_age_minutes
        == pytest.approx(5.0)
    )

    assert (
        report.largest_gap_minutes
        == pytest.approx(10.0)
    )


def test_degraded_temporal_quality():
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
        make_observation(
            35,
            20.0,
            prediction_time,
        ),
        make_observation(
            14,
            30.0,
            prediction_time,
        ),
    ]

    report = evaluate_temporal_quality(
        observations=observations,
        prediction_time=prediction_time,
        policy=policy,
    )

    assert (
        report.level
        == TemporalQualityLevel.DEGRADED
    )

    assert report.can_predict is True


def test_stale_latest_observation_is_unusable():
    prediction_time = datetime(
        2026,
        9,
        4,
        12,
        0,
        tzinfo=UTC,
    )

    observations = [
        make_observation(
            45,
            30.0,
            prediction_time,
        ),
        make_observation(
            30,
            40.0,
            prediction_time,
        ),
    ]

    report = evaluate_temporal_quality(
        observations=observations,
        prediction_time=prediction_time,
    )

    assert (
        report.level
        == TemporalQualityLevel.UNUSABLE
    )

    assert report.can_predict is False
    assert report.stale is True

    assert (
        "stale_latest_observation"
        in report.reasons
    )


def test_excessive_gap_is_unusable():
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
        make_observation(
            50,
            20.0,
            prediction_time,
        ),
        make_observation(
            10,
            40.0,
            prediction_time,
        ),
        make_observation(
            5,
            50.0,
            prediction_time,
        ),
    ]

    report = evaluate_temporal_quality(
        observations=observations,
        prediction_time=prediction_time,
        policy=policy,
    )

    assert (
        report.level
        == TemporalQualityLevel.UNUSABLE
    )

    assert report.excessive_gap is True
    assert report.can_predict is False


def test_future_observation_is_unusable():
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
                + timedelta(minutes=5)
            ),
            rainfall_mm_per_hr=30.0,
        )
    ]

    report = evaluate_temporal_quality(
        observations=observations,
        prediction_time=prediction_time,
    )

    assert (
        report.level
        == TemporalQualityLevel.UNUSABLE
    )

    assert (
        report.has_future_observation
        is True
    )


def test_duplicate_timestamp_is_unusable():
    prediction_time = datetime(
        2026,
        9,
        4,
        12,
        0,
        tzinfo=UTC,
    )

    timestamp = (
        prediction_time
        - timedelta(minutes=5)
    )

    observations = [
        RainfallObservation(
            timestamp=timestamp,
            rainfall_mm_per_hr=20.0,
        ),
        RainfallObservation(
            timestamp=timestamp,
            rainfall_mm_per_hr=30.0,
        ),
    ]

    report = evaluate_temporal_quality(
        observations=observations,
        prediction_time=prediction_time,
    )

    assert (
        report.level
        == TemporalQualityLevel.UNUSABLE
    )

    assert (
        report.has_duplicate_timestamp
        is True
    )


def test_time_aware_api_uses_elapsed_time():
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
                - timedelta(hours=1)
            ),
            rainfall_mm_per_hr=10.0,
        ),
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=30)
            ),
            rainfall_mm_per_hr=20.0,
        ),
    ]

    api = calculate_time_aware_api(
        observations=observations,
        prediction_time=prediction_time,
        decay_factor_per_hour=0.90,
    )

    first_depth = 10.0 * 0.5
    second_depth = 20.0 * 0.5

    expected = (
        first_depth
        * (0.90 ** 0.5)
        + second_depth
    )

    assert api == pytest.approx(
        expected
    )


def test_time_aware_api_changes_with_interval_duration():
    prediction_time = datetime(
        2026,
        9,
        4,
        12,
        0,
        tzinfo=UTC,
    )

    short_interval = [
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(minutes=30)
            ),
            rainfall_mm_per_hr=20.0,
        )
    ]

    long_interval = [
        RainfallObservation(
            timestamp=(
                prediction_time
                - timedelta(hours=1)
            ),
            rainfall_mm_per_hr=20.0,
        )
    ]

    short_api = calculate_time_aware_api(
        observations=short_interval,
        prediction_time=prediction_time,
    )

    long_api = calculate_time_aware_api(
        observations=long_interval,
        prediction_time=prediction_time,
    )

    assert long_api > short_api


def test_invalid_decay_factor_rejected():
    prediction_time = datetime(
        2026,
        9,
        4,
        12,
        0,
        tzinfo=UTC,
    )

    observations = [
        make_observation(
            5,
            20.0,
            prediction_time,
        )
    ]

    with pytest.raises(
        ValueError,
        match="decay_factor_per_hour",
    ):
        calculate_time_aware_api(
            observations=observations,
            prediction_time=prediction_time,
            decay_factor_per_hour=1.50,
        )


def test_empty_observations_rejected():
    prediction_time = datetime(
        2026,
        9,
        4,
        12,
        0,
        tzinfo=UTC,
    )

    with pytest.raises(
        ValueError,
        match=(
            "At least one rainfall observation"
        ),
    ):
        evaluate_temporal_quality(
            observations=[],
            prediction_time=prediction_time,
        )