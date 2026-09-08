from datetime import (
    datetime,
    timedelta,
    timezone,
)

import pytest

from pravaha_ml.anticipation.models import (
    RiskObservation,
    RiskTrajectoryPolicy,
    RiskTrend,
    ThresholdStatus,
    TrajectoryConfidenceLevel,
)
from pravaha_ml.anticipation.trajectory import (
    assess_risk_trajectory,
)


UTC = timezone.utc

BASE_TIME = datetime(
    2026,
    9,
    8,
    12,
    0,
    tzinfo=UTC,
)


def make_series(
    values: list[float],
    *,
    interval_minutes: float = 10.0,
    end_time: datetime = BASE_TIME,
) -> list[
    RiskObservation
]:
    start_time = (
        end_time
        - timedelta(
            minutes=(
                interval_minutes
                * (
                    len(values)
                    - 1
                )
            )
        )
    )

    return [
        RiskObservation(
            timestamp=(
                start_time
                + timedelta(
                    minutes=(
                        interval_minutes
                        * index
                    )
                )
            ),
            risk_score=value,
        )
        for (
            index,
            value,
        ) in enumerate(
            values
        )
    ]


def get_projection(
    assessment,
    threshold_name: str,
):
    return next(
        projection
        for projection
        in assessment.projections
        if (
            projection.threshold_name
            == threshold_name
        )
    )


def test_steady_rising_series_detected():
    assessment = assess_risk_trajectory(
        observations=make_series(
            [
                0.30,
                0.36,
                0.42,
                0.48,
                0.54,
                0.60,
            ]
        ),
        current_time=BASE_TIME,
    )

    assert assessment.trend in {
        RiskTrend.RISING,
        RiskTrend.RAPIDLY_RISING,
    }

    assert (
        assessment.slope_per_minute
        is not None
    )

    assert (
        assessment.slope_per_minute
        > 0.0
    )


def test_rapidly_rising_series_detected():
    assessment = assess_risk_trajectory(
        observations=make_series(
            [
                0.20,
                0.28,
                0.38,
                0.49,
                0.61,
                0.74,
            ]
        ),
        current_time=BASE_TIME,
    )

    assert (
        assessment.trend
        == RiskTrend.RAPIDLY_RISING
    )

    assert (
        "risk_rising_rapidly"
        in assessment.reasons
    )


def test_falling_series_detected():
    assessment = assess_risk_trajectory(
        observations=make_series(
            [
                0.70,
                0.64,
                0.58,
                0.52,
                0.46,
                0.40,
            ]
        ),
        current_time=BASE_TIME,
    )

    assert assessment.trend in {
        RiskTrend.FALLING,
        RiskTrend.RAPIDLY_FALLING,
    }


def test_stable_series_detected():
    assessment = assess_risk_trajectory(
        observations=make_series(
            [
                0.40,
                0.401,
                0.399,
                0.400,
                0.401,
                0.400,
            ]
        ),
        current_time=BASE_TIME,
    )

    assert (
        assessment.trend
        == RiskTrend.STABLE
    )


def test_insufficient_observation_count_blocks_analysis():
    assessment = assess_risk_trajectory(
        observations=make_series(
            [
                0.30,
                0.40,
                0.50,
            ]
        ),
        current_time=BASE_TIME,
    )

    assert (
        assessment.trend
        == RiskTrend.INSUFFICIENT_DATA
    )

    assert (
        "insufficient_observation_count"
        in assessment.reasons
    )


def test_insufficient_history_span_blocks_analysis():
    observations = make_series(
        [
            0.30,
            0.35,
            0.40,
            0.45,
        ],
        interval_minutes=2.0,
    )

    assessment = assess_risk_trajectory(
        observations=observations,
        current_time=BASE_TIME,
    )

    assert (
        assessment.trend
        == RiskTrend.INSUFFICIENT_DATA
    )

    assert (
        "insufficient_history_span"
        in assessment.reasons
    )


def test_stale_history_blocks_analysis():
    end_time = (
        BASE_TIME
        - timedelta(
            minutes=30
        )
    )

    assessment = assess_risk_trajectory(
        observations=make_series(
            [
                0.30,
                0.40,
                0.50,
                0.60,
            ],
            end_time=end_time,
        ),
        current_time=BASE_TIME,
    )

    assert (
        assessment.trend
        == RiskTrend.INSUFFICIENT_DATA
    )

    assert (
        "risk_history_stale"
        in assessment.reasons
    )


def test_future_latest_observation_blocks_analysis():
    future_end = (
        BASE_TIME
        + timedelta(
            minutes=5
        )
    )

    assessment = assess_risk_trajectory(
        observations=make_series(
            [
                0.30,
                0.40,
                0.50,
                0.60,
            ],
            end_time=future_end,
        ),
        current_time=BASE_TIME,
    )

    assert (
        assessment.trend
        == RiskTrend.INSUFFICIENT_DATA
    )

    assert (
        "latest_observation_is_in_future"
        in assessment.reasons
    )


def test_duplicate_timestamps_rejected():
    timestamp = BASE_TIME

    observations = [
        RiskObservation(
            timestamp=timestamp,
            risk_score=0.40,
        ),
        RiskObservation(
            timestamp=timestamp,
            risk_score=0.50,
        ),
        RiskObservation(
            timestamp=(
                timestamp
                - timedelta(
                    minutes=10
                )
            ),
            risk_score=0.30,
        ),
        RiskObservation(
            timestamp=(
                timestamp
                - timedelta(
                    minutes=20
                )
            ),
            risk_score=0.20,
        ),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "Duplicate risk observation timestamp"
        ),
    ):
        assess_risk_trajectory(
            observations=observations,
            current_time=BASE_TIME,
        )


def test_unsorted_observations_are_sorted_safely():
    observations = make_series(
        [
            0.30,
            0.36,
            0.42,
            0.48,
            0.54,
        ]
    )

    observations = list(
        reversed(
            observations
        )
    )

    assessment = assess_risk_trajectory(
        observations=observations,
        current_time=BASE_TIME,
    )

    assert assessment.trend in {
        RiskTrend.RISING,
        RiskTrend.RAPIDLY_RISING,
    }


def test_already_crossed_threshold_reported():
    assessment = assess_risk_trajectory(
        observations=make_series(
            [
                0.60,
                0.65,
                0.70,
                0.75,
                0.80,
                0.82,
            ]
        ),
        current_time=BASE_TIME,
    )

    high_projection = get_projection(
        assessment,
        "HIGH",
    )

    assert (
        high_projection.status
        == ThresholdStatus.ALREADY_CROSSED
    )

    assert (
        high_projection.estimated_minutes_to_crossing
        == pytest.approx(0.0)
    )


def test_rising_trend_can_project_future_threshold():
    assessment = assess_risk_trajectory(
        observations=make_series(
            [
                0.35,
                0.39,
                0.43,
                0.47,
                0.51,
                0.55,
                0.59,
                0.63,
            ]
        ),
        current_time=BASE_TIME,
    )

    high_projection = get_projection(
        assessment,
        "HIGH",
    )

    assert (
        high_projection.status
        == ThresholdStatus.PROJECTED
    )

    assert (
        high_projection.estimated_minutes_to_crossing
        is not None
    )

    assert (
        high_projection.estimated_minutes_to_crossing
        > 0.0
    )

    assert (
        high_projection.earliest_minutes
        is not None
    )

    assert (
        high_projection.latest_minutes
        is not None
    )


def test_falling_trend_does_not_project_higher_threshold():
    assessment = assess_risk_trajectory(
        observations=make_series(
            [
                0.65,
                0.60,
                0.55,
                0.50,
                0.45,
                0.40,
            ]
        ),
        current_time=BASE_TIME,
    )

    high_projection = get_projection(
        assessment,
        "HIGH",
    )

    assert (
        high_projection.status
        == ThresholdStatus.NOT_PROJECTED
    )


def test_projection_beyond_maximum_horizon_is_suppressed():
    policy = RiskTrajectoryPolicy(
        maximum_projection_minutes=20.0
    )

    assessment = assess_risk_trajectory(
        observations=make_series(
            [
                0.30,
                0.31,
                0.32,
                0.33,
                0.34,
                0.35,
                0.36,
                0.37,
            ]
        ),
        current_time=BASE_TIME,
        policy=policy,
    )

    severe_projection = get_projection(
        assessment,
        "SEVERE",
    )

    assert (
        severe_projection.status
        == ThresholdStatus.NOT_PROJECTED
    )


def test_noisy_series_can_block_projection():
    assessment = assess_risk_trajectory(
        observations=make_series(
            [
                0.30,
                0.52,
                0.34,
                0.58,
                0.39,
                0.62,
                0.41,
                0.64,
            ]
        ),
        current_time=BASE_TIME,
    )

    if (
        assessment.r_squared
        is not None
        and assessment.r_squared
        < RiskTrajectoryPolicy()
        .minimum_projection_r_squared
    ):
        assert (
            "trend_fit_too_noisy_for_projection"
            in assessment.reasons
        )

        high_projection = get_projection(
            assessment,
            "HIGH",
        )

        assert (
            high_projection.status
            != ThresholdStatus.PROJECTED
        )


def test_longer_clean_history_improves_trajectory_confidence():
    short = assess_risk_trajectory(
        observations=make_series(
            [
                0.40,
                0.44,
                0.48,
                0.52,
            ]
        ),
        current_time=BASE_TIME,
    )

    long = assess_risk_trajectory(
        observations=make_series(
            [
                0.28,
                0.32,
                0.36,
                0.40,
                0.44,
                0.48,
                0.52,
            ]
        ),
        current_time=BASE_TIME,
    )

    assert (
        long.trajectory_confidence
        >= short.trajectory_confidence
    )


def test_irregular_sampling_still_supported():
    observations = [
        RiskObservation(
            timestamp=(
                BASE_TIME
                - timedelta(
                    minutes=55
                )
            ),
            risk_score=0.30,
        ),
        RiskObservation(
            timestamp=(
                BASE_TIME
                - timedelta(
                    minutes=38
                )
            ),
            risk_score=0.38,
        ),
        RiskObservation(
            timestamp=(
                BASE_TIME
                - timedelta(
                    minutes=24
                )
            ),
            risk_score=0.45,
        ),
        RiskObservation(
            timestamp=(
                BASE_TIME
                - timedelta(
                    minutes=9
                )
            ),
            risk_score=0.53,
        ),
        RiskObservation(
            timestamp=BASE_TIME,
            risk_score=0.58,
        ),
    ]

    assessment = assess_risk_trajectory(
        observations=observations,
        current_time=BASE_TIME,
    )

    assert (
        assessment.trend
        != RiskTrend.INSUFFICIENT_DATA
    )


def test_acceleration_available_with_enough_points():
    assessment = assess_risk_trajectory(
        observations=make_series(
            [
                0.30,
                0.32,
                0.34,
                0.38,
                0.44,
                0.52,
                0.62,
                0.74,
            ]
        ),
        current_time=BASE_TIME,
    )

    assert (
        assessment.acceleration_per_minute_squared
        is not None
    )


def test_current_time_must_be_timezone_aware():
    with pytest.raises(
        ValueError,
        match=(
            "current_time must be timezone-aware"
        ),
    ):
        assess_risk_trajectory(
            observations=make_series(
                [
                    0.30,
                    0.40,
                    0.50,
                    0.60,
                ]
            ),
            current_time=datetime(
                2026,
                9,
                8,
                12,
                0,
            ),
        )


def test_invalid_policy_threshold_order_rejected():
    policy = RiskTrajectoryPolicy(
        watch_threshold=0.60,
        warning_threshold=0.40,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Risk thresholds must be ordered"
        ),
    ):
        assess_risk_trajectory(
            observations=make_series(
                [
                    0.30,
                    0.40,
                    0.50,
                    0.60,
                ]
            ),
            current_time=BASE_TIME,
            policy=policy,
        )


def test_confidence_level_is_exposed():
    assessment = assess_risk_trajectory(
        observations=make_series(
            [
                0.28,
                0.32,
                0.36,
                0.40,
                0.44,
                0.48,
                0.52,
                0.56,
            ]
        ),
        current_time=BASE_TIME,
    )

    assert assessment.confidence_level in {
        TrajectoryConfidenceLevel.HIGH,
        TrajectoryConfidenceLevel.MODERATE,
        TrajectoryConfidenceLevel.LOW,
    }