from datetime import datetime
from math import isfinite
from statistics import mean
from typing import Iterable

from pravaha_ml.anticipation.models import (
    RiskObservation,
    RiskTrajectoryAssessment,
    RiskTrajectoryPolicy,
    RiskTrend,
    ThresholdProjection,
    ThresholdStatus,
    TrajectoryConfidenceLevel,
)


def _validate_policy(
    policy: RiskTrajectoryPolicy,
) -> None:
    if policy.minimum_observations < 2:
        raise ValueError(
            "minimum_observations must be at least 2."
        )

    positive_fields = {
        "minimum_history_minutes": (
            policy.minimum_history_minutes
        ),
        "preferred_history_minutes": (
            policy.preferred_history_minutes
        ),
        "maximum_latest_age_minutes": (
            policy.maximum_latest_age_minutes
        ),
        "rapid_slope_per_minute": (
            policy.rapid_slope_per_minute
        ),
        "maximum_projection_minutes": (
            policy.maximum_projection_minutes
        ),
    }

    for field_name, value in positive_fields.items():
        if value <= 0.0:
            raise ValueError(
                f"{field_name} must be greater than 0."
            )

    if policy.stable_slope_per_minute < 0.0:
        raise ValueError(
            "stable_slope_per_minute cannot be negative."
        )

    if (
        policy.rapid_slope_per_minute
        <= policy.stable_slope_per_minute
    ):
        raise ValueError(
            "rapid_slope_per_minute must be greater than "
            "stable_slope_per_minute."
        )

    if not (
        0.0
        <= policy.minimum_projection_r_squared
        <= 1.0
    ):
        raise ValueError(
            "minimum_projection_r_squared must be "
            "between 0 and 1."
        )

    thresholds = [
        policy.watch_threshold,
        policy.warning_threshold,
        policy.high_threshold,
        policy.severe_threshold,
    ]

    if any(
        not 0.0 <= threshold <= 1.0
        for threshold in thresholds
    ):
        raise ValueError(
            "Risk thresholds must be between 0 and 1."
        )

    if not (
        policy.watch_threshold
        <= policy.warning_threshold
        <= policy.high_threshold
        <= policy.severe_threshold
    ):
        raise ValueError(
            "Risk thresholds must be ordered."
        )

    if not (
        0.0
        <= policy.moderate_confidence_threshold
        <= policy.high_confidence_threshold
        <= 1.0
    ):
        raise ValueError(
            "Trajectory confidence thresholds must be ordered "
            "between 0 and 1."
        )


def _prepare_observations(
    observations: Iterable[
        RiskObservation
    ],
) -> tuple[
    RiskObservation,
    ...
]:
    observations = tuple(
        sorted(
            observations,
            key=lambda item: item.timestamp,
        )
    )

    seen_timestamps: set[
        datetime
    ] = set()

    for observation in observations:
        if observation.timestamp in seen_timestamps:
            raise ValueError(
                "Duplicate risk observation timestamp detected."
            )

        seen_timestamps.add(
            observation.timestamp
        )

    return observations


def _minutes_between(
    earlier: datetime,
    later: datetime,
) -> float:
    return (
        later - earlier
    ).total_seconds() / 60.0


def _linear_fit(
    x_values: list[float],
    y_values: list[float],
) -> tuple[
    float,
    float,
    float,
]:
    """
    Ordinary least-squares linear regression.

    Returns:
        slope
        intercept
        r_squared
    """

    if len(x_values) != len(y_values):
        raise ValueError(
            "x_values and y_values must have equal length."
        )

    if len(x_values) < 2:
        raise ValueError(
            "At least two points are required for linear fit."
        )

    x_mean = mean(
        x_values
    )

    y_mean = mean(
        y_values
    )

    denominator = sum(
        (
            x_value
            - x_mean
        ) ** 2
        for x_value in x_values
    )

    if denominator <= 0.0:
        raise ValueError(
            "Risk observations must span more than one timestamp."
        )

    numerator = sum(
        (
            x_value
            - x_mean
        )
        * (
            y_value
            - y_mean
        )
        for (
            x_value,
            y_value,
        ) in zip(
            x_values,
            y_values,
        )
    )

    slope = (
        numerator
        / denominator
    )

    intercept = (
        y_mean
        - slope * x_mean
    )

    predicted = [
        intercept
        + slope * x_value
        for x_value in x_values
    ]

    residual_sum_squares = sum(
        (
            actual
            - prediction
        ) ** 2
        for (
            actual,
            prediction,
        ) in zip(
            y_values,
            predicted,
        )
    )

    total_sum_squares = sum(
        (
            actual
            - y_mean
        ) ** 2
        for actual in y_values
    )

    if total_sum_squares <= 1e-12:
        r_squared = 1.0
    else:
        r_squared = (
            1.0
            - residual_sum_squares
            / total_sum_squares
        )

    r_squared = max(
        0.0,
        min(
            float(r_squared),
            1.0,
        ),
    )

    return (
        float(slope),
        float(intercept),
        r_squared,
    )


def _classify_trend(
    slope_per_minute: float,
    policy: RiskTrajectoryPolicy,
) -> RiskTrend:
    if (
        slope_per_minute
        >= policy.rapid_slope_per_minute
    ):
        return (
            RiskTrend.RAPIDLY_RISING
        )

    if (
        slope_per_minute
        > policy.stable_slope_per_minute
    ):
        return RiskTrend.RISING

    if (
        slope_per_minute
        <= -policy.rapid_slope_per_minute
    ):
        return (
            RiskTrend.RAPIDLY_FALLING
        )

    if (
        slope_per_minute
        < -policy.stable_slope_per_minute
    ):
        return RiskTrend.FALLING

    return RiskTrend.STABLE


def _estimate_acceleration(
    *,
    minute_offsets: list[float],
    risk_scores: list[float],
) -> float | None:
    """
    Estimate whether the fitted risk slope itself is changing.

    We compare an earlier and recent half of the time series.

    This is deliberately simple and transparent. It is not a
    second-order physical forecast model.
    """

    if len(
        minute_offsets
    ) < 6:
        return None

    split_index = (
        len(minute_offsets)
        // 2
    )

    first_x = minute_offsets[
        :split_index
    ]

    first_y = risk_scores[
        :split_index
    ]

    second_x = minute_offsets[
        split_index:
    ]

    second_y = risk_scores[
        split_index:
    ]

    if (
        len(first_x) < 2
        or len(second_x) < 2
    ):
        return None

    first_slope, _, _ = _linear_fit(
        first_x,
        first_y,
    )

    second_slope, _, _ = _linear_fit(
        second_x,
        second_y,
    )

    first_midpoint = mean(
        first_x
    )

    second_midpoint = mean(
        second_x
    )

    midpoint_separation = (
        second_midpoint
        - first_midpoint
    )

    if midpoint_separation <= 0.0:
        return None

    return float(
        (
            second_slope
            - first_slope
        )
        / midpoint_separation
    )


def _sampling_regularity_score(
    observations: tuple[
        RiskObservation,
        ...
    ],
) -> float:
    if len(observations) < 3:
        return 0.5

    gaps = [
        _minutes_between(
            observations[index - 1].timestamp,
            observations[index].timestamp,
        )
        for index in range(
            1,
            len(observations),
        )
    ]

    average_gap = mean(
        gaps
    )

    if average_gap <= 0.0:
        return 0.0

    mean_absolute_deviation = mean(
        abs(
            gap
            - average_gap
        )
        for gap in gaps
    )

    irregularity = min(
        mean_absolute_deviation
        / average_gap,
        1.0,
    )

    return float(
        1.0
        - irregularity
    )


def _confidence_level(
    confidence: float,
    policy: RiskTrajectoryPolicy,
) -> TrajectoryConfidenceLevel:
    if (
        confidence
        >= policy.high_confidence_threshold
    ):
        return (
            TrajectoryConfidenceLevel.HIGH
        )

    if (
        confidence
        >= policy.moderate_confidence_threshold
    ):
        return (
            TrajectoryConfidenceLevel.MODERATE
        )

    if confidence > 0.0:
        return (
            TrajectoryConfidenceLevel.LOW
        )

    return (
        TrajectoryConfidenceLevel.INSUFFICIENT
    )


def _calculate_trajectory_confidence(
    *,
    observations: tuple[
        RiskObservation,
        ...
    ],
    history_span_minutes: float,
    latest_age_minutes: float,
    r_squared: float,
    policy: RiskTrajectoryPolicy,
) -> float:
    observation_factor = min(
        len(observations)
        / max(
            policy.minimum_observations * 2,
            1,
        ),
        1.0,
    )

    history_factor = min(
        history_span_minutes
        / policy.preferred_history_minutes,
        1.0,
    )

    freshness_factor = max(
        0.0,
        1.0
        - latest_age_minutes
        / policy.maximum_latest_age_minutes,
    )

    regularity_factor = (
        _sampling_regularity_score(
            observations
        )
    )

    confidence = (
        0.20 * observation_factor
        + 0.20 * history_factor
        + 0.25 * freshness_factor
        + 0.15 * regularity_factor
        + 0.20 * r_squared
    )

    return max(
        0.0,
        min(
            float(confidence),
            1.0,
        ),
    )


def _projection_uncertainty_fraction(
    *,
    r_squared: float,
    trajectory_confidence: float,
) -> float:
    """
    Convert weak trend evidence into wider ETA bounds.

    Best-quality trajectory:
        roughly ±10%

    Weak trajectory:
        may widen toward ±50%
    """

    evidence_quality = (
        0.50 * r_squared
        + 0.50 * trajectory_confidence
    )

    uncertainty_fraction = (
        0.50
        - 0.40 * evidence_quality
    )

    return max(
        0.10,
        min(
            uncertainty_fraction,
            0.50,
        ),
    )


def _build_projection(
    *,
    threshold_name: str,
    threshold_value: float,
    current_risk_score: float,
    slope_per_minute: float,
    r_squared: float,
    trajectory_confidence: float,
    can_project: bool,
    policy: RiskTrajectoryPolicy,
) -> ThresholdProjection:
    if current_risk_score >= threshold_value:
        return ThresholdProjection(
            threshold_name=threshold_name,
            threshold_value=threshold_value,
            status=ThresholdStatus.ALREADY_CROSSED,
            estimated_minutes_to_crossing=0.0,
            earliest_minutes=0.0,
            latest_minutes=0.0,
        )

    if not can_project:
        return ThresholdProjection(
            threshold_name=threshold_name,
            threshold_value=threshold_value,
            status=ThresholdStatus.NOT_PROJECTED,
            estimated_minutes_to_crossing=None,
            earliest_minutes=None,
            latest_minutes=None,
        )

    if slope_per_minute <= 0.0:
        return ThresholdProjection(
            threshold_name=threshold_name,
            threshold_value=threshold_value,
            status=ThresholdStatus.NOT_PROJECTED,
            estimated_minutes_to_crossing=None,
            earliest_minutes=None,
            latest_minutes=None,
        )

    estimated_minutes = (
        threshold_value
        - current_risk_score
    ) / slope_per_minute

    if (
        not isfinite(
            estimated_minutes
        )
        or estimated_minutes < 0.0
        or estimated_minutes
        > policy.maximum_projection_minutes
    ):
        return ThresholdProjection(
            threshold_name=threshold_name,
            threshold_value=threshold_value,
            status=ThresholdStatus.NOT_PROJECTED,
            estimated_minutes_to_crossing=None,
            earliest_minutes=None,
            latest_minutes=None,
        )

    uncertainty_fraction = (
        _projection_uncertainty_fraction(
            r_squared=r_squared,
            trajectory_confidence=(
                trajectory_confidence
            ),
        )
    )

    earliest = max(
        0.0,
        estimated_minutes
        * (
            1.0
            - uncertainty_fraction
        ),
    )

    latest = (
        estimated_minutes
        * (
            1.0
            + uncertainty_fraction
        )
    )

    return ThresholdProjection(
        threshold_name=threshold_name,
        threshold_value=threshold_value,
        status=ThresholdStatus.PROJECTED,
        estimated_minutes_to_crossing=float(
            estimated_minutes
        ),
        earliest_minutes=float(
            earliest
        ),
        latest_minutes=float(
            latest
        ),
    )


def _insufficient_assessment(
    *,
    observations: tuple[
        RiskObservation,
        ...
    ],
    current_time: datetime,
    reason: str,
    policy: RiskTrajectoryPolicy,
) -> RiskTrajectoryAssessment:
    current_risk_score = (
        observations[-1].risk_score
        if observations
        else 0.0
    )

    history_span = (
        _minutes_between(
            observations[0].timestamp,
            observations[-1].timestamp,
        )
        if len(observations) >= 2
        else 0.0
    )

    latest_age = (
        max(
            0.0,
            _minutes_between(
                observations[-1].timestamp,
                current_time,
            ),
        )
        if observations
        else 0.0
    )

    projections = tuple(
        ThresholdProjection(
            threshold_name=name,
            threshold_value=value,
            status=ThresholdStatus.INSUFFICIENT_DATA,
            estimated_minutes_to_crossing=None,
            earliest_minutes=None,
            latest_minutes=None,
        )
        for (
            name,
            value,
        ) in (
            (
                "WATCH",
                policy.watch_threshold,
            ),
            (
                "WARNING",
                policy.warning_threshold,
            ),
            (
                "HIGH",
                policy.high_threshold,
            ),
            (
                "SEVERE",
                policy.severe_threshold,
            ),
        )
    )

    return RiskTrajectoryAssessment(
        trend=RiskTrend.INSUFFICIENT_DATA,
        current_risk_score=float(
            current_risk_score
        ),
        slope_per_minute=None,
        acceleration_per_minute_squared=None,
        r_squared=None,
        trajectory_confidence=0.0,
        confidence_level=(
            TrajectoryConfidenceLevel.INSUFFICIENT
        ),
        history_span_minutes=float(
            history_span
        ),
        latest_age_minutes=float(
            latest_age
        ),
        observation_count=len(
            observations
        ),
        projections=projections,
        reasons=(
            reason,
        ),
    )


def assess_risk_trajectory(
    *,
    observations: Iterable[
        RiskObservation
    ],
    current_time: datetime,
    policy: RiskTrajectoryPolicy | None = None,
) -> RiskTrajectoryAssessment:
    """
    Analyze a time series of normalized risk scores.

    This function estimates:

        trend direction
        trend rate
        approximate acceleration
        short-horizon threshold crossing

    The threshold ETA is only exposed when:

        observations are sufficient
        history span is sufficient
        latest observation is fresh
        fitted trend is rising
        trend fit is sufficiently coherent

    This prevents PRAVAHA from turning noisy or stale data into
    confident evacuation-time claims.
    """

    if policy is None:
        policy = RiskTrajectoryPolicy()

    _validate_policy(
        policy
    )

    if current_time.tzinfo is None:
        raise ValueError(
            "current_time must be timezone-aware."
        )

    observations = _prepare_observations(
        observations
    )

    if len(observations) < policy.minimum_observations:
        return _insufficient_assessment(
            observations=observations,
            current_time=current_time,
            reason="insufficient_observation_count",
            policy=policy,
        )

    if (
        observations[-1].timestamp
        > current_time
    ):
        return _insufficient_assessment(
            observations=observations,
            current_time=current_time,
            reason="latest_observation_is_in_future",
            policy=policy,
        )

    history_span_minutes = (
        _minutes_between(
            observations[0].timestamp,
            observations[-1].timestamp,
        )
    )

    if (
        history_span_minutes
        < policy.minimum_history_minutes
    ):
        return _insufficient_assessment(
            observations=observations,
            current_time=current_time,
            reason="insufficient_history_span",
            policy=policy,
        )

    latest_age_minutes = max(
        0.0,
        _minutes_between(
            observations[-1].timestamp,
            current_time,
        ),
    )

    if (
        latest_age_minutes
        > policy.maximum_latest_age_minutes
    ):
        return _insufficient_assessment(
            observations=observations,
            current_time=current_time,
            reason="risk_history_stale",
            policy=policy,
        )

    first_timestamp = (
        observations[0].timestamp
    )

    minute_offsets = [
        _minutes_between(
            first_timestamp,
            observation.timestamp,
        )
        for observation in observations
    ]

    risk_scores = [
        observation.risk_score
        for observation in observations
    ]

    (
        slope_per_minute,
        _,
        r_squared,
    ) = _linear_fit(
        minute_offsets,
        risk_scores,
    )

    acceleration = (
        _estimate_acceleration(
            minute_offsets=minute_offsets,
            risk_scores=risk_scores,
        )
    )

    trend = _classify_trend(
        slope_per_minute,
        policy,
    )

    trajectory_confidence = (
        _calculate_trajectory_confidence(
            observations=observations,
            history_span_minutes=(
                history_span_minutes
            ),
            latest_age_minutes=(
                latest_age_minutes
            ),
            r_squared=r_squared,
            policy=policy,
        )
    )

    confidence_level = (
        _confidence_level(
            trajectory_confidence,
            policy,
        )
    )

    reasons: list[str] = []

    if trend == RiskTrend.RAPIDLY_RISING:
        reasons.append(
            "risk_rising_rapidly"
        )

    elif trend == RiskTrend.RISING:
        reasons.append(
            "risk_rising"
        )

    elif trend == RiskTrend.STABLE:
        reasons.append(
            "risk_stable"
        )

    elif trend == RiskTrend.FALLING:
        reasons.append(
            "risk_falling"
        )

    elif trend == RiskTrend.RAPIDLY_FALLING:
        reasons.append(
            "risk_falling_rapidly"
        )

    if (
        acceleration is not None
        and acceleration > 0.0001
    ):
        reasons.append(
            "risk_increase_accelerating"
        )

    if (
        r_squared
        < policy.minimum_projection_r_squared
    ):
        reasons.append(
            "trend_fit_too_noisy_for_projection"
        )

    if (
        trajectory_confidence
        < policy.moderate_confidence_threshold
    ):
        reasons.append(
            "trajectory_confidence_low"
        )

    can_project = (
        trend
        in {
            RiskTrend.RISING,
            RiskTrend.RAPIDLY_RISING,
        }
        and r_squared
        >= policy.minimum_projection_r_squared
        and trajectory_confidence
        >= policy.moderate_confidence_threshold
    )

    current_risk_score = (
        observations[-1].risk_score
    )

    projections = tuple(
        _build_projection(
            threshold_name=name,
            threshold_value=value,
            current_risk_score=current_risk_score,
            slope_per_minute=slope_per_minute,
            r_squared=r_squared,
            trajectory_confidence=(
                trajectory_confidence
            ),
            can_project=can_project,
            policy=policy,
        )
        for (
            name,
            value,
        ) in (
            (
                "WATCH",
                policy.watch_threshold,
            ),
            (
                "WARNING",
                policy.warning_threshold,
            ),
            (
                "HIGH",
                policy.high_threshold,
            ),
            (
                "SEVERE",
                policy.severe_threshold,
            ),
        )
    )

    if any(
        projection.status
        == ThresholdStatus.PROJECTED
        for projection in projections
    ):
        reasons.append(
            "future_risk_threshold_projected"
        )

    return RiskTrajectoryAssessment(
        trend=trend,
        current_risk_score=float(
            current_risk_score
        ),
        slope_per_minute=float(
            slope_per_minute
        ),
        acceleration_per_minute_squared=(
            float(acceleration)
            if acceleration is not None
            else None
        ),
        r_squared=float(
            r_squared
        ),
        trajectory_confidence=float(
            trajectory_confidence
        ),
        confidence_level=confidence_level,
        history_span_minutes=float(
            history_span_minutes
        ),
        latest_age_minutes=float(
            latest_age_minutes
        ),
        observation_count=len(
            observations
        ),
        projections=projections,
        reasons=tuple(
            reasons
        ),
    )