from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterable

from pravaha_ml.features.rainfall import (
    RainfallObservation,
)


class TemporalQualityLevel(str, Enum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    UNUSABLE = "UNUSABLE"


@dataclass(frozen=True)
class TemporalQualityPolicy:
    """
    Policy controlling whether rainfall history is sufficiently
    fresh and continuous for hydrological feature generation.

    max_gap_minutes:
        Maximum acceptable gap between consecutive rainfall
        observations.

    max_staleness_minutes:
        Maximum acceptable age of the latest rainfall observation
        relative to prediction_time.

    degraded_gap_fraction:
        Fraction of max_gap_minutes at which data begins to be
        marked DEGRADED.

    degraded_staleness_fraction:
        Fraction of max_staleness_minutes at which data begins
        to be marked DEGRADED.

    These are development policy parameters. They must eventually
    be calibrated against real source cadence and operational
    requirements.
    """

    max_gap_minutes: float = 30.0
    max_staleness_minutes: float = 15.0

    degraded_gap_fraction: float = 0.70
    degraded_staleness_fraction: float = 0.70


@dataclass(frozen=True)
class TemporalQualityReport:
    level: TemporalQualityLevel

    observation_count: int

    latest_observation_age_minutes: float
    largest_gap_minutes: float

    has_future_observation: bool
    has_duplicate_timestamp: bool

    stale: bool
    excessive_gap: bool

    can_predict: bool

    reasons: tuple[str, ...]


def _validate_policy(
    policy: TemporalQualityPolicy,
) -> None:
    if policy.max_gap_minutes <= 0.0:
        raise ValueError(
            "max_gap_minutes must be greater than 0."
        )

    if policy.max_staleness_minutes <= 0.0:
        raise ValueError(
            "max_staleness_minutes must be greater than 0."
        )

    if not 0.0 < policy.degraded_gap_fraction <= 1.0:
        raise ValueError(
            "degraded_gap_fraction must be between 0 and 1."
        )

    if not (
        0.0
        < policy.degraded_staleness_fraction
        <= 1.0
    ):
        raise ValueError(
            "degraded_staleness_fraction must be "
            "between 0 and 1."
        )


def _validate_prediction_time(
    prediction_time: datetime,
) -> None:
    if prediction_time.tzinfo is None:
        raise ValueError(
            "prediction_time must be timezone-aware."
        )


def _validate_observations(
    observations: Iterable[
        RainfallObservation
    ],
) -> list[RainfallObservation]:
    observations = list(
        observations
    )

    if not observations:
        raise ValueError(
            "At least one rainfall observation is required."
        )

    for observation in observations:
        if observation.timestamp.tzinfo is None:
            raise ValueError(
                "Rainfall observation timestamps must "
                "be timezone-aware."
            )

        if observation.rainfall_mm_per_hr < 0.0:
            raise ValueError(
                "rainfall_mm_per_hr cannot be negative."
            )

    return sorted(
        observations,
        key=lambda observation: observation.timestamp,
    )


def evaluate_temporal_quality(
    observations: Iterable[
        RainfallObservation
    ],
    prediction_time: datetime,
    policy: TemporalQualityPolicy | None = None,
) -> TemporalQualityReport:
    """
    Evaluate rainfall observation quality before hydrology
    calculations are allowed to proceed.

    UNUSABLE conditions:

        - future-dated observation
        - duplicate timestamps
        - latest observation is too stale
        - gap between observations exceeds policy

    DEGRADED means data is approaching the configured limit
    but remains usable.

    GOOD means no temporal-quality warning was detected.
    """

    if policy is None:
        policy = TemporalQualityPolicy()

    _validate_policy(
        policy
    )

    _validate_prediction_time(
        prediction_time
    )

    ordered = _validate_observations(
        observations
    )

    reasons: list[str] = []

    has_future_observation = any(
        observation.timestamp
        > prediction_time
        for observation in ordered
    )

    timestamps = [
        observation.timestamp
        for observation in ordered
    ]

    has_duplicate_timestamp = (
        len(timestamps)
        != len(set(timestamps))
    )

    if has_future_observation:
        reasons.append(
            "future_observation"
        )

    if has_duplicate_timestamp:
        reasons.append(
            "duplicate_timestamp"
        )

    latest_timestamp = max(
        timestamps
    )

    latest_observation_age_minutes = (
        prediction_time
        - latest_timestamp
    ).total_seconds() / 60.0

    if (
        latest_observation_age_minutes
        < 0.0
    ):
        latest_observation_age_minutes = 0.0

    gaps_minutes: list[float] = []

    for previous, current in zip(
        ordered,
        ordered[1:],
    ):
        gap_minutes = (
            current.timestamp
            - previous.timestamp
        ).total_seconds() / 60.0

        gaps_minutes.append(
            gap_minutes
        )

    largest_gap_minutes = (
        max(gaps_minutes)
        if gaps_minutes
        else 0.0
    )

    stale = (
        latest_observation_age_minutes
        > policy.max_staleness_minutes
    )

    excessive_gap = (
        largest_gap_minutes
        > policy.max_gap_minutes
    )

    if stale:
        reasons.append(
            "stale_latest_observation"
        )

    if excessive_gap:
        reasons.append(
            "excessive_observation_gap"
        )

    unusable = (
        has_future_observation
        or has_duplicate_timestamp
        or stale
        or excessive_gap
    )

    if unusable:
        return TemporalQualityReport(
            level=TemporalQualityLevel.UNUSABLE,
            observation_count=len(
                ordered
            ),
            latest_observation_age_minutes=float(
                latest_observation_age_minutes
            ),
            largest_gap_minutes=float(
                largest_gap_minutes
            ),
            has_future_observation=(
                has_future_observation
            ),
            has_duplicate_timestamp=(
                has_duplicate_timestamp
            ),
            stale=stale,
            excessive_gap=excessive_gap,
            can_predict=False,
            reasons=tuple(
                reasons
            ),
        )

    degraded_gap_threshold = (
        policy.max_gap_minutes
        * policy.degraded_gap_fraction
    )

    degraded_staleness_threshold = (
        policy.max_staleness_minutes
        * policy.degraded_staleness_fraction
    )

    degraded_gap = (
        largest_gap_minutes
        >= degraded_gap_threshold
    )

    degraded_staleness = (
        latest_observation_age_minutes
        >= degraded_staleness_threshold
    )

    if degraded_gap:
        reasons.append(
            "observation_gap_near_limit"
        )

    if degraded_staleness:
        reasons.append(
            "latest_observation_near_stale_limit"
        )

    if degraded_gap or degraded_staleness:
        level = (
            TemporalQualityLevel.DEGRADED
        )
    else:
        level = (
            TemporalQualityLevel.GOOD
        )

    return TemporalQualityReport(
        level=level,
        observation_count=len(
            ordered
        ),
        latest_observation_age_minutes=float(
            latest_observation_age_minutes
        ),
        largest_gap_minutes=float(
            largest_gap_minutes
        ),
        has_future_observation=False,
        has_duplicate_timestamp=False,
        stale=False,
        excessive_gap=False,
        can_predict=True,
        reasons=tuple(
            reasons
        ),
    )


def calculate_time_aware_api(
    observations: Iterable[
        RainfallObservation
    ],
    prediction_time: datetime,
    decay_factor_per_hour: float = 0.90,
) -> float:
    """
    Calculate antecedent precipitation using elapsed-time-aware
    exponential decay.

    Previous API decays according to the actual duration of each
    rainfall interval:

        decay = decay_factor_per_hour ** duration_hours

    This avoids treating a 5-minute gap and a 2-hour gap as the
    same amount of elapsed time.

    Rainfall intensity is interpreted using zero-order hold within
    each accepted observation interval.

    Large or stale intervals should therefore be screened using
    evaluate_temporal_quality before this function is used.
    """

    if not 0.0 <= decay_factor_per_hour <= 1.0:
        raise ValueError(
            "decay_factor_per_hour must be between 0 and 1."
        )

    _validate_prediction_time(
        prediction_time
    )

    ordered = _validate_observations(
        observations
    )

    if any(
        observation.timestamp
        > prediction_time
        for observation in ordered
    ):
        raise ValueError(
            "Rainfall observations cannot occur after "
            "prediction_time."
        )

    timestamps = [
        observation.timestamp
        for observation in ordered
    ]

    if len(timestamps) != len(
        set(timestamps)
    ):
        raise ValueError(
            "Rainfall observations cannot contain "
            "duplicate timestamps."
        )

    api_mm = 0.0

    for index, observation in enumerate(
        ordered
    ):
        interval_start = (
            observation.timestamp
        )

        if index + 1 < len(
            ordered
        ):
            interval_end = (
                ordered[
                    index + 1
                ].timestamp
            )
        else:
            interval_end = (
                prediction_time
            )

        if interval_end <= interval_start:
            continue

        duration_hours = (
            interval_end
            - interval_start
        ).total_seconds() / 3600.0

        rainfall_depth_mm = (
            observation.rainfall_mm_per_hr
            * duration_hours
        )

        decay = (
            decay_factor_per_hour
            ** duration_hours
        )

        api_mm = (
            api_mm * decay
            + rainfall_depth_mm
        )

    return float(
        api_mm
    )