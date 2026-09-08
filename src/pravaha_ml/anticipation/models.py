from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class RiskTrend(str, Enum):
    RAPIDLY_FALLING = "RAPIDLY_FALLING"
    FALLING = "FALLING"
    STABLE = "STABLE"
    RISING = "RISING"
    RAPIDLY_RISING = "RAPIDLY_RISING"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ThresholdStatus(str, Enum):
    ALREADY_CROSSED = "ALREADY_CROSSED"
    PROJECTED = "PROJECTED"
    NOT_PROJECTED = "NOT_PROJECTED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class TrajectoryConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True)
class RiskObservation:
    """
    One timestamped risk observation.

    risk_score must represent the same risk quantity throughout
    one trajectory series.

    Examples:
        catchment flash-flood risk
        landslide susceptibility
        road flood risk

    Do NOT mix different quantities in the same series.
    """

    timestamp: datetime
    risk_score: float

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware."
            )

        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError(
                "risk_score must be between 0 and 1."
            )


@dataclass(frozen=True)
class RiskTrajectoryPolicy:
    """
    Development policy for trend and threshold estimation.

    This is intentionally conservative.

    Threshold forecasts are short-horizon decision-support
    estimates, not guarantees that a threshold will be crossed.
    """

    minimum_observations: int = 4

    minimum_history_minutes: float = 15.0
    preferred_history_minutes: float = 60.0

    maximum_latest_age_minutes: float = 15.0

    stable_slope_per_minute: float = 0.001
    rapid_slope_per_minute: float = 0.006

    minimum_projection_r_squared: float = 0.55

    maximum_projection_minutes: float = 120.0

    watch_threshold: float = 0.30
    warning_threshold: float = 0.50
    high_threshold: float = 0.70
    severe_threshold: float = 0.85

    moderate_confidence_threshold: float = 0.55
    high_confidence_threshold: float = 0.78


@dataclass(frozen=True)
class ThresholdProjection:
    threshold_name: str
    threshold_value: float

    status: ThresholdStatus

    estimated_minutes_to_crossing: float | None

    earliest_minutes: float | None
    latest_minutes: float | None


@dataclass(frozen=True)
class RiskTrajectoryAssessment:
    """
    Result of analyzing one risk-score history.

    slope_per_minute:
        Fitted change in normalized risk score per minute.

    acceleration_per_minute_squared:
        Difference between recent and earlier fitted slopes,
        normalized by the time separating those windows.

    r_squared:
        Goodness-of-fit of the linear trend.

    trajectory_confidence:
        Confidence in the trajectory interpretation itself.

    This is separate from the underlying model's prediction
    confidence.
    """

    trend: RiskTrend

    current_risk_score: float

    slope_per_minute: float | None
    acceleration_per_minute_squared: float | None

    r_squared: float | None

    trajectory_confidence: float
    confidence_level: TrajectoryConfidenceLevel

    history_span_minutes: float
    latest_age_minutes: float

    observation_count: int

    projections: tuple[
        ThresholdProjection,
        ...
    ]

    reasons: tuple[str, ...]