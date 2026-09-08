from dataclasses import dataclass
from enum import Enum


class OperationalState(str, Enum):
    NORMAL = "NORMAL"
    WATCH = "WATCH"
    PREPARE = "PREPARE"
    WARNING = "WARNING"
    EVACUATION_SUPPORT = "EVACUATION_SUPPORT"
    ACTIVE_EVENT = "ACTIVE_EVENT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class AuthorityEventStatus(str, Enum):
    NONE = "NONE"
    WATCH = "WATCH"
    WARNING = "WARNING"
    EVACUATION_ORDER = "EVACUATION_ORDER"
    EVENT_CONFIRMED = "EVENT_CONFIRMED"


class DecisionConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True)
class AnticipatoryDecisionInputs:
    """
    Consolidated decision-support inputs for one ward/village.

    impact_score:
        Current local impact assessment in [0, 1].

    trajectory_rising:
        Whether the recent risk trajectory is rising.

    trajectory_rapidly_rising:
        Stronger escalation indicator.

    projected_high_threshold_minutes:
        Earliest credible estimate of when HIGH risk may be crossed.

    forecast_worst_consequence_score:
        Highest credible consequence score from evaluated
        forecast scenarios.

    forecast_worsening:
        Whether forecast scenarios indicate worsening conditions.

    cascade_intensity_score:
        Current or inferred multi-hazard cascade intensity.

    evacuation_ready:
        Whether current evacuation readiness is operationally
        sufficient.

    evacuation_clearance_minutes:
        Estimated time required to clear the exposed population.

    hazard_window_minutes:
        Conservative time available before the hazard threshold
        of concern.

    authority_event_status:
        Explicit external authority state.

    event_observed:
        True only when the hazardous event is directly observed.

    system_confidence:
        Consolidated confidence of the decision inputs.
    """

    unit_id: str

    impact_score: float

    trajectory_rising: bool
    trajectory_rapidly_rising: bool

    projected_high_threshold_minutes: float | None

    forecast_worst_consequence_score: float
    forecast_worsening: bool

    cascade_intensity_score: float

    evacuation_ready: bool

    evacuation_clearance_minutes: float | None
    hazard_window_minutes: float | None

    authority_event_status: AuthorityEventStatus

    event_observed: bool

    system_confidence: float

    def __post_init__(self) -> None:
        if not self.unit_id.strip():
            raise ValueError(
                "unit_id cannot be empty."
            )

        normalized = {
            "impact_score": self.impact_score,
            "forecast_worst_consequence_score": (
                self.forecast_worst_consequence_score
            ),
            "cascade_intensity_score": (
                self.cascade_intensity_score
            ),
            "system_confidence": (
                self.system_confidence
            ),
        }

        for field_name, value in normalized.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} must be between 0 and 1."
                )

        optional_times = {
            "projected_high_threshold_minutes": (
                self.projected_high_threshold_minutes
            ),
            "evacuation_clearance_minutes": (
                self.evacuation_clearance_minutes
            ),
            "hazard_window_minutes": (
                self.hazard_window_minutes
            ),
        }

        for field_name, value in optional_times.items():
            if value is not None and value < 0.0:
                raise ValueError(
                    f"{field_name} cannot be negative."
                )


@dataclass(frozen=True)
class AnticipatoryDecisionPolicy:
    minimum_reliable_confidence: float = 0.60

    watch_impact_threshold: float = 0.30
    warning_impact_threshold: float = 0.50
    high_impact_threshold: float = 0.70

    severe_forecast_threshold: float = 0.85
    significant_cascade_threshold: float = 0.60

    prepare_horizon_minutes: float = 60.0
    evacuation_support_horizon_minutes: float = 45.0

    tight_clearance_margin_minutes: float = 10.0

    high_confidence_threshold: float = 0.80
    moderate_confidence_threshold: float = 0.60


@dataclass(frozen=True)
class AnticipatoryDecision:
    unit_id: str

    operational_state: OperationalState

    confidence: float
    confidence_level: DecisionConfidenceLevel

    evacuation_margin_minutes: float | None

    authority_event_status: AuthorityEventStatus

    recommendation_code: str

    reasons: tuple[str, ...]