from dataclasses import dataclass
from enum import Enum


class HazardType(str, Enum):
    EXTREME_RAINFALL = "EXTREME_RAINFALL"
    SOIL_SATURATION = "SOIL_SATURATION"

    FLASH_FLOOD = "FLASH_FLOOD"
    LANDSLIDE = "LANDSLIDE"

    DRAIN_OVERLOAD = "DRAIN_OVERLOAD"
    STREAM_BLOCKAGE = "STREAM_BLOCKAGE"

    ROAD_FLOODING = "ROAD_FLOODING"
    ROAD_OBSTRUCTION = "ROAD_OBSTRUCTION"

    ROUTE_DEGRADATION = "ROUTE_DEGRADATION"


class HazardEvidenceStatus(str, Enum):
    """
    Strength/type of evidence behind one hazard state.

    OBSERVED:
        Directly observed/measured event or condition.

    PREDICTED:
        Output from a model.

    POSSIBLE_CASCADE:
        Plausible downstream consequence inferred from another
        hazard, but not independently observed or confirmed.

    AUTHORITY_CONFIRMED:
        Confirmed by competent external authority.
    """

    OBSERVED = "OBSERVED"
    PREDICTED = "PREDICTED"
    POSSIBLE_CASCADE = "POSSIBLE_CASCADE"
    AUTHORITY_CONFIRMED = "AUTHORITY_CONFIRMED"


class CascadeSeverity(str, Enum):
    LOW = "LOW"
    WATCH = "WATCH"
    WARNING = "WARNING"
    HIGH = "HIGH"
    SEVERE = "SEVERE"


@dataclass(frozen=True)
class HazardState:
    """
    One hazard/condition state entering the cascade engine.

    intensity_score:
        Normalized severity/intensity in [0, 1].

    confidence:
        Confidence in this state.

    evidence_status:
        Whether the state is observed, predicted, inferred as
        possible cascade, or authority confirmed.

    entity_id:
        Optional spatial/entity identifier such as:
            catchment
            slope zone
            drain
            road
            route
    """

    hazard_type: HazardType
    intensity_score: float
    confidence: float
    evidence_status: HazardEvidenceStatus

    entity_id: str | None = None

    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.intensity_score <= 1.0:
            raise ValueError(
                "intensity_score must be between 0 and 1."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        if (
            self.entity_id is not None
            and not self.entity_id.strip()
        ):
            raise ValueError(
                "entity_id cannot be empty when provided."
            )


@dataclass(frozen=True)
class CascadeRule:
    """
    Transparent directional relationship:

        source hazard
            ↓
        possible target hazard

    minimum_source_intensity:
        Source must reach this threshold before cascade inference.

    transfer_factor:
        Fraction of source intensity transferred to target
        possibility score.

    confidence_factor:
        Confidence degradation applied when inferring downstream
        consequence.

    This is a decision-support rule, not proof of physical event.
    """

    rule_id: str

    source_hazard: HazardType
    target_hazard: HazardType

    minimum_source_intensity: float

    transfer_factor: float
    confidence_factor: float

    reason_code: str

    def __post_init__(self) -> None:
        if not self.rule_id.strip():
            raise ValueError(
                "rule_id cannot be empty."
            )

        if not 0.0 <= self.minimum_source_intensity <= 1.0:
            raise ValueError(
                "minimum_source_intensity must be between 0 and 1."
            )

        if not 0.0 <= self.transfer_factor <= 1.0:
            raise ValueError(
                "transfer_factor must be between 0 and 1."
            )

        if not 0.0 <= self.confidence_factor <= 1.0:
            raise ValueError(
                "confidence_factor must be between 0 and 1."
            )

        if not self.reason_code.strip():
            raise ValueError(
                "reason_code cannot be empty."
            )


@dataclass(frozen=True)
class CascadeLink:
    """
    One inferred source -> target relationship.
    """

    rule_id: str

    source_hazard: HazardType
    target_hazard: HazardType

    source_entity_id: str | None

    target_intensity_score: float
    target_confidence: float

    status: HazardEvidenceStatus

    reason: str


@dataclass(frozen=True)
class HazardCascadeAssessment:
    """
    Consolidated result of cascade evaluation.
    """

    input_hazards: tuple[
        HazardState,
        ...
    ]

    inferred_hazards: tuple[
        HazardState,
        ...
    ]

    links: tuple[
        CascadeLink,
        ...
    ]

    highest_severity: CascadeSeverity

    critical_reasons: tuple[str, ...]