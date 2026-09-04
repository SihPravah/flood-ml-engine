from dataclasses import dataclass
from enum import Enum


class RoadFloodRiskLevel(str, Enum):
    LOW = "LOW"
    WATCH = "WATCH"
    WARNING = "WARNING"
    HIGH = "HIGH"
    SEVERE = "SEVERE"


class RoadRecommendation(str, Enum):
    """
    Operational recommendation for a road segment.

    PASSABLE:
        Current evidence supports relatively low flood risk.

    CAUTION:
        Elevated risk or insufficient confidence exists.

    AVOID:
        PRAVAHA recommends excluding this road from normal
        safe-route selection.

    CLOSED:
        Reserved only for an externally confirmed/authoritative
        road closure.
    """

    PASSABLE = "PASSABLE"
    CAUTION = "CAUTION"
    AVOID = "AVOID"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class RoadFloodInputs:
    """
    Dynamic and static evidence used to assess one road segment.

    All *_score and *_probability fields are normalized to [0, 1].

    nearest_drain_distance_m:
        Metric distance from this road segment to its most relevant
        drain/overflow source.

        IMPORTANT:
        This must eventually be calculated in an appropriate
        projected metric CRS. Do not use raw longitude/latitude
        Shapely distance and interpret degrees as metres.

    terrain_depression_score:
        0 -> not locally depressed
        1 -> strong local low-point/depression

    stream_proximity_score:
        Higher when the road is vulnerable to nearby river/stream
        influence.

    historical_waterlogging_score:
        Evidence from known historical waterlogging/flood events.

    road_surface_vulnerability_score:
        Static susceptibility caused by local road geometry,
        underpasses, poor surface drainage, etc.

    data_confidence:
        Reliability of the combined evidence.

    authority_closed:
        True only when closure is explicitly provided by a trusted
        external/authority source.
    """

    road_id: str

    catchment_risk_score: float

    drain_overflow_probability: float
    drain_capacity_utilization: float
    drain_overflow_discharge_m3_per_s: float
    nearest_drain_distance_m: float

    terrain_depression_score: float
    stream_proximity_score: float
    historical_waterlogging_score: float
    road_surface_vulnerability_score: float

    data_confidence: float

    authority_closed: bool = False

    def __post_init__(self) -> None:
        if not self.road_id.strip():
            raise ValueError(
                "road_id cannot be empty."
            )

        normalized_fields = {
            "catchment_risk_score": (
                self.catchment_risk_score
            ),
            "drain_overflow_probability": (
                self.drain_overflow_probability
            ),
            "terrain_depression_score": (
                self.terrain_depression_score
            ),
            "stream_proximity_score": (
                self.stream_proximity_score
            ),
            "historical_waterlogging_score": (
                self.historical_waterlogging_score
            ),
            "road_surface_vulnerability_score": (
                self.road_surface_vulnerability_score
            ),
            "data_confidence": (
                self.data_confidence
            ),
        }

        for field_name, value in normalized_fields.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} must be between 0 and 1."
                )

        if self.drain_capacity_utilization < 0.0:
            raise ValueError(
                "drain_capacity_utilization cannot be negative."
            )

        if self.drain_overflow_discharge_m3_per_s < 0.0:
            raise ValueError(
                "drain_overflow_discharge_m3_per_s "
                "cannot be negative."
            )

        if self.nearest_drain_distance_m < 0.0:
            raise ValueError(
                "nearest_drain_distance_m cannot be negative."
            )


@dataclass(frozen=True)
class RoadFloodRiskPolicy:
    """
    Development-time weighting policy.

    These are explicit heuristic weights, not scientifically
    calibrated universal constants.

    They must eventually be calibrated using real observed
    road-waterlogging/flood-event data.
    """

    drainage_weight: float = 0.30
    catchment_weight: float = 0.20
    terrain_weight: float = 0.20
    historical_weight: float = 0.15
    stream_weight: float = 0.10
    road_vulnerability_weight: float = 0.05

    drain_distance_decay_m: float = 75.0

    minimum_passable_confidence: float = 0.60

    watch_threshold: float = 0.30
    warning_threshold: float = 0.50
    high_threshold: float = 0.70
    severe_threshold: float = 0.85


@dataclass(frozen=True)
class RoadFloodAssessment:
    road_id: str

    risk_score: float
    risk_level: RoadFloodRiskLevel

    recommendation: RoadRecommendation

    data_confidence: float

    drainage_exposure_score: float
    catchment_component: float
    terrain_component: float
    historical_component: float
    stream_component: float
    road_vulnerability_component: float

    nearest_drain_distance_m: float

    inferred_avoidance: bool
    authoritative_closure: bool

    reasons: tuple[str, ...]