from dataclasses import dataclass
from enum import Enum


class ShelterStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    LIMITED = "LIMITED"
    FULL = "FULL"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


class EvacuationRouteStatus(str, Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    UNSAFE = "UNSAFE"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


class EvacuationReadinessLevel(str, Enum):
    READY = "READY"
    PREPARE = "PREPARE"
    DEGRADED = "DEGRADED"
    NOT_READY = "NOT_READY"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class ShelterOption:
    """
    One candidate shelter for an affected ward/village.

    total_capacity:
        Maximum supported occupancy.

    current_occupancy:
        Current known occupancy.

    data_confidence:
        Confidence in shelter availability/capacity information.

    PRAVAHA must not invent shelter capacity when it is unknown.
    """

    shelter_id: str
    shelter_name: str

    total_capacity: int | None
    current_occupancy: int | None

    status: ShelterStatus

    data_confidence: float

    def __post_init__(self) -> None:
        if not self.shelter_id.strip():
            raise ValueError(
                "shelter_id cannot be empty."
            )

        if not self.shelter_name.strip():
            raise ValueError(
                "shelter_name cannot be empty."
            )

        if (
            self.total_capacity is not None
            and self.total_capacity < 0
        ):
            raise ValueError(
                "total_capacity cannot be negative."
            )

        if (
            self.current_occupancy is not None
            and self.current_occupancy < 0
        ):
            raise ValueError(
                "current_occupancy cannot be negative."
            )

        if (
            self.total_capacity is not None
            and self.current_occupancy is not None
            and self.current_occupancy > self.total_capacity
        ):
            raise ValueError(
                "current_occupancy cannot exceed total_capacity."
            )

        if not 0.0 <= self.data_confidence <= 1.0:
            raise ValueError(
                "data_confidence must be between 0 and 1."
            )

    @property
    def available_capacity(
        self,
    ) -> int | None:
        if (
            self.total_capacity is None
            or self.current_occupancy is None
        ):
            return None

        return max(
            self.total_capacity
            - self.current_occupancy,
            0,
        )


@dataclass(frozen=True)
class EvacuationRouteOption:
    """
    Candidate route from affected area toward a shelter.

    route_status:
        Current PRAVAHA/authority route state.

    travel_time_minutes:
        Estimated route travel time.

    minimum_confidence:
        Minimum confidence among route segments.

    maximum_risk_score:
        Highest flood/landslide related risk on the route.

    authority_closed:
        Explicit authoritative closure signal.

    A route may be AI-marked UNSAFE without being authority CLOSED.
    """

    route_id: str
    shelter_id: str

    route_status: EvacuationRouteStatus

    travel_time_minutes: float

    maximum_risk_score: float
    minimum_confidence: float

    authority_closed: bool = False

    def __post_init__(self) -> None:
        if not self.route_id.strip():
            raise ValueError(
                "route_id cannot be empty."
            )

        if not self.shelter_id.strip():
            raise ValueError(
                "shelter_id cannot be empty."
            )

        if self.travel_time_minutes <= 0.0:
            raise ValueError(
                "travel_time_minutes must be greater than 0."
            )

        if not 0.0 <= self.maximum_risk_score <= 1.0:
            raise ValueError(
                "maximum_risk_score must be between 0 and 1."
            )

        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence must be between 0 and 1."
            )


@dataclass(frozen=True)
class EvacuationReadinessInputs:
    """
    Consolidated inputs for one administrative area.

    population_exposed:
        Number potentially requiring relocation.

        None means unknown.

    current_impact_score:
        Current ward/village impact score.

    current_impact_confidence:
        Confidence in that assessment.

    projected_threshold_minutes:
        Best estimate of minutes until an important future hazard
        threshold is crossed.

        None means no reliable threshold ETA is available.

    projected_threshold_earliest_minutes:
        Conservative early edge of the threshold window.

    trajectory_confidence:
        Confidence in the threshold trajectory.

    preparation_buffer_minutes:
        Operational buffer reserved before the predicted hazard
        threshold. This prevents treating the entire ETA as usable
        evacuation time.
    """

    unit_id: str

    population_exposed: int | None

    current_impact_score: float
    current_impact_confidence: float

    projected_threshold_minutes: float | None
    projected_threshold_earliest_minutes: float | None
    trajectory_confidence: float

    preparation_buffer_minutes: float = 10.0

    def __post_init__(self) -> None:
        if not self.unit_id.strip():
            raise ValueError(
                "unit_id cannot be empty."
            )

        if (
            self.population_exposed is not None
            and self.population_exposed < 0
        ):
            raise ValueError(
                "population_exposed cannot be negative."
            )

        normalized = {
            "current_impact_score": (
                self.current_impact_score
            ),
            "current_impact_confidence": (
                self.current_impact_confidence
            ),
            "trajectory_confidence": (
                self.trajectory_confidence
            ),
        }

        for field_name, value in normalized.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} must be between 0 and 1."
                )

        optional_times = {
            "projected_threshold_minutes": (
                self.projected_threshold_minutes
            ),
            "projected_threshold_earliest_minutes": (
                self.projected_threshold_earliest_minutes
            ),
        }

        for field_name, value in optional_times.items():
            if value is not None and value < 0.0:
                raise ValueError(
                    f"{field_name} cannot be negative."
                )

        if self.preparation_buffer_minutes < 0.0:
            raise ValueError(
                "preparation_buffer_minutes cannot be negative."
            )


@dataclass(frozen=True)
class EvacuationReadinessPolicy:
    minimum_reliable_confidence: float = 0.60

    safe_route_maximum_risk: float = 0.50
    safe_route_minimum_confidence: float = 0.60

    minimum_capacity_ratio_ready: float = 1.00
    minimum_capacity_ratio_degraded: float = 0.50

    urgent_threshold_minutes: float = 45.0


@dataclass(frozen=True)
class ShelterReadiness:
    shelter_id: str
    shelter_name: str

    available_capacity: int | None

    reachable: bool

    best_route_id: str | None
    best_route_status: EvacuationRouteStatus | None

    best_route_travel_time_minutes: float | None

    reasons: tuple[str, ...]


@dataclass(frozen=True)
class EvacuationReadinessAssessment:
    unit_id: str

    readiness_level: EvacuationReadinessLevel

    confidence: float

    population_exposed: int | None

    reachable_known_capacity: int
    capacity_ratio: float | None

    usable_time_window_minutes: float | None

    fastest_safe_route_minutes: float | None

    shelter_readiness: tuple[
        ShelterReadiness,
        ...
    ]

    reasons: tuple[str, ...]