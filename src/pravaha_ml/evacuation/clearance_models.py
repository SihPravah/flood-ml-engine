from dataclasses import dataclass
from enum import Enum


class ClearanceFeasibility(str, Enum):
    FEASIBLE = "FEASIBLE"
    CAPACITY_INSUFFICIENT = "CAPACITY_INSUFFICIENT"
    NO_USABLE_PATH = "NO_USABLE_PATH"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ClearanceMarginStatus(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    TIGHT = "TIGHT"
    INSUFFICIENT = "INSUFFICIENT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class EvacuationFlowPath:
    """
    One usable population movement path from an exposed area
    toward a shelter.

    route_throughput_people_per_minute:
        Planning estimate of how many people can move through the
        road/corridor per minute.

    shelter_intake_people_per_minute:
        Planning estimate of how rapidly the destination can admit
        arriving evacuees.

    Effective flow is limited by the weaker bottleneck.

    available_shelter_capacity:
        Remaining known capacity at the destination.

    These values must come from explicit assumptions, operational
    planning data, field estimates, or calibrated models. PRAVAHA
    must not present them as observed facts when they are estimated.
    """

    path_id: str
    route_id: str
    shelter_id: str

    travel_time_minutes: float

    route_throughput_people_per_minute: float
    shelter_intake_people_per_minute: float

    available_shelter_capacity: int

    data_confidence: float

    usable: bool = True
    authority_closed: bool = False

    def __post_init__(self) -> None:
        if not self.path_id.strip():
            raise ValueError(
                "path_id cannot be empty."
            )

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

        if self.route_throughput_people_per_minute <= 0.0:
            raise ValueError(
                "route_throughput_people_per_minute "
                "must be greater than 0."
            )

        if self.shelter_intake_people_per_minute <= 0.0:
            raise ValueError(
                "shelter_intake_people_per_minute "
                "must be greater than 0."
            )

        if self.available_shelter_capacity < 0:
            raise ValueError(
                "available_shelter_capacity cannot be negative."
            )

        if not 0.0 <= self.data_confidence <= 1.0:
            raise ValueError(
                "data_confidence must be between 0 and 1."
            )

    @property
    def effective_throughput_people_per_minute(
        self,
    ) -> float:
        return min(
            self.route_throughput_people_per_minute,
            self.shelter_intake_people_per_minute,
        )


@dataclass(frozen=True)
class EvacuationClearanceInputs:
    """
    Inputs for clearance-time estimation.

    population_to_move:
        Estimated number of people who may need relocation.

    mobilization_delay_minutes:
        Time before meaningful movement starts.

        This may include:
            warning dissemination
            household preparation
            responder mobilization
            vehicle boarding

    hazard_window_minutes:
        Conservative time until the hazard threshold of concern.

        Prefer the EARLIEST credible threshold estimate rather than
        only the central estimate.

        None means no reliable timing estimate is available.

    data_confidence:
        Confidence in population and mobilization assumptions.
    """

    unit_id: str

    population_to_move: int

    mobilization_delay_minutes: float

    hazard_window_minutes: float | None

    data_confidence: float

    def __post_init__(self) -> None:
        if not self.unit_id.strip():
            raise ValueError(
                "unit_id cannot be empty."
            )

        if self.population_to_move < 0:
            raise ValueError(
                "population_to_move cannot be negative."
            )

        if self.mobilization_delay_minutes < 0.0:
            raise ValueError(
                "mobilization_delay_minutes cannot be negative."
            )

        if (
            self.hazard_window_minutes is not None
            and self.hazard_window_minutes < 0.0
        ):
            raise ValueError(
                "hazard_window_minutes cannot be negative."
            )

        if not 0.0 <= self.data_confidence <= 1.0:
            raise ValueError(
                "data_confidence must be between 0 and 1."
            )


@dataclass(frozen=True)
class EvacuationClearancePolicy:
    """
    Development decision-support policy.

    minimum_reliable_confidence:
        Clearance estimates below this confidence are not treated
        as operationally reliable.

    sufficient_margin_minutes:
        Minimum positive buffer considered comfortable.

    tight_margin_minutes:
        Positive buffer below this value is considered TIGHT.
    """

    minimum_reliable_confidence: float = 0.60

    sufficient_margin_minutes: float = 15.0
    tight_margin_minutes: float = 5.0

    search_tolerance_minutes: float = 0.01
    maximum_search_minutes: float = 1440.0


@dataclass(frozen=True)
class PathClearanceContribution:
    path_id: str
    route_id: str
    shelter_id: str

    effective_throughput_people_per_minute: float
    available_capacity: int

    estimated_people_moved: int


@dataclass(frozen=True)
class EvacuationClearanceAssessment:
    unit_id: str

    feasibility: ClearanceFeasibility

    estimated_clearance_minutes: float | None

    hazard_window_minutes: float | None
    clearance_margin_minutes: float | None

    margin_status: ClearanceMarginStatus

    population_to_move: int
    reachable_capacity: int

    confidence: float

    path_contributions: tuple[
        PathClearanceContribution,
        ...
    ]

    reasons: tuple[str, ...]