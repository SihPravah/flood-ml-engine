from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from pravaha_ml.drainage.network import (
    DrainFlowState,
)
from pravaha_ml.features.hydrology_features import (
    HydrologyFeatures,
)
from pravaha_ml.inference.confidence import (
    ConfidenceAssessment,
)
from pravaha_ml.roads.spatial import (
    SpatialRoadRiskResult,
)


class CityOperationalStatus(str, Enum):
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    EMERGENCY = "EMERGENCY"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class CatchmentIntelligence:
    """
    Map-facing internal intelligence representation of one
    catchment.

    This is NOT yet a shared backend/frontend API contract.
    """

    catchment_id: str

    risk_score: float
    risk_level: str

    confidence: ConfidenceAssessment

    hydrology: HydrologyFeatures

    affected_population: int | None = None

    def __post_init__(self) -> None:
        if not self.catchment_id.strip():
            raise ValueError(
                "catchment_id cannot be empty."
            )

        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError(
                "risk_score must be between 0 and 1."
            )

        if not self.risk_level.strip():
            raise ValueError(
                "risk_level cannot be empty."
            )

        if (
            self.affected_population is not None
            and self.affected_population < 0
        ):
            raise ValueError(
                "affected_population cannot be negative."
            )


@dataclass(frozen=True)
class CityIntelligenceSummary:
    generated_at: datetime

    catchment_count: int
    drain_count: int
    road_count: int

    high_risk_catchments: int
    overflowing_drains: int
    roads_to_avoid: int
    confirmed_road_closures: int

    low_confidence_catchments: int
    low_confidence_roads: int

    operational_status: CityOperationalStatus


@dataclass(frozen=True)
class CityIntelligenceState:
    """
    Internal consolidated state used by future backend/map layers.

    It provides a single point from which the UI can retrieve
    detailed intelligence for:

        catchments
        drains
        roads
        overall city status
    """

    generated_at: datetime

    catchments: tuple[
        CatchmentIntelligence,
        ...
    ]

    drains: tuple[
        DrainFlowState,
        ...
    ]

    roads: tuple[
        SpatialRoadRiskResult,
        ...
    ]

    summary: CityIntelligenceSummary