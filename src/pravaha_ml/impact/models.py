from dataclasses import dataclass
from enum import Enum


class AdministrativeUnitType(str, Enum):
    WARD = "WARD"
    VILLAGE = "VILLAGE"


class ImpactLevel(str, Enum):
    LOW = "LOW"
    WATCH = "WATCH"
    WARNING = "WARNING"
    HIGH = "HIGH"
    SEVERE = "SEVERE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class RouteReadiness(str, Enum):
    AVAILABLE = "AVAILABLE"
    DEGRADED = "DEGRADED"
    UNSAFE = "UNSAFE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class AdministrativeExposure:
    """
    Static/dynamic exposure information for one ward or village.

    These values must come from explicit GIS/demographic sources.

    PRAVAHA must not silently invent population or facility counts.
    """

    unit_id: str
    unit_name: str
    unit_type: AdministrativeUnitType

    population: int | None

    road_count: int
    drain_count: int
    critical_facility_count: int

    data_confidence: float

    def __post_init__(self) -> None:
        if not self.unit_id.strip():
            raise ValueError(
                "unit_id cannot be empty."
            )

        if not self.unit_name.strip():
            raise ValueError(
                "unit_name cannot be empty."
            )

        if (
            self.population is not None
            and self.population < 0
        ):
            raise ValueError(
                "population cannot be negative."
            )

        count_fields = {
            "road_count": self.road_count,
            "drain_count": self.drain_count,
            "critical_facility_count": (
                self.critical_facility_count
            ),
        }

        for field_name, value in count_fields.items():
            if value < 0:
                raise ValueError(
                    f"{field_name} cannot be negative."
                )

        if not 0.0 <= self.data_confidence <= 1.0:
            raise ValueError(
                "data_confidence must be between 0 and 1."
            )


@dataclass(frozen=True)
class WardHazardInputs:
    """
    Hazard and infrastructure state relevant to one ward/village.
    """

    flood_risk_score: float
    flood_confidence: float

    landslide_risk_score: float
    landslide_confidence: float

    cascade_intensity_score: float
    cascade_confidence: float

    high_risk_road_count: int
    overflowing_drain_count: int

    route_readiness: RouteReadiness

    def __post_init__(self) -> None:
        normalized = {
            "flood_risk_score": self.flood_risk_score,
            "flood_confidence": self.flood_confidence,
            "landslide_risk_score": self.landslide_risk_score,
            "landslide_confidence": self.landslide_confidence,
            "cascade_intensity_score": (
                self.cascade_intensity_score
            ),
            "cascade_confidence": (
                self.cascade_confidence
            ),
        }

        for field_name, value in normalized.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} must be between 0 and 1."
                )

        if self.high_risk_road_count < 0:
            raise ValueError(
                "high_risk_road_count cannot be negative."
            )

        if self.overflowing_drain_count < 0:
            raise ValueError(
                "overflowing_drain_count cannot be negative."
            )


@dataclass(frozen=True)
class WardImpactPolicy:
    flood_weight: float = 0.40
    landslide_weight: float = 0.25
    cascade_weight: float = 0.20
    infrastructure_weight: float = 0.15

    minimum_reliable_confidence: float = 0.60

    watch_threshold: float = 0.30
    warning_threshold: float = 0.50
    high_threshold: float = 0.70
    severe_threshold: float = 0.85


@dataclass(frozen=True)
class WardImpactAssessment:
    unit_id: str
    unit_name: str
    unit_type: AdministrativeUnitType

    impact_score: float
    impact_level: ImpactLevel

    confidence: float
    reliable: bool

    population_exposed: int | None

    high_risk_road_count: int
    overflowing_drain_count: int
    critical_facility_count: int

    route_readiness: RouteReadiness

    reasons: tuple[str, ...]