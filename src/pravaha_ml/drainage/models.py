from dataclasses import dataclass
from enum import Enum


class DrainCondition(str, Enum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    POOR = "POOR"
    UNKNOWN = "UNKNOWN"


class DrainCapacityProvenance(str, Enum):
    VERIFIED = "VERIFIED"
    DERIVED = "DERIVED"
    ESTIMATED = "ESTIMATED"


class DrainRiskLevel(str, Enum):
    LOW = "LOW"
    WATCH = "WATCH"
    WARNING = "WARNING"
    HIGH = "HIGH"
    SEVERE = "SEVERE"


@dataclass(frozen=True)
class DrainStaticProfile:
    """
    Static hydraulic/infrastructure information for one drain.

    width_m:
        Internal hydraulic width.

    depth_m:
        Effective hydraulic depth.

    slope_fraction:
        Dimensionless longitudinal slope.

        Example:
            2% slope -> 0.02

    manning_roughness:
        Manning's n coefficient.

    blockage_fraction:
        Fraction of the hydraulic cross-section considered
        unavailable due to blockage/silt/debris.

        0.0 -> completely clear
        1.0 -> fully blocked

    condition:
        Infrastructure/maintenance condition.

    capacity_provenance:
        Whether geometric/capacity information is verified,
        derived, or estimated.
    """

    drain_id: str

    width_m: float
    depth_m: float
    slope_fraction: float
    manning_roughness: float

    blockage_fraction: float

    condition: DrainCondition
    capacity_provenance: DrainCapacityProvenance

    catchment_id: str | None = None

    def __post_init__(self) -> None:
        if not self.drain_id.strip():
            raise ValueError(
                "drain_id cannot be empty."
            )

        if self.width_m <= 0.0:
            raise ValueError(
                "width_m must be greater than 0."
            )

        if self.depth_m <= 0.0:
            raise ValueError(
                "depth_m must be greater than 0."
            )

        if self.slope_fraction <= 0.0:
            raise ValueError(
                "slope_fraction must be greater than 0."
            )

        if self.manning_roughness <= 0.0:
            raise ValueError(
                "manning_roughness must be greater than 0."
            )

        if not 0.0 <= self.blockage_fraction <= 1.0:
            raise ValueError(
                "blockage_fraction must be between 0 and 1."
            )


@dataclass(frozen=True)
class DrainDynamicLoad:
    """
    Dynamic hydraulic loading on a drain.

    estimated_inflow_m3_per_s:
        Current estimated incoming discharge.

    catchment_runoff_mm:
        Current hydrology-engine runoff feature.

    rainfall_mm_per_hr:
        Current representative rainfall intensity.

    data_confidence:
        Confidence in the dynamic inputs.
    """

    estimated_inflow_m3_per_s: float
    catchment_runoff_mm: float
    rainfall_mm_per_hr: float

    data_confidence: float

    def __post_init__(self) -> None:
        if self.estimated_inflow_m3_per_s < 0.0:
            raise ValueError(
                "estimated_inflow_m3_per_s cannot be negative."
            )

        if self.catchment_runoff_mm < 0.0:
            raise ValueError(
                "catchment_runoff_mm cannot be negative."
            )

        if self.rainfall_mm_per_hr < 0.0:
            raise ValueError(
                "rainfall_mm_per_hr cannot be negative."
            )

        if not 0.0 <= self.data_confidence <= 1.0:
            raise ValueError(
                "data_confidence must be between 0 and 1."
            )


@dataclass(frozen=True)
class DrainCapacityResult:
    drain_id: str

    theoretical_capacity_m3_per_s: float
    effective_capacity_m3_per_s: float

    blockage_reduction_fraction: float
    condition_factor: float


@dataclass(frozen=True)
class DrainRiskAssessment:
    drain_id: str

    estimated_inflow_m3_per_s: float
    effective_capacity_m3_per_s: float

    capacity_utilization: float
    overflow_probability: float

    risk_level: DrainRiskLevel

    data_confidence: float

    overflow_expected: bool

    reasons: tuple[str, ...]