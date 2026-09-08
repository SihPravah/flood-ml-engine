from dataclasses import dataclass
from enum import Enum


class LandslideRiskLevel(str, Enum):
    LOW = "LOW"
    WATCH = "WATCH"
    WARNING = "WARNING"
    HIGH = "HIGH"
    SEVERE = "SEVERE"


class LandslideDataProvenance(str, Enum):
    OBSERVED = "OBSERVED"
    VERIFIED = "VERIFIED"
    DERIVED = "DERIVED"
    ESTIMATED = "ESTIMATED"
    MISSING = "MISSING"


@dataclass(frozen=True)
class LandslideInputs:
    """
    Inputs used to estimate rainfall-triggered landslide
    susceptibility for one slope/terrain zone.

    slope_degrees:
        Terrain slope angle in degrees.

        This is NOT the same quantity as the dimensionless
        slope_fraction used by the Kirpich concentration-time
        equation.

    soil_saturation:
        Normalized wetness in [0, 1].

    rain_1h_mm:
        Rainfall accumulated over the most recent hour.

    rain_24h_mm:
        Rainfall accumulated over the most recent 24 hours.

    api_mm:
        Antecedent precipitation indicator.

    historical_landslide_score:
        Normalized evidence from historical landslide inventory.

        0:
            little/no known historical evidence

        1:
            very strong historical evidence

    land_cover_disturbance_score:
        Normalized susceptibility contribution from sparse
        vegetation, exposed soil, construction/disturbance, etc.

    data_confidence:
        Reliability of the input evidence.

    historical_inventory_provenance:
        Whether historical-landslide evidence is verified,
        derived, estimated, etc.
    """

    zone_id: str

    slope_degrees: float

    soil_saturation: float

    rain_1h_mm: float
    rain_24h_mm: float
    api_mm: float

    historical_landslide_score: float
    land_cover_disturbance_score: float

    data_confidence: float

    historical_inventory_provenance: (
        LandslideDataProvenance
    ) = LandslideDataProvenance.VERIFIED

    def __post_init__(self) -> None:
        if not self.zone_id.strip():
            raise ValueError(
                "zone_id cannot be empty."
            )

        if not 0.0 <= self.slope_degrees <= 90.0:
            raise ValueError(
                "slope_degrees must be between 0 and 90."
            )

        normalized_fields = {
            "soil_saturation": self.soil_saturation,
            "historical_landslide_score": (
                self.historical_landslide_score
            ),
            "land_cover_disturbance_score": (
                self.land_cover_disturbance_score
            ),
            "data_confidence": self.data_confidence,
        }

        for field_name, value in normalized_fields.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} must be between 0 and 1."
                )

        rainfall_fields = {
            "rain_1h_mm": self.rain_1h_mm,
            "rain_24h_mm": self.rain_24h_mm,
            "api_mm": self.api_mm,
        }

        for field_name, value in rainfall_fields.items():
            if value < 0.0:
                raise ValueError(
                    f"{field_name} cannot be negative."
                )


@dataclass(frozen=True)
class LandslideRiskPolicy:
    """
    Development-time susceptibility weighting policy.

    These values are explicit heuristics for pipeline development,
    not universal geotechnical constants.

    They must eventually be calibrated against historical
    landslide inventories and rainfall-triggered events.
    """

    slope_weight: float = 0.24
    soil_weight: float = 0.20
    short_rain_weight: float = 0.16
    antecedent_rain_weight: float = 0.16
    historical_weight: float = 0.18
    land_cover_weight: float = 0.06

    slope_reference_degrees: float = 45.0

    rain_1h_reference_mm: float = 75.0
    rain_24h_reference_mm: float = 200.0
    api_reference_mm: float = 150.0

    minimum_reliable_confidence: float = 0.60

    watch_threshold: float = 0.30
    warning_threshold: float = 0.50
    high_threshold: float = 0.70
    severe_threshold: float = 0.85


@dataclass(frozen=True)
class LandslideAssessment:
    zone_id: str

    susceptibility_score: float
    risk_level: LandslideRiskLevel

    confidence: float
    reliable: bool

    slope_component: float
    soil_component: float
    short_rain_component: float
    antecedent_rain_component: float
    historical_component: float
    land_cover_component: float

    reasons: tuple[str, ...]